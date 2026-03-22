"""
High-level simulator: build initial state, build li(t) functions, integrate until target length.
Supports single-file (legacy) and multi-file (LeafState + SegmentState).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Callable, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field

from .params import GrowthParams
from .model import growth_rhs, li_expt, li_elong
from .state import LeafState, CellPhase, ensure_li_geq_l_after_segments
from .segments import (
    SegmentState,
    build_segments_from_leaf,
    seg_to_cells_flat,
    collect_segments,
    join_fragments_one,
)
from .division import (
    cell_division_loop,
    update_phases_k1t_to_k2t,
    update_phases_k2t_to_k3t,
)


@dataclass
class SimulatorResult:
    """Result of a simulation run."""
    t: np.ndarray
    lr: np.ndarray      # (n_t, n_cells)
    segm: np.ndarray    # (n_t, n_segm)
    total_length: np.ndarray  # (n_t,) sum of first-file segment lengths
    params: GrowthParams
    final_leaf: Optional[LeafState] = field(default=None, repr=False)
    final_seg: Optional[SegmentState] = field(default=None, repr=False)
    lineage: Optional[List[Tuple[float, int, int, int, int, int]]] = None  # (ti, file_i, cell_j, parent_id, id_a, id_b)


def make_li_func(
    t0: float, li0: float, phase: str,
    params: GrowthParams,
    file_index: int = 0,
) -> Callable[[float], float]:
    """
    Return li(t) for one cell. K1t/K2t: slow growth (li_expt toward div_threshold).
    K3t: fast elongation (li_elong). Legacy 'expt'/'elong' also supported.
    Capped at params.max_cell_length[file_index] when set.
    """
    div_thresh = params.div_threshold[file_index]
    min_div = params.min_div_cell_size[file_index]
    max_elong = params.max_elong_cell_size[file_index]
    tc = params.t_cell_cycle
    te = params.t_cell_elongation
    cap_list = getattr(params, "max_cell_length", None)
    cap = cap_list[file_index] if cap_list and file_index < len(cap_list) else None

    use_expt = phase in ("K1t", "K2t") or phase == "expt"
    if use_expt:
        def li(t):
            raw = li_expt(t, li0, t0, div_thresh, min_div, tc)
            return min(raw, cap) if cap is not None else raw
    else:
        def li(t):
            raw = li_elong(t, li0, t0, div_thresh, max_elong, tc, te)
            return min(raw, cap) if cap is not None else raw
    return li


def build_single_file_initial(
    n_cells: int,
    params: GrowthParams,
    lr0: Optional[np.ndarray] = None,
    segm0: Optional[np.ndarray] = None,
    phase: str = "expt",
) -> Tuple[np.ndarray, List[Callable[[float], float]], List[List[int]]]:
    """
    One file: each segment = one cell. Initial lr and segm equal, li(t) = expt or elong.
    Returns: y0, li_funcs, seg_to_cells (seg_to_cells[k] = [k]).
    """
    file_ix = 0
    div_thresh = params.div_threshold[file_ix]
    min_div = params.min_div_cell_size[file_ix]
    if lr0 is None:
        lr0 = np.full(n_cells, min_div * 1.2)  # slightly above min
    if segm0 is None:
        segm0 = lr0.copy()
    y0 = np.concatenate([lr0, segm0])
    li_funcs = []
    for j in range(n_cells):
        f = make_li_func(0.0, lr0[j], phase, params, file_ix)
        li_funcs.append(f)
    seg_to_cells = [[k] for k in range(n_cells)]
    return y0, li_funcs, seg_to_cells


def run_step(
    t0: float, t1: float, y0: np.ndarray,
    params: GrowthParams,
    li_funcs: List[Callable[[float], float]],
    seg_to_cells: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate from t0 to t1, return (t, y) with y shape (n_t, n_state)."""
    n_cells = len(li_funcs)

    def rhs(t, y):
        return growth_rhs(t, y, params, li_funcs, seg_to_cells)

    sol = solve_ivp(
        rhs, (t0, t1), y0,
        method="RK45",
        dense_output=True,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")
    t = np.linspace(t0, t1, max(2, int((t1 - t0) * 2)))
    y = sol.sol(t)
    return t, y.T  # (n_t, n_state)


class SymplasticSimulator:
    """
    Run symplastic growth: integrate ODEs step-by-step until total length >= Ly.
    Simplified: single file, no cell division (fixed n_cells).
    """

    def __init__(
        self,
        params: GrowthParams,
        n_cells: int = 10,
        dt: float = 1.0,
    ):
        self.params = params
        self.n_cells = n_cells
        self.dt = dt
        self.y0, self.li_funcs, self.seg_to_cells = build_single_file_initial(
            n_cells, params
        )
        self.n_segm = len(self.seg_to_cells)

    def run(self, t_end: float, t0: float = 0.0) -> SimulatorResult:
        """Integrate from t0 to t_end (single shot, no stopping at Ly)."""
        t, y = run_step(
            t0, t_end, self.y0, self.params,
            self.li_funcs, self.seg_to_cells,
        )
        lr = y[:, : self.n_cells]
        segm = y[:, self.n_cells :]
        total_length = segm.sum(axis=1)
        return SimulatorResult(t=t, lr=lr, segm=segm, total_length=total_length, params=self.params)

    def run_until_length(self, Ly: float, t0: float = 0.0, max_steps: int = 500) -> SimulatorResult:
        """Step until total length >= Ly or max_steps."""
        y = self.y0.copy()
        t_cur = t0
        all_t = [t_cur]
        all_y = [y.copy()]
        for _ in range(max_steps):
            t_next = t_cur + self.dt
            t, y_arr = run_step(
                t_cur, t_next, y, self.params,
                self.li_funcs, self.seg_to_cells,
            )
            y = y_arr[-1]
            total = y[self.n_cells :].sum()
            all_t.append(t_next)
            all_y.append(y.copy())
            t_cur = t_next
            if total >= Ly:
                break
        all_t = np.array(all_t)
        all_y = np.array(all_y)
        lr = all_y[:, : self.n_cells]
        segm = all_y[:, self.n_cells :]
        total_length = segm.sum(axis=1)
        return SimulatorResult(
            t=all_t, lr=lr, segm=segm, total_length=total_length, params=self.params
        )


def _dli_dt_constant(phase: str, file_index: int, params: GrowthParams) -> float:
    """Analytical d(li)/dt: li_expt slope or li_elong slope. Used to avoid finite diff in ODE."""
    th = min(file_index, len(params.div_threshold) - 1)
    if th < 0:
        return 0.0
    div_thresh = params.div_threshold[th]
    min_div = params.min_div_cell_size[th]
    max_elong = params.max_elong_cell_size[th]
    tc, te = params.t_cell_cycle, params.t_cell_elongation
    use_expt = phase in ("K1t", "K2t") or phase == "expt"
    if use_expt:
        return (div_thresh - min_div) / tc if tc > 0 else 0.0
    denom = tc - te
    if denom <= 0:
        return 0.0
    return (div_thresh - max_elong) / denom


def build_ode_state_from_leaf(
    leaf: LeafState,
    seg: SegmentState,
    params: GrowthParams,
) -> Tuple[np.ndarray, List[Callable[[float], float]], List[List[int]], List[int], List[float]]:
    """
    Build flat ODE state y0, li_funcs, seg_to_cells, file_index_per_cell, dli_dt_const from LeafState and SegmentState.
    y0 = [lr_1..lr_N, segm_1..segm_K]. file_index_per_cell[i] = file index for cell i in flat order.
    dli_dt_const[i] = analytical d(li)/dt for cell i (avoids finite diff in RHS, faster).
    """
    lr_flat = leaf.to_lr_flat()
    segm = np.array(seg.lengths, dtype=np.float64)
    y0 = np.concatenate([lr_flat, segm])
    li_funcs = []
    file_index_per_cell = []
    dli_dt_const = []
    for file_i, f in enumerate(leaf.files):
        for c in f:
            phase = c.phase.value if isinstance(c.phase, CellPhase) else c.phase
            li_funcs.append(make_li_func(c.t0, c.li, phase, params, c.theta))
            file_index_per_cell.append(file_i)
            # Use same param index as make_li_func (theta), so dli_dt matches growth law
            th = min(c.theta, len(params.div_threshold) - 1)
            dli_dt_const.append(_dli_dt_constant(phase, th, params))
    seg_to_cells = seg_to_cells_flat(leaf, seg)
    return y0, li_funcs, seg_to_cells, file_index_per_cell, dli_dt_const


def apply_ode_result_to_leaf(
    y: np.ndarray,
    leaf: LeafState,
    seg: SegmentState,
    params: GrowthParams,
) -> None:
    """Update leaf.lr and seg.lengths from ODE state y; then assign l from segments (in place)."""
    n_cells = leaf.total_cells
    leaf.update_lr_from_flat(y[:n_cells])
    seg_len = np.asarray(y[n_cells:], dtype=np.float64)
    # Safeguard: non-positive or NaN segment lengths would give l=0 cells; clamp to min_fragment_length
    min_len = getattr(params, "min_fragment_length", 0.1)
    seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0), seg_len, min_len)
    seg.lengths[:] = seg_len.tolist()
    collect_segments(leaf, seg)


def run_step_multi(
    t0: float,
    t1: float,
    leaf: LeafState,
    seg: SegmentState,
    params: GrowthParams,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    **solve_ivp_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One ODE step for multi-file: integrate from t0 to t1, update leaf and seg in place.
    Returns (t_array, y_array) for recording. Solver options: method, rtol, atol, and any solve_ivp kwargs.
    """
    y0, li_funcs, seg_to_cells, file_index_per_cell, dli_dt_const = build_ode_state_from_leaf(leaf, seg, params)

    def rhs(t, y):
        return growth_rhs(t, y, params, li_funcs, seg_to_cells, file_index_per_cell, dli_dt_const)

    kwargs: Dict[str, Any] = {"dense_output": True, "method": method, "rtol": rtol, "atol": atol, **solve_ivp_kwargs}
    sol = solve_ivp(rhs, (t0, t1), y0, **kwargs)
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")
    t = np.linspace(t0, t1, max(2, int((t1 - t0) * 2)))
    y = sol.sol(t).T
    apply_ode_result_to_leaf(y[-1], leaf, seg, params)
    # Keep leaf.li in sync with growth law: c.li is the target length at current time
    cells = leaf.all_cells_flat()
    for i, c in enumerate(cells):
        if i < len(li_funcs):
            c.li = float(li_funcs[i](t1))
    return t, y


def run_until_length(
    params: GrowthParams,
    Ly: float,
    n_cells: int = 10,
    dt: float = 1.0,
    t0: float = 0.0,
    initial_leaf: Optional[LeafState] = None,
) -> SimulatorResult:
    """
    Convenience: create simulator and run until length Ly.
    If initial_leaf is provided (multi-file), seg is built from it and multi-file path is used.
    """
    if initial_leaf is not None:
        return run_until_length_multi(params, Ly, initial_leaf, dt=dt, t0=t0)
    sim = SymplasticSimulator(params, n_cells=n_cells, dt=dt)
    return sim.run_until_length(Ly, t0=t0, max_steps=max(1000, int(Ly)))


def run_until_length_multi(
    params: GrowthParams,
    Ly: float,
    initial_leaf: LeafState,
    dt: float = 1.0,
    t0: float = 0.0,
    max_steps: int = 2000,
    with_division: bool = True,
    rng: Optional[np.random.Generator] = None,
    on_step_callback: Optional[Callable[[LeafState, SegmentState, float], None]] = None,
    export_path: Optional[str] = None,
    record_lineage: bool = False,
    verbose: bool = False,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    **solver_kwargs: Any,
) -> SimulatorResult:
    """
    Multi-file run: step ODE until total length (first file) >= Ly.
    If with_division: after each ODE step run cell division (K1t, li>=smax),
    then JoinFragments1, then phase updates K1t->K2t, K2t->K3t.
    on_step_callback(leaf, seg, t) called after each step; if export_path is set, state is saved per step.
    If record_lineage, result.lineage contains (ti, file_i, cell_j, parent_id, id_a, id_b) per division.
    If verbose, print progress after each step (step index, t, total length, cell count).
    Solver options: method, rtol, atol, **solver_kwargs passed to solve_ivp.
    """
    if rng is None:
        rng = np.random.default_rng()
    leaf = initial_leaf.copy()
    seg = build_segments_from_leaf(leaf)
    leaf, seg = join_fragments_one(leaf, seg, epsilon=params.min_fragment_length)
    collect_segments(leaf, seg)
    ensure_li_geq_l_after_segments(leaf)

    lineage: List[Tuple[float, int, int, int, int, int]] = [] if record_lineage else None  # type: ignore[assignment]
    if record_lineage:
        lineage = []

    step_index = [0]  # mutable so _on_step can increment

    def _on_step(t_step: float) -> None:
        if on_step_callback is not None:
            on_step_callback(leaf, seg, t_step)
        if export_path:
            import pickle
            import os
            os.makedirs(export_path, exist_ok=True)
            step_file = os.path.join(export_path, f"step_{step_index[0]:04d}.pkl")
            with open(step_file, "wb") as f:
                pickle.dump({"leaf": leaf.copy(), "seg": seg, "t": t_step}, f)
            step_index[0] += 1

    all_t = [t0]
    all_lr = [leaf.to_lr_flat().copy()]
    all_segm = [np.array(seg.lengths, dtype=np.float64).copy()]
    total_lengths = [sum(c.l for c in leaf.files[0])]
    if on_step_callback or export_path:
        _on_step(t0)

    t_cur = t0
    for _ in range(max_steps):
        t_next = t_cur + dt
        run_step_multi(t_cur, t_next, leaf, seg, params, method=method, rtol=rtol, atol=atol, **solver_kwargs)

        # Снимок до деления: в момент t_next состояние после ODE ещё содержит материнскую клетку;
        # после деления — только дочерние. Два снимка на одном t_next дают на графике
        # исчезновение матери и появление дочерних в один и тот же момент времени.
        if on_step_callback or export_path:
            _on_step(t_next)

        if with_division:
            cell_division_loop(leaf, seg, params, t_next, rng, lineage_out=lineage)
            collect_segments(leaf, seg)
        leaf, seg = join_fragments_one(leaf, seg, epsilon=params.min_fragment_length)
        if with_division:
            update_phases_k1t_to_k2t(leaf, params)
            update_phases_k2t_to_k3t(leaf, params, t_next)

        # Rebuild segments from leaf so segment count stays O(cells) instead of blowing up after divisions.
        # Each division splits every segment containing the cell, so without this the ODE state grows unbounded.
        # Invariant: build_segments_from_leaf uses current leaf.l (cumsum boundaries); the new segment
        # lengths partition the same total length and l_from_segments(leaf, new_seg) recovers the same l,
        # so physics (posm, pturg, growth) is unchanged — we only re-discretize the same leaf state.
        l_before = leaf.to_l_flat().copy()
        seg = build_segments_from_leaf(leaf)
        collect_segments(leaf, seg)
        l_after = leaf.to_l_flat()
        if not np.allclose(l_before, l_after, rtol=1e-9, atol=1e-12):
            raise AssertionError(
                "Segment rebuild must preserve cell lengths l. "
                f"max |l_before - l_after| = {np.max(np.abs(l_before - l_after))}"
            )

        total_len = sum(c.l for c in leaf.files[0])
        all_t.append(t_next)
        all_lr.append(leaf.to_lr_flat().copy())
        all_segm.append(np.array(seg.lengths, dtype=np.float64).copy())
        total_lengths.append(total_len)
        if verbose:
            print(f"  шаг {len(total_lengths)}, t={t_next:.1f}, L={total_len:.2f}, клеток={leaf.total_cells}")
        if on_step_callback or export_path:
            _on_step(t_next)
        t_cur = t_next
        if total_len >= Ly:
            break

    # Pad to constant size for lr/segm arrays (variable after division)
    max_cells = max(arr.size for arr in all_lr)
    max_segm = max(arr.size for arr in all_segm)
    lr_padded = np.zeros((len(all_lr), max_cells))
    segm_padded = np.zeros((len(all_segm), max_segm))
    for i, arr in enumerate(all_lr):
        lr_padded[i, : arr.size] = arr
    for i, arr in enumerate(all_segm):
        segm_padded[i, : arr.size] = arr

    return SimulatorResult(
        t=np.array(all_t),
        lr=lr_padded,
        segm=segm_padded,
        total_length=np.array(total_lengths),
        params=params,
        final_leaf=leaf,
        final_seg=seg,
        lineage=lineage if record_lineage else None,
    )
