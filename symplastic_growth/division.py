"""
Cell division: find K1t cells with li >= Smax, split into two cells, update segments.
Phase updates: K1t -> K2t (by Sdivision), K2t -> K3t (by Smax, record t0).
"""

import numpy as np
from typing import Optional, Tuple, List
from .state import LeafState, CellState, CellPhase
from .params import GrowthParams
from .segments import SegmentState, l_from_segments
from .model import li_expt, li_elong


def sample_division_alpha(params: GrowthParams, rng: np.random.Generator) -> float:
    """Sample division ratio alpha in [alpha_min, alpha_max] from truncated normal."""
    from scipy.stats import truncnorm
    a = (params.alpha_min - params.alpha_mu) / max(params.alpha_sigma, 1e-9)
    b = (params.alpha_max - params.alpha_mu) / max(params.alpha_sigma, 1e-9)
    x = truncnorm.rvs(a, b, loc=params.alpha_mu, scale=params.alpha_sigma, random_state=rng)
    return float(np.clip(x, params.alpha_min, params.alpha_max))


def update_phases_k1t_to_k2t(leaf: LeafState, params: GrowthParams) -> None:
    """
    K1t -> K2t when cumulative length from start of file >= s_division[file].
    Modifies leaf in place.
    """
    n_files = min(leaf.n_cell_files, len(params.s_division))
    for file_i in range(n_files):
        cum = 0.0
        for cell_j, c in enumerate(leaf.files[file_i]):
            if c.phase == CellPhase.K1t and cum >= params.s_division[file_i]:
                c.phase = CellPhase.K2t
            cum += c.l


def update_phases_k2t_to_k3t(leaf: LeafState, params: GrowthParams, ti: float) -> None:
    """
    K2t -> K3t when li >= smax[theta]; set t0 = ti for elongation phase.
    Set c.li so that li_elong(ti) equals current c.li (no jump: li_elong(t, y0, t0, ...) = intercept + y0 at t=t0).
    Modifies leaf in place.
    """
    tc = params.t_cell_cycle
    te = params.t_cell_elongation
    denom = tc - te
    for file_i, f in enumerate(leaf.files):
        for c in f:
            if c.phase != CellPhase.K2t:
                continue
            th = min(c.theta, len(params.smax) - 1)
            if th < 0:
                continue
            if c.li >= params.smax[th]:
                c.phase = CellPhase.K3t
                c.t0 = ti
                if denom > 0:
                    div_thresh = params.div_threshold[th]
                    max_elong = params.max_elong_cell_size[th]
                    intercept = (tc * max_elong - te * div_thresh) / denom
                    c.li = c.li - intercept  # so li_elong(ti, c.li, ti, ...) = intercept + c.li = old c.li


def _li_at_time(c: CellState, file_i: int, ti: float, params: GrowthParams) -> float:
    """Current target length li(t) at time ti from growth law (expt or elong), capped by max_cell_length."""
    th = min(c.theta, len(params.div_threshold) - 1)
    if th < 0:
        raw = c.li
    else:
        div_thresh = params.div_threshold[th]
        min_div = params.min_div_cell_size[th]
        max_elong = params.max_elong_cell_size[th]
        tc, te = params.t_cell_cycle, params.t_cell_elongation
        if c.phase == CellPhase.K1t or c.phase == CellPhase.K2t:
            raw = li_expt(ti, c.li, c.t0, div_thresh, min_div, tc)
        elif c.phase == CellPhase.K3t:
            raw = li_elong(ti, c.li, c.t0, div_thresh, max_elong, tc, te)
        else:
            raw = c.li
    cap_list = getattr(params, "max_cell_length", None)
    if cap_list and file_i < len(cap_list):
        return min(raw, cap_list[file_i])
    return raw


def find_cell_to_divide(leaf: LeafState, params: GrowthParams, ti: float) -> Optional[Tuple[int, int]]:
    """
    Return (file_i, cell_j) for the first K1t cell with li(ti) >= smax[theta], or None.
    Uses growth law to evaluate li at current time ti.
    """
    for file_i in range(leaf.n_cell_files):
        for cell_j, c in enumerate(leaf.files[file_i]):
            if c.phase != CellPhase.K1t:
                continue
            th = min(c.theta, len(params.smax) - 1)
            if th < 0:
                continue
            li_now = _li_at_time(c, file_i, ti, params)
            if li_now >= params.smax[th]:
                return (file_i, cell_j)
    return None


def _max_cell_id(leaf: LeafState) -> int:
    return max((c.id for f in leaf.files for c in f), default=0)


def apply_division(
    leaf: LeafState,
    seg: SegmentState,
    file_i: int,
    cell_j: int,
    alpha: float,
    ti: float,
    params: GrowthParams,
) -> Tuple[float, int, int, int, int, int]:
    """
    Replace cell (file_i, cell_j) with two cells (alpha and 1-alpha split).
    Update seg: split every segment that contains this cell (multi-file: one cell spans many segments).
    Modifies leaf and seg in place.
    Returns (ti, file_i, cell_j, parent_id, id_a, id_b) for lineage.
    Raises RuntimeError if no segment contains this cell (leaf and seg would be out of sync).
    """
    k_list = [k for k in range(seg.n_segments) if seg.cell_index[k][file_i] == cell_j]
    if not k_list:
        raise RuntimeError(
            f"apply_division: cell (file_i={file_i}, cell_j={cell_j}) has no segment. "
            "Leaf and segment state are out of sync; every cell must be in at least one segment."
        )

    c = leaf.files[file_i][cell_j]
    parent_id = c.id
    w = c.width
    th = c.theta
    mid = _max_cell_id(leaf) + 1
    # Article 2.1.4: initial isosmotic lengths of daughters are d·limax and (1−d)·limax
    limax = params.smax[min(th, len(params.smax) - 1)]
    li_a = alpha * limax
    li_b = (1.0 - alpha) * limax

    cell_a = CellState(
        width=w,
        li=li_a,
        lr=alpha * c.lr,
        l=alpha * c.l,
        theta=th,
        t0=ti,
        phase=CellPhase.K1t,
        id=mid,
    )
    cell_b = CellState(
        width=w,
        li=li_b,
        lr=(1.0 - alpha) * c.lr,
        l=(1.0 - alpha) * c.l,
        theta=th,
        t0=ti,
        phase=CellPhase.K1t,
        id=mid + 1,
    )
    leaf.files[file_i][cell_j] = cell_a
    leaf.files[file_i].insert(cell_j + 1, cell_b)

    # Shift indices for cells after cell_j so new cell_b gets index cell_j+1
    for k in range(seg.n_segments):
        if seg.cell_index[k][file_i] > cell_j:
            seg.cell_index[k][file_i] += 1

    # Split every segment that contained the dividing cell (process high k first so indices stay valid)
    for k in reversed(k_list):
        L = seg.lengths[k]
        row_new = list(seg.cell_index[k])
        row_new[file_i] = cell_j + 1
        seg.lengths[k] = L * alpha
        seg.lengths.insert(k + 1, L * (1.0 - alpha))
        seg.cell_index.insert(k + 1, row_new)

    return (ti, file_i, cell_j, parent_id, mid, mid + 1)


def cell_division_loop(
    leaf: LeafState,
    seg: SegmentState,
    params: GrowthParams,
    ti: float,
    rng: np.random.Generator,
    max_divisions: int = 100,
    lineage_out: Optional[List[Tuple[float, int, int, int, int, int]]] = None,
) -> int:
    """
    Repeatedly find a K1t cell with li >= smax, apply division, until none left or max_divisions.
    Returns number of divisions performed. If lineage_out is provided, appends (ti, file_i, cell_j, parent_id, id_a, id_b) per division.
    """
    count = 0
    for _ in range(max_divisions):
        pos = find_cell_to_divide(leaf, params, ti)
        if pos is None:
            break
        file_i, cell_j = pos
        alpha = sample_division_alpha(params, rng)
        record = apply_division(leaf, seg, file_i, cell_j, alpha, ti, params)
        if lineage_out is not None:
            lineage_out.append(record)
        count += 1
    return count
