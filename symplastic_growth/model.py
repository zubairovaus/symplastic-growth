"""
Core ODEs of the symplastic growth model.
Equations: osmotic pressure, turgor, wall relaxation (lr), segment lengths (segm).
Optional: install numba for parallel RHS (uses more CPU cores).
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from .params import GrowthParams

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _cell_lr_rhs_parallel(
        l: np.ndarray, lr: np.ndarray, li_vals: np.ndarray, dli_dt: np.ndarray,
        r_arr: np.ndarray, s_cw_arr: np.ndarray,
        m_young: float, thresh: float, etha: float, k_heaviside: float,
    ) -> np.ndarray:
        """Parallel per-cell dlr_dt; called with precomputed arrays."""
        n = l.shape[0]
        out = np.empty(n)
        for i in prange(n):
            p = (m_young * s_cw_arr[i] / (r_arr[i] ** 2) * (l[i] - lr[i]) / lr[i]) if lr[i] > 0 else 0.0
            z1 = min(max(k_heaviside * (p - thresh), -500.0), 500.0)
            h1 = 1.0 / (1.0 + np.exp(-z1))
            z2 = min(max(k_heaviside * (li_vals[i] - l[i]), -500.0), 500.0)
            h2 = 1.0 / (1.0 + np.exp(-z2))
            dp = max(0.0, p - thresh)
            out[i] = etha * dli_dt[i] * (dp ** 3) * h1 * h2
        return out


def posm(li: float, l: float, alph: float) -> float:
    """Osmotic pressure: P_osm = alph * (li - l) / l."""
    if l <= 0:
        return 0.0
    return alph * (li - l) / l


def pturg(l: float, lr: float, r: float, m_young: float, s_cw: float) -> float:
    """Turgor (wall stress): P_turg = (m_young * s_cw / r^2) * (l - lr) / lr."""
    if lr <= 0:
        return 0.0
    return m_young * s_cw / (r ** 2) * (l - lr) / lr


def _posm_vec(li: np.ndarray, l: np.ndarray, alph: float) -> np.ndarray:
    """Vectorized osmotic pressure. li, l same shape."""
    out = np.where(l > 0, alph * (li - l) / l, 0.0)
    return np.asarray(out, dtype=np.float64)


def _pturg_vec(l: np.ndarray, lr: np.ndarray, r: np.ndarray, m_young: float, s_cw: np.ndarray) -> np.ndarray:
    """Vectorized turgor. l, lr, r, s_cw same shape."""
    out = np.where(lr > 0, m_young * s_cw / (r ** 2) * (l - lr) / lr, 0.0)
    return np.asarray(out, dtype=np.float64)


def _smooth_unit_step_vec(x: np.ndarray, k: float = 50.0) -> np.ndarray:
    """Vectorized smooth Heaviside."""
    z = np.clip(k * x, -500.0, 500.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


def smooth_unit_step(x: float, k: float = 50.0) -> float:
    """Smooth approximation to Heaviside: 1 / (1 + exp(-k*x)). Clipped to avoid overflow."""
    z = np.clip(k * x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def li_expt(t: float, y0: float, t0: float, div_thresh: float, min_div: float, t_cycle: float) -> float:
    """Slow growth phase: li grows linearly from y0 toward div_thresh."""
    return y0 + (t - t0) * (div_thresh - min_div) / t_cycle


def li_elong(
    t: float, y0: float, t0: float,
    div_thresh: float, max_elong: float,
    t_cycle: float, t_elong: float
) -> float:
    """
    Fast elongation phase (Flin from original):
    li = (DivThresh - MaxElong)/(Tcycle - Telong)*(t-t0) + (Tcycle*MaxElong - Telong*DivThresh)/(Tcycle - Telong) + y0.
    """
    denom = t_cycle - t_elong
    if denom <= 0:
        return y0
    slope = (div_thresh - max_elong) / denom
    intercept = (t_cycle * max_elong - t_elong * div_thresh) / denom
    return slope * (t - t0) + intercept + y0


def _current_lengths_from_segments(
    state: np.ndarray, n_cells: int, seg_to_cells: List[List[int]]
) -> np.ndarray:
    """
    Compute current length l for each cell from segment lengths (segm).
    Each segment contributes its full length to each cell in the segment (one per file);
    do NOT divide by len(cells), so total leaf length = sum(segm).
    """
    n_segm = len(seg_to_cells)
    segm = state[n_cells : n_cells + n_segm]
    l = np.zeros(n_cells)
    for k, cells in enumerate(seg_to_cells):
        for c in cells:
            if 0 <= c < n_cells:
                l[c] += segm[k]
    return l


def _file_index(flat_cell: int, n_cells: int, file_index_per_cell: List[int], params: GrowthParams) -> int:
    """File index for a cell by flat index. Uses file_index_per_cell if provided, else modulo (single-file)."""
    if file_index_per_cell is not None and flat_cell < len(file_index_per_cell):
        return file_index_per_cell[flat_cell]
    return flat_cell % params.n_cell_files


def _rhs_shared_arrays(
    t: float,
    state: np.ndarray,
    n_cells: int,
    params: GrowthParams,
    li_funcs: List[Callable[[float], float]],
    seg_to_cells: List[List[int]],
    file_index_per_cell: Optional[List[int]],
    dli_dt_const: Optional[List[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute l, li_vals, r_arr, s_cw_arr, dli_dt once for use in both derivatives."""
    l = _current_lengths_from_segments(state, n_cells, seg_to_cells)
    if dli_dt_const is not None and len(dli_dt_const) >= n_cells:
        dli_dt = np.array(dli_dt_const[:n_cells], dtype=np.float64)
    else:
        dli_dt = np.array([(li_funcs[i](t + 1e-6) - li_funcs[i](t - 1e-6)) / 2e-6 for i in range(n_cells)], dtype=np.float64)
    fi = np.array([_file_index(i, n_cells, file_index_per_cell, params) for i in range(n_cells)], dtype=np.intp)
    cw = params.cell_width
    r_arr = np.array([cw[f] if f < len(cw) else cw[-1] for f in fi], dtype=np.float64)
    s_cw_arr = 4.0 * r_arr
    li_vals = np.array([li_funcs[i](t) for i in range(n_cells)], dtype=np.float64)
    return l, li_vals, r_arr, s_cw_arr, dli_dt


def cell_li_derivative(
    t: float,
    state: np.ndarray,
    params: GrowthParams,
    li_funcs: List[Callable[[float], float]],
    seg_to_cells: List[List[int]],
    file_index_per_cell: Optional[List[int]] = None,
    dli_dt_const: Optional[List[float]] = None,
    _l: Optional[np.ndarray] = None,
    _li_vals: Optional[np.ndarray] = None,
    _r_arr: Optional[np.ndarray] = None,
    _s_cw_arr: Optional[np.ndarray] = None,
    _dli_dt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    d(lr)/dt for each cell: etha * (dli/dt) * (pturg - thresh)^3 * H(pturg - thresh) * H(li - l).
    If _l, _li_vals, etc. are provided (by growth_rhs), use them to avoid recomputation.
    """
    n_cells = len(li_funcs)
    lr = state[:n_cells]
    if _l is not None and _li_vals is not None and _r_arr is not None and _s_cw_arr is not None and _dli_dt is not None:
        l, li_vals, r_arr, s_cw_arr, dli_dt = _l, _li_vals, _r_arr, _s_cw_arr, _dli_dt
    else:
        l = _current_lengths_from_segments(state, n_cells, seg_to_cells)
        if dli_dt_const is not None and len(dli_dt_const) >= n_cells:
            dli_dt = np.array(dli_dt_const[:n_cells], dtype=np.float64)
        else:
            dli_dt = np.array([(li_funcs[i](t + 1e-6) - li_funcs[i](t - 1e-6)) / 2e-6 for i in range(n_cells)], dtype=np.float64)
        fi = np.array([_file_index(i, n_cells, file_index_per_cell, params) for i in range(n_cells)], dtype=np.intp)
        cw = params.cell_width
        r_arr = np.array([cw[f] if f < len(cw) else cw[-1] for f in fi], dtype=np.float64)
        s_cw_arr = 4.0 * r_arr
        li_vals = np.array([li_funcs[i](t) for i in range(n_cells)], dtype=np.float64)

    if _HAS_NUMBA:
        dlr_dt = _cell_lr_rhs_parallel(
            l, lr, li_vals, dli_dt, r_arr, s_cw_arr,
            params.m_young, params.thresh, params.etha, 50.0,
        )
    else:
        p = _pturg_vec(l, lr, r_arr, params.m_young, s_cw_arr)
        h1 = _smooth_unit_step_vec(p - params.thresh)
        h2 = _smooth_unit_step_vec(li_vals - l)
        dlr_dt = params.etha * dli_dt * np.maximum(0.0, p - params.thresh) ** 3 * h1 * h2
    return dlr_dt


def segment_derivative(
    t: float,
    state: np.ndarray,
    params: GrowthParams,
    li_funcs: List[Callable[[float], float]],
    seg_to_cells: List[List[int]],
    file_index_per_cell: Optional[List[int]] = None,
    _l: Optional[np.ndarray] = None,
    _li_vals: Optional[np.ndarray] = None,
    _r_arr: Optional[np.ndarray] = None,
    _s_cw_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Article Eq. (13): dλk/dt = (λk/N) * Σ (1/l·dl/dt)_free. If _l, _li_vals, etc. provided, use them.
    """
    n_cells = len(li_funcs)
    n_segm = len(seg_to_cells)
    lr = state[:n_cells]
    segm = state[n_cells:]
    if _l is not None and _li_vals is not None and _r_arr is not None and _s_cw_arr is not None:
        l, li_vals, r_arr, s_cw_arr = _l, _li_vals, _r_arr, _s_cw_arr
    else:
        l = _current_lengths_from_segments(state, n_cells, seg_to_cells)
        fi = np.array([_file_index(i, n_cells, file_index_per_cell, params) for i in range(n_cells)], dtype=np.intp)
        cw = params.cell_width
        r_arr = np.array([cw[f] if f < len(cw) else cw[-1] for f in fi], dtype=np.float64)
        s_cw_arr = 4.0 * r_arr
        li_vals = np.array([li_funcs[i](t) for i in range(n_cells)], dtype=np.float64)

    Lw = getattr(params, "Lw", 40.0)
    N = params.n_cell_files
    pom = _posm_vec(li_vals, l, params.alph)
    pt = _pturg_vec(l, lr, r_arr, params.m_young, s_cw_arr)
    contrib = r_arr * Lw * (pom - pt)
    d_segm = np.zeros(n_segm)
    for k, cells in enumerate(seg_to_cells):
        total = sum(contrib[c] for c in cells if c < n_cells)
        d_segm[k] = (segm[k] / N) * total if segm[k] > 0 else 0.0
    return d_segm


def growth_rhs(
    t: float,
    state: np.ndarray,
    params: GrowthParams,
    li_funcs: List[Callable[[float], float]],
    seg_to_cells: List[List[int]],
    file_index_per_cell: Optional[List[int]] = None,
    dli_dt_const: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Full RHS: [d(lr)/dt for all cells, d(segm)/dt for all segments].
    state = [lr_1..lr_n, segm_1..segm_K]. Precomputes l, li_vals, r, s_cw, dli_dt once for both derivatives.
    """
    n_cells = len(li_funcs)
    l, li_vals, r_arr, s_cw_arr, dli_dt = _rhs_shared_arrays(
        t, state, n_cells, params, li_funcs, seg_to_cells, file_index_per_cell, dli_dt_const
    )
    dlr = cell_li_derivative(
        t, state, params, li_funcs, seg_to_cells,
        file_index_per_cell, dli_dt_const,
        _l=l, _li_vals=li_vals, _r_arr=r_arr, _s_cw_arr=s_cw_arr, _dli_dt=dli_dt,
    )
    dsegm = segment_derivative(
        t, state, params, li_funcs, seg_to_cells, file_index_per_cell,
        _l=l, _li_vals=li_vals, _r_arr=r_arr, _s_cw_arr=s_cw_arr,
    )
    return np.concatenate([dlr, dsegm])
