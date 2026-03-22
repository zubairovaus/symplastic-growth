"""
Visualization: leaf geometry, DrawLeafOsm, DrawLeafTurgor, and unified draw_leaf API.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal, TYPE_CHECKING
from .state import LeafState
from .segments import SegmentState
from .params import GrowthParams

if TYPE_CHECKING:
    from .simulator import SimulatorResult


def leaf_geometry(
    leaf: LeafState,
    koef: float = 1.0,
) -> List[Tuple[float, float, float, float]]:
    """
    For each cell (in flat order: file 0, then file 1, ...) return rectangle (x_min, x_max, y_min, y_max).
    x = cumulative width of files (each file has width = cell width * koef).
    y = cumulative length l along the file.
    """
    rects = []
    x_left = 0.0
    for f in leaf.files:
        w = f[0].width * koef if f else 0.0
        x_right = x_left + w
        y_bottom = 0.0
        for c in f:
            y_top = y_bottom + c.l
            rects.append((x_left, x_right, y_bottom, y_top))
            y_bottom = y_top
        x_left = x_right
    return rects


def leaf_geometry_per_cell(
    leaf: LeafState,
    koef: float = 1.0,
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Like leaf_geometry but returns (file_i, cell_j, x_min, x_max, y_min, y_max) for each cell.
    """
    out = []
    x_left = 0.0
    for file_i, f in enumerate(leaf.files):
        w = f[0].width * koef if f else 0.0
        x_right = x_left + w
        y_bottom = 0.0
        for cell_j, c in enumerate(f):
            y_top = y_bottom + c.l
            out.append((file_i, cell_j, x_left, x_right, y_bottom, y_top))
            y_bottom = y_top
        x_left = x_right
    return out


def _color_map_rainbow(norm_val: float) -> Tuple[float, float, float]:
    """Map value in [0, 1] to RGB (rainbow: 0=red, 0.5=green, 1=blue)."""
    # Simple HSV-like: H = (1 - norm_val) * 2/3 (so 0->red, 1/3->yellow, 2/3->green, 1->blue)
    from matplotlib import colors
    return colors.hsv_to_rgb((2.0 / 3.0 * (1.0 - norm_val), 1.0, 1.0))


def draw_leaf(
    leaf: LeafState,
    seg: Optional[SegmentState] = None,
    params: Optional[GrowthParams] = None,
    mode: Literal["geometry", "osm", "turgor", "phase"] = "geometry",
    ax=None,
    koef: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
):
    """
    Draw the leaf: rectangles per cell, optional color by osm, turgor, or phase.
    - geometry: outline only (no color scale).
    - osm: color = alph * (li - l) / l, normalized to [vmin, vmax] or [1, 8].
    - turgor: color = m_young * s_cw / r^2 * (l - lr) / lr, normalized to [vmin, vmax] or [1, 200].
    - phase: color by cell cycle phase (K1t, K2t, K3t).
    Returns the matplotlib Axes (or creates one).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from .state import CellPhase

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))
    if params is None and mode not in ("geometry", "phase"):
        from .params import GrowthParams
        params = GrowthParams()

    geos = leaf_geometry_per_cell(leaf, koef=koef)
    if mode == "geometry":
        for _fi, _fj, xl, xr, yb, yt in geos:
            ax.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, fill=False, edgecolor="gray", linewidth=0.5))
    elif mode == "phase":
        phase_colors = {CellPhase.K1t: (0.9, 0.3, 0.2), CellPhase.K2t: (0.2, 0.7, 0.3), CellPhase.K3t: (0.2, 0.4, 0.9)}
        for file_i, cell_j, xl, xr, yb, yt in geos:
            c = leaf.get_cell(file_i, cell_j)
            ph = c.phase if isinstance(c.phase, CellPhase) else CellPhase(c.phase)
            rgb = phase_colors.get(ph, (0.6, 0.6, 0.6))
            ax.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, facecolor=rgb, edgecolor="white", linewidth=0.2))
    else:
        vals = []
        for file_i, cell_j, xl, xr, yb, yt in geos:
            c = leaf.get_cell(file_i, cell_j)
            if mode == "osm":
                v = params.alph * (c.li - c.l) / c.l if c.l > 0 else 0.0
            else:
                r = params.r(file_i)
                s_cw = params.effective_s_cw(file_i)
                v = params.m_young * s_cw / (r ** 2) * (c.l - c.lr) / c.lr if c.lr > 0 else 0.0
            vals.append(v)
        vals = np.array(vals)
        if vmin is None:
            vmin = 1.0 if mode == "osm" else 1.0
        if vmax is None:
            vmax = 8.0 if mode == "osm" else 200.0
        span = vmax - vmin
        for (file_i, cell_j, xl, xr, yb, yt), v in zip(geos, vals):
            norm = np.clip((v - vmin) / span, 0.0, 1.0) if span > 0 else 0.0
            rgb = _color_map_rainbow(norm)
            ax.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, facecolor=rgb, edgecolor="white", linewidth=0.2))
    total_width = sum(leaf.files[i][0].width for i in range(leaf.n_cell_files)) * koef if leaf.files else 1.0
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, leaf.total_length_per_file()[0] if leaf.files else 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (width)")
    ax.set_ylabel("y (length)")
    titles = {"geometry": "Leaf geometry", "osm": "Osmotic (li-l)/l", "turgor": "Turgor (l-lr)/lr", "phase": "Phase (K1t/K2t/K3t)"}
    ax.set_title(titles.get(mode, mode))
    return ax


def plot_lengths_evolution(
    res: "SimulatorResult",
    snapshots: Optional[List[Tuple[LeafState, SegmentState, float]]] = None,
    ax=None,
):
    """
    Plot total length L(t) and, if snapshots are provided, mean l, li, lr over time.
    res: SimulatorResult from run_until_length_multi (has t, total_length).
    snapshots: optional list of (leaf, seg, t) from on_step_callback.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    t = res.t
    L = res.total_length
    ax.plot(t, L, "k-", label="L (total length)", linewidth=2)
    if snapshots:
        times = [s[2] for s in snapshots]
        mean_l = []
        mean_li = []
        mean_lr = []
        for leaf, _seg, _t in snapshots:
            cells = leaf.all_cells_flat()
            if cells:
                mean_l.append(np.mean([c.l for c in cells]))
                mean_li.append(np.mean([c.li for c in cells]))
                mean_lr.append(np.mean([c.lr for c in cells]))
            else:
                mean_l.append(0.0)
                mean_li.append(0.0)
                mean_lr.append(0.0)
        ax.plot(times, mean_l, "b-", label="mean l", alpha=0.8)
        ax.plot(times, mean_li, "g-", label="mean li", alpha=0.8)
        ax.plot(times, mean_lr, "r-", label="mean lr", alpha=0.8)
    ax.set_xlabel("t (h)")
    ax.set_ylabel("length")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
