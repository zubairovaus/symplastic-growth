"""
Symplastic growth model — Python reimplementation and extension.
Core physics: osmotic pressure, turgor, cell wall mechanics, segment dynamics.
"""

from .params import GrowthParams
from .model import growth_rhs, cell_li_derivative, segment_derivative, posm, pturg
from .simulator import SymplasticSimulator, SimulatorResult, run_until_length, run_until_length_multi, run_step_multi
from .state import CellPhase, CellState, LeafState, create_initial_leaf, align_initial_file_lengths, ensure_li_geq_l_after_segments
from .segments import (
    SegmentState,
    build_segments_from_leaf,
    l_from_segments,
    seg_to_cells_flat,
    join_fragments_one,
    collect_segments,
)
from .visualization import leaf_geometry, leaf_geometry_per_cell, draw_leaf, plot_lengths_evolution
from .config import load_params, save_params, params_to_dict, params_from_dict
from .division import (
    cell_division_loop,
    apply_division,
    find_cell_to_divide,
    update_phases_k1t_to_k2t,
    update_phases_k2t_to_k3t,
    sample_division_alpha,
)

__all__ = [
    "GrowthParams",
    "growth_rhs",
    "cell_li_derivative",
    "segment_derivative",
    "posm",
    "pturg",
    "SymplasticSimulator",
    "run_until_length",
    "run_until_length_multi",
    "run_step_multi",
    "SimulatorResult",
    "CellPhase",
    "CellState",
    "LeafState",
    "create_initial_leaf",
    "align_initial_file_lengths",
    "ensure_li_geq_l_after_segments",
    "SegmentState",
    "build_segments_from_leaf",
    "l_from_segments",
    "seg_to_cells_flat",
    "join_fragments_one",
    "collect_segments",
    "leaf_geometry",
    "leaf_geometry_per_cell",
    "draw_leaf",
    "plot_lengths_evolution",
    "cell_division_loop",
    "apply_division",
    "find_cell_to_divide",
    "update_phases_k1t_to_k2t",
    "update_phases_k2t_to_k3t",
    "sample_division_alpha",
    "load_params",
    "save_params",
    "params_to_dict",
    "params_from_dict",
]
