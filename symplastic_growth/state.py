"""
State structures for the symplastic growth model: cells and leaf grid.
Compatible with the original Mathematica axiom/str representation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class CellPhase(str, Enum):
    """Cell cycle phase: meristem/pre-division, division zone, elongation."""
    K1t = "K1t"   # growth toward division threshold
    K2t = "K2t"   # in division zone
    K3t = "K3t"   # elongation (post-division)


@dataclass
class CellState:
    """State of one cell in the leaf grid."""
    width: float           # cell width (x-direction), e.g. cell_width[file]
    li: float              # target length from growth law
    lr: float              # relaxed wall length
    l: float               # current length (from segment sum)
    theta: int             # type index for thresholds (usually file index)
    t0: float              # time of last division or start
    phase: CellPhase = CellPhase.K1t
    id: int = 0            # optional id for tracking

    def __post_init__(self) -> None:
        if isinstance(self.phase, str):
            self.phase = CellPhase(self.phase)


@dataclass
class LeafState:
    """
    Grid of cells: list of files, each file is a list of CellState.
    files[i][j] = cell at file i, position j (along leaf).
    """
    files: List[List[CellState]] = field(default_factory=list)

    @property
    def n_cell_files(self) -> int:
        return len(self.files)

    @property
    def n_cells_per_file(self) -> List[int]:
        return [len(f) for f in self.files]

    @property
    def total_cells(self) -> int:
        return sum(self.n_cells_per_file)

    def get_cell(self, file_i: int, cell_j: int) -> CellState:
        return self.files[file_i][cell_j]

    def total_length_per_file(self) -> List[float]:
        """Total length (sum of l) per file."""
        return [sum(c.l for c in f) for f in self.files]

    def flat_index(self, file_i: int, cell_j: int) -> int:
        """Global flat index for cell (file_i, cell_j). Order: file 0, then file 1, ..."""
        return sum(len(self.files[i]) for i in range(file_i)) + cell_j

    def file_and_cell(self, flat_idx: int) -> Tuple[int, int]:
        """From flat index return (file_i, cell_j)."""
        for i, f in enumerate(self.files):
            if flat_idx < len(f):
                return i, flat_idx
            flat_idx -= len(f)
        raise IndexError(flat_idx)

    def all_cells_flat(self) -> List[CellState]:
        """List of all cells in flat order (file 0, file 1, ...)."""
        out = []
        for f in self.files:
            out.extend(f)
        return out

    def to_lr_flat(self) -> np.ndarray:
        """Vector of lr for all cells (flat order)."""
        return np.array([c.lr for c in self.all_cells_flat()])

    def to_l_flat(self) -> np.ndarray:
        """Vector of l for all cells (flat order)."""
        return np.array([c.l for c in self.all_cells_flat()])

    def update_lr_from_flat(self, lr_flat: np.ndarray) -> None:
        """Update all cell lr from flat vector (in place)."""
        idx = 0
        for f in self.files:
            for c in f:
                c.lr = float(lr_flat[idx])
                idx += 1

    def update_l_from_flat(self, l_flat: np.ndarray) -> None:
        """Update all cell l from flat vector (in place)."""
        idx = 0
        for f in self.files:
            for c in f:
                c.l = float(l_flat[idx])
                idx += 1

    def copy(self) -> "LeafState":
        """Deep copy of the leaf state."""
        new_files = []
        for f in self.files:
            new_files.append([
                CellState(
                    width=c.width, li=c.li, lr=c.lr, l=c.l,
                    theta=c.theta, t0=c.t0, phase=c.phase, id=c.id
                )
                for c in f
            ])
        return LeafState(files=new_files)


def create_initial_leaf(
    params: "GrowthParams",
    n_cells_per_file: Optional[List[int]] = None,
    phase: CellPhase = CellPhase.K1t,
    l_scale: float = 1.2,
    li_scale: float = 1.1,
    lr_scale: float = 0.97,
    rng: Optional[np.random.Generator] = None,
) -> LeafState:
    """
    Create initial LeafState with one or more files.
    Base length from min_div_cell_size * l_scale; then li = base*li_scale, lr = base*lr_scale, l = base.
    For growth we need posm > pturg (water in); lr_scale=0.97 gives lower initial pturg so the leaf
    grows from the first steps (0.95 still gave slight shrinkage over many steps).
    """
    from .params import GrowthParams
    if n_cells_per_file is None:
        n_cells_per_file = params.n_cells_per_file[: params.n_cell_files]
    if rng is None:
        rng = np.random.default_rng()
    files = []
    for i, n in enumerate(n_cells_per_file):
        th = params.div_threshold[min(i, len(params.div_threshold) - 1)]
        min_div = params.min_div_cell_size[min(i, len(params.min_div_cell_size) - 1)]
        w = params.cell_width[min(i, len(params.cell_width) - 1)]
        file_cells = []
        for j in range(n):
            # Random size per cell: lognormal so sizes stay positive, mean ≈ 1, spread ~15%
            base = min_div * l_scale * float(rng.lognormal(0.0, 0.15))
            file_cells.append(CellState(
                width=w,
                li=base * li_scale,
                lr=base * lr_scale,
                l=base,
                theta=i,
                t0=0.0,
                phase=phase,
                id=i * 1000 + j,
            ))
        files.append(file_cells)
    return LeafState(files=files)


def ensure_li_geq_l_after_segments(leaf: LeafState, li_margin: float = 1.05, lr_margin: float = 0.99) -> None:
    """
    After l are set from segments (collect_segments), ensure li > l and lr < l so no cell
    starts in strong compression. Sets li = max(li, l * li_margin); lr = max(lr, l * lr_margin)
    so lr is not much smaller than l (avoids huge pturg). Modifies leaf in place.
    """
    for f in leaf.files:
        for c in f:
            if c.l <= 0:
                continue
            c.li = max(c.li, c.l * li_margin)
            # So that (l-lr)/lr is bounded: lr at least l*lr_margin
            c.lr = max(c.lr, c.l * lr_margin)
            if c.lr >= c.l:
                c.lr = c.l * lr_margin


def align_initial_file_lengths(leaf: LeafState) -> None:
    """
    Align total length of all files to the minimum (axiom1-style).
    For each file: trim from the end (last cell(s)) or extend the last cell so sum(l) = min_total.
    Modifies leaf in place.
    """
    totals = leaf.total_length_per_file()
    if not totals:
        return
    target = min(totals)
    for file_i, f in enumerate(leaf.files):
        if not f:
            continue
        current = totals[file_i]
        diff = target - current
        if abs(diff) < 1e-12:
            continue
        if diff > 0:
            # Extend last cell
            last = f[-1]
            new_l = max(1e-6, last.l + diff)
            last.l = new_l
            last.lr = new_l
            last.li = max(last.li, new_l)
        else:
            # Trim from the end: remove or shorten last cell(s) until total == target.
            # Cells that would become zero are removed from the file.
            remain = current - target
            while remain > 1e-12 and len(f) > 0:
                c = f[-1]
                if remain >= c.l - 1e-12:
                    # Remove whole cell
                    remain -= c.l
                    f.pop()
                else:
                    # Shorten last cell
                    new_l = c.l - remain
                    new_l = max(1e-9, new_l)
                    c.l = new_l
                    c.lr = new_l
                    c.li = max(c.li, new_l)
                    remain = 0
                    break
