"""
Segment structure and operations: transverse segments across cell files,
JoinFragments1 (merge small segments), CollectSegments (assign l from segments to cells).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .state import LeafState, CellState


@dataclass
class SegmentState:
    """
    Segments: transverse slices across the leaf.
    - lengths[k] = length of segment k
    - cell_index[k][file_i] = index j of cell in file file_i that belongs to segment k
    So segment k spans cells (0, cell_index[k][0]), (1, cell_index[k][1]), ...
    """

    lengths: List[float]
    cell_index: List[List[int]]  # cell_index[k][file_i] = cell j in file file_i

    @property
    def n_segments(self) -> int:
        return len(self.lengths)

    @property
    def n_files(self) -> int:
        return len(self.cell_index[0]) if self.cell_index else 0


def build_segments_from_leaf(leaf: LeafState) -> "SegmentState":
    """
    Build initial segments from leaf state: align boundaries across files
    (union of cumulative lengths), then assign each segment to one cell per file.
    Mathematica: Union @@ Accumulate /@ lengths_per_file, then Differences; each segment
    maps to the cell index at that position in each file.

    Invariant: l_from_segments(leaf, seg) equals the current leaf.l (and sum(seg.lengths)
    equals total leaf length). So rebuilding segments only re-partitions the same lengths;
    used after divisions to keep segment count O(cells) without changing physics.
    """
    n_files = leaf.n_cell_files
    # Cumulative lengths per file: cumsum of l for each file
    cum_per_file = []
    for f in leaf.files:
        L = [c.l for c in f]
        cum_per_file.append(np.cumsum(L))
    # Union of all boundaries (sorted unique values)
    all_bounds = set()
    for cum in cum_per_file:
        all_bounds.add(0.0)
        for x in cum:
            all_bounds.add(float(x))
    bounds = sorted(all_bounds)
    seg_lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    # For each segment (between bounds[k] and bounds[k+1]), find which cell in each file
    # contains this interval. Cell j in file i contains [cum[i][j-1], cum[i][j]] (0-based: cum[i][-1] = end).
    cell_index = []
    for k in range(len(seg_lengths)):
        mid = (bounds[k] + bounds[k + 1]) / 2
        row = []
        for i in range(n_files):
            cum = cum_per_file[i]
            # Find j such that cum[j-1] < mid <= cum[j]
            j = 0
            for j in range(len(cum)):
                if mid <= cum[j]:
                    break
            else:
                j = len(cum) - 1
            row.append(j)
        cell_index.append(row)
    return SegmentState(lengths=seg_lengths, cell_index=cell_index)


def l_from_segments(leaf: LeafState, seg: SegmentState) -> np.ndarray:
    """
    Compute current length l for each cell from segment lengths.
    For cell (file_i, cell_j): l = sum of seg.lengths[k] for all k where seg.cell_index[k][file_i] == cell_j.
    Returns flat array of l (same order as leaf.all_cells_flat()).
    """
    n_files = leaf.n_cell_files
    n_cells_per_file = leaf.n_cells_per_file
    l_flat = np.zeros(leaf.total_cells)
    # Map flat index to (file_i, cell_j)
    for k in range(seg.n_segments):
        L = seg.lengths[k]
        for file_i, cell_j in enumerate(seg.cell_index[k]):
            if file_i >= n_files:
                continue
            if cell_j >= n_cells_per_file[file_i]:
                continue
            flat_idx = leaf.flat_index(file_i, cell_j)
            l_flat[flat_idx] += L
    return l_flat


def seg_to_cells_flat(leaf: LeafState, seg: SegmentState) -> List[List[int]]:
    """
    seg_to_cells[k] = list of flat indices of cells that belong to segment k.
    Used by the ODE layer (growth_rhs) to sum over cells in a segment.
    """
    result = []
    for k in range(seg.n_segments):
        indices = []
        for file_i, cell_j in enumerate(seg.cell_index[k]):
            if file_i < leaf.n_cell_files and cell_j < leaf.n_cells_per_file[file_i]:
                indices.append(leaf.flat_index(file_i, cell_j))
        result.append(indices)
    return result


def join_fragments_one(
    leaf: LeafState,
    seg: SegmentState,
    epsilon: float = 0.1,
) -> Tuple[LeafState, SegmentState]:
    """
    JoinFragments1: while min(seg.lengths) <= epsilon, merge smallest segment with next.
    Merging is only done when the neighbour has the same cell_index, so cell lengths
    stay consistent (no cell loses its length to another). Epsilon is the model's
    min_fragment_length (params.min_fragment_length): segments below this are not
    resolved; merging keeps the number of tiny segments bounded and numerics stable.
    Then update cell lengths l in leaf from new segment lengths (CollectSegments).
    Returns (leaf with updated l, new SegmentState). Leaf state's l are updated in place and returned.
    """
    lengths = list(seg.lengths)
    cell_index = [list(row) for row in seg.cell_index]

    def same_cell(ka: int, kb: int) -> bool:
        if ka < 0 or kb < 0 or ka >= len(cell_index) or kb >= len(cell_index):
            return False
        return cell_index[ka] == cell_index[kb]

    while lengths and min(lengths) <= epsilon:
        k_min = int(np.argmin(lengths))
        merged = False
        # Merge only with a neighbour that has the same cell_index, so we don't reassign
        # length to another cell (which would give some cells l=0).
        if k_min + 1 < len(lengths) and same_cell(k_min, k_min + 1):
            lengths[k_min] = lengths[k_min] + lengths[k_min + 1]
            lengths.pop(k_min + 1)
            cell_index.pop(k_min + 1)
            merged = True
        elif k_min > 0 and same_cell(k_min, k_min - 1):
            lengths[k_min - 1] = lengths[k_min - 1] + lengths[k_min]
            lengths.pop(k_min)
            cell_index.pop(k_min)
            merged = True
        if not merged:
            # Smallest segment has no neighbour with same cell_index; skip to avoid l=0
            break
    seg_new = SegmentState(lengths=lengths, cell_index=cell_index)
    l_flat = l_from_segments(leaf, seg_new)
    leaf.update_l_from_flat(l_flat)
    return leaf, seg_new


def collect_segments(leaf: LeafState, seg: SegmentState) -> None:
    """
    Assign each cell's l from segment lengths (in place).
    """
    l_flat = l_from_segments(leaf, seg)
    leaf.update_l_from_flat(l_flat)
