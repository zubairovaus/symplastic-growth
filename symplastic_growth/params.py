"""
Parameter container for the symplastic growth model.
All physical and phenomenological constants in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class GrowthParams:
    """Parameters for symplastic growth (walls, osmotics, cell cycle)."""

    # --- Mechanics & osmotics ---
    # m_young with effective_s_cw=4*r gives pturg scale K = m_young*4*r/r^2 = 4*m_young/r.
    # For r~19, K~211 at m_young=1000 → pturg >> posm (alph=10) and leaf shrinks. Use m_young~100 so K~21 and growth is possible.
    m_young: float = 100.0           # effective Young modulus (stress units); 1000 from original gives K~212 and shrinkage
    s_cw: float = 4.0                # wall cross-section factor (s_cw = 4*r*dr typically)
    alph: float = 10.0               # osmotic coefficient: P_osm = alph * (li - l) / l (e.g. Bar)
    etha: float = 0.15              # inverse relaxation time for wall growth (1/hour)
    thresh: float = 2.0               # stress threshold for lr growth (same units as P_turg)
    # Hydraulic conductivity Lw (article: 40 μm·h⁻¹·bar⁻¹). Eq. 10: (1/l)dl/dt = r·Lw·(posm - pturg).
    Lw: float = 40.0                  # μm·h⁻¹·bar⁻¹; segment ODE uses r·Lw·(posm - pturg) per cell

    # --- Cell cycle (per file/type) ---
    t_cell_cycle: float = 24.7       # hour
    t_cell_elongation: float = 71.09  # hour (elongation phase duration)
    div_threshold: List[float] = field(default_factory=lambda: [9.896, 8.95, 11.491, 13.153, 19.569])
    min_div_cell_size: Optional[List[float]] = None   # default 0.5 * div_threshold
    max_elong_cell_size: Optional[List[float]] = None # default 2 * ave_elong

    # --- Geometry (cell files = rows along leaf) ---
    # cell_width = characteristic size r: used in pturg (∝ 1/r²) and growth (∝ r·Lw·(posm−pturg)).
    # Biologically, cell width is often ~ (cell length)/4; set accordingly if matching real tissue.
    n_cell_files: int = 5
    cell_width: List[float] = field(default_factory=lambda: [18.9, 26.3, 20.1, 10.1, 6.7])
    n_cells_per_file: List[int] = field(default_factory=lambda: [18, 8, 8, 13, 32])
    max_cell_length: Optional[List[float]] = None   # cap li at this per file (original: maxCellLengthReal)
    elong_border: Optional[List[float]] = None     # from original ElongBorder (position along leaf; optional use)

    # --- Division (Phase 2) ---
    s_division: Optional[List[float]] = None   # cumulative length threshold for K1t -> K2t per file
    smax: Optional[List[float]] = None         # li threshold for division (default = div_threshold)
    alpha_min: float = 0.4                     # division ratio bounds (alpha, 1-alpha)
    alpha_max: float = 0.6
    alpha_mu: float = 0.5
    alpha_sigma: float = 0.1                   # truncated normal for division ratio

    # --- Scaling ---
    koef: float = 1.367

    # --- Segment resolution (JoinFragments1) ---
    # Fragments shorter than this are merged with an adjacent fragment of the same cell.
    # This defines the minimal resolved segment length: avoids unbounded tiny segments and
    # keeps the ODE state and numerics stable when many cell files produce many boundaries.
    min_fragment_length: float = 0.1

    def _pad_list(self, lst: List, target_len: int, name: str = "list") -> None:
        """Pad list to target_len with last value, or trim; modify in place if possible."""
        n = len(lst)
        if n < target_len:
            last = lst[-1] if n else 0
            lst.extend([last] * (target_len - n))
        elif n > target_len:
            lst[:] = lst[:target_len]

    def __post_init__(self) -> None:
        # Дополняем n_cells_per_file до длины n_cell_files (последнее значение повторяется)
        if len(self.n_cells_per_file) < self.n_cell_files:
            last_n = self.n_cells_per_file[-1] if self.n_cells_per_file else 4
            self.n_cells_per_file = list(self.n_cells_per_file) + [last_n] * (
                self.n_cell_files - len(self.n_cells_per_file)
            )
        elif len(self.n_cells_per_file) > self.n_cell_files:
            self.n_cells_per_file = self.n_cells_per_file[: self.n_cell_files]
        # Списки по файлам должны иметь длину n_cell_files (индекс file_i/theta может быть 0..n_cell_files-1)
        self.div_threshold = list(self.div_threshold)
        self._pad_list(self.div_threshold, self.n_cell_files)
        self.cell_width = list(self.cell_width)
        self._pad_list(self.cell_width, self.n_cell_files)
        if self.min_div_cell_size is None:
            self.min_div_cell_size = [0.5 * d for d in self.div_threshold]
        else:
            self.min_div_cell_size = list(self.min_div_cell_size)
            self._pad_list(self.min_div_cell_size, self.n_cell_files)
        if self.max_elong_cell_size is None:
            ave_elong = [13.80775, 12.781, 15.167, 14.01, 22.07]  # from original
            self.max_elong_cell_size = [2.0 * self.koef * e for e in ave_elong]
        if len(self.max_elong_cell_size) < self.n_cell_files:
            last = self.max_elong_cell_size[-1] if self.max_elong_cell_size else 30.0
            self.max_elong_cell_size = list(self.max_elong_cell_size) + [
                last
            ] * (self.n_cell_files - len(self.max_elong_cell_size))
        if self.smax is None:
            self.smax = list(self.div_threshold)
        else:
            self.smax = list(self.smax)
            self._pad_list(self.smax, self.n_cell_files)
        if self.s_division is None:
            # Default: from original DivBorder (first file ~179, etc.)
            self.s_division = [179.3, 164.6, 235.8, 255.9, 257.8][: self.n_cell_files]
            if len(self.s_division) < self.n_cell_files:
                self.s_division.extend([200.0] * (self.n_cell_files - len(self.s_division)))
        else:
            self.s_division = list(self.s_division)
            self._pad_list(self.s_division, self.n_cell_files)
        if self.max_cell_length is None:
            # From original maxCellLength (per file)
            self.max_cell_length = [157.76, 152.015, 111.296, 125.835, 136.336][: self.n_cell_files]
            if len(self.max_cell_length) < self.n_cell_files:
                self.max_cell_length.extend([150.0] * (self.n_cell_files - len(self.max_cell_length)))
        else:
            self.max_cell_length = list(self.max_cell_length)
            self._pad_list(self.max_cell_length, self.n_cell_files)
        if self.elong_border is None:
            self.elong_border = [398.23, 387.34, 476.39, 418.24, 463.56][: self.n_cell_files]
            if len(self.elong_border) < self.n_cell_files:
                last = self.elong_border[-1] if self.elong_border else 400.0
                self.elong_border.extend([last] * (self.n_cell_files - len(self.elong_border)))
        else:
            self.elong_border = list(self.elong_border)
            self._pad_list(self.elong_border, self.n_cell_files)

    def r(self, file_index: int) -> float:
        """Characteristic size (e.g. cell width) for file."""
        return self.cell_width[file_index]

    def effective_s_cw(self, file_index: int, dr: float = 1.0) -> float:
        """Wall section = 4 * r * dr."""
        return 4.0 * self.r(file_index) * dr
