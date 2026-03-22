"""
Generate training data for surrogate: run simulator with different parameters, collect (X, y).
X = parameter vector, y = scalar or vector output (e.g. final total length, or trajectory summary).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symplastic_growth import GrowthParams, run_until_length, run_until_length_multi, create_initial_leaf


@dataclass
class SimulationSample:
    """One run: input params (as vector) and output(s)."""
    x: np.ndarray   # input vector (e.g. [alph, etha, thresh, n_cells, ...])
    y: np.ndarray   # output: e.g. [final_length, time_to_Ly], or scalar
    meta: Optional[dict] = None


def _simulate_one(
    x: np.ndarray,
    *,
    keys: List[str],
    base: GrowthParams,
    Ly: float,
    n_cells: int,
    dt: float,
    outputs: str,
    use_multi_file: bool,
    n_profile_bins: int,
    max_steps: int,
    n_cell_files: int,
    n_cells_per_file: Optional[List[int]],
    lr_scale: float = 0.97,
    li_scale: float = 1.1,
) -> Optional["SimulationSample"]:
    """Run one simulation; return sample or None on failure."""
    params = _params_from_vector(x, keys, base)
    try:
        if use_multi_file:
            leaf0 = create_initial_leaf(
                params,
                n_cells_per_file=params.n_cells_per_file,
                lr_scale=lr_scale,
                li_scale=li_scale,
            )
            res = run_until_length_multi(
                params,
                Ly=Ly,
                initial_leaf=leaf0,
                dt=dt,
                max_steps=max_steps,
                with_division=True,
            )
        else:
            res = run_until_length(params, Ly=Ly, n_cells=n_cells, dt=dt, t0=0.0)
    except Exception:
        return None

    total_length = res.total_length[-1]
    t_final = res.t[-1]
    if outputs == "length":
        y = np.array([total_length])
    elif outputs == "time":
        y = np.array([t_final])
    elif outputs == "length_and_time_and_profile":
        profile = cell_length_profile_from_leaf(res.final_leaf, n_bins=n_profile_bins)
        # Пустые бины (nan/0) заменяем на малую константу, чтобы GP не упирался в границы (ConvergenceWarning)
        profile = np.nan_to_num(profile, nan=0.0)
        profile = np.maximum(profile, 1e-3)
        y = np.concatenate([[total_length, t_final], profile])
    else:
        y = np.array([total_length, t_final])

    meta = {
        "Ly": Ly,
        "n_cells": n_cells,
        "use_multi_file": use_multi_file,
        "Lw": params.Lw,
        "max_steps": max_steps,
    }
    if outputs == "length_and_time_and_profile":
        meta["n_profile_bins"] = n_profile_bins
    if use_multi_file:
        meta["n_cell_files"] = n_cell_files
        meta["n_cells_per_file"] = list(n_cells_per_file or params.n_cells_per_file)
    return SimulationSample(x=x.astype(np.float64), y=y, meta=meta)


def params_to_vector(
    params: GrowthParams,
    *,
    use_alph: bool = True,
    use_etha: bool = True,
    use_thresh: bool = True,
    use_m_young: bool = False,
    use_koef: bool = False,
) -> np.ndarray:
    """Encode key parameters as a fixed-size vector for surrogate input."""
    parts = []
    if use_alph:
        parts.append(params.alph)
    if use_etha:
        parts.append(params.etha)
    if use_thresh:
        parts.append(params.thresh)
    if use_m_young:
        parts.append(params.m_young)
    if use_koef:
        parts.append(params.koef)
    return np.array(parts, dtype=np.float64)


def vector_to_params(
    x: np.ndarray,
    base: GrowthParams,
    *,
    use_alph: bool = True,
    use_etha: bool = True,
    use_thresh: bool = True,
    use_m_young: bool = False,
    use_koef: bool = False,
) -> GrowthParams:
    """Build GrowthParams from vector and base (copy non-encoded from base)."""
    import dataclasses
    d = dataclasses.asdict(base)
    idx = 0
    if use_alph:
        d["alph"] = float(x[idx]); idx += 1
    if use_etha:
        d["etha"] = float(x[idx]); idx += 1
    if use_thresh:
        d["thresh"] = float(x[idx]); idx += 1
    if use_m_young:
        d["m_young"] = float(x[idx]); idx += 1
    if use_koef:
        d["koef"] = float(x[idx]); idx += 1
    return GrowthParams(**d)


def _params_from_vector(x: np.ndarray, keys: List[str], base: GrowthParams) -> GrowthParams:
    """Build GrowthParams from base and vector x (keys order)."""
    from dataclasses import fields
    kwargs = {}
    for f in fields(GrowthParams):
        if f.name in keys:
            idx = keys.index(f.name)
            kwargs[f.name] = float(x[idx])
        else:
            kwargs[f.name] = getattr(base, f.name)
    return GrowthParams(**kwargs)


def cell_length_profile_from_leaf(leaf, n_bins: int = 10) -> np.ndarray:
    """
    Вычислить профиль длины клеток вдоль листа: разбить лист на n_bins интервалов
    по расстоянию от основания и в каждом интервале взять среднюю длину клеток.
    Клетка попадает в бин по положению своего начала, кроме последнего бина:
    в последний бин входят все клетки, чей отрезок [начало, начало+длина] пересекается
    с последним интервалом, чтобы в последнем бине всегда были клетки (кончик листа).
    """
    if not leaf.files:
        return np.full(n_bins, np.nan)
    # Позиция от основания и длина клетки по первому файлу (вдоль оси листа)
    positions = []
    lengths = []
    cum = 0.0
    for c in leaf.files[0]:
        positions.append(cum)
        lengths.append(c.l)
        cum += c.l
    L_total = cum
    if not positions or L_total <= 0:
        return np.full(n_bins, np.nan)
    positions = np.array(positions)
    lengths = np.array(lengths)
    bin_edges = np.linspace(0, L_total, n_bins + 1)
    profile = np.zeros(n_bins)
    cell_ends = positions + lengths
    for k in range(n_bins):
        if k < n_bins - 1:
            # обычный бин: клетка входит по положению начала
            in_bin = (positions >= bin_edges[k]) & (positions < bin_edges[k + 1])
        else:
            # последний бин: все клетки, пересекающиеся с [bin_edges[-2], L_total]
            in_bin = (cell_ends > bin_edges[k]) & (positions < bin_edges[k + 1])
        if in_bin.any():
            profile[k] = np.mean(lengths[in_bin])
        else:
            profile[k] = np.nan
    return profile


def generate_training_data(
    n_samples: int,
    Ly: float = 500.0,
    n_cells: int = 8,
    dt: float = 1.0,
    param_bounds: Optional[dict] = None,
    random_state: Optional[int] = None,
    outputs: str = "length_and_time",  # "length", "time", "length_and_time", "length_and_time_and_profile"
    use_multi_file: bool = True,
    n_cell_files: int = 2,
    n_cells_per_file: Optional[List[int]] = None,
    n_profile_bins: int = 10,
    show_progress: bool = True,
    Lw: Optional[float] = 5.0,
    max_steps: int = 500,
    n_jobs: int = 1,
    use_physics_verification_params: bool = True,
) -> List[SimulationSample]:
    """
    Sample parameter vectors, run simulator for each, return list of SimulationSample.
    By default use_multi_file=True: мультифайловый лист с делением клеток (клетки влияют друг на друга).
    param_bounds: e.g. {"alph": (5, 15), "etha": (0.05, 0.3), "thresh": (1, 4)}.
    outputs="length_and_time_and_profile": y = [L, t_f, profile_bin_1, ..., profile_bin_n] для калибровки по экспериментальному профилю.
    show_progress: if True, show progress (tqdm or print per simulation).
    Lw: гидравлическая проводимость (статья 40; при 5 симуляция быстрее, как в 03_physics_verification). None = дефолт GrowthParams (40).
    max_steps: максимум шагов ОДУ на один запуск (уменьшить для ускорения, но часть запусков может не достичь Ly).
    use_physics_verification_params: если True и use_multi_file — те же параметры, что в §6 ноутбука 03 (m_young=10, s_division, min_fragment_length=0.4, lr_scale=0.99, li_scale=1.2 и др.).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    rng = np.random.default_rng(random_state)
    lr_scale, li_scale = 0.97, 1.1
    if use_multi_file:
        n_cells_per_file = n_cells_per_file or [4] * n_cell_files
        base_kw = dict(n_cell_files=n_cell_files, n_cells_per_file=list(n_cells_per_file))
        if Lw is not None:
            base_kw["Lw"] = Lw
        if use_physics_verification_params:
            # Как в §6 ноутбука 03_physics_verification (мульти-файловая симуляция: рост + деление)
            base_kw["m_young"] = 10.0
            base_kw["smax"] = [12.0] * n_cell_files
            base_kw["div_threshold"] = [12.0] * n_cell_files
            base_kw["s_division"] = [10.0] * n_cell_files
            base_kw["min_div_cell_size"] = [5.0] * n_cell_files
            base_kw["cell_width"] = [1.5] * n_cell_files
            base_kw["min_fragment_length"] = 0.4
            lr_scale, li_scale = 0.99, 1.2
        base = GrowthParams(**base_kw)
    else:
        base_kw = {}
        if Lw is not None:
            base_kw["Lw"] = Lw
        base = GrowthParams(**base_kw) if base_kw else GrowthParams()
    if param_bounds is None:
        param_bounds = {
            "alph": (5.0, 15.0),
            "etha": (0.05, 0.3),
            "thresh": (1.0, 4.0),
        }
    keys = list(param_bounds.keys())
    bounds_arr = np.array([param_bounds[k] for k in keys])
    xs = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], size=(n_samples, bounds_arr.shape[0]))
    samples: List[SimulationSample] = []

    # Sequential (default)
    if n_jobs <= 1:
        iterator = range(n_samples)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Simulations", unit="run")
        elif show_progress:
            import time
            t0 = time.perf_counter()

        for i in iterator:
            if show_progress and tqdm is None:
                print(f"  Simulation {i + 1}/{n_samples} ...", flush=True)
            s = _simulate_one(
                xs[i],
                keys=keys,
                base=base,
                Ly=Ly,
                n_cells=n_cells,
                dt=dt,
                outputs=outputs,
                use_multi_file=use_multi_file,
                n_profile_bins=n_profile_bins,
                max_steps=max_steps,
                n_cell_files=n_cell_files,
                n_cells_per_file=n_cells_per_file,
                lr_scale=lr_scale,
                li_scale=li_scale,
            )
            if s is None:
                if show_progress and tqdm is None:
                    print("    (run failed, skipping)", flush=True)
                continue
            samples.append(s)

        if show_progress and tqdm is None and samples:
            elapsed = time.perf_counter() - t0
            print(f"  Done: {len(samples)} successful runs in {elapsed:.1f} s", flush=True)
        if samples and len(samples) < n_samples and show_progress:
            print(f"  Note: {n_samples - len(samples)} run(s) failed (numerical/ODE errors).", flush=True)
        return samples

    # Parallel over samples (CPU)
    import concurrent.futures as cf
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=n_samples, desc=f"Simulations (n_jobs={n_jobs})", unit="run")
    elif show_progress:
        print(f"Parallel generation: n_jobs={n_jobs}, n_samples={n_samples}", flush=True)

    with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _simulate_one,
                xs[i],
                keys=keys,
                base=base,
                Ly=Ly,
                n_cells=n_cells,
                dt=dt,
                outputs=outputs,
                use_multi_file=use_multi_file,
                n_profile_bins=n_profile_bins,
                max_steps=max_steps,
                n_cell_files=n_cell_files,
                n_cells_per_file=n_cells_per_file,
                lr_scale=lr_scale,
                li_scale=li_scale,
            )
            for i in range(n_samples)
        ]
        for fut in cf.as_completed(futures):
            s = fut.result()
            if s is not None:
                samples.append(s)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()
    if show_progress:
        print(f"  Done: {len(samples)}/{n_samples} successful runs.", flush=True)
        if len(samples) < n_samples:
            print(f"  Note: {n_samples - len(samples)} run(s) failed (numerical/ODE errors).", flush=True)
    return samples


def samples_to_arrays(samples: List[SimulationSample]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack samples into X (n, d) and Y (n, out_dim)."""
    if not samples:
        return np.zeros((0, 0)), np.zeros((0, 0))
    X = np.stack([s.x for s in samples])
    Y = np.stack([s.y for s in samples])
    return X, Y
