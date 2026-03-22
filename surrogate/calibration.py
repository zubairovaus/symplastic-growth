"""
Калибровка параметров модели по экспериментальному профилю длины клеток вдоль листа.
Суррогат, обученный на выходах (L, t_f, profile_bin_1, ..., profile_bin_n), предсказывает
профиль по параметрам; минимизируем расхождение с экспериментальным профилем.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from .data import SimulationSample, samples_to_arrays, cell_length_profile_from_leaf


def profile_loss(
    y_pred_profile: np.ndarray,
    y_exp_profile: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Потери между предсказанным и экспериментальным профилем (например MSE).
    y_pred_profile, y_exp_profile: массивы формы (n_bins,).
    weights: вес по бинам (например больше вес у основания листа).
    """
    diff = np.nan_to_num(y_pred_profile - y_exp_profile, nan=0.0)
    if weights is not None:
        diff = diff * np.asarray(weights)
    return float(np.mean(diff ** 2))


def calibrate_with_surrogate(
    surrogate,
    param_bounds: dict,
    experimental_profile: np.ndarray,
    n_profile_bins: int,
    param_keys: Optional[List[str]] = None,
    Y_index_offset: int = 2,
    method: str = "minimize",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Подбор параметров по экспериментальному профилю с помощью суррогата.

    surrogate: обученная модель, предсказывающая [L, t_f, profile_1, ..., profile_n].
    param_bounds: e.g. {"alph": (5, 15), "etha": (0.05, 0.3), "thresh": (1, 4)}.
    experimental_profile: массив длины n_profile_bins (средние длины клеток по бинам вдоль листа).
    n_profile_bins: число бинов профиля (должно совпадать с обучением суррогата).
    Y_index_offset: индекс начала профиля в выходе (2 для [L, t_f, profile...]).
    method: "minimize" (scipy L-BFGS-B) или "random_search" (перебор случайных точек).

    Возвращает (x_best, loss_best).
    """
    from scipy.optimize import minimize
    keys = param_keys or list(param_bounds.keys())
    bounds_arr = np.array([param_bounds[k] for k in keys])
    exp = np.asarray(experimental_profile, dtype=np.float64)
    if exp.size != n_profile_bins:
        raise ValueError(f"experimental_profile length {exp.size} != n_profile_bins {n_profile_bins}")

    def objective(x: np.ndarray) -> float:
        x_2d = x.reshape(1, -1)
        y_pred = surrogate.predict(x_2d)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        pred_profile = y_pred[0, Y_index_offset : Y_index_offset + n_profile_bins]
        return profile_loss(pred_profile, exp)

    if method == "minimize":
        x0 = np.mean(bounds_arr, axis=1)
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(bounds_arr[:, 0], bounds_arr[:, 1])),
        )
        return res.x, float(res.fun)
    else:
        rng = np.random.default_rng(random_state)
        best_loss = np.inf
        best_x = np.mean(bounds_arr, axis=1)
        for _ in range(200):
            x = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            loss = objective(x)
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()
        return best_x, best_loss


def experimental_profile_to_bins(
    distances_from_base: np.ndarray,
    cell_lengths: np.ndarray,
    n_bins: int,
    L_total: Optional[float] = None,
) -> np.ndarray:
    """
    Преобразовать экспериментальные данные (расстояние от основания, длина клетки)
    в профиль из n_bins: средняя длина клеток в каждом интервале по расстоянию.
    Для согласованности с симулятором последний бин строится по пересечению отрезка
    клетки [d, d+length] с последним интервалом, чтобы в последнем бине всегда были клетки.

    distances_from_base: массив расстояний начала клетки от основания листа.
    cell_lengths: длины клеток.
    n_bins: число бинов.
    L_total: полная длина листа (если None, берётся max(distances_from_base) + один шаг).
    """
    d = np.asarray(distances_from_base, dtype=float)
    l_arr = np.asarray(cell_lengths, dtype=float)
    if L_total is None:
        L_total = float(np.nanmax(d)) + 1e-6
    bin_edges = np.linspace(0, L_total, n_bins + 1)
    profile = np.zeros(n_bins)
    cell_ends = d + l_arr
    for k in range(n_bins):
        if k < n_bins - 1:
            in_bin = (d >= bin_edges[k]) & (d < bin_edges[k + 1])
        else:
            in_bin = (cell_ends > bin_edges[k]) & (d < bin_edges[k + 1])
        if np.any(in_bin):
            profile[k] = np.nanmean(l_arr[in_bin])
        else:
            profile[k] = np.nan
    return np.nan_to_num(profile, nan=0.0)
