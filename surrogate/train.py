"""
Training and evaluation of surrogates.
"""

import numpy as np
from typing import List, Union, Type
from .data import SimulationSample, samples_to_arrays
from .surrogate import SurrogateModel, GPSurrogate, MLPSurrogate


def train_surrogate(
    samples: List[SimulationSample],
    model_class: Type[SurrogateModel] = GPSurrogate,
    **model_kwargs,
) -> SurrogateModel:
    """Build X, y from samples and fit surrogate."""
    X, Y = samples_to_arrays(samples)
    if X.size == 0 or Y.size == 0:
        raise ValueError("No samples to train on")
    model = model_class(**model_kwargs)
    model.fit(X, Y)
    return model


def evaluate_surrogate(
    model: SurrogateModel,
    samples: List[SimulationSample],
) -> dict:
    """Compute MAE and RMSE per output dimension."""
    X, Y_true = samples_to_arrays(samples)
    Y_pred = model.predict(X)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    mae = np.abs(Y_true - Y_pred).mean(axis=0)
    rmse = np.sqrt(((Y_true - Y_pred) ** 2).mean(axis=0))
    return {"mae": mae, "rmse": rmse, "mae_per_output": list(mae), "rmse_per_output": list(rmse)}
