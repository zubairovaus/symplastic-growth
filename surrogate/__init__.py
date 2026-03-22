"""
Surrogate modeling for the symplastic growth simulator.
Train a fast approximator (GP or NN) to replace expensive ODE runs.
"""

from .data import (
    generate_training_data,
    SimulationSample,
    samples_to_arrays,
    cell_length_profile_from_leaf,
)
from .surrogate import SurrogateModel, GPSurrogate, MLPSurrogate
from .train import train_surrogate, evaluate_surrogate
from .calibration import (
    calibrate_with_surrogate,
    profile_loss,
    experimental_profile_to_bins,
)

__all__ = [
    "generate_training_data",
    "SimulationSample",
    "samples_to_arrays",
    "cell_length_profile_from_leaf",
    "SurrogateModel",
    "GPSurrogate",
    "MLPSurrogate",
    "train_surrogate",
    "evaluate_surrogate",
    "calibrate_with_surrogate",
    "profile_loss",
    "experimental_profile_to_bins",
]
