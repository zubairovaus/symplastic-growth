"""
Surrogate models: Gaussian Process and MLP to approximate simulator output.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class SurrogateModel(ABC):
    """Abstract surrogate: train on (X, y), predict for new X."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateModel":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (mean, std). Std is None if not supported."""
        return self.predict(X), None


class GPSurrogate(SurrogateModel):
    """
    Gaussian Process surrogate (one GP per output). Supports uncertainty (std) per prediction.
    """

    def __init__(self, kernel=None):
        self.kernel = kernel
        self._gps = []  # list of GPR, one per output
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        from sklearn.base import clone
        from sklearn.exceptions import ConvergenceWarning
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_out = y.shape[1]
        if self.kernel is None:
            kernel = (
                C(1.0, (1e-3, 1e6))
                * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e5))
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
            )
        else:
            kernel = self.kernel
        self._gps = []
        for j in range(n_out):
            gp = GaussianProcessRegressor(
                kernel=clone(kernel),
                n_restarts_optimizer=5,
                alpha=1e-6,
                normalize_y=True,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp.fit(X, y[:, j])
            self._gps.append(gp)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GPSurrogate not fitted")
        out = np.column_stack([gp.predict(X) for gp in self._gps])
        return out if out.shape[1] > 1 else out.ravel()

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._fitted:
            raise RuntimeError("GPSurrogate not fitted")
        means = []
        stds = []
        for gp in self._gps:
            m, s = gp.predict(X, return_std=True)
            means.append(m)
            stds.append(s)
        mean = np.column_stack(means) if len(means) > 1 else np.array(means[0])
        std = np.column_stack(stds) if len(stds) > 1 else np.array(stds[0])
        return mean, std


class MLPSurrogate(SurrogateModel):
    """
    MLP regressor (scikit-learn). Fast, no native uncertainty; can use ensemble for that.
    """

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 64), max_iter: int = 500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._net = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSurrogate":
        from sklearn.neural_network import MLPRegressor
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._net = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=42,
        )
        self._net.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MLPSurrogate not fitted")
        out = self._net.predict(X)
        return out if out.ndim > 1 else out.ravel()
