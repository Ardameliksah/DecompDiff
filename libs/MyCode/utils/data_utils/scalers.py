"""
Scaler abstractions for time series normalization.

Each scaler implements a common interface:
  - fit(data)        -> fit on raw data (2D: [num_samples, num_features])
  - transform(data)  -> normalize
  - inverse_transform(data) -> denormalize
  - fit_transform(data)     -> fit + transform in one call
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler as _SklearnMinMax, StandardScaler as _SklearnStandard


class BaseScaler(ABC):
    """Interface that every scaler must implement."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "BaseScaler":
        ...

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


class ZScoreScaler(BaseScaler):
    """
    Standard (Z-score) normalization: (x - mean) / std.
    Wraps sklearn.preprocessing.StandardScaler.
    Range: unbounded.
    """

    def __init__(self):
        self._scaler = _SklearnStandard()

    def fit(self, data: np.ndarray) -> "ZScoreScaler":
        self._scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self._scaler.transform(data).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(data).astype(np.float32)


class MinMaxNormScaler(BaseScaler):
    """
    MinMax normalization to [0, 1] with optional remap to [-1, 1].
    Wraps sklearn.preprocessing.MinMaxScaler.

    Args:
        neg_one_to_one: If True, remap [0,1] -> [-1,1] after MinMax scaling.
    """

    def __init__(self, neg_one_to_one: bool = True):
        self._scaler = _SklearnMinMax()
        self.neg_one_to_one = neg_one_to_one

    def fit(self, data: np.ndarray) -> "MinMaxNormScaler":
        self._scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        scaled = self._scaler.transform(data)
        if self.neg_one_to_one:
            scaled = scaled * 2 - 1
        return scaled.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        d = np.array(data, dtype=np.float64)
        if self.neg_one_to_one:
            d = (d + 1) * 0.5
        return self._scaler.inverse_transform(d).astype(np.float32)
