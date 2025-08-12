from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class RegimeClassifier(ABC):
    """Abstract base for macro regime classifiers.

    Implementations should be lightweight and interpretable. All methods must
    be side-effect free beyond storing fitted parameters on self.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RegimeClassifier":
        """Fit the classifier on features X and optional labels y.

        Args:
            X: Feature matrix (rows=time, cols=features). Must be index-aligned but
               content cleaning (NaN handling) is up to the implementation.
            y: Optional label series for supervised calibration.

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete regime labels for X.

        Returns a 1D numpy array of length len(X).
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-regime probabilities for X.

        Returns a 2D numpy array of shape (len(X), n_regimes). Implementations with
        hard labels only may return one-hot probabilities.
        """


