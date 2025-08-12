from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base_classifier import RegimeClassifier


class KMeansClassifier(RegimeClassifier):
    """KMeans-based unsupervised regime classifier.

    Performs k-means on scaled features; probabilities are soft assignments via
    inverse distance weighting. Smoothing can be applied externally.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KMeansClassifier":
        Z = self._clean(X)
        Zs = self.scaler.fit_transform(Z.values)
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.model.fit(Zs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        Zs = self.scaler.transform(self._clean(X).values)
        return self.model.predict(Zs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        Z = self._clean(X)
        Zs = self.scaler.transform(Z.values)
        centers = self.model.cluster_centers_
        # inverse distance weights
        d = np.linalg.norm(Zs[:, None, :] - centers[None, :, :], axis=2)
        d = np.where(d == 0, 1e-6, d)
        w = 1.0 / d
        proba = w / w.sum(axis=1, keepdims=True)
        return proba

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        Z = Z.replace([np.inf, -np.inf], np.nan)
        Z = Z.ffill().bfill()
        return Z.select_dtypes(include=[np.number])


