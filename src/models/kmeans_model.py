from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult


logger = logging.getLogger(__name__)


class KMeansModel(RegimeModel):
    def __init__(self, n_clusters: int = 4, cfg: Optional[Dict] = None):
        self.n_clusters = n_clusters
        self.cfg = cfg or {}

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            logger.warning("sklearn missing for KMeans: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "sklearn missing"})

        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        km = KMeans(n_clusters=int(self.n_clusters), random_state=42)
        labels_idx = km.fit_predict(Z)
        labels = pd.Series(labels_idx, index=X.index)

        # fabricate uniform probabilities or distance-based soft assignment
        try:
            # Use distance to centroid to compute softmax-like confidences
            centroids = km.cluster_centers_
            dists = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)
            inv = 1 / (dists + 1e-9)
            probs = inv / inv.sum(axis=1, keepdims=True)
        except Exception:
            probs = np.ones((len(X), int(self.n_clusters))) / float(self.n_clusters)

        proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])
        return RegimeResult(labels=labels, proba=proba, diagnostics={})


