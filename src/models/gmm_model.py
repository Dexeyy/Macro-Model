from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult


logger = logging.getLogger(__name__)


class GMMModel(RegimeModel):
    def __init__(self, n_components: int = 4, cfg: Optional[Dict] = None):
        self.n_components = n_components
        self.cfg = cfg or {}

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            logger.warning("sklearn missing for GMM: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "sklearn missing"})

        scaler = StandardScaler()
        Z = scaler.fit_transform(X)

        gm = GaussianMixture(n_components=int(self.n_components), covariance_type=self.cfg.get("covariance_type", "full"), random_state=42)
        gm.fit(Z)
        states = gm.predict(Z)
        probs = gm.predict_proba(Z)
        proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])
        labels = pd.Series(states, index=X.index)
        return RegimeResult(labels=labels, proba=proba, diagnostics={"bic": gm.bic(Z)})


