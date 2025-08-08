from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult


logger = logging.getLogger(__name__)


class HMMModel(RegimeModel):
    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            logger.warning("hmmlearn not available, falling back to uniform regime assignment: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "hmmlearn missing"})

        scaler = StandardScaler()
        Z = scaler.fit_transform(X)

        range_cfg = (self.cfg.get("hmm") or {}).get("n_states_range", [2, 6])
        min_states, max_states = 2, 6
        if isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2:
            try:
                min_states = max(2, int(range_cfg[0]))
                max_states = max(min_states, int(range_cfg[1]))
            except Exception:
                pass

        best_bic = np.inf
        best_model = None
        best_k = None
        for k in range(min_states, max_states + 1):
            try:
                model = hmm.GaussianHMM(n_components=k, covariance_type=(self.cfg.get("hmm") or {}).get("covariance_type", "full"), random_state=42)
                model.fit(Z)
                logL = model.score(Z)
                n_params = k * (Z.shape[1] + Z.shape[1] * (Z.shape[1] + 1) / 2) + k - 1
                bic = -2 * logL + n_params * np.log(len(Z))
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_k = k
            except Exception as exc:
                logger.debug("HMM fit failed for k=%s: %s", k, exc)

        if best_model is None:
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "no_hmm_model"})

        states = best_model.predict(Z)
        try:
            probs = best_model.predict_proba(Z)
        except Exception:
            # build pseudo probabilities from one-hot states
            probs = np.eye(best_k)[states]

        proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])
        labels = pd.Series(states, index=X.index)
        return RegimeResult(labels=labels, proba=proba, diagnostics={"k": best_k, "bic": best_bic})


