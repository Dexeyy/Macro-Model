from __future__ import annotations

import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult


logger = logging.getLogger(__name__)


class MSDynModel(RegimeModel):
    """Markov-Switching Dynamic Regression proxy on MacroMomentum.

    MacroMomentum = mean(F_Growth, F_Inflation, F_Liquidity) where available,
    otherwise mean across available numeric columns in X as a fallback. If PCA
    macro factor is available (e.g., PC_Macro or average of PC_*), prefer it.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def _compute_macro_momentum(self, X: pd.DataFrame, original: Optional[pd.DataFrame]) -> pd.Series:
        df = original if original is not None else X
        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Prefer a single macro PCA factor if present
        for c in ("PC_Macro",):
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                return s

        # If multiple PC_* exist, average them
        pc_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("PC_")]
        if pc_cols:
            s = df[pc_cols].select_dtypes("number").mean(axis=1)
            return s

        # Otherwise use F_* composites if available
        f_cols = [c for c in ("F_Growth", "F_Inflation", "F_Liquidity") if c in df.columns]
        if f_cols:
            s = df[f_cols].select_dtypes("number").mean(axis=1)
            return s

        # Fallback: mean across numeric columns in X
        num = X.select_dtypes("number")
        if not num.empty:
            return num.mean(axis=1)
        return pd.Series(dtype=float)

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        original: Optional[pd.DataFrame] = kwargs.get("original_data")

        # Build the target MacroMomentum series
        y = self._compute_macro_momentum(X, original)
        y = pd.to_numeric(y, errors="coerce").dropna()
        if y.empty:
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "no_macro_momentum"})

        # Align X index to y
        idx = y.index

        # Use statsmodels MarkovRegression as the stable proxy
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except Exception as exc:
            logger.warning("statsmodels MarkovRegression unavailable: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "no_statsmodels"})

        regimes = int((self.cfg.get("msdyn") or {}).get("n_states", 2))
        # Configure switching mean; allow switching variance as well for robustness
        try:
            mod = MarkovRegression(y.values, k_regimes=regimes, trend="c", switching_variance=True)
            res = mod.fit(disp=False)
        except Exception as exc:
            logger.warning("MarkovRegression fit failed: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "fit_failed"})

        # Smoothed probabilities and most likely state
        smoothed = res.smoothed_marginal_probabilities
        # smoothed is shape (k, T); transpose to (T, k)
        probs = np.vstack([smoothed[i] for i in range(regimes)]).T
        proba_df = pd.DataFrame(probs, index=idx, columns=[f"state_{i}" for i in range(regimes)])
        states = probs.argmax(axis=1)
        labels = pd.Series(states, index=idx)

        # Reindex to original X index with forward/back-fill for any gaps
        proba_df = proba_df.reindex(X.index).ffill().bfill()
        labels = labels.reindex(X.index).ffill().bfill().astype(int)

        return RegimeResult(labels=labels, proba=proba_df, diagnostics={"llf": float(getattr(res, "llf", np.nan)), "k": regimes})


