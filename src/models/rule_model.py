from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult


logger = logging.getLogger(__name__)


class RuleModel(RegimeModel):
    """Very simple rule-based classifier using a few key indicators.

    Produces labels 0..K-1; optional probability matrix via one-hot.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        # Heuristics based on growth vs inflation vs conditions
        g = X.filter(regex="gdp|growth|indpro|YieldCurve|gdp_growth|GDP_YoY", axis=1)
        infl = X.filter(regex="cpi|infl|PCE|CPI_YoY|core", axis=1)
        fin = X.filter(regex="credit|spread|risk|FinConditions|NFCI|MOVE", axis=1)

        score_growth = g.mean(axis=1).fillna(0)
        score_infl = infl.mean(axis=1).fillna(0)
        score_fin = fin.mean(axis=1).fillna(0)

        # Map heuristic combinations to regimes
        # 0: expansion, 1: recession, 2: stagflation, 3: recovery
        labels = []
        for i in range(len(X)):
            sg = score_growth.iloc[i]
            si = score_infl.iloc[i]
            sf = score_fin.iloc[i]
            if sg > 0 and si < 0 and sf < 0:
                lbl = 0  # expansion
            elif sg < 0 and sf > 0:
                lbl = 1  # recession
            elif si > 0 and sg <= 0:
                lbl = 2  # stagflation-like
            else:
                lbl = 3  # recovery/neutral
            labels.append(lbl)

        labels_s = pd.Series(labels, index=X.index)

        # Optional one-hot probabilities
        K = int(np.max(labels_s.values)) + 1 if len(labels_s) else 1
        proba = np.zeros((len(X), K))
        for i, lbl in enumerate(labels):
            proba[i, int(lbl)] = 1.0
        proba_df = pd.DataFrame(proba, index=X.index, columns=[f"state_{i}" for i in range(K)])

        return RegimeResult(labels=labels_s, proba=proba_df, diagnostics={})


