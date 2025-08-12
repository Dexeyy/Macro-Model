from __future__ import annotations

from typing import Optional, Iterable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .base_classifier import RegimeClassifier


class HMMClassifier(RegimeClassifier):
    """Gaussian HMM regime classifier with AIC/BIC model selection.

    - Fits GaussianHMM on scaled features
    - Selects number of states by BIC over a provided range
    - Provides hard labels (Viterbi) and posterior probabilities (predict_proba)
    """

    def __init__(
        self,
        n_states: int | Iterable[int] = (2, 3, 4, 5),
        covariance_type: str = "full",
        random_state: int = 42,
        max_iter: int = 200,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model: Optional[GaussianHMM] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HMMClassifier":
        Z = self._clean(X)
        Zs = self.scaler.fit_transform(Z.values)
        candidates = list(self.n_states) if not isinstance(self.n_states, int) else [self.n_states]
        best_bic = np.inf
        best = None
        for k in candidates:
            try:
                hmm = GaussianHMM(
                    n_components=int(k),
                    covariance_type=self.covariance_type,
                    n_iter=self.max_iter,
                    random_state=self.random_state,
                )
                hmm.fit(Zs)
                bic = self._bic(hmm, Zs)
                if bic < best_bic:
                    best_bic = bic
                    best = hmm
            except Exception:
                continue
        self.model = best
        if self.model is None:
            # fallback to a trivial 2-state model
            self.model = GaussianHMM(n_components=2, covariance_type=self.covariance_type, n_iter=self.max_iter, random_state=self.random_state).fit(Zs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        Zs = self.scaler.transform(self._clean(X).values)
        return self.model.predict(Zs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        Zs = self.scaler.transform(self._clean(X).values)
        log_gamma = self.model.predict_proba(Zs)
        return np.asarray(log_gamma)

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        Z = Z.replace([np.inf, -np.inf], np.nan)
        Z = Z.ffill().bfill()
        return Z.select_dtypes(include=[np.number])

    @staticmethod
    def _bic(model: GaussianHMM, X: np.ndarray) -> float:
        n_params = model.n_components * (model.n_components - 1) + model.n_components * X.shape[1] * (
            X.shape[1] + 1
        ) / 2
        log_l = model.score(X)
        return -2 * log_l + n_params * np.log(X.shape[0])


