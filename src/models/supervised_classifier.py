from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base_classifier import RegimeClassifier


class SupervisedClassifier(RegimeClassifier):
    """Supervised logistic classifier for recession vs expansion (or multi-class via OVR).

    Uses sklearn LogisticRegression with cross-validation over C; features are scaled.
    """

    def __init__(self, multi_class: str = "ovr", cv_splits: int = 4, random_state: int = 42) -> None:
        self.multi_class = multi_class
        self.cv_splits = int(cv_splits)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[LogisticRegression] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SupervisedClassifier":
        if y is None or y.dropna().empty:
            # No labels -> default to predicting expansion
            self.model = LogisticRegression()
            # still fit scaler
            _ = self.scaler.fit(self._clean(X).values)
            return self
        Z = self._clean(X)
        yb = pd.Series(y).reindex(Z.index).ffill().bfill().values
        Zs = self.scaler.fit_transform(Z.values)
        lr = LogisticRegression(max_iter=500, multi_class=self.multi_class, random_state=self.random_state)
        cv = TimeSeriesSplit(n_splits=max(2, self.cv_splits))
        gs = GridSearchCV(lr, {"C": [0.1, 1.0, 10.0]}, cv=cv, scoring="f1_macro", n_jobs=1)
        gs.fit(Zs, yb)
        self.model = gs.best_estimator_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=int)
        Zs = self.scaler.transform(self._clean(X).values)
        return self.model.predict(Zs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.tile(np.array([[1.0, 0.0]]), (len(X), 1))
        Zs = self.scaler.transform(self._clean(X).values)
        proba = self.model.predict_proba(Zs)
        return np.asarray(proba)

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        Z = Z.replace([np.inf, -np.inf], np.nan)
        Z = Z.ffill().bfill()
        return Z.select_dtypes(include=[np.number])


