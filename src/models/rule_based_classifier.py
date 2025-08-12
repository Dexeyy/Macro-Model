from __future__ import annotations

from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .base_classifier import RegimeClassifier


class RuleBasedClassifier(RegimeClassifier):
    """Rule-based macro regime classifier with optional calibration.

    Parameters are thresholds and weights applied on a small set of interpretable
    indicators; multi-class regimes are defined by rule precedence. Calibration
    maximizes F1 score against provided labels via a simple grid search.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        smoothing_window: int = 1,
        require_consecutive: int = 1,
    ) -> None:
        self.thresholds = thresholds or {}
        self.weights = weights or {}
        self.smoothing_window = int(smoothing_window)
        self.require_consecutive = int(require_consecutive)
        self._feature_list: Tuple[str, ...] = tuple()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RuleBasedClassifier":
        X = self._clean_X(X)
        self._feature_list = tuple(X.columns)
        if y is None or y.dropna().empty:
            return self
        # Grid search thresholds if labels provided (binary recession vs expansion assumed)
        yb = pd.Series(y).reindex(X.index).fillna(method="ffill").fillna(method="bfill")
        candidates = self._grid_thresholds(X)
        best_f1 = -np.inf
        best = self.thresholds
        for thr in candidates:
            preds = self._score_rules(X, thr) >= 0.0
            f1 = self._f1(yb.astype(int).values, preds.astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best = thr
        self.thresholds = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xc = self._clean_X(X)
        score = self._score_rules(Xc, self.thresholds)
        labels = np.where(score >= 0.0, 1, 0)  # 1=expansion, 0=recession
        if self.smoothing_window > 1:
            labels = self._smooth(labels, self.smoothing_window, self.require_consecutive)
        return labels

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        labels = self.predict(X)
        # Convert to two-class probabilities using a margin heuristic
        p1 = np.clip(pd.Series(labels).rolling(3, min_periods=1).mean().values, 0.0, 1.0)
        proba = np.vstack([1.0 - p1, p1]).T
        return proba

    # ---- internals ----
    def _clean_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        Z = Z.replace([np.inf, -np.inf], np.nan)
        Z = Z.ffill().bfill()
        return Z.select_dtypes(include=[np.number])

    def _score_rules(self, X: pd.DataFrame, thr: Dict[str, float]) -> pd.Series:
        # Simple weighted sum of sign-threshold comparisons over available indicators
        s = pd.Series(0.0, index=X.index)
        for name, t in thr.items():
            if name not in X.columns:
                continue
            w = float(self.weights.get(name, 1.0))
            s = s + w * ((X[name] >= t).astype(float) - 0.5)
        return s

    def _grid_thresholds(self, X: pd.DataFrame) -> Tuple[Dict[str, float], ...]:
        # Use quantiles of each feature as candidate thresholds
        grids = []
        for col in X.columns[:8]:  # cap to 8 most relevant features to keep light
            qs = X[col].quantile([0.2, 0.4, 0.6, 0.8]).dropna().values.tolist()
            grids.append((col, qs))
        # Cartesian product over small grid
        candidates: list[Dict[str, float]] = []
        def backtrack(i: int, acc: Dict[str, float]):
            if i == len(grids):
                candidates.append(dict(acc))
                return
            name, vals = grids[i]
            for v in vals[:2]:  # keep grid tiny for CI speed
                acc[name] = float(v)
                backtrack(i + 1, acc)
        backtrack(0, {})
        # Fallback if no numeric columns
        return tuple(candidates or [self.thresholds or {}])

    @staticmethod
    def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = float(((y_true == 1) & (y_pred == 1)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _smooth(labels: np.ndarray, window: int, require_consecutive: int) -> np.ndarray:
        out = labels.copy()
        if require_consecutive <= 1:
            # rolling majority
            roll = pd.Series(out).rolling(window, min_periods=max(1, window // 2)).mean()
            return (roll >= 0.5).astype(int).values
        # require N consecutive to switch
        cur = out[0]
        cnt = 0
        for i in range(1, len(out)):
            if out[i] != cur:
                cnt += 1
                if cnt >= require_consecutive:
                    cur = out[i]
                    cnt = 0
            else:
                cnt = 0
            out[i] = cur
        return out


