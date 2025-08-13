from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import RegimeModel, RegimeResult
from .postprocess import apply_min_duration, confirm_by_probability


logger = logging.getLogger(__name__)


class HSMMModel(RegimeModel):
    """Hidden Semi-Markov Model with explicit duration handling.

    Preference order:
      1) If a native HSMM library is available (hsmmlearn or pomegranate HiddenSemiMarkovModel), use it.
      2) Otherwise, fit a standard HMM (hmmlearn) and apply explicit min-duration smoothing on the
         Viterbi-decoded path, optionally using probability confirmation.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult:
        # Try native HSMM libraries (best-effort; fall back immediately on any error)
        try:
            from hsmmlearn.hsmm import GaussianHSMM  # type: ignore
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            Z = scaler.fit_transform(X)

            hsmm_cfg = (self.cfg.get("hsmm") or {})
            n_states = int(hsmm_cfg.get("n_states", 4))
            # Minimal HSMM usage; duration distributions default to geometric-like if not specified
            model = GaussianHSMM(n_components=n_states, random_state=42)
            model.fit(Z)
            states = model.predict(Z)
            try:
                probs = model.predict_proba(Z)
            except Exception:
                probs = np.eye(n_states)[states]

            proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])

            # Duration enforcement even when native HSMM is used (acts as an extra guard)
            min_k = int(hsmm_cfg.get("min_duration", 3))
            labels = apply_min_duration(pd.Series(states, index=X.index), k=min_k)
            return RegimeResult(labels=labels, proba=proba, diagnostics={"engine": "hsmmlearn"})
        except Exception:
            pass

        try:
            # Pomegranate HiddenSemiMarkovModel (if available)
            from pomegranate import HiddenMarkovModel  # type: ignore
            # pomegranate HSMM support varies; use HMM as base then duration enforcement
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            Z = scaler.fit_transform(X)

            # Build a simple HMM as a base using pomegranate
            # Note: We do not craft complex emissions; this is a graceful fallback path
            hmm = HiddenMarkovModel.from_samples(distribution="multivariate_normal", n_components=4, X=[Z])
            states = np.array(hmm.predict(Z))
            # Soft probabilities are not trivial to extract consistently; approximate with one-hot
            probs = np.eye(int(states.max() + 1))[states]
            proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])

            hsmm_cfg = (self.cfg.get("hsmm") or {})
            min_k = int(hsmm_cfg.get("min_duration", 3))
            labels = apply_min_duration(pd.Series(states, index=X.index), k=min_k)
            return RegimeResult(labels=labels, proba=proba, diagnostics={"engine": "pomegranate_hmm_base"})
        except Exception:
            pass

        # Fallback: hmmlearn + explicit duration correction
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            logger.warning("hmmlearn not available for HSMM fallback: %s", exc)
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "no_backend"})

        scaler = StandardScaler()
        Z = scaler.fit_transform(X)

        range_cfg = (self.cfg.get("hsmm") or {}).get("n_states_range", [2, 6])
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
                model = hmm.GaussianHMM(n_components=k, covariance_type=(self.cfg.get("hsmm") or {}).get("covariance_type", "full"), random_state=42)
                model.fit(Z)
                logL = model.score(Z)
                n_params = k * (Z.shape[1] + Z.shape[1] * (Z.shape[1] + 1) / 2) + k - 1
                bic = -2 * logL + n_params * np.log(len(Z))
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_k = k
            except Exception as exc:
                logger.debug("HSMM fallback HMM fit failed for k=%s: %s", k, exc)

        if best_model is None:
            labels = pd.Series([0] * len(X), index=X.index)
            proba = pd.DataFrame(np.ones((len(X), 1)), index=X.index, columns=["state_0"])
            return RegimeResult(labels=labels, proba=proba, diagnostics={"error": "no_hmm_model"})

        states = best_model.predict(Z)
        try:
            probs = best_model.predict_proba(Z)
        except Exception:
            probs = np.eye(best_k)[states]

        proba = pd.DataFrame(probs, index=X.index, columns=[f"state_{i}" for i in range(probs.shape[1])])
        hsmm_cfg = (self.cfg.get("hsmm") or {})
        thr = float(hsmm_cfg.get("prob_threshold", (self.cfg.get("hmm") or {}).get("prob_threshold", 0.7)))
        consec = int(hsmm_cfg.get("confirm_consecutive", (self.cfg.get("ensemble") or {}).get("confirm_consecutive", 2)))
        min_k = int(hsmm_cfg.get("min_duration", (self.cfg.get("hmm") or {}).get("min_duration", 3)))

        # Probability confirmation followed by explicit min duration
        confirmed = confirm_by_probability(proba, threshold=thr, consecutive=consec)
        labels = apply_min_duration(confirmed, k=min_k)
        return RegimeResult(labels=labels, proba=proba, diagnostics={"engine": "hmmlearn_fallback", "k": best_k, "bic": best_bic})


