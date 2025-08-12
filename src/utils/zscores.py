from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd


def robust_zscore_rolling(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute robust rolling z-scores using median/MAD with safe fallbacks.

    z_t = (x_t - median_t) / (1.4826 * MAD_t), where median/MAD are over a past rolling window.
    Fallback to mean/std for windows where MAD is ~0. Replace 0 std by NaN.

    Notes
    - Uses pd.Series.rolling(window, min_periods=min_periods) to prevent look-ahead.
    - Keeps original index and dtype float.
    """
    if not isinstance(s, pd.Series):
        raise TypeError("robust_zscore_rolling expects a pandas Series")

    s_float = pd.to_numeric(s, errors="coerce").astype(float)
    roll = s_float.rolling(window=window, min_periods=min_periods)

    med = roll.median()
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)

    denom_mad = 1.4826 * mad
    # Identify places where MAD is effectively zero or NaN
    mad_bad = (denom_mad.isna()) | (denom_mad == 0)

    # Mean/std fallback computed with same rolling window and min_periods
    mean = roll.mean()
    std = roll.std(ddof=0)
    std_repl = std.replace(0.0, np.nan)

    z_mad = (s_float - med) / denom_mad.replace(0.0, np.nan)
    z_std = (s_float - mean) / std_repl

    out = z_mad.copy()
    out[mad_bad] = z_std[mad_bad]
    return out


_DEFAULT_WINDOWS: Dict[str, tuple[int, int]] = {
    "fast": (36, 18),
    "typical": (60, 24),
    "slow": (120, 36),
}


def _score_candidate(z: pd.Series) -> float:
    """Score a candidate z-series by coverage, smoothness, and low sign flips.

    score = 0.5*coverage + 0.3*smoothness + 0.2*low_flips
    coverage = share non-NaN
    smoothness = 1 / (1 + median(|Δz|))
    flips = average 12m rolling sign-change count; transformed: 1/(1+flips)
    """
    if z is None or z.empty:
        return 0.0
    z = pd.to_numeric(z, errors="coerce")
    n = len(z)
    if n == 0:
        return 0.0
    coverage = float(z.notna().mean())
    dz = z.diff().abs()
    smoothness = float(1.0 / (1.0 + np.nanmedian(dz.values))) if np.isfinite(np.nanmedian(dz.values)) else 0.0

    sign = np.sign(z.fillna(0.0))
    sign_change = (sign != sign.shift(1)).astype(float)
    flips_12m = sign_change.rolling(12, min_periods=6).sum()
    flips = float(np.nanmean(flips_12m.values)) if np.isfinite(np.nanmean(flips_12m.values)) else 0.0
    low_flips = 1.0 / (1.0 + flips)

    return 0.5 * coverage + 0.3 * smoothness + 0.2 * low_flips


def pick_window_minp(s: pd.Series) -> Dict[Literal["window", "min_periods"], int]:
    """Auto-tune window and min_periods for monthly data.

    Candidates: (36,18), (60,24), (120,36). Pick by scoring robust z-series.
    If ties, prefer larger window for stability.
    """
    candidates = [(36, 18), (60, 24), (120, 36)]
    best_score = -np.inf
    best = candidates[0]
    for w, m in candidates:
        try:
            z = robust_zscore_rolling(s, window=w, min_periods=m)
            sc = _score_candidate(z)
            if (sc > best_score) or (np.isclose(sc, best_score) and w > best[0]):
                best_score = sc
                best = (w, m)
        except Exception:
            continue
    return {"window": int(best[0]), "min_periods": int(best[1])}


def default_window_minp_for_type(series_type: str | None) -> Dict[Literal["window", "min_periods"], int]:
    """Return sensible defaults by type; unknown types follow 0.4*window rule.

    If type is None or not in known buckets, choose window=60 and min_periods=min(36, floor(0.4*window)).
    """
    if series_type in _DEFAULT_WINDOWS:
        w, m = _DEFAULT_WINDOWS[series_type]
        return {"window": int(w), "min_periods": int(m)}
    # Unknown: rule of thumb
    w = 60
    m = min(36, int(np.floor(0.4 * w)))
    return {"window": int(w), "min_periods": int(m)}


