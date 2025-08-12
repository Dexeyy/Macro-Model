from __future__ import annotations

import math
from typing import Dict, List, Union
from scipy import stats

import numpy as np
import pandas as pd


def chow_mean_change_test(series: pd.Series, switch_idx: int, window: int = 6) -> Dict:
    """Simple two-window mean-change test around a switch index.

    Not a formal Chow test (no regression), but compares means in windows
    [switch_idx-window, switch_idx) and [switch_idx, switch_idx+window).

    Returns dict with before_mean, after_mean, diff, t_stat, p_value (normal approx),
    and passed_chow boolean at alpha=0.1.
    """
    result = {
        "before_mean": np.nan,
        "after_mean": np.nan,
        "diff": np.nan,
        "t_stat": np.nan,
        "p_value": 1.0,
        "passed_chow": False,
    }
    if series is None or series.empty:
        return result
    n = len(series)
    if switch_idx <= 0 or switch_idx >= n:
        return result
    w = int(max(1, window))
    a = max(0, switch_idx - w)
    b = min(n, switch_idx + w)
    before = series.iloc[a:switch_idx].dropna()
    after = series.iloc[switch_idx:b].dropna()
    if len(before) < 2 or len(after) < 2:
        return result
    m1, m2 = float(before.mean()), float(after.mean())
    s1, s2 = float(before.std(ddof=1)), float(after.std(ddof=1))
    result["before_mean"], result["after_mean"], result["diff"] = m1, m2, m2 - m1
    # pooled std and t-statistic
    n1, n2 = len(before), len(after)
    sp2 = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    if sp2 <= 0:
        result["t_stat"] = 0.0
        result["p_value"] = 1.0
        return result
    denom = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    t = (m2 - m1) / denom if denom > 0 else 0.0
    result["t_stat"] = t
    # Normal approximation for two-tailed p-value
    # Phi(x) ~ 0.5 * (1 + erf(x/sqrt(2)))
    z = abs(t)
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    result["p_value"] = p
    result["passed_chow"] = p < 0.1
    return result


def duration_sanity(labels: pd.Series) -> Dict:
    """Compute contiguous run durations and stability flags.

    Returns dict with backward-compatible keys and new summary:
      - mean_duration, median_duration, min_duration, flagged_short
      - mean, median, too_short_share, passed
    """
    out = {
        "mean_duration": np.nan,
        "median_duration": np.nan,
        "min_duration": np.nan,
        "flagged_short": False,
    }
    if labels is None or labels.empty:
        return out
    vals = labels.to_list()
    if len(vals) == 0:
        return out
    runs = []
    start = 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            runs.append(i - start)
            start = i
    runs.append(len(vals) - start)
    arr = np.array(runs, dtype=float)
    mean_d = float(np.mean(arr)) if arr.size else np.nan
    med_d = float(np.median(arr)) if arr.size else np.nan
    min_d = float(np.min(arr)) if arr.size else np.nan
    too_short = float(np.mean(arr < 2)) if arr.size else np.nan
    out["mean_duration"] = mean_d
    out["median_duration"] = med_d
    out["min_duration"] = min_d
    out["flagged_short"] = bool(np.any(arr < 2))  # heuristic
    # New summary keys
    out["mean"] = mean_d
    out["median"] = med_d
    out["too_short_share"] = too_short
    out["passed"] = bool((arr.size > 0) and (min_d >= 2))
    return out


def detect_breaks(series: pd.Series, pen: Union[str, float] = "aic") -> List[pd.Timestamp]:
    """Detect change points using ruptures (PELT, l2 cost).

    pen: 'aic' | numeric (penalty value). Returns list of break timestamps.
    Falls back to empty list if ruptures is unavailable or series too short.
    """
    try:
        import ruptures as rpt  # type: ignore
    except Exception:
        return []


def chow_mean_variance_test(series: pd.Series, switch_idx: int, window: int = 6) -> Dict:
    """Combined mean and variance change test around a switch.

    Returns dict with p_value (max of mean/variance p-values) and passed (both pass at alpha=0.1).
    """
    out = {"p_value": 1.0, "passed": False, "p_mean": 1.0, "p_var": 1.0}
    if series is None or series.empty:
        return out
    n = len(series)
    if switch_idx <= 0 or switch_idx >= n:
        return out
    w = int(max(1, window))
    a = max(0, switch_idx - w)
    b = min(n, switch_idx + w)
    before = series.iloc[a:switch_idx].dropna()
    after = series.iloc[switch_idx:b].dropna()
    if len(before) < 3 or len(after) < 3:
        return out
    # Mean change: two-sample t-test (Welch)
    try:
        tstat, p_mean = stats.ttest_ind(before.values, after.values, equal_var=False, nan_policy="omit")
        p_mean = float(np.nan_to_num(p_mean, nan=1.0))
    except Exception:
        p_mean = 1.0
    # Variance change: F-test using sample variances
    try:
        s1, s2 = float(np.var(before.values, ddof=1)), float(np.var(after.values, ddof=1))
        if s1 <= 0 or s2 <= 0:
            p_var = 1.0
        else:
            f = s1 / s2 if s1 >= s2 else s2 / s1
            df1, df2 = len(before) - 1, len(after) - 1
            # two-tailed
            p_var = 2.0 * min(1.0 - stats.f.cdf(f, df1, df2), stats.f.cdf(1.0 / f, df1, df2))
            p_var = float(np.clip(p_var, 0.0, 1.0))
    except Exception:
        p_var = 1.0
    out["p_mean"] = p_mean
    out["p_var"] = p_var
    out["p_value"] = max(p_mean, p_var)
    out["passed"] = bool((p_mean < 0.1) and (p_var < 0.1))
    return out

    if series is None or series.dropna().size < 20:
        return []

    y = series.dropna().astype(float)
    algo = rpt.Pelt(model="l2")
    try:
        if isinstance(pen, str) and pen.lower() == "aic":
            # heuristic: use linear penalty scaled by log(n)
            penalty = 2.0 * np.log(len(y))
        else:
            penalty = float(pen)
        bks = algo.fit(y.values).predict(pen=penalty)
        # ruptures returns end indices of segments; exclude the last endpoint
        idxs = [i for i in bks if i < len(y)]
        return [y.index[i - 1] for i in idxs if i > 0]
    except Exception:
        return []

