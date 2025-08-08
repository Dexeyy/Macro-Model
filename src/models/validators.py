from __future__ import annotations

import math
from typing import Dict

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
    """Compute contiguous run durations with basic sanity flags.

    Returns dict: mean_duration, median_duration, min_duration, flagged_short (bool)
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
    out["mean_duration"] = float(np.mean(arr)) if arr.size else np.nan
    out["median_duration"] = float(np.median(arr)) if arr.size else np.nan
    out["min_duration"] = float(np.min(arr)) if arr.size else np.nan
    out["flagged_short"] = bool(np.any(arr < 2))  # heuristic
    return out


