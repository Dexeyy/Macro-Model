from __future__ import annotations

import pandas as pd


def apply_min_duration(labels: pd.Series, k: int) -> pd.Series:
    """Merge runs shorter than k periods with neighboring regimes.

    Strategy: For any contiguous run with length < k, reassign those indices to the
    previous regime if it exists; otherwise to the next regime.
    """
    if labels.empty or k <= 1:
        return labels

    s = labels.astype("Int64").copy()
    vals = s.to_list()
    idx = list(s.index)

    # Identify runs (start, end, value)
    runs = []
    start = 0
    for i in range(1, len(vals)):
        if pd.isna(vals[i]) or pd.isna(vals[i - 1]) or vals[i] != vals[i - 1]:
            runs.append((start, i - 1, vals[i - 1]))
            start = i
    runs.append((start, len(vals) - 1, vals[-1]))

    # Process short runs
    for r_idx, (a, b, v) in enumerate(runs):
        length = b - a + 1
        if length >= k:
            continue
        # Decide neighbor label
        prev_label = runs[r_idx - 1][2] if r_idx - 1 >= 0 else None
        next_label = runs[r_idx + 1][2] if r_idx + 1 < len(runs) else None
        fill_label = prev_label if prev_label is not None else next_label
        if fill_label is None:
            continue
        for j in range(a, b + 1):
            vals[j] = fill_label

    return pd.Series(vals, index=idx, dtype="Int64")


def confirm_by_probability(proba: pd.DataFrame, threshold: float, consecutive: int) -> pd.Series:
    """Stabilize labels using probability confirmation.

    - Start from argmax labels each period
    - Switch to a new label only after observing that label's probability >= threshold
      for `consecutive` consecutive periods.
    - Returns integer labels aligned to proba.index
    """
    if proba is None or proba.empty:
        return pd.Series(dtype="Int64")

    argmax = proba.values.argmax(axis=1)
    maxp = proba.values.max(axis=1)

    out = []
    current = int(argmax[0]) if len(argmax) else None
    pending = None
    count = 0

    for i in range(len(argmax)):
        cand = int(argmax[i])
        p = float(maxp[i])
        if current is None:
            current = cand
            out.append(current)
            continue
        if cand == current or p < float(threshold):
            # reset confirmation if same as current or insufficient confidence
            pending = None
            count = 0
            out.append(current)
            continue
        # candidate different and confident enough
        if pending is None or pending != cand:
            pending = cand
            count = 1
        else:
            count += 1
        if count >= max(1, int(consecutive)):
            current = cand
            pending = None
            count = 0
        out.append(current)

    return pd.Series(out, index=proba.index, dtype="Int64")


