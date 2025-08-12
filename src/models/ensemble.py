from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd


_REGEX_NUM = re.compile(r"(\d+)$")


def _parse_state_int(label: str | int | float) -> int | None:
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, float) and float(label).is_integer():
        return int(label)
    if not isinstance(label, str):
        return None
    # expect formats like 'Regime_2'
    m = _REGEX_NUM.search(label)
    return int(m.group(1)) if m else None


def align_states(df: pd.DataFrame, labels_col: str, reference: str = "F_Growth") -> Dict[int, int]:
    """Map each state's raw id -> rank by mean of reference factor.

    If reference is missing or insufficient data, returns identity mapping.
    """
    if labels_col not in df.columns:
        return {}
    labels = df[labels_col].dropna()
    states = sorted({s for s in labels.map(_parse_state_int).dropna().unique()})
    if not states:
        return {}
    if reference not in df.columns:
        return {s: s for s in states}

    means: list[tuple[int, float]] = []
    for s in states:
        mask = labels.map(_parse_state_int) == s
        series = df.loc[mask, reference].dropna()
        if series.empty:
            m = np.nan
        else:
            m = float(series.mean())
        means.append((s, m))

    # Sort by mean ascending; assign rank 0..K-1
    means_sorted = sorted(means, key=lambda t: (np.nan_to_num(t[1], nan=0.0)))
    mapping: Dict[int, int] = {state: rank for rank, (state, _) in enumerate(means_sorted)}
    return mapping


def average_probabilities(prob_mats: List[pd.DataFrame]) -> pd.DataFrame:
    """Average aligned probability matrices.

    Expects each matrix to have columns named 'state_<i>' already aligned.
    Returns averaged probabilities over intersected index, with columns unified to
    the superset of states across inputs (missing filled with 0 before averaging).
    """
    mats = [p.copy() for p in prob_mats if isinstance(p, pd.DataFrame) and not p.empty]
    if len(mats) < 2:
        return pd.DataFrame()
    # inner join on dates
    idx = mats[0].index
    for m in mats[1:]:
        idx = idx.intersection(m.index)
    if len(idx) == 0:
        return pd.DataFrame()
    mats = [m.loc[idx] for m in mats]

    # unify columns to superset of state_i
    all_cols = sorted(set().union(*[set(m.columns) for m in mats]))
    mats = [m.reindex(columns=all_cols, fill_value=0.0) for m in mats]

    avg = sum(mats) / float(len(mats))
    # renormalize rows to 1 to guard against fill zeros
    row_sum = avg.sum(axis=1).replace(0.0, np.nan)
    avg = avg.div(row_sum, axis=0).fillna(0.0)
    return avg


def ensemble_labels(probs: pd.DataFrame) -> pd.Series:
    if probs is None or probs.empty:
        return pd.Series(dtype="Int64")
    argmax = probs.values.argmax(axis=1)
    return pd.Series(argmax, index=probs.index, dtype="Int64")


