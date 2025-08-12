from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import load_yaml_config
from src.utils.zscores import (
    robust_zscore_rolling,
    default_window_minp_for_type,
    pick_window_minp,
)


def _series_type_of(name: str, cfg: Dict | None) -> str | None:
    if cfg and isinstance(cfg.get("series_types"), dict):
        t = cfg["series_types"].get(name)
        return str(t) if isinstance(t, str) else None
    if cfg and isinstance(cfg.get("zscore"), dict):
        z = cfg["zscore"]
        if isinstance(z.get("series_types"), dict):
            t = z["series_types"].get(name)
            return str(t) if isinstance(t, str) else None
    return None


def build_factor(
    df: pd.DataFrame,
    bases: List[str],
    mode: Literal["RT", "RETRO"],
    z_window: int | None = None,
    z_min: int | None = None,
    min_k: int = 2,
) -> Tuple[pd.Series, List[str], pd.Series]:
    """Build a transform-adaptive composite factor using robust rolling z-scores.

    Returns
    - factor: averaged robust z across chosen cols (NaN where coverage < min_k)
    - used_cols: resolved input column names that existed and were non-constant
    - coverage: per-date count of non-NaN inputs used
    """
    cfg = load_yaml_config() or {}
    priority_rt = ["_YoY", "_QoQAnn", "_MA3_YoY", "_MoM", ""]
    priority_retro = ["_YoY", "_MA3_YoY", "_QoQAnn", "_MoM", ""]
    priority = priority_rt if str(mode).upper() == "RT" else priority_retro

    resolved: List[str] = []
    for base in bases:
        chosen = None
        for suffix in priority:
            cand = f"{base}{suffix}"
            if cand in df.columns:
                ser = pd.to_numeric(df[cand], errors="coerce")
                if ser.notna().any():
                    chosen = cand
                    break
        if chosen is None and base in df.columns:
            ser = pd.to_numeric(df[base], errors="coerce")
            if ser.notna().any():
                chosen = base
        if chosen is not None:
            resolved.append(chosen)

    if not resolved:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    z_cols: Dict[str, pd.Series] = {}
    for col in resolved:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            continue
        if z_window is None or z_min is None:
            t = _series_type_of(col, cfg)
            params = default_window_minp_for_type(t) if t else pick_window_minp(s)
        else:
            params = {"window": int(z_window), "min_periods": int(z_min)}
        z = robust_zscore_rolling(s, window=int(params["window"]), min_periods=int(params["min_periods"]))
        z_cols[col] = z

    if not z_cols:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    Z = pd.concat(z_cols, axis=1)
    coverage = Z.notna().sum(axis=1)
    factor = Z.mean(axis=1)
    factor = factor.where(coverage >= int(min_k))
    return factor.astype(float), list(z_cols.keys()), coverage.astype("Int64")


