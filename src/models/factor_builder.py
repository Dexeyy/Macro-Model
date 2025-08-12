from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.utils.helpers import load_yaml_config
from src.utils.zscores import robust_zscore_rolling, default_window_minp_for_type, pick_window_minp


@dataclass
class FactorSpec:
    name: str
    bases: List[str]
    min_k: int = 2


def _series_type_of(name: str, cfg: Dict | None) -> str | None:
    if cfg and isinstance(cfg.get("series_types"), dict):
        t = cfg["series_types"].get(name)
        return str(t) if isinstance(t, str) else None
    z = cfg.get("zscore") if cfg else None
    if isinstance(z, dict) and isinstance(z.get("series_types"), dict):
        t = z["series_types"].get(name)
        return str(t) if isinstance(t, str) else None
    return None


def _resolve_transform(df: pd.DataFrame, base: str, mode: str) -> str | None:
    priority_rt = ["_YoY", "_MA3_YoY", "_MoM", "_QoQAnn", ""]
    priority_retro = ["_YoY", "_MA3_YoY", "_QoQAnn", "_MoM", ""]
    priority = priority_rt if str(mode).upper() == "RT" else priority_retro
    for suffix in priority:
        cand = f"{base}{suffix}"
        if cand in df.columns and pd.to_numeric(df[cand], errors="coerce").notna().any():
            return cand
    if base in df.columns and pd.to_numeric(df[base], errors="coerce").notna().any():
        return base
    return None


def build_factor(df: pd.DataFrame, spec: FactorSpec, mode: str = "RETRO") -> Tuple[pd.Series, List[str], pd.Series]:
    """Build a factor as average of robust z-scores over resolved base transforms.

    Applies per-series rolling windows (median/MAD fallback to mean/std) and masks
    the factor where coverage < min_k.
    """
    cfg = load_yaml_config() or {}
    chosen: List[str] = []
    for base in spec.bases:
        col = _resolve_transform(df, base, mode)
        if col is not None:
            chosen.append(col)
    if not chosen:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    z_cols: Dict[str, pd.Series] = {}
    for col in chosen:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            continue
        t = _series_type_of(col, cfg)
        params = default_window_minp_for_type(t) if t else pick_window_minp(s)
        z_cols[col] = robust_zscore_rolling(s, window=int(params["window"]), min_periods=int(params["min_periods"]))
    if not z_cols:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    Z = pd.concat(z_cols, axis=1)
    coverage = Z.notna().sum(axis=1)
    factor = Z.mean(axis=1).where(coverage >= int(spec.min_k))
    return factor.astype(float), list(z_cols.keys()), coverage.astype("Int64")


def build_from_config(df: pd.DataFrame, mode: str = "RETRO") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build factors defined in YAML config.

    Config structure example (config/regimes.yaml):
      factors:
        F_Growth:
          bases: ["INDPRO", "PAYEMS", "GDPC1", "CUMFNS"]
          min_k: 2
        F_Inflation:
          bases: ["CPIAUCSL", "PCEPI", "PPICMM"]
      zscore:
        series_types:
          CPIAUCSL: typical
          NFCI: fast

    Returns:
      (factor_df, coverage_df) where coverage_df has columns <factor>_count and <factor>_ratio
    """
    cfg = load_yaml_config() or {}
    factors_cfg = cfg.get("factors") or {}
    results: Dict[str, pd.Series] = {}
    coverage_fields: Dict[str, pd.DataFrame] = {}
    for name, spec_cfg in factors_cfg.items():
        bases = list(spec_cfg.get("bases", []))
        if not bases:
            continue
        min_k = int(spec_cfg.get("min_k", 2))
        f, used, cov = build_factor(df, FactorSpec(name=name, bases=bases, min_k=min_k), mode=mode)
        results[name] = f
        denom = max(1, len(used))
        coverage_fields[name] = pd.DataFrame({
            f"{name}_count": cov,
            f"{name}_ratio": cov.astype(float) / denom,
        })
    factors_df = pd.DataFrame(results) if results else pd.DataFrame(index=df.index)
    coverage_df = pd.concat(coverage_fields.values(), axis=1) if coverage_fields else pd.DataFrame(index=df.index)
    return factors_df, coverage_df


