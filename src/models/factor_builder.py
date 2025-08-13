from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import pandas as pd

from src.utils.helpers import load_yaml_config
from config.config import SERIES_SIGN, SUB_BUCKETS, PRUNE_CORR_THRESHOLD, MIN_SUB_BUCKET_COVERAGE
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


def _apply_sign(series: pd.Series, name: str, series_sign: Dict[str, int] | None = None) -> pd.Series:
    """Multiply a series by its configured sign. Falls back to +1.

    The sign key is matched on the base series name (without transform suffixes).
    """
    if series_sign is None:
        series_sign = SERIES_SIGN or {}
    # infer base name by splitting at first underscore if present
    base = str(name).split("_")[0]
    sign = int(series_sign.get(base, 1))
    try:
        return pd.to_numeric(series, errors="coerce") * sign
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def _signal_strength(z: pd.Series) -> float:
    # coverage ratio times std deviation
    cov = float(z.notna().mean())
    sd = float(pd.to_numeric(z, errors="coerce").dropna().std(ddof=0) or 0.0)
    return cov * sd


def _prune_correlated(cols: List[str], Z: pd.DataFrame, threshold: float) -> List[str]:
    if not cols:
        return []
    # rank by signal strength
    ranks = sorted(cols, key=lambda c: _signal_strength(Z[c]), reverse=True)
    kept: List[str] = []
    for c in ranks:
        if not kept:
            kept.append(c)
            continue
        corr_ok = True
        for k in kept:
            try:
                cc = float(Z[[c, k]].dropna().corr().iloc[0, 1])
            except Exception:
                cc = 0.0
            if abs(cc) >= threshold:
                corr_ok = False
                break
        if corr_ok:
            kept.append(c)
    return kept


def _group_by_subbucket(cols: Iterable[str], sub_buckets: Dict[str, Dict[str, List[str]]], theme_hint: str | None = None) -> Dict[str, List[str]]:
    """Map columns to sub-buckets using configured lists.

    If theme_hint provided, use that node of SUB_BUCKETS; else attempt to find
    matching sub-buckets across themes.
    """
    mapping: Dict[str, List[str]] = {}
    buckets = {}
    if theme_hint and theme_hint in sub_buckets:
        buckets = sub_buckets[theme_hint]
    else:
        # merge all for loose matching
        for v in sub_buckets.values():
            buckets.update(v)
    inv: Dict[str, str] = {}
    for sb, bases in buckets.items():
        for b in bases:
            inv[b] = sb
    for c in cols:
        base = str(c).split("_")[0]
        sb = inv.get(base, "misc")
        mapping.setdefault(sb, []).append(c)
    return mapping


def build_factor(
    df: pd.DataFrame,
    spec: FactorSpec,
    mode: str = "RETRO",
    *,
    prune_threshold: float | None = None,
    series_sign_override: Dict[str, int] | None = None,
    sub_buckets_override: Dict[str, Dict[str, List[str]]] | None = None,
    min_sub_bucket_coverage: int | None = None,
) -> Tuple[pd.Series, List[str], pd.Series]:
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

    # Apply sign before z-scoring and compute rolling robust z-scores
    z_cols: Dict[str, pd.Series] = {}
    for col in chosen:
        s = _apply_sign(df[col], col, series_sign_override or SERIES_SIGN)
        if s.nunique(dropna=True) <= 1:
            continue
        t = _series_type_of(col, cfg)
        params = default_window_minp_for_type(t) if t else pick_window_minp(s)
        z_cols[col] = robust_zscore_rolling(s, window=int(params["window"]), min_periods=int(params["min_periods"]))
    if not z_cols:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    Z = pd.concat(z_cols, axis=1)

    # 1) Correlation pruning
    thresh = float(prune_threshold if prune_threshold is not None else PRUNE_CORR_THRESHOLD)
    kept_cols = _prune_correlated(list(Z.columns), Z, thresh)
    Z = Z[kept_cols]

    # 2) Sub-bucket weighting
    groups = _group_by_subbucket(kept_cols, sub_buckets_override or SUB_BUCKETS)
    sub_means = {}
    min_cov = int(min_sub_bucket_coverage if min_sub_bucket_coverage is not None else MIN_SUB_BUCKET_COVERAGE)
    for sb, cols_sb in groups.items():
        block = Z[cols_sb]
        cov = block.notna().sum(axis=1)
        sub_means[sb] = block.mean(axis=1).where(cov >= min_cov)
    if not sub_means:
        coverage = Z.notna().sum(axis=1)
        factor = Z.mean(axis=1).where(coverage >= int(spec.min_k))
        return factor.astype(float), kept_cols, coverage.astype("Int64")
    SB = pd.DataFrame(sub_means)
    sb_cov = SB.notna().sum(axis=1)
    factor = SB.mean(axis=1).where(sb_cov >= int(spec.min_k))
    # coverage counts reflect contributing series after masking
    coverage = pd.Series(0, index=df.index, dtype="Int64")
    for sb, cols_sb in groups.items():
        block = Z[cols_sb]
        coverage = coverage.add(block.notna().sum(axis=1).astype("Int64"), fill_value=0)
    coverage = coverage.astype("Int64")
    return factor.astype(float), kept_cols, coverage


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


