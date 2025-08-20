from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SERIES_ALIASES = {
    # Housing
    "MORTGAGE_RATE": ["MORTGAGE30US", "MORTG30US", "MORTG", "MORTGAGE30"],
    "HOUSE_PRICE": ["CSUSHPINSA", "USSTHPI", "HPI", "MSPUS", "HOUSTPRICE"],
    "HOUSE_INCOME": ["MEHOINUSA672N", "DSPIC96", "A229RX0", "AHEMAN", "AHETPI"],
    "PERMITS": ["PERMIT", "PERMITS", "PERMIT1"],
    "STARTS": ["HOUST", "HOUSTF", "HOUSTNSA", "HOUSTS"],
    "NAHB": ["NAHB", "NHSPST", "HMI"],
    # Inflation
    "CPI_YOY": ["CPI_YoY", "CPIYOY"],
    "CPI_LVL": ["CPIAUCSL", "CPIAUCSL_SA", "CPIAUCNS"],
    "CORECPI_YOY": ["CoreCPI_YoY", "CPI_Core_YoY", "CORECPIYOY"],
    "CORECPI_LVL": ["CPILFESL", "CPILFENS"],
    # Extras
    "M2_LVL": ["M2SL", "WM2NS"],
}


def aliases_for(key: str) -> List[str]:
    """Return alias list for a logical key, extended from config if available."""
    base = list(SERIES_ALIASES.get(key, []))
    try:
        # Lazy import to avoid heavy deps at module import time
        from config import config as CFG  # type: ignore
        # Extend from FEATURE_GROUPS/SUB_BUCKETS where sensible
        if key == "PERMITS":
            # Sub-bucket permits plus FRED_SERIES keys if present
            permits = []
            try:
                permits = list(getattr(CFG, "SUB_BUCKETS", {}).get("Housing", {}).get("permits", []))
            except Exception:
                permits = []
            base.extend([c for c in permits if c not in base])
        if key == "STARTS":
            starts = []
            try:
                starts = list(getattr(CFG, "SUB_BUCKETS", {}).get("Housing", {}).get("construction", []))
            except Exception:
                starts = []
            base.extend([c for c in starts if c not in base])
        if key == "CPI_LVL":
            if "CPIAUCSL" not in base:
                base.append("CPIAUCSL")
        if key == "CORECPI_LVL":
            if "CPILFESL" not in base:
                base.append("CPILFESL")
        if key == "HOUSE_INCOME":
            # Add proxies from config if present
            for proxy in ["RPI", "W875RX1", "DSPIC96"]:
                if proxy not in base:
                    base.append(proxy)
    except Exception:
        pass
    return base


# Loose keyword patterns per key for substring matching when exact aliases fail
SERIES_KEYWORDS = {
    "MORTGAGE_RATE": ["mortgage", "mortg", "m30"],
    "HOUSE_PRICE": ["case", "shiller", "csushp", "house price", "hpi", "fhfa", "mspus"],
    "HOUSE_INCOME": ["income", "wage", "earnings", "disp", "personal income"],
    "PERMITS": ["permit"],
    "STARTS": ["houst", "starts"],
    "NAHB": ["nahb", "hmi"],
    "CPI_YOY": ["cpi", "yoy"],
    "CPI_LVL": ["cpiaucsl", "cpi"],
    "CORECPI_YOY": ["core", "cpi", "yoy", "ex food"],
    "CORECPI_LVL": ["cpilfe", "core", "ex food"],
}


def _find_by_keywords(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    pats = SERIES_KEYWORDS.get(key, [])
    if not pats:
        return None
    best_col = None
    best_nn = -1
    for col in df.columns:
        name = str(col).lower()
        if any(p in name for p in pats):
            t = pd.to_numeric(df[col], errors="coerce")
            nn = int(t.notna().sum())
            if nn > best_nn:
                best_nn = nn
                best_col = col
    if best_col is None:
        return None
    return pd.to_numeric(df[best_col], errors="coerce")


def _coalesce_series(candidates: List[pd.Series]) -> Optional[pd.Series]:
    """Combine multiple candidate series by taking the first non-null at each timestamp.

    Priority is the list order. Index is normalized to month-end.
    """
    if not candidates:
        return None
    prepared: List[pd.Series] = []
    for s in candidates:
        if s is None:
            continue
        x = _to_month_end(s)
        if isinstance(x, pd.Series) and x.notna().any():
            prepared.append(pd.to_numeric(x, errors="coerce"))
    if not prepared:
        return None
    base = prepared[0].copy()
    for s in prepared[1:]:
        try:
            base = base.combine_first(s)
        except Exception:
            # Align then combine
            idx = base.index.union(s.index)
            base = base.reindex(idx).combine_first(s.reindex(idx))
    return base


def _to_month_end(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce")
    try:
        if not isinstance(x.index, pd.DatetimeIndex):
            idx = pd.to_datetime(x.index, errors="coerce")
            x = pd.Series(x.values, index=idx)
        # Normalize to month-end without introducing extra NaNs
        x.index = x.index.to_period("M").to_timestamp("M")
    except Exception:
        # best effort
        try:
            x.index = pd.to_datetime(x.index)
        except Exception:
            pass
    return x


def align_monthly(*series: pd.Series, ffill_limit: int = 6) -> Tuple[List[pd.Series], bool]:
    """Align input series to month-end and return intersection-only alignment.

    - Converts to month-end frequency via .asfreq('M')
    - Forward-fills up to ffill_limit months
    - Returns (aligned_series_list, has_overlap)
    """
    prepared: List[pd.Series] = []
    for s in series:
        if s is None:
            prepared.append(pd.Series(dtype=float))
            continue
        x = _to_month_end(s)
        if ffill_limit and ffill_limit > 0:
            x = x.ffill(limit=int(ffill_limit))
        prepared.append(x)

    # Determine intersection of non-null dates across provided series
    # Skip completely empty/non-informative series to avoid eliminating overlap
    non_null_indices: List[pd.DatetimeIndex] = []
    for x in prepared:
        if isinstance(x.index, pd.DatetimeIndex):
            idx_nn = x.dropna().index
            if idx_nn.size > 0:
                non_null_indices.append(idx_nn)
    if not non_null_indices:
        return prepared, False
    idx = non_null_indices[0]
    for ni in non_null_indices[1:]:
        idx = idx.intersection(ni)
    # Relaxed overlap: require at least 3 months when only two inputs are present,
    # default to 6+ otherwise.
    min_required = 3 if len(non_null_indices) <= 2 else 6
    has_overlap = len(idx) >= min_required
    if has_overlap:
        prepared = [x.reindex(idx) for x in prepared]
    return prepared, has_overlap


def first_present(df: pd.DataFrame, names: Sequence[str]) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    # case-insensitive fallback
    lower_map = {str(c).lower(): c for c in df.columns}
    for n in names:
        if str(n).lower() in lower_map:
            col = lower_map[str(n).lower()]
            return pd.to_numeric(df[col], errors="coerce")
    return None


def yoy_from_level(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce")
    return x.pct_change(12) * 100.0


def level_from_yoy(s: pd.Series, base: Optional[float] = None) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce")
    # Reconstruct approximate level via cumulative product
    if base is None:
        base = 100.0
    lvl = pd.Series(index=x.index, dtype=float)
    lvl.iloc[0] = float(base)
    for i in range(1, len(x)):
        try:
            lvl.iloc[i] = lvl.iloc[i - 1] * (1.0 + (x.iloc[i] if np.isfinite(x.iloc[i]) else 0.0))
        except Exception:
            lvl.iloc[i] = np.nan
    return lvl


def ensure_series(
    df: pd.DataFrame,
    target_name: str,
    synonyms: Sequence[str],
    level_synonyms: Sequence[str],
    make_yoy: bool = False,
) -> Optional[pd.Series]:
    # 1) exact or synonym
    s = first_present(df, [target_name] + list(synonyms))
    if s is not None:
        return s
    # 2) derive from levels if requested
    if make_yoy and level_synonyms:
        lvl = first_present(df, list(level_synonyms))
        if lvl is not None:
            return yoy_from_level(lvl)
    # 3) last resort: return first level if not making YoY
    if not make_yoy and level_synonyms:
        lvl = first_present(df, list(level_synonyms))
        if lvl is not None:
            return lvl
    return None


def ensure_series_by_key(df: pd.DataFrame, target_key: str) -> Optional[pd.Series]:
    """Resolve a canonical series using alias keys. If *_YOY requested and
    only level exists, derive YoY via pct_change(12)."""
    # 1) try direct/alias candidates and coalesce to fill gaps
    cand_names = aliases_for(target_key)
    cand_series: List[pd.Series] = []
    for name in cand_names:
        if name in df.columns or str(name).lower() in {str(c).lower() for c in df.columns}:
            s = first_present(df, [name])
            if s is not None:
                cand_series.append(s)
    s_alias = _coalesce_series(cand_series)
    if s_alias is not None:
        return s_alias
    # 2) loose keyword match
    s_kw = _find_by_keywords(df, target_key)
    if s_kw is not None:
        return s_kw
    # derive YoY for CPI series
    if target_key == "CPI_YOY":
        lvl_names = aliases_for("CPI_LVL")
        candidates = [first_present(df, [n]) for n in lvl_names]
        lvl = _coalesce_series([c for c in candidates if c is not None]) or _find_by_keywords(df, "CPI_LVL")
        return yoy_from_level(lvl) if lvl is not None else None
    if target_key == "CORECPI_YOY":
        lvl_names = aliases_for("CORECPI_LVL")
        candidates = [first_present(df, [n]) for n in lvl_names]
        lvl = _coalesce_series([c for c in candidates if c is not None]) or _find_by_keywords(df, "CORECPI_LVL")
        return yoy_from_level(lvl) if lvl is not None else None
    return None


