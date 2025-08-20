from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.drawing.image import Image as XLImage


def safe_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in cols if c in df.columns]


def rolling_percentile_cone(s: pd.Series, window: int = 60, q: Tuple[float, float, float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
    if s is None or len(s) == 0:
        return pd.DataFrame(index=pd.Index([], name=s.index.name if hasattr(s, 'index') else None))
    x = pd.to_numeric(s, errors="coerce")
    def _perc(y: pd.Series, p: float) -> float:
        try:
            arr = y.values.astype(float)
            if not np.isfinite(arr).any():
                return np.nan
            with np.errstate(all="ignore"):
                return float(np.nanpercentile(arr, p * 100))
        except Exception:
            return np.nan
    p10 = x.rolling(window, min_periods=max(6, window // 6)).apply(lambda y: _perc(y, q[0]), raw=False)
    p50 = x.rolling(window, min_periods=max(6, window // 6)).apply(lambda y: _perc(y, q[1]), raw=False)
    p90 = x.rolling(window, min_periods=max(6, window // 6)).apply(lambda y: _perc(y, q[2]), raw=False)
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}).reindex(index=x.index)


def breadth_z(df_z: pd.DataFrame) -> pd.DataFrame:
    if df_z is None or df_z.empty:
        return pd.DataFrame(columns=["pct_pos", "pct_pos1"])
    z = df_z.apply(pd.to_numeric, errors="coerce")
    denom = (z.notna()).sum(axis=1).replace(0, np.nan)
    pct_pos = (z.gt(0).sum(axis=1) / denom * 100).astype(float)
    pct_pos1 = (z.gt(1).sum(axis=1) / denom * 100).astype(float)
    return pd.DataFrame({"pct_pos": pct_pos, "pct_pos1": pct_pos1}).fillna(0.0)


def yoy(x: pd.Series) -> pd.Series:
    if x is None or len(x) == 0:
        return pd.Series(dtype=float)
    s = pd.to_numeric(x, errors="coerce")
    return s.pct_change(12).replace([np.inf, -np.inf], np.nan)


def lead_lag_corr(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    out = []
    if x is None or y is None or len(x) < 3 or len(y) < 3:
        return pd.DataFrame(columns=["lag", "corr"])
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if x.empty or y.empty:
        return pd.DataFrame(columns=["lag", "corr"])
    idx = x.index.intersection(y.index)
    x = x.reindex(idx)
    y = y.reindex(idx)
    for k in range(-max_lag, max_lag + 1):
        try:
            if k < 0:
                r = x.corr(y.shift(-k))
            else:
                r = x.shift(k).corr(y)
            out.append((k, float(r)))
        except Exception:
            out.append((k, np.nan))
    return pd.DataFrame(out, columns=["lag", "corr"]).dropna()


def zscore(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or abs(sd) < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def to_periodic_rf(rf_annual: float, periods: int) -> float:
    if periods <= 0:
        return 0.0
    try:
        return (1.0 + float(rf_annual)) ** (1.0 / float(periods)) - 1.0
    except Exception:
        return 0.0


def credit_spread(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "credit_spread" in df.columns:
        return pd.to_numeric(df["credit_spread"], errors="coerce")
    if "BAA" in df.columns and "AAA" in df.columns:
        a = pd.to_numeric(df["BAA"], errors="coerce")
        b = pd.to_numeric(df["AAA"], errors="coerce")
        return (a - b).dropna()
    return None


def make_move(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "MOVE" in df.columns:
        return pd.to_numeric(df["MOVE"], errors="coerce")
    return None


def theme_composite(df: pd.DataFrame, components: Sequence[str], *, series_sign: Optional[dict] = None) -> Optional[pd.Series]:
    """Create a composite series from a list of component column names.

    Strategy:
    - Prefer *_ZScore columns if available for each component
    - Else use raw columns and z-score them robustly
    - Return the mean across available component z-scores
    """
    if df is None or df.empty or not components:
        return None
    z_cols = []
    for c in components:
        zc = f"{c}_ZScore"
        if zc in df.columns:
            z_cols.append(zc)
        elif c in df.columns:
            z_cols.append(c)
    if not z_cols:
        # fallback: try any column that matches key fragment
        for c in components:
            matches = [col for col in df.columns if str(col).lower() == str(c).lower()]
            z_cols.extend(matches)
    if not z_cols:
        return None
    sub = df[z_cols].apply(pd.to_numeric, errors="coerce")
    # If we got raw columns (not *_ZScore), z-score them
    if not all(col.endswith("_ZScore") for col in z_cols):
        sub = sub.apply(zscore, axis=0)
    # Apply orientation if provided
    if series_sign:
        oriented = {}
        for col in sub.columns:
            base = col[:-7] if col.endswith("_ZScore") else col
            oriented[col] = sub[col] * float(series_sign.get(base, 1))
        sub = pd.DataFrame(oriented)
    comp = sub.mean(axis=1).dropna()
    return comp if not comp.empty else None


def theme_composite_from_cfg(df: pd.DataFrame, cfg, theme_key: str) -> Optional[pd.Series]:
    """Build a balanced composite using config groups/sub-buckets.

    Order of precedence for component lists:
    1) cfg.GROUPS[theme_key]
    2) cfg.THEME_GROUPS[theme_key]
    3) cfg.FEATURE_GROUPS[ThemeName]
    4) cfg.SUB_BUCKETS[ThemeName] (averaged across sub-buckets equally)
    """
    theme_key_l = str(theme_key).lower()
    theme_name_map = {
        "growth": "Growth",
        "inflation": "Inflation",
        "risk": "Risk",
        "credit": "Risk",
        "housing": "Housing",
        "fx": "FX",
        "external": "FX",
    }
    theme_name = theme_name_map.get(theme_key_l, theme_key)

    # Try flat groups first
    group = getattr(cfg, "GROUPS", {}).get(theme_key_l, []) or getattr(cfg, "THEME_GROUPS", {}).get(theme_key_l, [])
    if group:
        return theme_composite(df, group, series_sign=getattr(cfg, "SERIES_SIGN", None))

    # Try FEATURE_GROUPS with proper-cased key
    feat = getattr(cfg, "FEATURE_GROUPS", {}).get(theme_name, [])
    if feat:
        return theme_composite(df, feat, series_sign=getattr(cfg, "SERIES_SIGN", None))

    # Try SUB_BUCKETS balanced
    subs = getattr(cfg, "SUB_BUCKETS", {}).get(theme_name, {})
    if subs:
        sub_series = []
        for _, cols in subs.items():
            s = theme_composite(df, cols, series_sign=getattr(cfg, "SERIES_SIGN", None))
            if s is not None and not s.empty:
                sub_series.append(s)
        if sub_series:
            base = pd.concat(sub_series, axis=1)
            return base.mean(axis=1).dropna()
    return None


def save_fig(fig, outdir: Path, filename: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    try:
        print(f"Saved chart: {path}")
    except Exception:
        pass
    return path


def insert_image(ws: Worksheet, image_path: Path, anchor_cell: str, *, max_width: int | None = None, max_height: int | None = None) -> None:
    try:
        img = XLImage(str(image_path))
        try:
            ow, oh = int(img.width), int(img.height)
        except Exception:
            ow = oh = None
        if ow and oh and (max_width or max_height):
            sw = (max_width / ow) if max_width else 1.0
            sh = (max_height / oh) if max_height else 1.0
            s = min(sw, sh)
            if s < 1.0:
                img.width = int(ow * s)
                img.height = int(oh * s)
        # Ensure the anchor cell exists by writing a harmless value
        try:
            _ = ws[anchor_cell].value
        except Exception:
            pass
        ws.add_image(img, anchor_cell)
        try:
            print(f"Inserted image at {anchor_cell}: {image_path}")
        except Exception:
            pass
    except Exception:
        pass


def choose_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column.

    Strategy:
    - Exact match in order provided
    - Fallback: case-insensitive substring match
    """
    cols = list(df.columns)
    # Exact
    for c in candidates:
        if c in cols:
            return c
    # Case-insensitive exact
    lower_map = {str(col).lower(): col for col in cols}
    for c in candidates:
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    # Substring fallback
    for c in candidates:
        lc = str(c).lower()
        for col in cols:
            if lc in str(col).lower():
                return col
    return None


def find_series(df: pd.DataFrame, keywords: Sequence[str], *, excludes: Sequence[str] | None = None, min_non_na: int = 12) -> Optional[str]:
    """Find a column whose name contains any of the keywords (case-insensitive)
    and has at least min_non_na numeric observations. Exclude names that contain
    any of the excludes fragments.
    """
    if df is None or df.empty:
        return None
    excludes = excludes or []
    for col in df.columns:
        name = str(col)
        n = name.lower()
        if any(k.lower() in n for k in keywords) and not any(x.lower() in n for x in excludes):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= min_non_na:
                return col
    return None


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if len(x) == 0:
        return x
    base = x.dropna()
    if base.empty:
        return x
    return x / (base.iloc[0] if base.iloc[0] != 0 else 1.0)


