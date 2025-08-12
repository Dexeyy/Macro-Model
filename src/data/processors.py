import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Optional, Tuple
import os
from config import config  # configuration (FRED series, paths)
from src.utils.contracts import validate_frame, ProcessedMacroFrame, PerformanceFrame
from src.excel.excel_live import group_features
from src.features.economic_indicators import build_theme_composites
from src.features.pca_factors import fit_theme_pca
from src.utils.helpers import load_yaml_config, get_regimes_config
from src.utils.zscores import robust_zscore_rolling, pick_window_minp, default_window_minp_for_type
from typing import Literal
from src.data.vintage_fetcher import apply_publication_lags as apply_publication_lags_v2
from src.utils.factors import build_factor as public_build_factor
############ New robust preparation utilities ############

def _winsorize_series(s: pd.Series, zmax: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd is None or pd.isna(sd) or sd == 0:
        return s
    lo = mu - zmax * sd
    hi = mu + zmax * sd
    return s.clip(lower=lo, upper=hi)


def _hampel_series(s: pd.Series, window: int, zmax: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    roll = s.rolling(window=window, min_periods=max(1, int(window * 0.5)))
    med = roll.median()
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)
    denom = 1.4826 * mad.replace(0.0, np.nan)
    z = (s - med) / denom
    out = s.copy()
    mask = z.abs() > float(zmax)
    out[mask] = med[mask]
    return out


def prepare_missing(
    df: pd.DataFrame,
    tcode_map: Dict[str, int] | None = None,
    *,
    outlier_method: str = "hampel",
    zmax: float = 6.0,
    hampel_window: int = 36,
) -> pd.DataFrame:
    """Apply Stock–Watson tcodes, series-mapping, and robust outlier cleaning.

    - Applies differencing/log transforms per tcode_map (if provided)
    - Harmonizes financial condition series names (VIXCLS->VIX, ^MOVE->MOVE, BAA-AAA)
    - Drops constant or all-NaN columns
    - Applies Hampel (median/MAD) or winsorize cleaning after transforms
    """
    if df is None or df.empty:
        return df
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            pass
    data = data.sort_index()

    # Series mapping first to ensure tcodes apply to final names as needed
    data = _harmonize_financial_condition_names(data)

    # Apply tcodes if provided
    if tcode_map:
        def safe_log(s: pd.Series) -> pd.Series:
            x = pd.to_numeric(s, errors="coerce")
            x = x.where(x > 0)
            return np.log(x)
        transformed = {}
        for col, tcode in tcode_map.items():
            if col not in data.columns:
                continue
            s = pd.to_numeric(data[col], errors="coerce")
            try:
                if tcode == 1:
                    transformed[col] = s
                elif tcode == 2:
                    transformed[col] = s.diff(1)
                elif tcode == 4:
                    transformed[col] = s.diff(4)
                elif tcode == 5:
                    transformed[col] = safe_log(s)
                elif tcode == 6:
                    transformed[col] = safe_log(s).diff(1)
                elif tcode == 7:
                    transformed[col] = safe_log(s).diff(1).diff(1)
                else:
                    transformed[col] = s
            except Exception:
                transformed[col] = s
        if transformed:
            for k, v in transformed.items():
                data[k] = v

    # Drop constant or entirely NaN columns
    to_drop = []
    for c in list(data.columns):
        s = pd.to_numeric(data[c], errors="coerce")
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            to_drop.append(c)
    if to_drop:
        data = data.drop(columns=to_drop, errors="ignore")

    # Outlier cleaning on remaining numeric columns
    num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    if num_cols:
        if outlier_method.lower() in ("winsorize", "winsor", "clip"):
            for c in num_cols:
                data[c] = _winsorize_series(data[c], zmax=float(zmax))
        else:
            for c in num_cols:
                data[c] = _hampel_series(data[c], window=int(hampel_window), zmax=float(zmax))

    return data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LagAligner:
    """
    Aligns time series data with different publication lags.
    
    This class handles the common problem in macroeconomic analysis where
    different indicators are published with varying lags. It ensures that
    only data that would have been available at a given point in time is used.
    """
    
    def __init__(self, base_date=None):
        """
        Initialize the LagAligner.
        
        Args:
            base_date: Optional reference date for alignment
        """
        self.base_date = base_date
        self.lag_dict = {}
        self.aligned_data = None
    
    def add_series(self, name: str, series: pd.Series, lag_months: int = 0):
        """
        Add a time series with its publication lag.
        
        Args:
            name: Name of the series
            series: The time series data
            lag_months: Publication lag in months
        """
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")
        
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        
        self.lag_dict[name] = {
            'series': series,
            'lag_months': lag_months
        }
        logger.info(f"Added series '{name}' with {lag_months} month(s) lag")
        return self
    
    def align_data(self, target_date=None):
        """
        Align all series based on their publication lags.
        
        Args:
            target_date: Date to align data to (defaults to base_date or latest date)
            
        Returns:
            DataFrame with aligned data
        """
        if not self.lag_dict:
            logger.warning("No series added to align")
            return pd.DataFrame()
        
        # Determine target date if not provided
        if target_date is None:
            if self.base_date is not None:
                target_date = self.base_date
            else:
                # Find the latest date across all series
                latest_dates = []
                for info in self.lag_dict.values():
                    if not info['series'].empty:
                        latest_dates.append(info['series'].index.max())
                
                if latest_dates:
                    target_date = max(latest_dates)
                else:
                    logger.warning("No valid dates found in any series")
                    return pd.DataFrame()
        
        # Create aligned series
        aligned_series = {}
        for name, info in self.lag_dict.items():
            series = info['series']
            lag = info['lag_months']
            
            # Calculate the effective date based on lag
            effective_date = pd.Timestamp(target_date) - pd.DateOffset(months=lag)
            
            # Get the value at or before the effective date
            if not series.empty:
                # Find the closest date at or before the effective date
                valid_dates = series.index[series.index <= effective_date]
                if not valid_dates.empty:
                    closest_date = valid_dates.max()
                    aligned_series[name] = series.loc[closest_date]
                else:
                    # Early sample often lacks data; avoid log spam
                    logger.debug(f"No data available for '%s' before %s", name, effective_date)
                    aligned_series[name] = np.nan
            else:
                logger.debug(f"Series '%s' is empty", name)
                aligned_series[name] = np.nan
        
        # Create DataFrame with aligned data
        self.aligned_data = pd.Series(aligned_series, name=target_date)
        return self.aligned_data
    
    def align_all_dates(self, start_date=None, end_date=None, freq='ME'):
        """
        Align data for a range of dates.
        
        Args:
            start_date: Start date for alignment
            end_date: End date for alignment
            freq: Frequency for date range
            
        Returns:
            DataFrame with aligned data for all dates
        """
        if not self.lag_dict:
            logger.warning("No series added to align")
            return pd.DataFrame()
        
        # Determine date range if not provided
        if start_date is None or end_date is None:
            all_dates = []
            for info in self.lag_dict.values():
                if not info['series'].empty:
                    all_dates.extend(info['series'].index)
            
            all_dates = pd.DatetimeIndex(all_dates).unique().sort_values()
            
            if all_dates.empty:
                logger.warning("No valid dates found in any series")
                return pd.DataFrame()
            
            if start_date is None:
                start_date = all_dates.min()
            
            if end_date is None:
                end_date = all_dates.max()
        
        # Create date range (month-end by default)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Align data for each date
        aligned_data_list = []
        for date in date_range:
            aligned_series = self.align_data(date)
            aligned_data_list.append(aligned_series)
        
        # Combine into DataFrame
        self.aligned_data = pd.DataFrame(aligned_data_list)
        return self.aligned_data


def _get_publication_lags_from_yaml() -> Dict[str, int]:
    """Best-effort load of per-series publication lags (in months) from YAML.

    Expects a mapping under key 'publication_lags' like:
      publication_lags:
        CPIAUCSL: 1
        UNRATE: 0
    """
    # Prefer unified regimes config loader to ensure consistent source
    cfg = get_regimes_config() or {}
    raw = cfg.get("publication_lags") or {}
    if not isinstance(raw, dict):
        return {}
    # Ensure int months
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    try:
        logger.info("Loaded %d publication lag entries", len(out))
    except Exception:
        pass
    return out


def apply_publication_lags(df: pd.DataFrame, lag_map: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Align a DataFrame to real-time availability using LagAligner.

    - If lag_map is None, attempts to read from YAML; defaults to 0 if unknown.
    - Works column-wise and returns a DataFrame of aligned values at monthly frequency.
    """
    if df is None or df.empty:
        return df

    # Ensure datetime index and monthly frequency ordering
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            return df
    data = data.sort_index()

    if lag_map is None:
        lag_map = _get_publication_lags_from_yaml()

    aligner = LagAligner()
    for col in data.columns:
        try:
            # Skip derived/engineered columns; only align raw/base series
            name = str(col)
            if (
                name.startswith("F_") or name.startswith("PC_") or
                name.endswith("_YoY") or name.endswith("_MoM") or
                "_MA" in name or name.endswith("_ZScore")
            ):
                continue
            series = data[col]
            # Default to 0 months if not specified
            lag_months = int(lag_map.get(col, 0))
            aligner.add_series(col, series, lag_months=lag_months)
        except Exception:
            # Skip non-series-like columns silently
            continue

    aligned = aligner.align_all_dates(start_date=data.index.min(), end_date=data.index.max(), freq="ME")
    # Preserve original column order where possible
    if not aligned.empty:
        # Join aligned base series back into full feature set
        base_cols = [c for c in data.columns if c in aligned.columns]
        out = data.copy()
        out[base_cols] = aligned[base_cols]
        # Fallback: if a base column became entirely NaN after alignment, restore original
        fully_nan_cols = [c for c in base_cols if out[c].notna().sum() == 0 and data[c].notna().sum() > 0]
        if fully_nan_cols:
            logger.warning("RT alignment produced all-NaN for %d columns; restoring unlagged values: %s",
                           len(fully_nan_cols), fully_nan_cols[:8])
            for c in fully_nan_cols:
                out[c] = data[c]
        out.index.name = data.index.name
        return out
    return data


def process_macro_data(macro_data_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean & enrich raw macro series for downstream modeling and Excel dashboards.
    Steps:
    - Resample to month-end and forward-fill
    - Compute YoY and MoM for raw series in config.FRED_SERIES
    - Add 3M and 6M moving averages for YoY/MoM series
    - Add 24M rolling z-score for all numeric columns
    - Provide common alias columns (e.g., CPI_YoY, GDP_YoY)
    - Build per-theme composites and optional PCA factors
    - Persist a parquet snapshot for reuse
    """
    monthly = macro_data_raw.resample("ME").last().ffill()

    fred_codes = list(getattr(config, "FRED_SERIES", {}).keys())
    # Build transforms in a dict to avoid fragmentation and silence pct_change fill warnings
    _new_tf_cols: Dict[str, pd.Series] = {}
    for code in fred_codes:
        if code in monthly.columns:
            try:
                s = monthly[code]
                _new_tf_cols[f"{code}_YoY"] = s.pct_change(12, fill_method=None)
                _new_tf_cols[f"{code}_MoM"] = s.pct_change(1, fill_method=None)
            except Exception as exc:
                logger.warning("YoY/MoM calc failed for %s: %s", code, exc)
    if _new_tf_cols:
        monthly = monthly.join(pd.DataFrame(_new_tf_cols), how="left")

    pct_cols = [c for c in monthly.columns if c.endswith("_YoY") or c.endswith("_MoM")]
    if pct_cols:
        roll3 = monthly[pct_cols].rolling(3, min_periods=1).mean().add_suffix("_3M_MA")
        roll6 = monthly[pct_cols].rolling(6, min_periods=1).mean().add_suffix("_6M_MA")
        monthly = monthly.join(roll3, how="left").join(roll6, how="left")

    numeric_cols = [c for c in monthly.columns if pd.api.types.is_numeric_dtype(monthly[c])]
    zscore_cols: Dict[str, pd.Series] = {}
    for col in numeric_cols:
        mean24 = monthly[col].rolling(24, min_periods=1).mean()
        std24 = monthly[col].rolling(24, min_periods=1).std()
        z = np.where(std24 != 0, (monthly[col] - mean24) / std24, 0)
        zscore_cols[f"{col}_ZScore"] = pd.Series(z, index=monthly.index)
    if zscore_cols:
        monthly = monthly.join(pd.DataFrame(zscore_cols), how="left")

    alias_map = {"CPI_YoY": "CPIAUCSL_YoY", "GDP_YoY": "GDPC1_YoY"}
    for alias, source in alias_map.items():
        if alias not in monthly.columns and source in monthly.columns:
            monthly[alias] = monthly[source]

    try:
        validate_frame(monthly, ProcessedMacroFrame, validate=False, where="process_macro_data")
    except Exception:
        logger.debug("ProcessedMacroFrame validation raised unexpectedly", exc_info=True)

    try:
        mapping = group_features(monthly)
        theme_key_map = {
            "Growth & Labour": "growth",
            "Inflation & Liquidity": "inflation",
            "Credit & Risk": "credit_risk",
            "Housing": "housing",
            "FX & Commodities": "external",
        }
        canonical_mapping: Dict[str, List[str]] = {}
        for k, cols in mapping.items():
            canon = theme_key_map.get(k, k)
            canonical_mapping.setdefault(canon, []).extend(cols)

        from src.features.economic_indicators import build_theme_composites
        composites = build_theme_composites(monthly, canonical_mapping)
        base_features = monthly.join(composites, how="left")

        cfg = load_yaml_config() or {}
        use_pca = bool((cfg.get("themes") or {}).get("use_pca") or cfg.get("use_pca"))
        features_with_f = base_features
        if use_pca:
            # Optional PCA hooks (not enabled by default)
            pca_map = {
                "growth": "PC_Growth",
                "inflation": "PC_Inflation",
                "liquidity": "PC_Liquidity",
                "credit_risk": "PC_CreditRisk",
                "housing": "PC_Housing",
                "external": "PC_External",
            }
            # Placeholder for PCA injection if enabled later
            features_with_f = base_features  # keep as-is for now

        # Persist legacy combined parquet for backward compatibility
        legacy_path = os.path.join("Data", "processed", "macro_features.parquet")
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        features_with_f.to_parquet(legacy_path)
        logger.info("Saved macro features (legacy) to %s", legacy_path)

        # Build and persist both retro and real-time advanced feature sets
        try:
            features_retro = create_advanced_features(features_with_f, mode="retro")
            features_rt = create_advanced_features(features_with_f, mode="rt")
            out_dir = os.path.join("Data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            retro_path = os.path.join(out_dir, "macro_features_retro.parquet")
            rt_path = os.path.join(out_dir, "macro_features_rt.parquet")
            features_retro.to_parquet(retro_path)
            features_rt.to_parquet(rt_path)
            logger.info("Saved retro features -> %s", retro_path)
            logger.info("Saved real-time features -> %s", rt_path)
        except Exception as sub_exc:
            logger.warning("Failed to create/save RT/retro feature sets: %s", sub_exc)
    except Exception as exc:
        logger.warning("Failed to build/persist theme composites: %s", exc)
        features_with_f = monthly

    return features_with_f

def create_advanced_features(data: pd.DataFrame, mode: str = "retro") -> pd.DataFrame:
    # This function is defined later in the file with full RT discipline and factor rebuild.
    # Keep a thin delegator here if earlier imports reference it.
    return create_advanced_features.__wrapped__(data, mode)  # type: ignore

def _harmonize_financial_condition_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common synonyms for Financial Conditions inputs in-place safe copy.

    - If 'VIX' missing and 'VIXCLS' present -> rename VIXCLS -> VIX
    - If 'MOVE' missing and '^MOVE' present -> rename '^MOVE' -> MOVE
    - CorporateBondSpread mapping:
        * If 'CorporateBondSpread' missing and 'credit_spread' exists, copy it
        * Else if both 'BAA' and 'AAA' exist, create CorporateBondSpread = BAA - AAA
    """
    out = df.copy()
    if "VIX" not in out.columns and "VIXCLS" in out.columns:
        out["VIX"] = pd.to_numeric(out["VIXCLS"], errors="coerce")
    if "MOVE" not in out.columns and "^MOVE" in out.columns:
        out["MOVE"] = pd.to_numeric(out["^MOVE"], errors="coerce")
    if "CorporateBondSpread" not in out.columns:
        if "credit_spread" in out.columns:
            out["CorporateBondSpread"] = pd.to_numeric(out["credit_spread"], errors="coerce")
        elif "BAA" in out.columns and "AAA" in out.columns:
            try:
                out["CorporateBondSpread"] = pd.to_numeric(out["BAA"], errors="coerce") - pd.to_numeric(out["AAA"], errors="coerce")
            except Exception:
                pass
    return out


def _series_type_of(name: str, cfg: Optional[Dict] = None) -> Optional[str]:
    """Lookup series type bucket from YAML config (fast/typical/slow)."""
    # Normalize transformed names to base
    base = str(name)
    for suf in ("_YoY", "_MoM", "_3M_MA", "_6M_MA", "_MA3_YoY", "_QoQAnn", "_ZScore"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    bucket_map = None
    if cfg and isinstance(cfg.get("series_types"), dict):
        bucket_map = cfg.get("series_types")
    elif cfg and isinstance(cfg.get("zscore"), dict) and isinstance(cfg["zscore"].get("series_types"), dict):
        bucket_map = cfg["zscore"]["series_types"]
    if isinstance(bucket_map, dict):
        t = bucket_map.get(base)
        if isinstance(t, str):
            return t
    return None


def _choose_window_params(col: str, s: pd.Series, cfg: Optional[Dict]) -> Dict[Literal["window", "min_periods"], int]:
    series_type = _series_type_of(col, cfg)
    if series_type:
        return default_window_minp_for_type(series_type)
    return pick_window_minp(s)


def _build_factor(
    df: pd.DataFrame,
    bases: list[str],
    *,
    mode: Literal["RT", "RETRO"],
    z_window: int | None = None,
    z_min: int | None = None,
    min_k: int = 2,
) -> tuple[pd.Series, list[str], pd.Series]:
    """Transform-adaptive factor builder with robust rolling z and coverage-aware averaging.

    Returns
    - factor: averaged robust z across chosen cols (NaN where coverage < min_k)
    - used_cols: resolved input column names that existed and were non-constant
    - coverage: per-date count of non-NaN inputs used
    """
    cfg = load_yaml_config() or {}
    priority_rt = ["_YoY", "_QoQAnn", "_MA3_YoY", "_MoM", ""]
    priority_retro = ["_YoY", "_MA3_YoY", "_QoQAnn", "_MoM", ""]
    priority = priority_rt if str(mode).upper() == "RT" else priority_retro

    resolved: list[str] = []
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
            # Fallback to base level
            if pd.to_numeric(df[base], errors="coerce").notna().any():
                chosen = base
        if chosen is not None:
            resolved.append(chosen)

    if not resolved:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    # Compute robust z per series
    z_cols: dict[str, pd.Series] = {}
    for col in resolved:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.nunique(dropna=True) <= 1:
                    continue
        params = {"window": z_window, "min_periods": z_min}
        if z_window is None or z_min is None:
            picked = _choose_window_params(col, s, cfg)
            params = {"window": int(z_window or picked["window"]), "min_periods": int(z_min or picked["min_periods"])}
        z = robust_zscore_rolling(s, window=int(params["window"]), min_periods=int(params["min_periods"]))
        z_cols[col] = z

    if not z_cols:
        return pd.Series(index=df.index, dtype=float), [], pd.Series(0, index=df.index, dtype="Int64")

    Z = pd.concat(z_cols, axis=1)
    coverage = Z.notna().sum(axis=1)
    factor = Z.mean(axis=1)
    factor = factor.where(coverage >= int(min_k))
    return factor.astype(float), list(z_cols.keys()), coverage.astype("Int64")


def create_advanced_features(data: pd.DataFrame, mode: str = "retro") -> pd.DataFrame:

    """
    Create advanced macroeconomic features for regime detection.
    
    Args:
        data: DataFrame with macro data
        
    Returns:
        DataFrame with added advanced features
    """
    try:
        # Normalize to monthly index first to ensure consistent transforms
        base = data.copy()
        if not isinstance(base.index, pd.DatetimeIndex):
            try:
                base.index = pd.to_datetime(base.index)
            except Exception:
                pass
        base = base.sort_index()
        try:
            base = base.resample("ME").last()
            base = base.ffill()
        except Exception:
            # if resample fails, proceed with original index
            pass
        # Align to publication lags if requested (real-time track)
        rt_mode = isinstance(mode, str) and mode.lower() in ("rt", "real-time", "realtime")
        if rt_mode:
            try:
                # Use the new v2 lag alignment which accepts explicit lags if provided via YAML
                base = apply_publication_lags_v2(base, None)
                logger.info("Applied publication-lag alignment for real-time feature set")
            except Exception as exc:
                logger.warning("Lag alignment failed; proceeding without alignment: %s", exc)

        # Create a working copy for feature engineering
        result_data = base.copy()

        # In true real-time mode, recompute transforms and composites from the aligned raw panel
        if rt_mode:
            try:
                # Drop any previously computed derived columns so we can rebuild them on aligned raw
                drop_cols = [
                    c for c in result_data.columns
                    if (
                        c.endswith("_YoY") or c.endswith("_MoM") or
                        c.endswith("_3M_MA") or c.endswith("_6M_MA") or
                        c.endswith("_ZScore") or c == "CPI_YoY"
                    ) and not c.startswith("F_")
                ]
                if drop_cols:
                    result_data = result_data.drop(columns=drop_cols, errors="ignore")

                # Recompute YoY/MoM for configured FRED series on aligned data (batched to avoid fragmentation)
                fred_codes = list(getattr(config, "FRED_SERIES", {}).keys())
                _rt_tf_cols: Dict[str, pd.Series] = {}
                for code in fred_codes:
                    if code in result_data.columns:
                        try:
                            s = pd.to_numeric(result_data[code], errors="coerce").reindex(result_data.index)
                            s = s.ffill(limit=1)
                            _rt_tf_cols[f"{code}_YoY"] = s.pct_change(12, fill_method=None)
                            _rt_tf_cols[f"{code}_MoM"] = s.pct_change(1, fill_method=None)
                        except Exception as exc:
                            logger.debug("RT YoY/MoM calc failed for %s: %s", code, exc)
                if _rt_tf_cols:
                    result_data = result_data.join(pd.DataFrame(_rt_tf_cols), how="left")

                # Rolling 3M/6M means for rate series
                pct_cols = [c for c in result_data.columns if c.endswith("_YoY") or c.endswith("_MoM")]
                if pct_cols:
                    roll3 = result_data[pct_cols].rolling(3, min_periods=1).mean().add_suffix("_3M_MA")
                    roll6 = result_data[pct_cols].rolling(6, min_periods=1).mean().add_suffix("_6M_MA")
                    result_data = result_data.join(roll3, how="left").join(roll6, how="left")

                # Restore common aliases expected elsewhere
                alias_map = {"CPI_YoY": "CPIAUCSL_YoY", "GDP_YoY": "GDPC1_YoY"}
                for alias, source in alias_map.items():
                    if alias not in result_data.columns and source in result_data.columns:
                        result_data[alias] = result_data[source]

                # Provide widely used engineered helpers used by themes
                if "BAA" in result_data.columns and "AAA" in result_data.columns and "credit_spread" not in result_data.columns:
                    try:
                        result_data["credit_spread"] = pd.to_numeric(result_data["BAA"], errors="coerce") - pd.to_numeric(result_data["AAA"], errors="coerce")
                    except Exception:
                        pass

                # Recompute z-scores later via factor builder; skip full-matrix legacy ZScore columns

                # Rebuild theme composites F_* from aligned inputs, with guarded backfill for sparse families
                try:
                    mapping = group_features(result_data)
                    theme_key_map = {
                        "Growth & Labour": "growth",
                        "Inflation & Liquidity": "inflation",
                        "Credit & Risk": "credit_risk",
                        "Housing": "housing",
                        "FX & Commodities": "external",
                    }
                    canonical_mapping: Dict[str, List[str]] = {}
                    for k, cols in mapping.items():
                        canon = theme_key_map.get(k, k)
                        canonical_mapping.setdefault(canon, []).extend(cols)

                    # Optional: if critical families are empty in RT, merge retro/base just for those inputs
                    try:
                        # Track columns prior to backfill so we can recompute transforms for any newly added bases
                        _cols_before_backfill = set(result_data.columns)
                        families = {
                                "inflation": [
                                    "CPIAUCSL", "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "PPICMM"
                                ],
                                "credit": [
                                    "AAA", "BAA"
                                ],
                                "housing": [
                                    "HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW",
                                    "PERMIT", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW",
                                ],
                        }
                        # 1) Try retro backfill
                        retro_path = os.path.join("Data", "processed", "macro_features_retro.parquet")
                        retro_df = pd.read_parquet(retro_path) if os.path.exists(retro_path) else None
                        # 2) If still empty, try base macro_features (fresh fetch)
                        base_path = os.path.join("Data", "processed", "macro_features.parquet")
                        base_df = pd.read_parquet(base_path) if os.path.exists(base_path) else None
                        touched_bases: set[str] = set()
                        for fam, base_list in families.items():
                            # Evaluate coverage using any of these bases (even if missing entirely)
                            present_cols = [c for c in base_list if c in result_data.columns]
                            has_any_data = bool(present_cols) and result_data[present_cols].notna().sum().sum() > 0
                            if not has_any_data:
                                backfilled = False
                                # Try retro then base; add columns if missing, or replace if empty
                                if retro_df is not None:
                                    for c in base_list:
                                        if c in retro_df.columns:
                                            if c not in result_data.columns or result_data.get(c, pd.Series(dtype=float)).notna().sum() == 0:
                                                result_data[c] = retro_df[c]
                                                backfilled = True
                                                touched_bases.add(c)
                                if not backfilled and base_df is not None:
                                    for c in base_list:
                                        if c in base_df.columns:
                                            if c not in result_data.columns or result_data.get(c, pd.Series(dtype=float)).notna().sum() == 0:
                                                result_data[c] = base_df[c]
                                                backfilled = True
                                                touched_bases.add(c)
                                if backfilled:
                                    logger.warning("Backfilled %s family from retro/base for composite stabilization", fam)
                        # Recompute transforms for any touched bases (added or replaced)
                        if touched_bases:
                            _rt_extra_tf: Dict[str, pd.Series] = {}
                            for code in sorted(touched_bases):
                                try:
                                    if code in result_data.columns:
                                        s = pd.to_numeric(result_data[code], errors="coerce").reindex(result_data.index)
                                        s = s.ffill(limit=1)
                                        if s.notna().any():
                                            _rt_extra_tf[f"{code}_YoY"] = s.pct_change(12, fill_method=None)
                                            _rt_extra_tf[f"{code}_MoM"] = s.pct_change(1, fill_method=None)
                                except Exception:
                                    continue
                            if _rt_extra_tf:
                                result_data = result_data.join(pd.DataFrame(_rt_extra_tf), how="left")
                    except Exception:
                        logger.debug("Family backfill guard skipped", exc_info=True)

                    # Build transform-adaptive F_* factors using robust z and coverage awareness
                    existing_f = [c for c in result_data.columns if c.startswith("F_")]
                    if existing_f:
                        result_data = result_data.drop(columns=existing_f, errors="ignore")
                    # Harmonize financial condition names prior to building
                    result_data = _harmonize_financial_condition_names(result_data)

                    # Define compact, strong base lists per factor (coverage-aware at build)
                    bases_map: Dict[str, list[str]] = {
                        "F_Growth": ["INDPRO", "PAYEMS", "GDPC1", "CUMFNS"],
                        "F_Inflation": ["CPIAUCSL", "PCEPI", "PPICMM"],
                        "F_CreditRisk": ["AAA", "BAA", "credit_spread"],
                        # Include national + regionals to maximize RT coverage
                        "F_Housing": [
                            "HOUST", "PERMIT",
                            "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW",
                            "PERMITNE", "PERMITMW", "PERMITS", "PERMITW",
                        ],
                        "F_External": ["DCOILWTICO", "TWEXAFEGSMTH", "EXJPUS"],
                    }

                    # Build each factor and collect coverage diagnostics
                    coverage_diag = {}
                    used_cols_log = {}
                    # Load cfg once for logging/tuning
                    cfg = load_yaml_config() or {}
                    for fname, bases in bases_map.items():
                        f, used_cols, cov = public_build_factor(
                            result_data, bases,
                            mode="RT" if rt_mode else "RETRO",
                            z_window=None, z_min=None,
                            min_k=2,
                        )
                        # bridge single-month gaps
                        result_data[fname] = f.ffill(limit=1)
                        coverage_diag[fname] = pd.DataFrame({
                            "count": cov,
                            "ratio": cov.astype(float) / max(1, len(used_cols) or 1),
                            "low_coverage": (cov < 2),
                        })
                        used_cols_log[fname] = used_cols
                        # Log tuned params per used column
                        try:
                            param_log = {}
                            for col in used_cols:
                                ty = _series_type_of(col, cfg)
                                if ty:
                                    params = default_window_minp_for_type(ty)
                                else:
                                    params = pick_window_minp(pd.to_numeric(result_data[col], errors="coerce"))
                                param_log[col] = (int(params["window"]), int(params["min_periods"]))
                            if param_log:
                                logger.info("Tuned window/min_periods for %s: %s", fname, param_log)
                        except Exception:
                            logger.debug("Param logging failed for %s", fname, exc_info=True)
                    # Financial Conditions composite from harmonized inputs
                    fin_inputs = [c for c in ["NFCI", "VIX", "MOVE", "CorporateBondSpread"] if c in result_data.columns]
                    if fin_inputs:
                        fin_cols = {}
                        # ensure cfg defined
                        cfg = cfg or (load_yaml_config() or {})
                        for col in fin_inputs:
                            ty = _series_type_of(col, cfg)
                            params = default_window_minp_for_type(ty) if ty else pick_window_minp(result_data[col])
                            fin_cols[col] = robust_zscore_rolling(pd.to_numeric(result_data[col], errors="coerce"),
                                                                  window=params["window"], min_periods=params["min_periods"])
                        FinZ = pd.concat(fin_cols, axis=1)
                        fin_cov = FinZ.notna().sum(axis=1)
                        fin_factor = FinZ.mean(axis=1).where(fin_cov >= 2)
                        result_data["FinConditions_Composite"] = fin_factor.ffill(limit=1)
                        coverage_diag["FinConditions_Composite"] = pd.DataFrame({
                            "count": fin_cov,
                            "ratio": fin_cov.astype(float) / max(1, len(fin_cols)),
                            "low_coverage": (fin_cov < 2),
                        })
                        used_cols_log["FinConditions_Composite"] = list(fin_cols.keys())
                        # Log window/min for FinConditions
                        try:
                            param_log = {}
                            for col in fin_cols.keys():
                                ty = _series_type_of(col, cfg)
                                if ty:
                                    params = default_window_minp_for_type(ty)
                                else:
                                    params = pick_window_minp(pd.to_numeric(result_data[col], errors="coerce"))
                                param_log[col] = (int(params["window"]), int(params["min_periods"]))
                            if param_log:
                                logger.info("Tuned window/min_periods for FinConditions: %s", param_log)
                        except Exception:
                            logger.debug("Param logging failed for FinConditions", exc_info=True)

                    # Emit concise logs
                    for k, cols in used_cols_log.items():
                        logger.info("Factor %s used inputs: %s", k, cols)
                    # Warn on early low coverage in first 24 months
                    for fac, diag in coverage_diag.items():
                        first_win = diag.iloc[:24]
                        if not first_win.empty and (first_win["ratio"] < 0.5).any():
                            logger.warning("Low coverage (<50%%) for %s in early sample", fac)
                    # Save last 12 months coverage snapshot
                    try:
                        snap = {k: v.tail(12).assign(factor=k) for k, v in coverage_diag.items()}
                        snap_df = pd.concat(snap.values(), axis=0)
                        out_path = os.path.join("Output", "diagnostics")
                        os.makedirs(out_path, exist_ok=True)
                        snap_df.to_csv(os.path.join(out_path, "coverage_snapshot.csv"))
                        # Save full coverage diagnostics
                        full = {k: v.assign(factor=k) for k, v in coverage_diag.items()}
                        full_df = pd.concat(full.values(), axis=0)
                        full_df.to_csv(os.path.join(out_path, "coverage_full.csv"))
                    except Exception:
                        logger.debug("Failed to persist coverage snapshot", exc_info=True)
                    logger.info("Rebuilt F_* and FinConditions with robust z and coverage diagnostics")
                except Exception as exc:
                    logger.warning("Failed to rebuild RT composites: %s", exc)
            except Exception:
                logger.debug("RT recomputation block raised", exc_info=True)
        
        # Yield curve dynamics
        if all(col in result_data.columns for col in ['DGS10', 'DGS2']):
            # Yield curve slope: Difference between 10Y and 2Y Treasury yields
            # - Positive: Normal yield curve (expansion)
            # - Negative: Inverted yield curve (recession signal)
            result_data['YieldCurve_Slope'] = result_data['DGS10'] - result_data['DGS2']
            logger.info("Created YieldCurve_Slope feature")
            
            # Yield curve curvature: Measures the non-linearity of the yield curve
            # - High positive: Steep in middle, flat at ends (mid-cycle)
            # - Negative: Humped yield curve (late cycle)
            if 'DGS5' in result_data.columns:
                result_data['YieldCurve_Curvature'] = 2*result_data['DGS5'] - result_data['DGS2'] - result_data['DGS10']
                logger.info("Created YieldCurve_Curvature feature")
            
            # Slope momentum: Rate of change in yield curve slope
            # - Positive: Steepening yield curve (early expansion)
            # - Negative: Flattening yield curve (late cycle)
            result_data['YieldCurve_Slope_Mom'] = result_data['YieldCurve_Slope'].diff(3)
            logger.info("Created YieldCurve_Slope_Mom feature")
        
        # Inflation expectations and real rates
        if all(col in result_data.columns for col in ['DGS10', 'T10YIE']):
            # Real 10Y rate: Nominal yield minus inflation expectations
            # - High positive: Restrictive monetary policy
            # - Negative: Accommodative policy, often during crises
            result_data['RealRate_10Y'] = result_data['DGS10'] - result_data['T10YIE']
            logger.info("Created RealRate_10Y feature")
            
            # Real rate momentum: Change in real rates
            # - Rising: Tightening financial conditions
            # - Falling: Easing financial conditions
            result_data['RealRate_10Y_Mom'] = result_data['RealRate_10Y'].diff(3)
            logger.info("Created RealRate_10Y_Mom feature")
        
        # Financial conditions composite with synonyms/fallbacks
        fin_series: Dict[str, pd.Series] = {}
        if 'NFCI' in result_data.columns:
            fin_series['NFCI'] = pd.to_numeric(result_data['NFCI'], errors='coerce')
        # VIX: prefer 'VIX', else 'VIXCLS'
        if 'VIX' in result_data.columns:
            fin_series['VIX'] = pd.to_numeric(result_data['VIX'], errors='coerce')
        elif 'VIXCLS' in result_data.columns:
            fin_series['VIX'] = pd.to_numeric(result_data['VIXCLS'], errors='coerce')
        # MOVE: include if present
        if 'MOVE' in result_data.columns:
            fin_series['MOVE'] = pd.to_numeric(result_data['MOVE'], errors='coerce')
        # Corporate bond spread: use provided, or credit_spread, or (BAA-AAA)
        if 'CorporateBondSpread' in result_data.columns:
            fin_series['CorporateBondSpread'] = pd.to_numeric(result_data['CorporateBondSpread'], errors='coerce')
        elif 'credit_spread' in result_data.columns:
            fin_series['CorporateBondSpread'] = pd.to_numeric(result_data['credit_spread'], errors='coerce')
        elif 'BAA' in result_data.columns and 'AAA' in result_data.columns:
            try:
                fin_series['CorporateBondSpread'] = pd.to_numeric(result_data['BAA'], errors='coerce') - pd.to_numeric(result_data['AAA'], errors='coerce')
            except Exception:
                pass
        if fin_series:
            try:
                fin_df = pd.DataFrame(fin_series)
                def _z(x: pd.Series) -> pd.Series:
                    denom = x.std()
                    if denom is None or pd.isna(denom) or denom == 0:
                        denom = 1.0
                    return (x - x.mean()) / denom
                fin_data = fin_df.apply(_z)
                result_data['FinConditions_Composite'] = fin_data.mean(axis=1)
                logger.info(f"Created FinConditions_Composite feature using {list(fin_series.keys())}")
            except Exception as e:
                logger.warning(f"Error creating FinConditions_Composite: {e}")
        
        # Growth momentum
        growth_cols = ['GDP_YoY', 'INDPRO_YoY', 'NFP_YoY']
        available_growth_cols = [col for col in growth_cols if col in result_data.columns]
        if available_growth_cols:
            # Calculate 3-month change in growth metrics
            # - Positive: Accelerating growth (early/mid expansion)
            # - Negative: Decelerating growth (late cycle/contraction)
            for col in available_growth_cols:
                try:
                    result_data[f'{col}_Mom'] = result_data[col].diff(3)
                    logger.info(f"Created momentum feature for {col}")
                except Exception as e:
                    logger.warning(f"Error creating momentum feature for {col}: {e}")
        
        # Liquidity measures
        if 'M2SL' in result_data.columns:
            try:
                # M2 growth rate: Year-over-year change in money supply
                # - High: Expansionary monetary policy
                # - Low/Negative: Contractionary monetary policy
                result_data['M2_YoY'] = result_data['M2SL'].pct_change(12, fill_method=None) * 100
                logger.info("Created M2_YoY feature")
                
                # Real M2 growth: Money supply growth adjusted for inflation
                # - High: Expansionary in real terms
                # - Low/Negative: Contractionary in real terms
                if 'CPI_YoY' in result_data.columns:
                    result_data['RealM2_Growth'] = result_data['M2_YoY'] - result_data['CPI_YoY']
                    logger.info("Created RealM2_Growth feature")
            except Exception as e:
                logger.warning(f"Error creating M2 growth features: {e}")
        
        logger.info("Successfully created advanced features (%s)", mode)
        return result_data
        
    except Exception as e:
        logger.error(f"Error creating advanced features: {e}")
        return data

def calculate_returns(prices_df):
    """
    Calculate returns from price data.
    
    Args:
        prices_df: DataFrame with price data
        
    Returns:
        DataFrame with returns
    """
    try:
        # Resample to month-end and calculate returns
        returns_monthly = prices_df.resample('ME').last().pct_change()
        
        # Handle missing values
        returns_monthly = returns_monthly.ffill(limit=3)
        returns_monthly = returns_monthly.dropna(how='all')
        
        logger.info(f"Successfully calculated returns with shape: {returns_monthly.shape}")
        return returns_monthly
    
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return None

def merge_macro_and_asset_data(macro_data, asset_returns, regime_col):
    """
    Merge macro data and asset returns.
    
    Args:
        macro_data: DataFrame with macro data
        asset_returns: DataFrame with asset returns
        regime_col: Name of the regime column
        
    Returns:
        DataFrame with merged data
    """
    try:
        # Ensure both DataFrames have DatetimeIndex
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data.index = pd.to_datetime(macro_data.index)
        if not isinstance(asset_returns.index, pd.DatetimeIndex):
            asset_returns.index = pd.to_datetime(asset_returns.index)

        # Normalize to month-end timestamps to ensure alignment
        macro_norm = macro_data.copy()
        asset_norm = asset_returns.copy()
        macro_norm.index = macro_norm.index.to_period('M').to_timestamp('M')
        asset_norm.index = asset_norm.index.to_period('M').to_timestamp('M')

        # Perform the merge on normalized indexes
        data_for_analysis = macro_norm[[regime_col]].merge(
            asset_norm,
            left_index=True,
            right_index=True,
            how="inner",
        )

        # Drop rows with missing regime label
        data_for_analysis = data_for_analysis.dropna(subset=[regime_col])

        # Drop rows where all asset returns are missing, but keep rows with partial coverage
        asset_cols = list(asset_norm.columns)
        if asset_cols:
            mask_all_na_assets = data_for_analysis[asset_cols].isna().all(axis=1)
            data_for_analysis = data_for_analysis.loc[~mask_all_na_assets]

        logger.info(
            f"Successfully merged macro and asset data with shape: {data_for_analysis.shape}"
        )
        return data_for_analysis
    
    except Exception as e:
        logger.error(f"Error merging macro and asset data: {e}")
        return None

def calculate_regime_performance(data_for_analysis, regime_col):
    """
    Calculate performance metrics by regime.
    
    Args:
        data_for_analysis: DataFrame with regime and asset returns
        regime_col: Name of the regime column
        
    Returns:
        DataFrame with performance metrics by regime
    """
    try:
        # Get asset columns (all columns except the regime column)
        asset_columns = [col for col in data_for_analysis.columns if col != regime_col]
        
        if not asset_columns:
            logger.error("No asset columns found in data_for_analysis")
            return None
        
        # Group by regime and calculate performance metrics
        regime_performance_monthly = data_for_analysis.groupby(regime_col)[asset_columns].agg(
            ['mean', 'std', 'count']
        )
        
        # Create annualized metrics
        annualized_data_dict = {}
        risk_free_rate_annual = 0.0
        
        for asset_col in asset_columns:
            monthly_mean = regime_performance_monthly[(asset_col, 'mean')]
            monthly_std = regime_performance_monthly[(asset_col, 'std')]
            months_count = regime_performance_monthly[(asset_col, 'count')]
            
            # Annualize returns and volatility
            ann_mean_return = monthly_mean * 12
            ann_std_dev = monthly_std * np.sqrt(12)
            
            # Store in dictionary
            annualized_data_dict[(asset_col, 'Ann_Mean_Return')] = ann_mean_return
            annualized_data_dict[(asset_col, 'Ann_Std_Dev')] = ann_std_dev
            annualized_data_dict[(asset_col, 'Months_Count')] = months_count
            
            # Calculate Sharpe ratios where volatility > 0
            current_asset_sharpe_ratios = pd.Series(np.nan, index=regime_performance_monthly.index)
            valid_std_dev_mask = pd.notna(ann_std_dev) & (ann_std_dev != 0)
            
            current_asset_sharpe_ratios.loc[valid_std_dev_mask] = \
                (ann_mean_return[valid_std_dev_mask] - risk_free_rate_annual) / ann_std_dev[valid_std_dev_mask]
            
            annualized_data_dict[(asset_col, 'Ann_Sharpe_Ratio')] = current_asset_sharpe_ratios
        
        # Create final annualized performance DataFrame
        regime_performance_annualized = pd.DataFrame(annualized_data_dict)
        regime_performance_annualized.columns.names = ['Asset', 'Metric']
        
        logger.info(f"Successfully calculated regime performance metrics")

        # Non-breaking validation (warnings only)
        try:
            validate_frame(regime_performance_annualized, PerformanceFrame, validate=False, where="calculate_regime_performance")
        except Exception:
            logger.debug("PerformanceFrame validation raised unexpectedly", exc_info=True)

        return regime_performance_annualized
    
    except Exception as e:
        logger.error(f"Error calculating regime performance: {e}")
        return None

def normalize_features(data, feature_columns=None, scaler=None):
    """
    Normalize features using StandardScaler.
    
    Args:
        data: DataFrame with features
        feature_columns: List of columns to normalize (if None, all numeric columns)
        scaler: Pre-fitted scaler (if None, a new one will be created)
        
    Returns:
        Tuple of (normalized DataFrame, scaler)
    """
    try:
        # If no feature columns specified, use all numeric columns
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create a copy of the data
        normalized_data = data.copy()
        
        # Create or use provided scaler
        if scaler is None:
            scaler = StandardScaler()
            # Fit the scaler on the data
            scaler.fit(data[feature_columns])
        
        # Transform the data
        normalized_data[feature_columns] = scaler.transform(data[feature_columns])
        
        logger.info(f"Successfully normalized {len(feature_columns)} features")
        return normalized_data, scaler
    
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return data, None

def create_feature_groups(data, feature_groups):
    """
    Create aggregate features from feature groups.
    
    Args:
        data: DataFrame with features
        feature_groups: Dictionary mapping group names to lists of feature columns
        
    Returns:
        DataFrame with added feature group aggregates
    """
    try:
        # Create a copy of the data
        result_data = data.copy()
        
        # Process each feature group
        for group_name, columns in feature_groups.items():
            # Filter to columns that actually exist in the data
            valid_columns = [col for col in columns if col in data.columns]
            
            if not valid_columns:
                logger.warning(f"No valid columns found for group '{group_name}'")
                continue
            
            # Calculate mean of the group
            result_data[f'{group_name}_Mean'] = data[valid_columns].mean(axis=1)
            
            # Calculate first principal component (simplified version)
            if len(valid_columns) > 1:
                try:
                    from sklearn.decomposition import PCA
                    # Handle NaN values
                    group_data = data[valid_columns].fillna(data[valid_columns].mean())
                    # Apply PCA
                    pca = PCA(n_components=1)
                    pca_result = pca.fit_transform(group_data)
                    result_data[f'{group_name}_PC1'] = pca_result.flatten()
                    logger.info(f"Created PCA for group '{group_name}'")
                except Exception as e:
                    logger.warning(f"Error creating PCA for group '{group_name}': {e}")
            
            logger.info(f"Created aggregates for feature group '{group_name}'")
        
        return result_data
    
    except Exception as e:
        logger.error(f"Error creating feature groups: {e}")
        return data

def create_interaction_features(data, feature_pairs):
    """
    Create interaction features between pairs of features.
    
    Args:
        data: DataFrame with features
        feature_pairs: List of tuples of feature column pairs
        
    Returns:
        DataFrame with added interaction features
    """
    try:
        # Create a copy of the data
        result_data = data.copy()
        
        # Process each feature pair
        for col1, col2 in feature_pairs:
            if col1 in data.columns and col2 in data.columns:
                # Create interaction feature
                interaction_name = f"{col1}_x_{col2}"
                result_data[interaction_name] = data[col1] * data[col2]
                logger.info(f"Created interaction feature '{interaction_name}'")
            else:
                if col1 not in data.columns:
                    logger.warning(f"Column '{col1}' not found in data")
                if col2 not in data.columns:
                    logger.warning(f"Column '{col2}' not found in data")
        
        return result_data
    
    except Exception as e:
        logger.error(f"Error creating interaction features: {e}")
        return data

def handle_outliers(data, columns=None, method='winsorize', threshold=3.0):
    """
    Handle outliers in the data.
    
    Args:
        data: DataFrame with features
        columns: List of columns to process (if None, all numeric columns)
        method: Method to handle outliers ('winsorize', 'clip', or 'remove')
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with handled outliers
    """
    try:
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create a copy of the data
        result_data = data.copy()
        
        # Process each column
        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data")
                continue
            
            # Calculate Z-scores
            mean = data[col].mean()
            std = data[col].std()
            
            if std == 0:
                logger.warning(f"Standard deviation is zero for column '{col}', skipping")
                continue
            
            z_scores = (data[col] - mean) / std
            
            # Handle outliers based on method
            if method == 'winsorize':
                # Winsorize: cap values at threshold
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                result_data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Winsorized outliers in column '{col}'")
                
            elif method == 'clip':
                # Clip: replace outliers with NaN
                outlier_mask = abs(z_scores) > threshold
                result_data.loc[outlier_mask, col] = np.nan
                logger.info(f"Clipped {outlier_mask.sum()} outliers in column '{col}'")
                
            elif method == 'remove':
                # Remove: drop rows with outliers
                outlier_mask = abs(z_scores) > threshold
                if outlier_mask.any():
                    result_data = result_data.loc[~outlier_mask]
                    logger.info(f"Removed {outlier_mask.sum()} rows with outliers in column '{col}'")
            
            else:
                logger.warning(f"Unknown outlier handling method: '{method}'")
        
        return result_data
    
    except Exception as e:
        logger.error(f"Error handling outliers: {e}")
        return data 




