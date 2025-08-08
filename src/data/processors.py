import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Optional, Tuple

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
                    logger.warning(f"No data available for '{name}' before {effective_date}")
                    aligned_series[name] = np.nan
            else:
                logger.warning(f"Series '{name}' is empty")
                aligned_series[name] = np.nan
        
        # Create DataFrame with aligned data
        self.aligned_data = pd.Series(aligned_series, name=target_date)
        return self.aligned_data
    
    def align_all_dates(self, start_date=None, end_date=None, freq='M'):
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
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Align data for each date
        aligned_data_list = []
        for date in date_range:
            aligned_series = self.align_data(date)
            aligned_data_list.append(aligned_series)
        
        # Combine into DataFrame
        self.aligned_data = pd.DataFrame(aligned_data_list)
        return self.aligned_data

<<<<<<< Updated upstream
def process_macro_data(macro_data_raw):
=======
import os
from config import config  # added import at top earlier
from src.utils.contracts import validate_frame, ProcessedMacroFrame, PerformanceFrame
from src.excel.excel_live import group_features
from src.features.economic_indicators import build_theme_composites
from src.features.pca_factors import fit_theme_pca
from src.utils.helpers import load_yaml_config

def process_macro_data(macro_data_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean & enrich raw macro series.

    Steps
    -----
    1.  Monthly resample (end-of-month).
    2.  Compute *YoY* and *MoM* % changes for every **original** series listed in
        ``config.FRED_SERIES`` that exists in the dataset.
    3.  Rolling moving-averages (3- and 6-month) for every numeric column that
        *already* contains a YoY or MoM value (to avoid averaging the raw price
        levels of different scales).
    4.  24-month rolling z-score for all numeric columns.
>>>>>>> Stashed changes
    """
    Process raw macro data:
    - Resample to monthly frequency
    - Calculate YoY and MoM changes
    - Calculate moving averages
    - Calculate Z-scores
    
    Args:
        macro_data_raw: Raw macro data from FRED
        
    Returns:
        DataFrame with processed macro data
    """
    try:
        # Resample to monthly frequency
        macro_data_monthly = macro_data_raw.resample('M').last()
        macro_data_monthly = macro_data_monthly.ffill()
        
        logger.info("Successfully resampled data to monthly frequency")
        
        # Calculate YoY and MoM changes
        for col in ['CPI', 'CoreCPI', 'PPI', 'GDP', 'NFP', 'RetailSales', 'INDPRO', 'WageGrowth']:
            if col in macro_data_monthly.columns:
                try:
                    macro_data_monthly[f'{col}_YoY'] = macro_data_monthly[col].pct_change(12) * 100
                    macro_data_monthly[f'{col}_MoM'] = macro_data_monthly[col].pct_change(1) * 100
                    logger.info(f"Successfully calculated changes for {col}")
                except Exception as e:
                    logger.warning(f"Error calculating changes for {col}: {e}")

<<<<<<< Updated upstream
        # Calculate GDP Gap
        if 'GDP' in macro_data_monthly.columns and 'RealPotentialGDP' in macro_data_monthly.columns:
            try:
                macro_data_monthly['GDP_Gap'] = (
                    (macro_data_monthly['GDP'] / macro_data_monthly['RealPotentialGDP']) - 1
                ) * 100
                logger.info("Successfully calculated GDP Gap")
            except Exception as e:
                logger.warning(f"Error calculating GDP Gap: {e}")
=======
    # ------------------------------------------------------------------
    # 1. Resample to month-end and forward-fill
    # ------------------------------------------------------------------
    monthly = macro_data_raw.resample("ME").last().ffill()
>>>>>>> Stashed changes

        # Calculate moving averages
        for col in ['UNRATE', 'CPI_YoY', 'UMCSENT']:
            if col in macro_data_monthly.columns:
                try:
                    macro_data_monthly[f'{col}_3M_MA'] = macro_data_monthly[col].rolling(window=3, min_periods=1).mean()
                    macro_data_monthly[f'{col}_6M_MA'] = macro_data_monthly[col].rolling(window=6, min_periods=1).mean()
                    logger.info(f"Successfully calculated moving averages for {col}")
                except Exception as e:
                    logger.warning(f"Error calculating moving averages for {col}: {e}")

        # Calculate Z-scores
        for col in macro_data_monthly.columns:
            if pd.api.types.is_numeric_dtype(macro_data_monthly[col]):
                try:
                    rolling_mean = macro_data_monthly[col].rolling(window=24, min_periods=1).mean()
                    rolling_std = macro_data_monthly[col].rolling(window=24, min_periods=1).std()
                    
                    # Safe division to handle zeros in std
                    macro_data_monthly[f'{col}_ZScore'] = np.where(
                        rolling_std != 0,
                        (macro_data_monthly[col] - rolling_mean) / rolling_std,
                        0
                    )
                    logger.info(f"Successfully calculated Z-score for {col}")
                except Exception as e:
                    logger.warning(f"Error calculating Z-score for {col}: {e}")

<<<<<<< Updated upstream
        # Drop NaNs
        macro_data_featured = macro_data_monthly.dropna(
            subset=[col for col in macro_data_monthly.columns if 'YoY' in col or 'MA' in col],
            how='all'
        )
        
        logger.info("Successfully created macro_data_featured")
        return macro_data_featured
    
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

def create_advanced_features(data):
=======
    # ------------------------------------------------------------------
    # 3. Moving averages for rate/percentage series (YoY / MoM columns)
    # ------------------------------------------------------------------
    pct_cols = [c for c in monthly.columns if c.endswith("_YoY") or c.endswith("_MoM")]
    if pct_cols:
        roll3 = monthly[pct_cols].rolling(3, min_periods=1).mean()
        roll3.columns = [f"{c}_3M_MA" for c in pct_cols]
        roll6 = monthly[pct_cols].rolling(6, min_periods=1).mean()
        roll6.columns = [f"{c}_6M_MA" for c in pct_cols]
        monthly = monthly.join(roll3, how="left").join(roll6, how="left")

    # ------------------------------------------------------------------
    # 4. 24-month rolling z-score for all numeric columns
    #    Build in a dict and join once to avoid fragmentation warnings
    # ------------------------------------------------------------------
    zscore_cols = {}
    numeric_cols = [c for c in monthly.columns if pd.api.types.is_numeric_dtype(monthly[c])]
    for col in numeric_cols:
        mean24 = monthly[col].rolling(24, min_periods=1).mean()
        std24 = monthly[col].rolling(24, min_periods=1).std()
        z = np.where(std24 != 0, (monthly[col] - mean24) / std24, 0)
        zscore_cols[f"{col}_ZScore"] = pd.Series(z, index=monthly.index)
    if zscore_cols:
        monthly = monthly.join(pd.DataFrame(zscore_cols), how="left")

    # Provide simple aliases expected elsewhere in the codebase
    alias_map = {
        "CPI_YoY": "CPIAUCSL_YoY",
        "GDP_YoY": "GDPC1_YoY",
    }
    for alias, source in alias_map.items():
        if alias not in monthly.columns and source in monthly.columns:
            monthly[alias] = monthly[source]

    logger.info("process_macro_data: finished feature engineering (%d cols)", len(monthly.columns))

    # Non-breaking validation (warnings only)
    try:
        validate_frame(monthly, ProcessedMacroFrame, validate=False, where="process_macro_data")
    except Exception:
        # Should not raise when validate=False, but guard anyway
        logger.debug("ProcessedMacroFrame validation raised unexpectedly", exc_info=True)

    # ------------------------------------------------------------------
    # Build and persist per-theme composites alongside raw features
    # ------------------------------------------------------------------
    try:
        mapping = group_features(monthly)
        # Convert dashboard-oriented group names to canonical theme keys
        theme_key_map = {
            "Growth & Labour": "growth",
            "Inflation & Liquidity": "inflation",  # liquidity will be captured by keywords too
            "Credit & Risk": "credit_risk",
            "Housing": "housing",
            "FX & Commodities": "external",
        }
        canonical_mapping = {}
        for k, cols in mapping.items():
            canon = theme_key_map.get(k, k)
            canonical_mapping.setdefault(canon, []).extend(cols)

        composites = build_theme_composites(monthly, canonical_mapping)
        features_with_f = monthly.join(composites, how="left")

        # Optional PCA per theme (controlled by YAML config: use_pca)
        cfg = load_yaml_config() or {}
        use_pca = bool((cfg.get("themes") or {}).get("use_pca") or cfg.get("use_pca"))
        if use_pca:
            pca_map = {
                "growth": "PC_Growth",
                "inflation": "PC_Inflation",
                "liquidity": "PC_Liquidity",
                "credit_risk": "PC_CreditRisk",
                "housing": "PC_Housing",
                "external": "PC_External",
            }
            for theme_key, pc_name in pca_map.items():
                cols = canonical_mapping.get(theme_key, [])
                if not cols:
                    continue
                pc1, _params = fit_theme_pca(features_with_f, cols, n_components=1)
                features_with_f[pc_name] = pc1
        # Save parquet with raw + engineered + F_*
        import os
        out_path = os.path.join("Data", "processed", "macro_features.parquet")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        features_with_f.to_parquet(out_path)
        logger.info("Saved macro features with composites to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to build/persist theme composites: %s", exc)
        features_with_f = monthly

    return features_with_f

def create_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
>>>>>>> Stashed changes
    """
    Create advanced macroeconomic features for regime detection.
    
    Args:
        data: DataFrame with macro data
        
    Returns:
        DataFrame with added advanced features
    """
    try:
        # Create a copy of the data to avoid modifying the original
        result_data = data.copy()
        
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
        
        # Financial conditions composite
        fin_cols = ['NFCI', 'VIX', 'MOVE', 'CorporateBondSpread']
        available_fin_cols = [col for col in fin_cols if col in result_data.columns]
        if available_fin_cols:
            try:
                # Standardize and average financial stress indicators
                # - Positive: Tight financial conditions (stress)
                # - Negative: Easy financial conditions (complacency)
                fin_data = result_data[available_fin_cols].apply(lambda x: (x - x.mean()) / x.std())
                result_data['FinConditions_Composite'] = fin_data.mean(axis=1)
                logger.info(f"Created FinConditions_Composite feature using {available_fin_cols}")
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
                result_data['M2_YoY'] = result_data['M2SL'].pct_change(12) * 100
                logger.info("Created M2_YoY feature")
                
                # Real M2 growth: Money supply growth adjusted for inflation
                # - High: Expansionary in real terms
                # - Low/Negative: Contractionary in real terms
                if 'CPI_YoY' in result_data.columns:
                    result_data['RealM2_Growth'] = result_data['M2_YoY'] - result_data['CPI_YoY']
                    logger.info("Created RealM2_Growth feature")
            except Exception as e:
                logger.warning(f"Error creating M2 growth features: {e}")
        
        logger.info("Successfully created advanced features")
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
        
        # Perform the merge
        data_for_analysis = macro_data[[regime_col]].merge(
            asset_returns,
            left_index=True,
            right_index=True,
            how="inner"  # 'inner' keeps only dates present in both DataFrames
        )
        
        # Drop any rows with NaN values
        data_for_analysis = data_for_analysis.dropna()
        
        logger.info(f"Successfully merged macro and asset data with shape: {data_for_analysis.shape}")
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