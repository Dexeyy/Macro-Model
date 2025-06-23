"""
Comprehensive Data Cleaning and Normalization Framework

This module provides a unified framework for cleaning, normalizing, and transforming
data from different sources (FRED, Yahoo Finance, etc.) to ensure consistency
and reliability for macro-economic regime analysis.

Key Features:
- Missing value handling with multiple strategies
- Outlier detection and treatment
- Timestamp standardization across sources
- Unit conversion and standardization
- Data type validation and conversion
- Extensible architecture for new data sources
- Comprehensive logging and validation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingValueStrategy(Enum):
    """Strategies for handling missing values"""
    DROP = "drop"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE_LINEAR = "interpolate_linear"
    INTERPOLATE_TIME = "interpolate_time"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"

class OutlierMethod(Enum):
    """Methods for outlier detection and treatment"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    MODIFIED_Z_SCORE = "modified_z_score"

class OutlierTreatment(Enum):
    """Treatment options for detected outliers"""
    REMOVE = "remove"
    WINSORIZE = "winsorize"
    CAP = "cap"
    TRANSFORM = "transform"
    FLAG = "flag"

class NormalizationMethod(Enum):
    """Normalization methods"""
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    UNIT_VECTOR = "unit_vector"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations"""
    # Missing value handling
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.FORWARD_FILL
    missing_value_limit: Optional[int] = None  # Max consecutive NaN to fill
    constant_fill_value: float = 0.0
    knn_neighbors: int = 5
    
    # Outlier handling
    outlier_method: OutlierMethod = OutlierMethod.Z_SCORE
    outlier_treatment: OutlierTreatment = OutlierTreatment.WINSORIZE
    outlier_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)
    
    # Normalization
    normalization_method: NormalizationMethod = NormalizationMethod.STANDARD
    feature_range: Tuple[float, float] = (0, 1)
    
    # Data validation
    validate_dtypes: bool = True
    validate_ranges: bool = True
    validate_timestamps: bool = True
    
    # Logging
    verbose: bool = True

@dataclass
class DataSourceConfig:
    """Configuration for specific data sources"""
    name: str
    expected_frequency: str = "D"  # Daily, Monthly, etc.
    timezone: str = "UTC"
    date_format: Optional[str] = None
    expected_columns: List[str] = field(default_factory=list)
    column_mappings: Dict[str, str] = field(default_factory=dict)
    unit_conversions: Dict[str, float] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

class DataCleaner:
    """
    Main data cleaning and normalization class
    
    Provides comprehensive data cleaning capabilities with support for
    multiple data sources and configurable cleaning strategies.
    """
    
    def __init__(self, config: CleaningConfig = None):
        """
        Initialize the DataCleaner
        
        Args:
            config: CleaningConfig object with cleaning parameters
        """
        self.config = config or CleaningConfig()
        self.source_configs: Dict[str, DataSourceConfig] = {}
        self.fitted_scalers: Dict[str, Any] = {}
        self.cleaning_history: List[Dict] = []
        
        # Initialize imputers
        self._init_imputers()
        
        logger.info("DataCleaner initialized")
    
    def _init_imputers(self):
        """Initialize imputers for different strategies"""
        self.imputers = {
            MissingValueStrategy.MEAN: SimpleImputer(strategy='mean'),
            MissingValueStrategy.MEDIAN: SimpleImputer(strategy='median'),
            MissingValueStrategy.MODE: SimpleImputer(strategy='most_frequent'),
            MissingValueStrategy.CONSTANT: SimpleImputer(
                strategy='constant', 
                fill_value=self.config.constant_fill_value
            ),
            MissingValueStrategy.KNN: KNNImputer(n_neighbors=self.config.knn_neighbors)
        }
    
    def register_data_source(self, source_config: DataSourceConfig):
        """
        Register a data source configuration
        
        Args:
            source_config: DataSourceConfig object
        """
        self.source_configs[source_config.name] = source_config
        logger.info(f"Registered data source: {source_config.name}")
    
    def clean_data(self, 
                   data: pd.DataFrame, 
                   source_name: Optional[str] = None,
                   custom_config: Optional[CleaningConfig] = None) -> pd.DataFrame:
        """
        Main method to clean data
        
        Args:
            data: Input DataFrame
            source_name: Name of registered data source
            custom_config: Override default config for this operation
            
        Returns:
            Cleaned DataFrame
        """
        config = custom_config or self.config
        
        if self.config.verbose:
            logger.info(f"Starting data cleaning for {len(data)} rows, {len(data.columns)} columns")
        
        # Start cleaning history entry
        cleaning_entry = {
            'timestamp': datetime.now(),
            'source_name': source_name,
            'input_shape': data.shape,
            'operations': []
        }
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        try:
            # Step 1: Source-specific preprocessing
            if source_name and source_name in self.source_configs:
                cleaned_data = self._apply_source_config(cleaned_data, source_name)
                cleaning_entry['operations'].append('source_config_applied')
            
            # Step 2: Data type validation and conversion
            if config.validate_dtypes:
                cleaned_data = self._validate_and_convert_dtypes(cleaned_data, source_name)
                cleaning_entry['operations'].append('dtype_validation')
            
            # Step 3: Timestamp standardization
            if config.validate_timestamps:
                cleaned_data = self._standardize_timestamps(cleaned_data, source_name)
                cleaning_entry['operations'].append('timestamp_standardization')
            
            # Step 4: Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data, config)
            cleaning_entry['operations'].append('missing_values_handled')
            
            # Step 5: Handle outliers
            cleaned_data = self._handle_outliers(cleaned_data, config)
            cleaning_entry['operations'].append('outliers_handled')
            
            # Step 6: Data range validation
            if config.validate_ranges and source_name:
                cleaned_data = self._validate_ranges(cleaned_data, source_name)
                cleaning_entry['operations'].append('range_validation')
            
            # Step 7: Normalization (optional, can be done separately)
            # cleaned_data = self._normalize_data(cleaned_data, config)
            
            # Finalize cleaning history
            cleaning_entry['output_shape'] = cleaned_data.shape
            cleaning_entry['success'] = True
            
            if self.config.verbose:
                logger.info(f"Data cleaning completed. Shape: {data.shape} -> {cleaned_data.shape}")
            
        except Exception as e:
            cleaning_entry['success'] = False
            cleaning_entry['error'] = str(e)
            logger.error(f"Error during data cleaning: {e}")
            raise
        
        finally:
            self.cleaning_history.append(cleaning_entry)
        
        return cleaned_data
    
    def _apply_source_config(self, data: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Apply source-specific configuration"""
        config = self.source_configs[source_name]
        result = data.copy()
        
        # Apply column mappings
        if config.column_mappings:
            result = result.rename(columns=config.column_mappings)
            logger.info(f"Applied column mappings for {source_name}")
        
        # Apply unit conversions
        for column, conversion_factor in config.unit_conversions.items():
            if column in result.columns:
                result[column] = result[column] * conversion_factor
                logger.info(f"Applied unit conversion to {column}: factor {conversion_factor}")
        
        return result
    
    def _validate_and_convert_dtypes(self, data: pd.DataFrame, source_name: Optional[str]) -> pd.DataFrame:
        """Validate and convert data types"""
        result = data.copy()
        
        # Get expected data types if source is registered
        expected_dtypes = {}
        if source_name and source_name in self.source_configs:
            expected_dtypes = self.source_configs[source_name].data_types
        
        # Auto-detect and fix numeric columns with mixed types
        for column in result.columns:
            if result[column].dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(result[column], errors='coerce')
                    # If most values are numeric, convert the column
                    if numeric_series.notna().sum() / len(result) > 0.8:
                        result[column] = numeric_series
                        logger.info(f"Auto-converted mixed-type column {column} to numeric")
                except Exception:
                    pass
        
        # Apply expected data types if source is registered
        for column in result.columns:
            if column in expected_dtypes:
                expected_dtype = expected_dtypes[column]
                try:
                    if expected_dtype == 'datetime64[ns]':
                        result[column] = pd.to_datetime(result[column])
                    elif expected_dtype in ['float64', 'float32']:
                        result[column] = pd.to_numeric(result[column], errors='coerce')
                    elif expected_dtype in ['int64', 'int32']:
                        # Convert to float first to handle NaN, then to int where possible
                        result[column] = pd.to_numeric(result[column], errors='coerce')
                        # Only convert to int if no NaN values
                        if not result[column].isna().any():
                            result[column] = result[column].astype(expected_dtype)
                    
                    logger.info(f"Converted {column} to {expected_dtype}")
                except Exception as e:
                    logger.warning(f"Could not convert {column} to {expected_dtype}: {e}")
        
        return result
    
    def _standardize_timestamps(self, data: pd.DataFrame, source_name: Optional[str]) -> pd.DataFrame:
        """Standardize timestamp formats and timezones"""
        result = data.copy()
        
        # Get timezone info if source is registered
        target_timezone = "UTC"
        if source_name and source_name in self.source_configs:
            target_timezone = self.source_configs[source_name].timezone
        
        # Handle index if it's datetime
        if isinstance(result.index, pd.DatetimeIndex):
            if result.index.tz is not None:
                # Convert to UTC
                result.index = result.index.tz_convert('UTC')
            else:
                # Assume it's already in target timezone and localize
                result.index = result.index.tz_localize('UTC')
            
            logger.info("Standardized timestamp index to UTC")
        
        # Handle datetime columns
        datetime_columns = result.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        for col in datetime_columns:
            if hasattr(result[col].dtype, 'tz') and result[col].dtype.tz is not None:
                result[col] = result[col].dt.tz_convert('UTC')
            else:
                result[col] = result[col].dt.tz_localize('UTC')
            
            logger.info(f"Standardized timestamp column {col} to UTC")
        
        return result
    
    def _handle_missing_values(self, data: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        result = data.copy()
        strategy = config.missing_value_strategy
        
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        
        if strategy == MissingValueStrategy.DROP:
            result = result.dropna()
            logger.info("Dropped rows with missing values")
            
        elif strategy == MissingValueStrategy.FORWARD_FILL:
            result[numeric_columns] = result[numeric_columns].ffill(limit=config.missing_value_limit)
            logger.info("Applied forward fill to missing values")
            
        elif strategy == MissingValueStrategy.BACKWARD_FILL:
            result[numeric_columns] = result[numeric_columns].bfill(limit=config.missing_value_limit)
            logger.info("Applied backward fill to missing values")
            
        elif strategy == MissingValueStrategy.INTERPOLATE_LINEAR:
            result[numeric_columns] = result[numeric_columns].interpolate(method='linear')
            logger.info("Applied linear interpolation to missing values")
            
        elif strategy == MissingValueStrategy.INTERPOLATE_TIME:
            if isinstance(result.index, pd.DatetimeIndex):
                result[numeric_columns] = result[numeric_columns].interpolate(method='time')
                logger.info("Applied time-based interpolation to missing values")
            else:
                logger.warning("Time interpolation requires datetime index, falling back to linear")
                result[numeric_columns] = result[numeric_columns].interpolate(method='linear')
        
        elif strategy in [MissingValueStrategy.MEAN, MissingValueStrategy.MEDIAN, 
                         MissingValueStrategy.MODE, MissingValueStrategy.CONSTANT, 
                         MissingValueStrategy.KNN]:
            imputer = self.imputers[strategy]
            if len(numeric_columns) > 0:
                result[numeric_columns] = imputer.fit_transform(result[numeric_columns])
                logger.info(f"Applied {strategy.value} imputation to missing values")
        
        return result
    
    def _handle_outliers(self, data: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
        """Handle outliers based on detection method and treatment"""
        result = data.copy()
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = result[column].dropna()
            if len(series) == 0:
                continue
            
            # Detect outliers
            outlier_mask = self._detect_outliers(series, config)
            
            if outlier_mask.sum() == 0:
                continue
            
            # Apply treatment
            if config.outlier_treatment == OutlierTreatment.REMOVE:
                # Remove outlier rows
                result = result.loc[~outlier_mask]
                logger.info(f"Removed {outlier_mask.sum()} outlier rows in {column}")
                
            elif config.outlier_treatment == OutlierTreatment.WINSORIZE:
                # Winsorize outliers
                lower_percentile = config.winsorize_limits[0]
                upper_percentile = 1 - config.winsorize_limits[1]
                
                lower_bound = series.quantile(lower_percentile)
                upper_bound = series.quantile(upper_percentile)
                
                result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Winsorized outliers in {column}")
                
            elif config.outlier_treatment == OutlierTreatment.CAP:
                # Cap outliers at threshold
                if config.outlier_method == OutlierMethod.Z_SCORE:
                    mean_val = series.mean()
                    std_val = series.std()
                    lower_bound = mean_val - config.outlier_threshold * std_val
                    upper_bound = mean_val + config.outlier_threshold * std_val
                    
                    result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in {column} using z-score")
                
            elif config.outlier_treatment == OutlierTreatment.FLAG:
                # Add outlier flag column
                result[f"{column}_outlier_flag"] = outlier_mask
                logger.info(f"Added outlier flag for {column}")
        
        return result
    
    def _detect_outliers(self, series: pd.Series, config: CleaningConfig) -> pd.Series:
        """Detect outliers using specified method"""
        
        if config.outlier_method == OutlierMethod.Z_SCORE:
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > config.outlier_threshold
            
        elif config.outlier_method == OutlierMethod.MODIFIED_Z_SCORE:
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > config.outlier_threshold
            
        elif config.outlier_method == OutlierMethod.IQR:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.iqr_multiplier * IQR
            upper_bound = Q3 + config.iqr_multiplier * IQR
            return (series < lower_bound) | (series > upper_bound)
            
        elif config.outlier_method == OutlierMethod.ISOLATION_FOREST:
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(series.values.reshape(-1, 1))
                return outlier_predictions == -1
            except ImportError:
                logger.warning("Isolation Forest not available, falling back to z-score")
                return self._detect_outliers(series, 
                    CleaningConfig(outlier_method=OutlierMethod.Z_SCORE))
        
        return pd.Series(False, index=series.index)
    
    def _validate_ranges(self, data: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Validate data ranges against expected values"""
        result = data.copy()
        config = self.source_configs[source_name]
        
        for column, (min_val, max_val) in config.valid_ranges.items():
            if column in result.columns:
                # Flag values outside valid range
                invalid_mask = (result[column] < min_val) | (result[column] > max_val)
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} values outside valid range for {column}")
                    # Option: Set invalid values to NaN
                    result.loc[invalid_mask, column] = np.nan
        
        return result
    
    def normalize_data(self, 
                      data: pd.DataFrame, 
                      method: Optional[NormalizationMethod] = None,
                      columns: Optional[List[str]] = None,
                      fit_scaler: bool = True) -> Tuple[pd.DataFrame, Any]:
        """
        Normalize data using specified method
        
        Args:
            data: Input DataFrame
            method: Normalization method
            columns: Columns to normalize (if None, all numeric columns)
            fit_scaler: Whether to fit a new scaler or use existing
            
        Returns:
            Tuple of (normalized DataFrame, fitted scaler)
        """
        method = method or self.config.normalization_method
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        result = data.copy()
        
        # Select appropriate scaler
        if method == NormalizationMethod.STANDARD:
            scaler = StandardScaler()
        elif method == NormalizationMethod.MIN_MAX:
            scaler = MinMaxScaler(feature_range=self.config.feature_range)
        elif method == NormalizationMethod.ROBUST:
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Fit and transform
        if fit_scaler:
            scaler.fit(data[columns])
            self.fitted_scalers[f"{method.value}_scaler"] = scaler
        
        result[columns] = scaler.transform(data[columns])
        
        logger.info(f"Applied {method.value} normalization to {len(columns)} columns")
        
        return result, scaler
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """Get summary of cleaning operations performed"""
        if not self.cleaning_history:
            return pd.DataFrame()
        
        summary_data = []
        for entry in self.cleaning_history:
            summary_data.append({
                'timestamp': entry['timestamp'],
                'source_name': entry['source_name'],
                'input_rows': entry['input_shape'][0],
                'input_cols': entry['input_shape'][1],
                'output_rows': entry.get('output_shape', (0, 0))[0],
                'output_cols': entry.get('output_shape', (0, 0))[1],
                'operations': ', '.join(entry['operations']),
                'success': entry['success']
            })
        
        return pd.DataFrame(summary_data)

# Utility functions for common data source configurations

def create_fred_config() -> DataSourceConfig:
    """Create configuration for FRED data"""
    return DataSourceConfig(
        name="FRED",
        expected_frequency="M",
        timezone="UTC",
        data_types={
            'date': 'datetime64[ns]',
            'value': 'float64'
        },
        valid_ranges={
            'UNRATE': (0, 20),  # Unemployment rate
            'CPIAUCSL': (0, 500),  # CPI
            'GDP': (0, 30000),  # GDP in billions
        }
    )

def create_yahoo_finance_config() -> DataSourceConfig:
    """Create configuration for Yahoo Finance data"""
    return DataSourceConfig(
        name="YahooFinance",
        expected_frequency="D",
        timezone="UTC",
        column_mappings={
            'Adj Close': 'AdjClose'
        },
        data_types={
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'AdjClose': 'float64',
            'Volume': 'int64'
        },
        valid_ranges={
            'Volume': (0, 1e12)  # Reasonable volume range
        }
    )

# Example usage and testing functions

def create_sample_cleaning_pipeline() -> DataCleaner:
    """Create a sample cleaning pipeline with common configurations"""
    config = CleaningConfig(
        missing_value_strategy=MissingValueStrategy.FORWARD_FILL,
        outlier_method=OutlierMethod.Z_SCORE,
        outlier_treatment=OutlierTreatment.WINSORIZE,
        normalization_method=NormalizationMethod.STANDARD,
        verbose=True
    )
    
    cleaner = DataCleaner(config)
    
    # Register common data sources
    cleaner.register_data_source(create_fred_config())
    cleaner.register_data_source(create_yahoo_finance_config())
    
    return cleaner

if __name__ == "__main__":
    # Example usage
    print("Data Cleaning and Normalization Framework")
    print("========================================")
    
    # Create sample data with issues
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'indicator': np.random.randn(100)
    })
    
    # Introduce some issues
    sample_data.loc[10:12, 'price'] = np.nan  # Missing values
    sample_data.loc[50, 'price'] = 1000  # Outlier
    sample_data.loc[75:77, 'volume'] = np.nan  # Missing values
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Missing values per column:\n{sample_data.isnull().sum()}")
    
    # Create and use cleaner
    cleaner = create_sample_cleaning_pipeline()
    cleaned_data = cleaner.clean_data(sample_data)
    
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print(f"Missing values per column:\n{cleaned_data.isnull().sum()}")
    
    # Show cleaning summary
    summary = cleaner.get_cleaning_summary()
    print(f"\nCleaning Summary:\n{summary}")