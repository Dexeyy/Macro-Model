"""
Test suite for the Data Cleaning and Normalization Framework

This test suite validates the functionality of the data cleaning framework
including missing value handling, outlier detection, timestamp standardization,
and data source specific configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data.data_cleaner import (
    DataCleaner, CleaningConfig, DataSourceConfig,
    MissingValueStrategy, OutlierMethod, OutlierTreatment, NormalizationMethod,
    create_fred_config, create_yahoo_finance_config, create_sample_cleaning_pipeline
)

def create_sample_data_with_issues():
    """Create sample data with various data quality issues"""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100).astype(float),
        'indicator': np.random.randn(100),
        'returns': np.random.randn(100) * 0.02
    })
    
    # Introduce missing values
    data.loc[10:12, 'price'] = np.nan
    data.loc[25:27, 'volume'] = np.nan
    data.loc[50, 'indicator'] = np.nan
    
    # Introduce outliers
    data.loc[45, 'price'] = 1000  # Extreme outlier
    data.loc[60, 'volume'] = 100000  # Volume outlier
    data.loc[75, 'returns'] = 0.5  # Return outlier
    
    # Add some inconsistent data types
    data.loc[80, 'volume'] = '5000'  # String in numeric column
    
    return data

def test_basic_initialization():
    """Test basic DataCleaner initialization"""
    print("Testing basic DataCleaner initialization...")
    
    # Test default initialization
    cleaner = DataCleaner()
    assert cleaner.config is not None
    assert len(cleaner.source_configs) == 0
    assert len(cleaner.fitted_scalers) == 0
    assert len(cleaner.cleaning_history) == 0
    
    # Test custom config initialization
    config = CleaningConfig(
        missing_value_strategy=MissingValueStrategy.MEAN,
        outlier_method=OutlierMethod.IQR,
        verbose=False
    )
    cleaner = DataCleaner(config)
    assert cleaner.config.missing_value_strategy == MissingValueStrategy.MEAN
    assert cleaner.config.outlier_method == OutlierMethod.IQR
    assert cleaner.config.verbose == False
    
    print("✓ Basic initialization tests passed")

def test_data_source_registration():
    """Test data source registration"""
    print("Testing data source registration...")
    
    cleaner = DataCleaner()
    
    # Test FRED config registration
    fred_config = create_fred_config()
    cleaner.register_data_source(fred_config)
    
    assert 'FRED' in cleaner.source_configs
    assert cleaner.source_configs['FRED'].name == 'FRED'
    assert cleaner.source_configs['FRED'].expected_frequency == 'M'
    
    # Test Yahoo Finance config registration
    yahoo_config = create_yahoo_finance_config()
    cleaner.register_data_source(yahoo_config)
    
    assert 'YahooFinance' in cleaner.source_configs
    assert cleaner.source_configs['YahooFinance'].name == 'YahooFinance'
    assert cleaner.source_configs['YahooFinance'].expected_frequency == 'D'
    
    print("✓ Data source registration tests passed")

def test_missing_value_handling():
    """Test different missing value handling strategies"""
    print("Testing missing value handling strategies...")
    
    data = create_sample_data_with_issues()
    original_missing_count = data.isnull().sum().sum()
    
    # Test forward fill
    config = CleaningConfig(missing_value_strategy=MissingValueStrategy.FORWARD_FILL)
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    assert cleaned_data.isnull().sum().sum() < original_missing_count
    print(f"  Forward fill: {original_missing_count} -> {cleaned_data.isnull().sum().sum()} missing values")
    
    # Test mean imputation
    config = CleaningConfig(missing_value_strategy=MissingValueStrategy.MEAN)
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    # Should have no missing values in numeric columns after mean imputation
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    assert cleaned_data[numeric_columns].isnull().sum().sum() == 0
    print(f"  Mean imputation: {original_missing_count} -> {cleaned_data.isnull().sum().sum()} missing values")
    
    # Test interpolation
    config = CleaningConfig(missing_value_strategy=MissingValueStrategy.INTERPOLATE_LINEAR)
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    assert cleaned_data.isnull().sum().sum() < original_missing_count
    print(f"  Linear interpolation: {original_missing_count} -> {cleaned_data.isnull().sum().sum()} missing values")
    
    print("✓ Missing value handling tests passed")

def test_outlier_detection_and_treatment():
    """Test outlier detection and treatment methods"""
    print("Testing outlier detection and treatment...")
    
    data = create_sample_data_with_issues()
    
    # Test Z-score outlier detection with winsorization
    config = CleaningConfig(
        outlier_method=OutlierMethod.Z_SCORE,
        outlier_treatment=OutlierTreatment.WINSORIZE,
        outlier_threshold=2.0
    )
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    # Check that extreme outlier (price=1000) has been winsorized
    assert cleaned_data['price'].max() < 1000
    print(f"  Z-score + Winsorize: Max price {data['price'].max():.2f} -> {cleaned_data['price'].max():.2f}")
    
    # Test IQR outlier detection with capping
    config = CleaningConfig(
        outlier_method=OutlierMethod.IQR,
        outlier_treatment=OutlierTreatment.WINSORIZE,
        iqr_multiplier=1.5
    )
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    # Should reduce extreme values - convert original data to numeric for comparison
    original_volume_max = pd.to_numeric(data['volume'], errors='coerce').max()
    assert cleaned_data['volume'].max() < original_volume_max
    print(f"  IQR + Winsorize: Max volume {original_volume_max:.0f} -> {cleaned_data['volume'].max():.0f}")
    
    # Test outlier flagging
    config = CleaningConfig(
        outlier_method=OutlierMethod.Z_SCORE,
        outlier_treatment=OutlierTreatment.FLAG,
        outlier_threshold=2.0
    )
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)
    
    # Should have outlier flag columns
    flag_columns = [col for col in cleaned_data.columns if '_outlier_flag' in col]
    assert len(flag_columns) > 0
    print(f"  Outlier flagging: Added {len(flag_columns)} flag columns")
    
    print("✓ Outlier detection and treatment tests passed")

def test_timestamp_standardization():
    """Test timestamp standardization"""
    print("Testing timestamp standardization...")
    
    # Create data with datetime index
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'value': np.random.randn(50),
        'timestamp': dates
    }, index=dates)
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(data)
    
    # Check that index has timezone info
    if hasattr(cleaned_data.index, 'tz'):
        assert cleaned_data.index.tz is not None
        print("  ✓ Timestamp index standardized with timezone")
    
    # Check datetime columns
    datetime_columns = cleaned_data.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for col in datetime_columns:
        if hasattr(cleaned_data[col].dtype, 'tz'):
            print(f"  ✓ Column {col} standardized with timezone")
    
    print("✓ Timestamp standardization tests passed")

def test_normalization():
    """Test data normalization methods"""
    print("Testing data normalization...")
    
    data = create_sample_data_with_issues()
    cleaner = DataCleaner()
    
    # Test standard normalization
    normalized_data, scaler = cleaner.normalize_data(
        data, 
        method=NormalizationMethod.STANDARD,
        columns=['price', 'volume', 'indicator']
    )
    
    # Check that normalized columns have approximately zero mean and unit variance
    for col in ['price', 'volume', 'indicator']:
        if col in normalized_data.columns:
            mean_val = normalized_data[col].mean()
            std_val = normalized_data[col].std()
            assert abs(mean_val) < 0.1, f"Mean of {col} should be ~0, got {mean_val}"
            assert abs(std_val - 1.0) < 0.1, f"Std of {col} should be ~1, got {std_val}"
    
    print("  ✓ Standard normalization: mean ~0, std ~1")
    
    # Test MinMax normalization
    normalized_data, scaler = cleaner.normalize_data(
        data, 
        method=NormalizationMethod.MIN_MAX,
        columns=['price', 'volume']
    )
    
    # Check that values are in [0, 1] range
    for col in ['price', 'volume']:
        if col in normalized_data.columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            assert min_val >= 0, f"Min of {col} should be >= 0, got {min_val}"
            assert max_val <= 1, f"Max of {col} should be <= 1, got {max_val}"
    
    print("  ✓ MinMax normalization: values in [0, 1] range")
    
    print("✓ Normalization tests passed")

def test_source_specific_processing():
    """Test source-specific data processing"""
    print("Testing source-specific processing...")
    
    # Create sample Yahoo Finance-like data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    yahoo_data = pd.DataFrame({
        'Open': np.random.randn(50) + 100,
        'High': np.random.randn(50) + 102,
        'Low': np.random.randn(50) + 98,
        'Close': np.random.randn(50) + 100,
        'Adj Close': np.random.randn(50) + 100,
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Test with registered Yahoo Finance config
    cleaner = DataCleaner()
    cleaner.register_data_source(create_yahoo_finance_config())
    
    cleaned_data = cleaner.clean_data(yahoo_data, source_name='YahooFinance')
    
    # Check column mapping was applied
    assert 'AdjClose' in cleaned_data.columns
    assert 'Adj Close' not in cleaned_data.columns
    print("  ✓ Column mapping applied: 'Adj Close' -> 'AdjClose'")
    
    # Check data types
    for col in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
        if col in cleaned_data.columns:
            assert cleaned_data[col].dtype in ['float64', 'float32']
    
    print("  ✓ Data types validated and converted")
    
    print("✓ Source-specific processing tests passed")

def test_cleaning_history():
    """Test cleaning history tracking"""
    print("Testing cleaning history tracking...")
    
    data = create_sample_data_with_issues()
    cleaner = DataCleaner()
    
    # Perform multiple cleaning operations
    cleaned_data1 = cleaner.clean_data(data)
    cleaned_data2 = cleaner.clean_data(data, source_name=None)
    
    # Check history
    assert len(cleaner.cleaning_history) == 2
    
    summary = cleaner.get_cleaning_summary()
    assert len(summary) == 2
    assert 'timestamp' in summary.columns
    assert 'input_rows' in summary.columns
    assert 'output_rows' in summary.columns
    assert 'operations' in summary.columns
    assert 'success' in summary.columns
    
    print(f"  ✓ Tracked {len(cleaner.cleaning_history)} cleaning operations")
    print("  ✓ Cleaning summary generated successfully")
    
    print("✓ Cleaning history tests passed")

def test_full_pipeline():
    """Test the complete cleaning pipeline"""
    print("Testing complete cleaning pipeline...")
    
    # Create comprehensive test data
    data = create_sample_data_with_issues()
    
    # Use the sample cleaning pipeline
    cleaner = create_sample_cleaning_pipeline()
    
    # Clean the data
    cleaned_data = cleaner.clean_data(data)
    
    # Verify improvements
    original_missing = data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    
    print(f"  Missing values: {original_missing} -> {cleaned_missing}")
    print(f"  Data shape: {data.shape} -> {cleaned_data.shape}")
    
    # Check that basic cleaning was performed
    assert cleaned_missing <= original_missing
    
    # Test normalization
    normalized_data, scaler = cleaner.normalize_data(cleaned_data)
    
    print(f"  Normalization applied to {len(cleaned_data.select_dtypes(include=[np.number]).columns)} columns")
    
    # Get cleaning summary
    summary = cleaner.get_cleaning_summary()
    print(f"  Cleaning operations logged: {len(summary)}")
    
    print("✓ Full pipeline test passed")

def run_all_tests():
    """Run all tests"""
    print("Running Data Cleaning Framework Tests")
    print("=" * 50)
    
    try:
        test_basic_initialization()
        print()
        
        test_data_source_registration()
        print()
        
        test_missing_value_handling()
        print()
        
        test_outlier_detection_and_treatment()
        print()
        
        test_timestamp_standardization()
        print()
        
        test_normalization()
        print()
        
        test_source_specific_processing()
        print()
        
        test_cleaning_history()
        print()
        
        test_full_pipeline()
        print()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Data Cleaning Framework is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)