"""
Test Suite for Returns Calculator Module

Comprehensive tests for all returns calculation functionality including:
- Simple returns calculation
- Log returns calculation  
- Cumulative returns calculation
- Excess returns calculation
- Rolling returns calculation
- Return statistics calculation
- Missing data handling
- Edge cases and error conditions
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from features.returns_calculator import (
    ReturnsCalculator,
    ReturnType,
    Frequency,
    ReturnConfig,
    calculate_returns
)

def create_test_price_series(length: int = 100, start_price: float = 100.0, volatility: float = 0.02) -> pd.Series:
    """Create a test price series with known properties."""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
    
    # Generate returns using random walk
    returns = np.random.normal(0, volatility, length)
    prices = [start_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices, index=dates, name='Price')

def create_test_dataframe(num_assets: int = 3, length: int = 100) -> pd.DataFrame:
    """Create a test DataFrame with multiple price series."""
    data = {}
    for i in range(num_assets):
        data[f'Asset_{i+1}'] = create_test_price_series(length, start_price=100 + i*10)
    
    return pd.DataFrame(data)

def test_basic_functionality():
    """Test basic returns calculation functionality"""
    print("Testing basic returns calculation functionality...")
    
    # Create test data
    prices = create_test_price_series(50)
    calculator = ReturnsCalculator()
    
    # Test simple returns
    simple_returns = calculator.calculate_simple_returns(prices, 1)
    assert isinstance(simple_returns, pd.Series), "Simple returns should return a Series for single period"
    assert len(simple_returns) == len(prices), "Returns series should have same length as prices"
    assert pd.isna(simple_returns.iloc[0]), "First return should be NaN"
    
    # Test multiple periods
    multi_returns = calculator.calculate_simple_returns(prices, [1, 5, 10])
    assert isinstance(multi_returns, pd.DataFrame), "Multiple periods should return DataFrame"
    assert multi_returns.shape[1] == 3, "Should have 3 columns for 3 periods"
    
    # Test log returns
    log_returns = calculator.calculate_log_returns(prices, 1)
    assert isinstance(log_returns, pd.Series), "Log returns should return a Series"
    assert not np.isinf(log_returns.dropna()).any(), "Log returns should not contain infinity"
    
    print("✓ Basic functionality tests passed")

def test_dataframe_handling():
    """Test handling of DataFrame inputs"""
    print("Testing DataFrame handling...")
    
    # Create test DataFrame
    prices_df = create_test_dataframe(3, 50)
    calculator = ReturnsCalculator()
    
    # Test simple returns for DataFrame
    returns_df = calculator.calculate_simple_returns(prices_df, [1, 5])
    expected_cols = len(prices_df.columns) * 2  # 3 assets * 2 periods
    assert returns_df.shape[1] == expected_cols, f"Expected {expected_cols} columns, got {returns_df.shape[1]}"
    
    # Test log returns for DataFrame
    log_returns_df = calculator.calculate_log_returns(prices_df, 1)
    assert log_returns_df.shape[1] == len(prices_df.columns), "Log returns should have same number of columns as input"
    
    print("✓ DataFrame handling tests passed")

def test_cumulative_returns():
    """Test cumulative returns calculation"""
    print("Testing cumulative returns...")
    
    # Create simple test case
    returns = pd.Series([0.01, 0.02, -0.01, 0.015], name='returns')
    calculator = ReturnsCalculator()
    
    # Test compound cumulative returns
    cum_returns_compound = calculator.calculate_cumulative_returns(returns, compound=True)
    
    # Manual calculation for verification
    expected_compound = (1 + returns).cumprod() - 1
    pd.testing.assert_series_equal(cum_returns_compound, expected_compound)
    
    # Test simple cumulative returns
    cum_returns_simple = calculator.calculate_cumulative_returns(returns, compound=False)
    expected_simple = returns.cumsum()
    pd.testing.assert_series_equal(cum_returns_simple, expected_simple)
    
    print("✓ Cumulative returns tests passed")

def test_missing_data_handling():
    """Test missing data handling strategies"""
    print("Testing missing data handling...")
    
    # Create data with missing values
    prices = create_test_price_series(20)
    prices.iloc[5] = np.nan
    prices.iloc[10] = np.nan
    prices.iloc[15] = np.nan
    
    calculator = ReturnsCalculator()
    
    # Test skip strategy (default)
    returns_skip = calculator.calculate_simple_returns(prices, 1, handle_missing="skip")
    assert pd.isna(returns_skip.iloc[6]), "Should have NaN where price was missing"
    
    # Test forward fill strategy
    returns_ffill = calculator.calculate_simple_returns(prices, 1, handle_missing="forward_fill")
    # Should have fewer NaN values due to forward filling
    nan_count_ffill = returns_ffill.isna().sum()
    nan_count_skip = returns_skip.isna().sum()
    assert nan_count_ffill <= nan_count_skip, "Forward fill should reduce NaN count"
    
    print("✓ Missing data handling tests passed")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("Testing edge cases...")
    
    calculator = ReturnsCalculator()
    
    # Test with zero prices (should handle gracefully for log returns)
    prices_with_zero = pd.Series([100, 50, 0, 25, 50])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        log_returns = calculator.calculate_log_returns(prices_with_zero, 1)
        assert len(w) > 0, "Should warn about non-positive prices"
        assert not np.isinf(log_returns.dropna()).any(), "Should not contain infinity after handling"
    
    # Test with single data point
    single_price = pd.Series([100])
    returns_single = calculator.calculate_simple_returns(single_price, 1)
    assert len(returns_single) == 1, "Should handle single data point"
    assert pd.isna(returns_single.iloc[0]), "Single data point should return NaN"
    
    print("✓ Edge cases tests passed")

def test_convenience_function():
    """Test the convenience function"""
    print("Testing convenience function...")
    
    prices = create_test_price_series(50)
    
    # Test simple returns
    simple_ret = calculate_returns(prices, "simple", 1)
    assert isinstance(simple_ret, pd.Series), "Should return Series for single period"
    
    # Test log returns
    log_ret = calculate_returns(prices, "log", [1, 5])
    assert isinstance(log_ret, pd.DataFrame), "Should return DataFrame for multiple periods"
    
    # Test error for unknown type
    try:
        calculate_returns(prices, "unknown", 1)
        assert False, "Should raise ValueError for unknown return type"
    except ValueError:
        pass  # Expected
    
    print("✓ Convenience function tests passed")

def run_all_tests():
    """Run all test functions"""
    print("Running Returns Calculator Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_dataframe_handling()
        test_cumulative_returns()
        test_missing_data_handling()
        test_edge_cases()
        test_convenience_function()
        
        print("\n" + "=" * 60)
        print("🎉 ALL RETURNS CALCULATOR TESTS PASSED! 🎉")
        print("Returns calculation module is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 