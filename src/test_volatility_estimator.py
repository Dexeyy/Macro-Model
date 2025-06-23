#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Volatility Estimation Module

This test suite validates the functionality of the volatility estimation
components including historical, rolling, EWMA volatility calculations,
and advanced estimators like Parkinson and Garman-Klass.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('.'))

from src.features.volatility_estimator import (
    VolatilityMethod,
    AnnualizationFactor,
    VolatilityConfig,
    VolatilityEstimator,
    calculate_simple_volatility,
    calculate_rolling_vol,
    calculate_ewma_vol
)

class TestVolatilityEstimator:
    """Test class for volatility estimation functionality"""
    
    def __init__(self):
        """Initialize test data and configurations"""
        print("Initializing Volatility Estimator Test Suite")
        print("=" * 60)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Create test data
        self.dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        self.returns = np.random.normal(0.0005, 0.02, len(self.dates))
        self.prices = pd.Series(100 * np.exp(np.cumsum(self.returns)), 
                               index=self.dates, name='test_price')
        
        # Create OHLC data for advanced estimators
        self.create_ohlc_data()
        
        # Initialize volatility estimator
        self.config = VolatilityConfig(
            window=30,
            min_periods=20,
            annualize=True,
            frequency=AnnualizationFactor.DAILY
        )
        self.vol_estimator = VolatilityEstimator(self.config)
        
        self.test_results = {}
        
    def create_ohlc_data(self):
        """Create OHLC data for testing advanced volatility estimators"""
        # Simple OHLC simulation
        daily_vol = 0.02 / np.sqrt(252)  # Daily volatility
        intraday_noise = np.random.normal(0, daily_vol/4, len(self.prices))
        
        self.open_prices = self.prices * (1 + np.random.normal(0, daily_vol/8, len(self.prices)))
        self.high_prices = self.prices * (1 + np.abs(np.random.normal(0, daily_vol/4, len(self.prices))))
        self.low_prices = self.prices * (1 - np.abs(np.random.normal(0, daily_vol/4, len(self.prices))))
        self.close_prices = self.prices  # Use prices as close
        
    def test_historical_volatility(self):
        """Test historical volatility calculation"""
        print("\\n1. Testing Historical Volatility Calculation")
        print("-" * 40)
        
        try:
            # Test with default configuration
            hist_vol = self.vol_estimator.calculate_historical_volatility(self.prices)
            
            # Test with different parameters
            hist_vol_30d = self.vol_estimator.calculate_historical_volatility(
                self.prices, window=30, annualize=True
            )
            hist_vol_no_annualize = self.vol_estimator.calculate_historical_volatility(
                self.prices, annualize=False
            )
            
            # Validate results
            assert isinstance(hist_vol, (float, np.float64)), "Historical volatility should be a float"
            assert hist_vol > 0, "Historical volatility should be positive"
            assert hist_vol_30d > 0, "30-day historical volatility should be positive"
            assert hist_vol_no_annualize > 0, "Non-annualized volatility should be positive"
            assert hist_vol > hist_vol_no_annualize, "Annualized vol should be higher"
            
            print(f"   ✓ Historical Volatility: {hist_vol:.4f}")
            print(f"   ✓ 30-day Historical Vol: {hist_vol_30d:.4f}")
            print(f"   ✓ Non-annualized Vol: {hist_vol_no_annualize:.4f}")
            
            self.test_results['historical_volatility'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Historical volatility test failed: {str(e)}")
            self.test_results['historical_volatility'] = 'FAIL'
    
    def test_rolling_volatility(self):
        """Test rolling volatility calculation"""
        print("\\n2. Testing Rolling Volatility Calculation")
        print("-" * 40)
        
        try:
            # Test rolling volatility
            rolling_vol = self.vol_estimator.calculate_rolling_volatility(self.prices)
            
            # Test with different window sizes
            rolling_vol_10d = self.vol_estimator.calculate_rolling_volatility(
                self.prices, window=10
            )
            rolling_vol_60d = self.vol_estimator.calculate_rolling_volatility(
                self.prices, window=60
            )
            
            # Validate results
            assert isinstance(rolling_vol, pd.Series), "Rolling volatility should be a Series"
            assert len(rolling_vol) == len(self.prices), "Series length should match"
            assert rolling_vol.dropna().min() > 0, "All volatilities should be positive"
            
            print(f"   ✓ Rolling Vol (30d) current: {rolling_vol.iloc[-1]:.4f}")
            print(f"   ✓ Rolling Vol (10d) current: {rolling_vol_10d.iloc[-1]:.4f}")
            print(f"   ✓ Rolling Vol (60d) current: {rolling_vol_60d.iloc[-1]:.4f}")
            print(f"   ✓ Valid observations: {rolling_vol.dropna().shape[0]:,}")
            
            self.test_results['rolling_volatility'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Rolling volatility test failed: {str(e)}")
            self.test_results['rolling_volatility'] = 'FAIL'
    
    def test_ewma_volatility(self):
        """Test EWMA volatility calculation"""
        print("\\n3. Testing EWMA Volatility Calculation")
        print("-" * 40)
        
        try:
            # Test EWMA volatility with span
            ewma_vol = self.vol_estimator.calculate_ewma_volatility(self.prices, span=30)
            
            # Test with alpha parameter
            ewma_vol_alpha = self.vol_estimator.calculate_ewma_volatility(
                self.prices, alpha=0.1
            )
            
            # Test different spans
            ewma_vol_10 = self.vol_estimator.calculate_ewma_volatility(
                self.prices, span=10
            )
            ewma_vol_60 = self.vol_estimator.calculate_ewma_volatility(
                self.prices, span=60
            )
            
            # Validate results
            assert isinstance(ewma_vol, pd.Series), "EWMA volatility should be a Series"
            assert len(ewma_vol) == len(self.prices), "Series length should match"
            assert ewma_vol.dropna().min() > 0, "All EWMA volatilities should be positive"
            
            print(f"   ✓ EWMA Vol (span=30) current: {ewma_vol.iloc[-1]:.4f}")
            print(f"   ✓ EWMA Vol (alpha=0.1) current: {ewma_vol_alpha.iloc[-1]:.4f}")
            print(f"   ✓ EWMA Vol (span=10) current: {ewma_vol_10.iloc[-1]:.4f}")
            print(f"   ✓ EWMA Vol (span=60) current: {ewma_vol_60.iloc[-1]:.4f}")
            
            self.test_results['ewma_volatility'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ EWMA volatility test failed: {str(e)}")
            self.test_results['ewma_volatility'] = 'FAIL'
    
    def test_parkinson_volatility(self):
        """Test Parkinson volatility estimator"""
        print("\\n4. Testing Parkinson Volatility Estimator")
        print("-" * 40)
        
        try:
            # Test Parkinson volatility
            parkinson_vol = self.vol_estimator.calculate_parkinson_volatility(
                self.high_prices, self.low_prices
            )
            
            # Test with different window
            parkinson_vol_10d = self.vol_estimator.calculate_parkinson_volatility(
                self.high_prices, self.low_prices, window=10
            )
            
            # Validate results
            assert isinstance(parkinson_vol, pd.Series), "Parkinson volatility should be a Series"
            assert parkinson_vol.dropna().min() > 0, "All Parkinson volatilities should be positive"
            
            print(f"   ✓ Parkinson Vol (30d) current: {parkinson_vol.iloc[-1]:.4f}")
            print(f"   ✓ Parkinson Vol (10d) current: {parkinson_vol_10d.iloc[-1]:.4f}")
            print(f"   ✓ Valid observations: {parkinson_vol.dropna().shape[0]:,}")
            
            self.test_results['parkinson_volatility'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Parkinson volatility test failed: {str(e)}")
            self.test_results['parkinson_volatility'] = 'FAIL'
    
    def test_garman_klass_volatility(self):
        """Test Garman-Klass volatility estimator"""
        print("\\n5. Testing Garman-Klass Volatility Estimator")
        print("-" * 40)
        
        try:
            # Test Garman-Klass volatility
            gk_vol = self.vol_estimator.calculate_garman_klass_volatility(
                self.open_prices, self.high_prices, self.low_prices, self.close_prices
            )
            
            # Test with different window
            gk_vol_20d = self.vol_estimator.calculate_garman_klass_volatility(
                self.open_prices, self.high_prices, self.low_prices, self.close_prices, 
                window=20
            )
            
            # Validate results
            assert isinstance(gk_vol, pd.Series), "Garman-Klass volatility should be a Series"
            assert gk_vol.dropna().min() > 0, "All GK volatilities should be positive"
            
            print(f"   ✓ Garman-Klass Vol (30d) current: {gk_vol.iloc[-1]:.4f}")
            print(f"   ✓ Garman-Klass Vol (20d) current: {gk_vol_20d.iloc[-1]:.4f}")
            print(f"   ✓ Valid observations: {gk_vol.dropna().shape[0]:,}")
            
            self.test_results['garman_klass_volatility'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Garman-Klass volatility test failed: {str(e)}")
            self.test_results['garman_klass_volatility'] = 'FAIL'
    
    def test_volatility_statistics(self):
        """Test volatility statistics calculation"""
        print("\\n6. Testing Volatility Statistics")
        print("-" * 40)
        
        try:
            # Calculate rolling volatility for statistics
            rolling_vol = self.vol_estimator.calculate_rolling_volatility(self.prices)
            
            # Calculate statistics
            vol_stats = self.vol_estimator.calculate_volatility_statistics(rolling_vol)
            
            # Validate results
            assert isinstance(vol_stats, dict), "Statistics should be a dictionary"
            required_stats = ['mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']
            for stat in required_stats:
                assert stat in vol_stats, f"Missing statistic: {stat}"
                assert not np.isnan(vol_stats[stat]) or stat in ['persistence'], f"Invalid {stat} value"
            
            print(f"   ✓ Mean Volatility: {vol_stats['mean']:.4f}")
            print(f"   ✓ Median Volatility: {vol_stats['median']:.4f}")
            print(f"   ✓ Volatility Std Dev: {vol_stats['std']:.4f}")
            print(f"   ✓ Min/Max Volatility: {vol_stats['min']:.4f} / {vol_stats['max']:.4f}")
            print(f"   ✓ Skewness: {vol_stats['skewness']:.4f}")
            print(f"   ✓ Kurtosis: {vol_stats['kurtosis']:.4f}")
            print(f"   ✓ Persistence (AR1): {vol_stats['persistence']:.4f}")
            
            self.test_results['volatility_statistics'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Volatility statistics test failed: {str(e)}")
            self.test_results['volatility_statistics'] = 'FAIL'
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        print("\\n7. Testing Convenience Functions")
        print("-" * 40)
        
        try:
            # Test convenience functions
            simple_vol = calculate_simple_volatility(self.prices)
            rolling_vol_conv = calculate_rolling_vol(self.prices)
            ewma_vol_conv = calculate_ewma_vol(self.prices)
            
            # Validate results
            assert isinstance(simple_vol, (float, np.float64)), "Simple volatility should be float"
            assert isinstance(rolling_vol_conv, pd.Series), "Rolling vol should be Series"
            assert isinstance(ewma_vol_conv, pd.Series), "EWMA vol should be Series"
            
            print(f"   ✓ Simple Volatility: {simple_vol:.4f}")
            print(f"   ✓ Rolling Vol (current): {rolling_vol_conv.iloc[-1]:.4f}")
            print(f"   ✓ EWMA Vol (current): {ewma_vol_conv.iloc[-1]:.4f}")
            
            self.test_results['convenience_functions'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Convenience functions test failed: {str(e)}")
            self.test_results['convenience_functions'] = 'FAIL'
    
    def test_configuration_system(self):
        """Test volatility configuration system"""
        print("\\n8. Testing Configuration System")
        print("-" * 40)
        
        try:
            # Test different configurations
            config_daily = VolatilityConfig(
                window=21,
                frequency=AnnualizationFactor.DAILY,
                annualize=True
            )
            
            config_monthly = VolatilityConfig(
                window=12,
                frequency=AnnualizationFactor.MONTHLY,
                annualize=True
            )
            
            # Test estimators with different configs
            estimator_daily = VolatilityEstimator(config_daily)
            estimator_monthly = VolatilityEstimator(config_monthly)
            
            vol_daily = estimator_daily.calculate_historical_volatility(self.prices)
            vol_monthly = estimator_monthly.calculate_historical_volatility(self.prices)
            
            # Validate configurations
            assert config_daily.frequency == AnnualizationFactor.DAILY
            assert config_monthly.frequency == AnnualizationFactor.MONTHLY
            assert vol_daily > 0 and vol_monthly > 0
            
            print(f"   ✓ Daily Config Vol: {vol_daily:.4f}")
            print(f"   ✓ Monthly Config Vol: {vol_monthly:.4f}")
            print(f"   ✓ Config window (daily): {config_daily.window}")
            print(f"   ✓ Config window (monthly): {config_monthly.window}")
            
            self.test_results['configuration_system'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Configuration system test failed: {str(e)}")
            self.test_results['configuration_system'] = 'FAIL'
    
    def run_all_tests(self):
        """Run all volatility estimator tests"""
        print("\\nStarting Comprehensive Volatility Estimator Tests")
        print("=" * 60)
        
        # Run individual tests
        self.test_historical_volatility()
        self.test_rolling_volatility()
        self.test_ewma_volatility()
        self.test_parkinson_volatility()
        self.test_garman_klass_volatility()
        self.test_volatility_statistics()
        self.test_convenience_functions()
        self.test_configuration_system()
        
        # Print summary
        print("\\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result == 'PASS')
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result == 'PASS' else "✗"
            print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {result}")
        
        print("-" * 60)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\\n🎉 ALL TESTS PASSED! Volatility Estimator is working correctly.")
        else:
            print(f"\\n⚠️  {total-passed} test(s) failed. Please review the implementation.")
        
        return passed == total


def main():
    """Main test execution function"""
    print("Volatility Estimation Module - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run tests
    test_suite = TestVolatilityEstimator()
    success = test_suite.run_all_tests()
    
    print(f"\\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\\n✅ Volatility Estimator Module is ready for use!")
    else:
        print("\\n❌ Some tests failed. Please review the implementation.")
    
    return success


if __name__ == "__main__":
    main() 