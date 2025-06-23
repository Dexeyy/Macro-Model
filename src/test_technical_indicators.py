#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Technical Indicators Module

This test suite validates the functionality of the technical indicators
library including moving averages, momentum oscillators, volatility bands,
and signal generation capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('.'))

from src.features.technical_indicators import (
    MovingAverageType,
    SignalType,
    IndicatorConfig,
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

class TestTechnicalIndicators:
    """Test class for technical indicators functionality"""
    
    def __init__(self):
        """Initialize test data and configurations"""
        print("Initializing Technical Indicators Test Suite")
        print("=" * 60)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Create test data
        self.dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        self.returns = np.random.normal(0.0005, 0.02, len(self.dates))
        self.prices = 100 * np.exp(np.cumsum(self.returns))
        
        # Create OHLC data
        self.create_ohlc_data()
        
        # Initialize technical indicators
        self.config = IndicatorConfig()
        self.tech_indicators = TechnicalIndicators(self.config)
        
        self.test_results = {}
        
    def create_ohlc_data(self):
        """Create OHLC data for testing"""
        noise = np.random.normal(0, 0.005, len(self.prices))
        
        self.high = pd.Series(self.prices * (1 + np.abs(noise)), 
                             index=self.dates, name='high')
        self.low = pd.Series(self.prices * (1 - np.abs(noise)), 
                            index=self.dates, name='low')
        self.close = pd.Series(self.prices, index=self.dates, name='close')
        self.open = pd.Series(
            np.concatenate([[self.prices[0]], self.prices[:-1]]), 
            index=self.dates, name='open'
        )
        self.volume = pd.Series(
            np.random.randint(1000000, 10000000, len(self.dates)),
            index=self.dates, name='volume'
        )
        
    def test_moving_averages(self):
        """Test moving average calculations"""
        print("\\n1. Testing Moving Averages")
        print("-" * 40)
        
        try:
            # Test Simple Moving Average
            sma_20 = self.tech_indicators.calculate_sma(self.close, 20)
            sma_50 = self.tech_indicators.calculate_sma(self.close, 50)
            
            # Test Exponential Moving Average
            ema_12 = self.tech_indicators.calculate_ema(self.close, 12)
            ema_26 = self.tech_indicators.calculate_ema(self.close, 26)
            
            # Test Weighted Moving Average
            wma_20 = self.tech_indicators.calculate_wma(self.close, 20)
            
            # Validate results
            assert isinstance(sma_20, pd.Series), "SMA should be a Series"
            assert len(sma_20) == len(self.close), "SMA length should match input"
            assert sma_20.dropna().min() > 0, "SMA values should be positive"
            
            assert isinstance(ema_12, pd.Series), "EMA should be a Series"
            assert len(ema_12) == len(self.close), "EMA length should match input"
            
            assert isinstance(wma_20, pd.Series), "WMA should be a Series"
            assert len(wma_20) == len(self.close), "WMA length should match input"
            
            print(f"   ✓ SMA(20) current: {sma_20.iloc[-1]:.2f}")
            print(f"   ✓ SMA(50) current: {sma_50.iloc[-1]:.2f}")
            print(f"   ✓ EMA(12) current: {ema_12.iloc[-1]:.2f}")
            print(f"   ✓ EMA(26) current: {ema_26.iloc[-1]:.2f}")
            print(f"   ✓ WMA(20) current: {wma_20.iloc[-1]:.2f}")
            
            self.test_results['moving_averages'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Moving averages test failed: {str(e)}")
            self.test_results['moving_averages'] = 'FAIL'
    
    def test_momentum_indicators(self):
        """Test momentum oscillators"""
        print("\\n2. Testing Momentum Indicators")
        print("-" * 40)
        
        try:
            # Test RSI
            rsi = self.tech_indicators.calculate_rsi(self.close)
            rsi_21 = self.tech_indicators.calculate_rsi(self.close, 21)
            
            # Test MACD
            macd = self.tech_indicators.calculate_macd(self.close)
            
            # Test Stochastic
            stoch = self.tech_indicators.calculate_stochastic(
                self.high, self.low, self.close
            )
            
            # Test Williams %R
            williams_r = self.tech_indicators.calculate_williams_r(
                self.high, self.low, self.close
            )
            
            # Validate RSI
            assert isinstance(rsi, pd.Series), "RSI should be a Series"
            assert rsi.dropna().min() >= 0, "RSI should be >= 0"
            assert rsi.dropna().max() <= 100, "RSI should be <= 100"
            
            # Validate MACD
            assert isinstance(macd, dict), "MACD should return dictionary"
            assert 'macd' in macd and 'signal' in macd and 'histogram' in macd
            
            # Validate Stochastic
            assert isinstance(stoch, dict), "Stochastic should return dictionary"
            assert 'stoch_k' in stoch and 'stoch_d' in stoch
            
            # Validate Williams %R
            assert isinstance(williams_r, pd.Series), "Williams %R should be a Series"
            assert williams_r.dropna().min() >= -100, "Williams %R should be >= -100"
            assert williams_r.dropna().max() <= 0, "Williams %R should be <= 0"
            
            print(f"   ✓ RSI(14) current: {rsi.iloc[-1]:.2f}")
            print(f"   ✓ RSI(21) current: {rsi_21.iloc[-1]:.2f}")
            print(f"   ✓ MACD current: {macd['macd'].iloc[-1]:.4f}")
            print(f"   ✓ MACD Signal: {macd['signal'].iloc[-1]:.4f}")
            print(f"   ✓ Stochastic %K: {stoch['stoch_k'].iloc[-1]:.2f}")
            print(f"   ✓ Stochastic %D: {stoch['stoch_d'].iloc[-1]:.2f}")
            print(f"   ✓ Williams %R: {williams_r.iloc[-1]:.2f}")
            
            self.test_results['momentum_indicators'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Momentum indicators test failed: {str(e)}")
            self.test_results['momentum_indicators'] = 'FAIL'
    
    def test_volatility_indicators(self):
        """Test volatility-based indicators"""
        print("\\n3. Testing Volatility Indicators")
        print("-" * 40)
        
        try:
            # Test Bollinger Bands
            bb = self.tech_indicators.calculate_bollinger_bands(self.close)
            bb_custom = self.tech_indicators.calculate_bollinger_bands(
                self.close, period=25, std_dev=2.5
            )
            
            # Test ATR
            atr = self.tech_indicators.calculate_atr(self.high, self.low, self.close)
            
            # Validate Bollinger Bands
            assert isinstance(bb, dict), "Bollinger Bands should return dictionary"
            required_keys = ['upper', 'middle', 'lower', 'bandwidth', 'percent_b']
            for key in required_keys:
                assert key in bb, f"Missing Bollinger Band key: {key}"
            
            # Check band relationships
            assert (bb['upper'] >= bb['middle']).all(), "Upper band should be >= middle"
            assert (bb['middle'] >= bb['lower']).all(), "Middle band should be >= lower"
            
            # Validate ATR
            assert isinstance(atr, pd.Series), "ATR should be a Series"
            assert atr.dropna().min() >= 0, "ATR should be >= 0"
            
            print(f"   ✓ BB Upper Band: {bb['upper'].iloc[-1]:.2f}")
            print(f"   ✓ BB Middle Band: {bb['middle'].iloc[-1]:.2f}")
            print(f"   ✓ BB Lower Band: {bb['lower'].iloc[-1]:.2f}")
            print(f"   ✓ BB Bandwidth: {bb['bandwidth'].iloc[-1]:.2f}%")
            print(f"   ✓ BB %B: {bb['percent_b'].iloc[-1]:.2f}")
            print(f"   ✓ ATR current: {atr.iloc[-1]:.2f}")
            
            self.test_results['volatility_indicators'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Volatility indicators test failed: {str(e)}")
            self.test_results['volatility_indicators'] = 'FAIL'
    
    def test_trend_indicators(self):
        """Test trend-based indicators"""
        print("\\n4. Testing Trend Indicators")
        print("-" * 40)
        
        try:
            # Test CCI
            cci = self.tech_indicators.calculate_cci(self.high, self.low, self.close)
            
            # Validate CCI
            assert isinstance(cci, pd.Series), "CCI should be a Series"
            assert not cci.dropna().empty, "CCI should have valid values"
            
            print(f"   ✓ CCI current: {cci.iloc[-1]:.2f}")
            print(f"   ✓ CCI valid observations: {cci.dropna().shape[0]:,}")
            
            self.test_results['trend_indicators'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Trend indicators test failed: {str(e)}")
            self.test_results['trend_indicators'] = 'FAIL'
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        print("\\n5. Testing Signal Generation")
        print("-" * 40)
        
        try:
            # Test MA crossover signals
            ma_signals = self.tech_indicators.generate_ma_crossover_signals(
                self.close, 12, 26
            )
            
            # Test RSI signals
            rsi_signals = self.tech_indicators.generate_rsi_signals(
                self.close, oversold=30, overbought=70
            )
            
            # Validate signals
            assert isinstance(ma_signals, pd.Series), "MA signals should be a Series"
            assert isinstance(rsi_signals, pd.Series), "RSI signals should be a Series"
            
            # Check signal values
            unique_ma_signals = ma_signals.unique()
            valid_signals = [-1, 0, 1]
            assert all(signal in valid_signals for signal in unique_ma_signals), \
                   "MA signals should only contain -1, 0, 1"
            
            unique_rsi_signals = rsi_signals.unique()
            assert all(signal in valid_signals for signal in unique_rsi_signals), \
                   "RSI signals should only contain -1, 0, 1"
            
            # Count signals
            ma_buy_signals = (ma_signals == 1).sum()
            ma_sell_signals = (ma_signals == -1).sum()
            rsi_buy_signals = (rsi_signals == 1).sum()
            rsi_sell_signals = (rsi_signals == -1).sum()
            
            print(f"   ✓ MA Crossover - Buy signals: {ma_buy_signals}")
            print(f"   ✓ MA Crossover - Sell signals: {ma_sell_signals}")
            print(f"   ✓ RSI - Buy signals: {rsi_buy_signals}")
            print(f"   ✓ RSI - Sell signals: {rsi_sell_signals}")
            print(f"   ✓ Total signals generated: {ma_signals.abs().sum() + rsi_signals.abs().sum()}")
            
            self.test_results['signal_generation'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Signal generation test failed: {str(e)}")
            self.test_results['signal_generation'] = 'FAIL'
    
    def test_multiple_indicators(self):
        """Test multiple indicators calculation"""
        print("\\n6. Testing Multiple Indicators Calculation")
        print("-" * 40)
        
        try:
            # Calculate multiple indicators at once
            all_indicators = self.tech_indicators.calculate_multiple_indicators(
                self.high, self.low, self.close, self.volume
            )
            
            # Validate results
            assert isinstance(all_indicators, pd.DataFrame), \
                   "Multiple indicators should return DataFrame"
            assert len(all_indicators) == len(self.close), \
                   "DataFrame length should match input"
            assert all_indicators.shape[1] > 10, \
                   "Should calculate multiple indicators"
            
            # Check for key indicators
            expected_columns = [
                'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 
                'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                'stoch_k', 'stoch_d', 'williams_r', 'cci'
            ]
            
            for col in expected_columns:
                assert col in all_indicators.columns, f"Missing indicator: {col}"
            
            print(f"   ✓ Indicators calculated: {all_indicators.shape[1]}")
            print(f"   ✓ Data points: {all_indicators.shape[0]:,}")
            print(f"   ✓ Sample indicators:")
            for col in expected_columns[:5]:
                value = all_indicators[col].iloc[-1]
                if not pd.isna(value):
                    print(f"     - {col}: {value:.4f}")
            
            self.test_results['multiple_indicators'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Multiple indicators test failed: {str(e)}")
            self.test_results['multiple_indicators'] = 'FAIL'
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        print("\\n7. Testing Convenience Functions")
        print("-" * 40)
        
        try:
            # Test convenience functions
            sma_conv = calculate_sma(self.close, 20)
            ema_conv = calculate_ema(self.close, 12)
            rsi_conv = calculate_rsi(self.close, 14)
            macd_conv = calculate_macd(self.close, 12, 26, 9)
            bb_conv = calculate_bollinger_bands(self.close, 20, 2.0)
            
            # Validate results
            assert isinstance(sma_conv, pd.Series), "SMA convenience should return Series"
            assert isinstance(ema_conv, pd.Series), "EMA convenience should return Series"
            assert isinstance(rsi_conv, pd.Series), "RSI convenience should return Series"
            assert isinstance(macd_conv, dict), "MACD convenience should return dict"
            assert isinstance(bb_conv, dict), "BB convenience should return dict"
            
            print(f"   ✓ SMA convenience: {sma_conv.iloc[-1]:.2f}")
            print(f"   ✓ EMA convenience: {ema_conv.iloc[-1]:.2f}")
            print(f"   ✓ RSI convenience: {rsi_conv.iloc[-1]:.2f}")
            print(f"   ✓ MACD convenience: {macd_conv['macd'].iloc[-1]:.4f}")
            print(f"   ✓ BB convenience: {bb_conv['upper'].iloc[-1]:.2f}")
            
            self.test_results['convenience_functions'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Convenience functions test failed: {str(e)}")
            self.test_results['convenience_functions'] = 'FAIL'
    
    def test_configuration_system(self):
        """Test indicator configuration system"""
        print("\\n8. Testing Configuration System")
        print("-" * 40)
        
        try:
            # Test different configurations
            config_default = IndicatorConfig()
            config_custom = IndicatorConfig(
                short_period=10,
                long_period=30,
                rsi_period=21,
                bb_period=25,
                bb_std_dev=2.5
            )
            
            # Test indicators with different configs
            indicators_default = TechnicalIndicators(config_default)
            indicators_custom = TechnicalIndicators(config_custom)
            
            rsi_default = indicators_default.calculate_rsi(self.close)
            rsi_custom = indicators_custom.calculate_rsi(self.close)
            
            bb_default = indicators_default.calculate_bollinger_bands(self.close)
            bb_custom = indicators_custom.calculate_bollinger_bands(self.close)
            
            # Validate configurations
            assert config_default.rsi_period == 14, "Default RSI period should be 14"
            assert config_custom.rsi_period == 21, "Custom RSI period should be 21"
            assert config_custom.bb_std_dev == 2.5, "Custom BB std dev should be 2.5"
            
            print(f"   ✓ Default RSI(14): {rsi_default.iloc[-1]:.2f}")
            print(f"   ✓ Custom RSI(21): {rsi_custom.iloc[-1]:.2f}")
            print(f"   ✓ Default BB Upper: {bb_default['upper'].iloc[-1]:.2f}")
            print(f"   ✓ Custom BB Upper: {bb_custom['upper'].iloc[-1]:.2f}")
            print(f"   ✓ Config validation passed")
            
            self.test_results['configuration_system'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Configuration system test failed: {str(e)}")
            self.test_results['configuration_system'] = 'FAIL'
    
    def run_all_tests(self):
        """Run all technical indicators tests"""
        print("\\nStarting Comprehensive Technical Indicators Tests")
        print("=" * 60)
        
        # Run individual tests
        self.test_moving_averages()
        self.test_momentum_indicators()
        self.test_volatility_indicators()
        self.test_trend_indicators()
        self.test_signal_generation()
        self.test_multiple_indicators()
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
            print("\\n🎉 ALL TESTS PASSED! Technical Indicators Library is working correctly.")
        else:
            print(f"\\n⚠️  {total-passed} test(s) failed. Please review the implementation.")
        
        return passed == total


def main():
    """Main test execution function"""
    print("Technical Indicators Library - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run tests
    test_suite = TestTechnicalIndicators()
    success = test_suite.run_all_tests()
    
    print(f"\\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\\n✅ Technical Indicators Library is ready for use!")
    else:
        print("\\n❌ Some tests failed. Please review the implementation.")
    
    return success


if __name__ == "__main__":
    main() 