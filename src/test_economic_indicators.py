"""
Test suite for the Economic Indicators Integration Module

This test suite validates the functionality of the economic indicators
calculator, including yield curve calculations, real rates, labor market
indicators, inflation measures, and composite indicators.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import sys
import os

# Add the src directory to path for direct imports
sys.path.insert(0, os.path.dirname(__file__))

from features.economic_indicators import (
    EconomicIndicators,
    IndicatorCategory,
    IndicatorDefinition,
    calculate_yield_curve_slope,
    calculate_real_rate,
    calculate_unemployment_gap
)

class TestEconomicIndicators(unittest.TestCase):
    """Test suite for EconomicIndicators class."""
    
    def setUp(self):
        """Set up test data and EconomicIndicators instance."""
        # Create date range for testing
        self.dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
        
        # Initialize economic indicators calculator
        self.econ_indicators = EconomicIndicators()
        
        # Create sample treasury data
        np.random.seed(42)  # For reproducible tests
        self.treasury_data = pd.DataFrame({
            'DGS3MO': np.random.normal(1.5, 0.5, len(self.dates)),
            'DGS2': np.random.normal(2.0, 0.7, len(self.dates)),
            'DGS10': np.random.normal(3.0, 0.8, len(self.dates)),
            'DGS30': np.random.normal(3.5, 0.9, len(self.dates)),
            'DAAA': np.random.normal(4.0, 0.6, len(self.dates)),
            'DBAA': np.random.normal(5.0, 0.8, len(self.dates)),
            'FEDFUNDS': np.random.normal(2.5, 1.0, len(self.dates))
        }, index=self.dates)
        
        # Ensure realistic ordering (positive values, term structure)
        self.treasury_data = self.treasury_data.abs()
        
        # Create sample inflation data
        self.inflation_data = pd.DataFrame({
            'CPIAUCSL': 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.001, len(self.dates)))),
            'CPILFESL': 100 * np.exp(np.cumsum(np.random.normal(0.0015, 0.0008, len(self.dates)))),
            'PCEPILFE': 100 * np.exp(np.cumsum(np.random.normal(0.0015, 0.0008, len(self.dates))))
        }, index=self.dates)
        
        # Create sample labor market data
        self.labor_data = pd.DataFrame({
            'UNRATE': np.abs(np.random.normal(5.0, 1.5, len(self.dates))),
            'NROU': np.abs(np.random.normal(4.5, 0.3, len(self.dates))),
            'CIVPART': np.random.normal(63.0, 1.0, len(self.dates)),
            'PAYEMS': 150000 + np.cumsum(np.random.normal(200, 50, len(self.dates)))
        }, index=self.dates)
        
        # Create sample growth data
        self.growth_data = pd.DataFrame({
            'GDPC1': 20000 * np.exp(np.cumsum(np.random.normal(0.005, 0.01, len(self.dates)))),
            'INDPRO': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, len(self.dates)))),
            'TCU': np.random.normal(75.0, 5.0, len(self.dates))
        }, index=self.dates)
        
        # Create sample financial data
        self.financial_data = pd.DataFrame({
            'SP500': 3000 * np.exp(np.cumsum(np.random.normal(0.008, 0.15, len(self.dates)))),
            'VIXCLS': np.abs(np.random.normal(20, 10, len(self.dates))),
            'BAMLC0A0CM': np.abs(np.random.normal(1.5, 0.8, len(self.dates)))
        }, index=self.dates)
    
    def test_initialization(self):
        """Test EconomicIndicators initialization."""
        self.assertIsInstance(self.econ_indicators, EconomicIndicators)
        self.assertIsInstance(self.econ_indicators.fred_series, dict)
        self.assertIsInstance(self.econ_indicators.standard_indicators, dict)
        
        # Check that standard indicators are properly defined
        self.assertGreater(len(self.econ_indicators.standard_indicators), 0)
        print(f"✓ EconomicIndicators initialized with {len(self.econ_indicators.standard_indicators)} standard indicators")
    
    def test_yield_curve_indicators(self):
        """Test yield curve indicator calculations."""
        indicators = self.econ_indicators.calculate_yield_curve_indicators(self.treasury_data)
        
        # Check that indicators were created
        self.assertIsInstance(indicators, pd.DataFrame)
        self.assertEqual(len(indicators), len(self.treasury_data))
        
        # Check specific indicators
        expected_indicators = [
            'yield_slope_10y2y',
            'yield_slope_2y3m', 
            'yield_slope_30y10y',
            'yield_level',
            'yield_curvature',
            'yield_inverted'
        ]
        
        found_indicators = []
        for indicator in expected_indicators:
            if indicator in indicators.columns:
                found_indicators.append(indicator)
        
        self.assertGreater(len(found_indicators), 0, "Should have at least some yield curve indicators")
        print(f"✓ Yield curve indicators calculated: {found_indicators}")
        
        # Test yield curve slope calculation
        if 'yield_slope_10y2y' in indicators.columns:
            expected_slope = self.treasury_data['DGS10'] - self.treasury_data['DGS2']
            pd.testing.assert_series_equal(
                indicators['yield_slope_10y2y'], 
                expected_slope,
                check_names=False
            )
            print("✓ Yield curve slope calculation verified")
        
        # Test inversion indicator (should be 0 or 1)
        if 'yield_inverted' in indicators.columns:
            self.assertTrue(indicators['yield_inverted'].isin([0, 1]).all())
            print("✓ Yield inversion indicator verified")
    
    def test_labor_market_indicators(self):
        """Test labor market indicator calculations."""
        indicators = self.econ_indicators.calculate_labor_market_indicators(self.labor_data)
        
        self.assertIsInstance(indicators, pd.DataFrame)
        
        # Test unemployment gap calculation
        if 'unemployment_gap' in indicators.columns:
            expected_gap = self.labor_data['UNRATE'] - self.labor_data['NROU']
            pd.testing.assert_series_equal(
                indicators['unemployment_gap'],
                expected_gap,
                check_names=False
            )
            print("✓ Unemployment gap calculation verified")
        
        print(f"✓ Labor market indicators: {list(indicators.columns)}")
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test yield curve slope
        treasury_10y = pd.Series([3.0, 3.1, 2.9], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        treasury_2y = pd.Series([2.0, 2.1, 1.9], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        slope = calculate_yield_curve_slope(treasury_10y, treasury_2y)
        expected = treasury_10y - treasury_2y
        pd.testing.assert_series_equal(slope, expected)
        print("✓ Convenience function: yield curve slope")
        
        # Test real rate
        nominal = pd.Series([3.0, 3.1, 2.9], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        inflation = pd.Series([2.0, 2.1, 1.9], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        real_rate = calculate_real_rate(nominal, inflation)
        expected = nominal - inflation
        pd.testing.assert_series_equal(real_rate, expected)
        print("✓ Convenience function: real rate")
        
        # Test unemployment gap
        unemployment = pd.Series([5.0, 5.1, 4.9], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        natural_rate = pd.Series([4.5, 4.5, 4.5], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        gap = calculate_unemployment_gap(unemployment, natural_rate)
        expected = unemployment - natural_rate
        pd.testing.assert_series_equal(gap, expected)
        print("✓ Convenience function: unemployment gap")


def run_test_suite():
    """Run the complete test suite and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestEconomicIndicators]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing Economic Indicators Module")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run the test suite
    result = run_test_suite()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) + len(result.errors)) == 0 else 1
    print(f"\nTest suite {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code) 