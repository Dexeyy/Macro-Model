"""
Test script for Performance Analytics Module

This script tests all functionality of the PerformanceAnalytics module
including core metrics, drawdown analysis, regime attribution, benchmark
comparison, and visualization functions.

Author: AI Assistant
Date: 2025-06-22
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.performance_analytics import PerformanceAnalytics, quick_performance_analysis, compare_portfolios

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create sample returns with different regimes
    returns = []
    regimes = []
    
    # Bull market periods (higher returns, lower volatility)
    bull_periods = [(0, 300), (600, 900), (1200, 1400)]
    # Bear market periods (negative returns, higher volatility)
    bear_periods = [(300, 400), (900, 1000)]
    # Neutral periods (moderate returns, moderate volatility)
    
    current_regime = 'neutral'
    for i in range(n_days):
        # Determine regime
        if any(start <= i < end for start, end in bull_periods):
            current_regime = 'bull'
            ret = np.random.normal(0.001, 0.015)  # Higher returns, lower vol
        elif any(start <= i < end for start, end in bear_periods):
            current_regime = 'bear'
            ret = np.random.normal(-0.0005, 0.025)  # Negative returns, higher vol
        else:
            current_regime = 'neutral'
            ret = np.random.normal(0.0003, 0.018)  # Moderate returns and vol
        
        returns.append(ret)
        regimes.append(current_regime)
    
    # Create DataFrames
    portfolio_returns = pd.Series(returns, index=dates, name='portfolio')
    regime_series = pd.Series(regimes, index=dates, name='regime')
    
    # Create benchmark returns (slightly lower performance)
    benchmark_returns = portfolio_returns * 0.8 + np.random.normal(0, 0.005, len(portfolio_returns))
    benchmark_returns.name = 'benchmark'
    
    # Create asset prices for portfolio calculation testing
    n_assets = 4
    asset_names = ['STOCK_A', 'STOCK_B', 'BOND_A', 'BOND_B']
    
    prices_data = {}
    for asset in asset_names:
        # Different asset characteristics
        if 'STOCK' in asset:
            asset_returns = np.random.normal(0.0008, 0.02, n_days)
        else:  # Bonds
            asset_returns = np.random.normal(0.0003, 0.008, n_days)
        
        # Convert to prices - fix the NaN issue by starting with a base price
        asset_returns_series = pd.Series(asset_returns, index=dates)
        price_series = (1 + asset_returns_series).cumprod() * 100
        prices_data[asset] = price_series
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    # Create sample weights
    weights = pd.Series([0.3, 0.3, 0.2, 0.2], index=asset_names)
    
    print(f"✅ Created sample data:")
    print(f"   - Portfolio returns: {len(portfolio_returns)} observations")
    print(f"   - Regimes: {regime_series.value_counts().to_dict()}")
    print(f"   - Asset prices: {prices_df.shape}")
    print(f"   - Benchmark returns: {len(benchmark_returns)} observations")
    
    return portfolio_returns, regime_series, benchmark_returns, prices_df, weights

def test_subtask_10_1(analyzer, portfolio_returns, prices_df, weights):
    """Test Subtask 10.1: Core Performance Metrics Calculation."""
    print("\n🧮 Testing Subtask 10.1: Core Performance Metrics Calculation")
    
    try:
        # Test 1: Calculate portfolio returns from prices and weights
        print("   Testing portfolio returns calculation...")
        calculated_returns = analyzer.calculate_returns(prices_df, weights)
        print(f"   ✅ Portfolio returns calculated: {len(calculated_returns)} observations")
        
        # Test 2: Calculate performance metrics
        print("   Testing performance metrics calculation...")
        metrics = analyzer.calculate_performance_metrics(portfolio_returns)
        
        # Validate key metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'win_rate', 'var_95', 'es_95'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print("   ✅ Performance metrics calculated successfully:")
        print(f"      - Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"      - Volatility: {metrics['volatility']:.2%}")
        print(f"      - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"      - Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"      - Win Rate: {metrics['win_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Subtask 10.1: {str(e)}")
        return False

def test_subtask_10_2(analyzer, portfolio_returns):
    """Test Subtask 10.2: Drawdown Analysis Framework."""
    print("\n📉 Testing Subtask 10.2: Drawdown Analysis Framework")
    
    try:
        # Test drawdown analysis
        print("   Testing drawdown analysis...")
        drawdown_df, drawdown_series = analyzer.analyze_drawdowns(portfolio_returns, top_n=5)
        
        print(f"   ✅ Drawdown analysis completed:")
        print(f"      - Drawdown periods identified: {len(drawdown_df)}")
        print(f"      - Drawdown series length: {len(drawdown_series)}")
        
        if not drawdown_df.empty:
            worst_dd = drawdown_df.iloc[0]
            print(f"      - Worst drawdown: {worst_dd['max_drawdown']:.2%}")
            print(f"      - Duration to trough: {worst_dd['duration_to_trough']} days")
        
        # Test underwater curve
        print("   Testing underwater curve calculation...")
        underwater = analyzer.calculate_underwater_curve(portfolio_returns)
        print(f"   ✅ Underwater curve calculated: {len(underwater)} observations")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Subtask 10.2: {str(e)}")
        return False

def test_subtask_10_3(analyzer, portfolio_returns, regime_series):
    """Test Subtask 10.3: Regime-Based Performance Attribution."""
    print("\n🎯 Testing Subtask 10.3: Regime-Based Performance Attribution")
    
    try:
        # Test regime performance attribution
        print("   Testing regime performance attribution...")
        attribution = analyzer.regime_performance_attribution(portfolio_returns, regime_series)
        
        assert 'overall_performance' in attribution
        assert 'regime_performance' in attribution
        assert 'regime_transitions' in attribution
        
        print("   ✅ Regime attribution analysis completed:")
        print(f"      - Overall performance calculated")
        print(f"      - Regime performance for {len(attribution['regime_performance'])} regimes")
        print(f"      - Regime transitions: {attribution['regime_transitions']['transition_count']}")
        
        # Print regime performance summary
        for regime, perf in attribution['regime_performance'].items():
            print(f"      - {regime}: {perf['annualized_return']:.2%} return, {perf['days']} days")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Subtask 10.3: {str(e)}")
        return False

def test_subtask_10_4(analyzer, portfolio_returns, benchmark_returns):
    """Test Subtask 10.4: Benchmark Comparison Methods."""
    print("\n📊 Testing Subtask 10.4: Benchmark Comparison Methods")
    
    try:
        # Test single benchmark comparison
        print("   Testing benchmark comparison...")
        comparison = analyzer.compare_to_benchmark(
            portfolio_returns, benchmark_returns, "Market Index"
        )
        
        required_fields = [
            'portfolio_metrics', 'benchmark_metrics', 'excess_return',
            'tracking_error', 'information_ratio', 'beta', 'alpha'
        ]
        
        for field in required_fields:
            assert field in comparison, f"Missing field: {field}"
        
        print("   ✅ Benchmark comparison completed:")
        print(f"      - Excess Return: {comparison['excess_return']:.2%}")
        print(f"      - Information Ratio: {comparison['information_ratio']:.3f}")
        print(f"      - Beta: {comparison['beta']:.3f}")
        print(f"      - Alpha: {comparison['alpha']:.2%}")
        print(f"      - Tracking Error: {comparison['tracking_error']:.2%}")
        
        # Test multi-benchmark comparison
        print("   Testing multi-benchmark comparison...")
        benchmarks = {
            'Index_A': benchmark_returns,
            'Index_B': benchmark_returns * 1.1 + np.random.normal(0, 0.002, len(benchmark_returns))
        }
        
        multi_comparison = analyzer.multi_benchmark_comparison(portfolio_returns, benchmarks)
        print(f"   ✅ Multi-benchmark comparison: {len(multi_comparison)} benchmarks")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Subtask 10.4: {str(e)}")
        return False

def test_subtask_10_5(analyzer, portfolio_returns, regime_series, benchmark_returns):
    """Test Subtask 10.5: Performance Visualization Functions."""
    print("\n📈 Testing Subtask 10.5: Performance Visualization Functions")
    
    try:
        # Test performance summary plot
        print("   Testing performance summary plot...")
        fig1 = analyzer.plot_performance_summary(portfolio_returns, regime_series)
        assert fig1.data is not None
        print("   ✅ Performance summary plot created")
        
        # Test regime performance comparison
        print("   Testing regime performance comparison plot...")
        attribution = analyzer.regime_performance_attribution(portfolio_returns, regime_series)
        fig2 = analyzer.plot_regime_performance_comparison(attribution)
        assert fig2.data is not None
        print("   ✅ Regime performance comparison plot created")
        
        # Test drawdown analysis plot
        print("   Testing drawdown analysis plot...")
        fig3 = analyzer.plot_drawdown_analysis(portfolio_returns, regime_series)
        assert fig3.data is not None
        print("   ✅ Drawdown analysis plot created")
        
        # Test benchmark comparison plot
        print("   Testing benchmark comparison plot...")
        comparison = analyzer.compare_to_benchmark(portfolio_returns, benchmark_returns)
        fig4 = analyzer.plot_benchmark_comparison(comparison)
        assert fig4.data is not None
        print("   ✅ Benchmark comparison plot created")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Subtask 10.5: {str(e)}")
        return False

def test_utility_functions(analyzer, portfolio_returns, regime_series, benchmark_returns):
    """Test utility functions and comprehensive report generation."""
    print("\n🔧 Testing Utility Functions")
    
    try:
        # Test comprehensive performance report
        print("   Testing comprehensive performance report...")
        report = analyzer.generate_performance_report(
            portfolio_returns, regime_series, benchmark_returns, "Test Benchmark"
        )
        
        assert 'timestamp' in report
        assert 'performance_metrics' in report
        assert 'drawdown_analysis' in report
        assert 'regime_attribution' in report
        assert 'benchmark_comparison' in report
        
        print("   ✅ Comprehensive performance report generated")
        
        # Test metrics export to DataFrame
        print("   Testing metrics export to DataFrame...")
        df = analyzer.export_metrics_to_dataframe(report['performance_metrics'])
        assert not df.empty
        print(f"   ✅ Metrics exported to DataFrame: {df.shape}")
        
        # Test quick analysis function
        print("   Testing quick analysis function...")
        quick_results = quick_performance_analysis(portfolio_returns, regime_series)
        assert 'metrics' in quick_results
        assert 'regime_attribution' in quick_results
        print("   ✅ Quick analysis function working")
        
        # Test portfolio comparison function
        print("   Testing portfolio comparison function...")
        portfolios = {
            'Portfolio_A': portfolio_returns,
            'Portfolio_B': portfolio_returns * 0.9 + np.random.normal(0, 0.005, len(portfolio_returns))
        }
        comparison_df = compare_portfolios(portfolios, benchmark_returns)
        assert not comparison_df.empty
        print(f"   ✅ Portfolio comparison: {comparison_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Utility Functions: {str(e)}")
        return False

def main():
    """Main testing function."""
    print("🚀 Starting Performance Analytics Module Test")
    print("=" * 60)
    
    # Create sample data
    portfolio_returns, regime_series, benchmark_returns, prices_df, weights = create_sample_data()
    
    # Initialize analyzer
    analyzer = PerformanceAnalytics(risk_free_rate=0.02, annualization_factor=252)
    print(f"\n✅ PerformanceAnalytics initialized")
    
    # Test all subtasks
    results = []
    
    results.append(test_subtask_10_1(analyzer, portfolio_returns, prices_df, weights))
    results.append(test_subtask_10_2(analyzer, portfolio_returns))
    results.append(test_subtask_10_3(analyzer, portfolio_returns, regime_series))
    results.append(test_subtask_10_4(analyzer, portfolio_returns, benchmark_returns))
    results.append(test_subtask_10_5(analyzer, portfolio_returns, regime_series, benchmark_returns))
    results.append(test_utility_functions(analyzer, portfolio_returns, regime_series, benchmark_returns))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Subtask 10.1: Core Performance Metrics",
        "Subtask 10.2: Drawdown Analysis Framework", 
        "Subtask 10.3: Regime-Based Performance Attribution",
        "Subtask 10.4: Benchmark Comparison Methods",
        "Subtask 10.5: Performance Visualization Functions",
        "Utility Functions"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Performance Analytics Module is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 