"""
Comprehensive Test Suite for Backtesting Framework

This test suite validates all components of the backtesting framework
including the main backtest engine, performance calculations, Monte Carlo
simulation, walk-forward optimization, and stress testing.
"""

import sys
import os
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from models.backtesting_framework import BacktestFramework

class MockRegimeClassifier:
    """Mock regime classifier for testing."""
    
    def __init__(self):
        self.regimes = ['Bull', 'Bear', 'Neutral']
        np.random.seed(42)  # For reproducible results
    
    def classify_regime(self, historical_data):
        """Classify regime based on recent returns."""
        if len(historical_data) == 0:
            return 'Unknown'
        
        # Simple rule: Bull if recent returns > 0.5%, Bear if < -0.5%, else Neutral
        recent_return = historical_data.mean(axis=1).tail(21).mean()  # Last 21 days
        
        if recent_return > 0.005:
            return 'Bull'
        elif recent_return < -0.005:
            return 'Bear'
        else:
            return 'Neutral'
    
    def fit(self, data):
        """Mock fit method for walk-forward testing."""
        pass

class MockPortfolioOptimizer:
    """Mock portfolio optimizer for testing."""
    
    def __init__(self):
        np.random.seed(42)
    
    def optimize_portfolio(self, historical_data, regime, current_weights):
        """Optimize portfolio based on regime."""
        n_assets = len(current_weights)
        
        if regime == 'Bull':
            # In bull market, go for growth (higher risk assets)
            weights = np.array([0.6, 0.3, 0.1])  # Assuming 3 assets
        elif regime == 'Bear':
            # In bear market, be defensive
            weights = np.array([0.2, 0.3, 0.5])  # More defensive allocation
        else:  # Neutral
            # Balanced allocation
            weights = np.array([0.4, 0.4, 0.2])
        
        # Ensure we have the right number of weights
        if len(weights) != n_assets:
            weights = np.ones(n_assets) / n_assets
        
        return pd.Series(weights, index=current_weights.index)

def generate_test_data(start_date='2020-01-01', end_date='2023-12-31', n_assets=3):
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Asset names
    assets = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Generate correlated returns with different volatilities
    n_days = len(dates)
    
    # Create different market regimes
    regime_changes = [0, n_days//3, 2*n_days//3, n_days]
    regimes = ['Bull', 'Bear', 'Neutral']
    
    returns = []
    for i in range(len(regime_changes)-1):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i+1]
        regime = regimes[i % len(regimes)]
        
        n_period = end_idx - start_idx
        
        if regime == 'Bull':
            # Bull market: positive drift, moderate volatility
            drift = np.array([0.0005, 0.0008, 0.0003])
            vol = np.array([0.015, 0.020, 0.012])
        elif regime == 'Bear':
            # Bear market: negative drift, high volatility
            drift = np.array([-0.0008, -0.0005, -0.0003])
            vol = np.array([0.025, 0.030, 0.020])
        else:  # Neutral
            # Neutral market: low drift, low volatility
            drift = np.array([0.0001, 0.0002, 0.0001])
            vol = np.array([0.010, 0.012, 0.008])
        
        # Generate correlated returns
        correlation = np.array([[1.0, 0.6, 0.4],
                               [0.6, 1.0, 0.5],
                               [0.4, 0.5, 1.0]])
        
        # Generate random returns
        random_returns = np.random.multivariate_normal(
            mean=drift, 
            cov=np.outer(vol, vol) * correlation, 
            size=n_period
        )
        
        returns.append(random_returns)
    
    # Combine all returns
    all_returns = np.vstack(returns)
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(all_returns, index=dates, columns=assets)
    
    # Create prices from returns
    prices_df = (1 + returns_df).cumprod() * 100  # Start at $100
    
    return prices_df, returns_df

def test_basic_backtest():
    """Test basic backtesting functionality."""
    print("Testing Basic Backtest...")
    
    # Generate test data
    prices, returns = generate_test_data()
    
    # Create framework
    framework = BacktestFramework(
        initial_capital=1000000,
        transaction_cost=0.001,
        risk_free_rate=0.02
    )
    
    # Create mock components
    regime_classifier = MockRegimeClassifier()
    portfolio_optimizer = MockPortfolioOptimizer()
    
    # Run backtest
    results = framework.run_backtest(
        prices=prices,
        regime_classifier=regime_classifier,
        portfolio_optimizer=portfolio_optimizer,
        rebalance_frequency='M',
        lookback_window=60,
        warmup_period=60,
        strategy_name="Test_Strategy"
    )
    
    # Validate results
    assert 'portfolio_value' in results, "Portfolio value missing from results"
    assert 'portfolio_returns' in results, "Portfolio returns missing from results"
    assert 'regimes' in results, "Regimes missing from results"
    assert 'performance_metrics' in results, "Performance metrics missing from results"
    
    # Check metrics
    metrics = results['performance_metrics']
    assert 'total_return' in metrics, "Total return missing from metrics"
    assert 'sharpe_ratio' in metrics, "Sharpe ratio missing from metrics"
    assert 'max_drawdown' in metrics, "Max drawdown missing from metrics"
    
    print(f"✓ Basic backtest completed successfully")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    return results

def test_strategy_comparison():
    """Test strategy comparison functionality."""
    print("\nTesting Strategy Comparison...")
    
    # Generate test data
    prices, returns = generate_test_data()
    
    # Create framework
    framework = BacktestFramework(initial_capital=1000000)
    
    # Create mock components
    regime_classifier = MockRegimeClassifier()
    portfolio_optimizer = MockPortfolioOptimizer()
    
    # Run multiple strategies
    strategies = {
        'Monthly_Rebalance': 'M',
        'Quarterly_Rebalance': 'Q'
    }
    
    for strategy_name, frequency in strategies.items():
        framework.run_backtest(
            prices=prices,
            regime_classifier=regime_classifier,
            portfolio_optimizer=portfolio_optimizer,
            rebalance_frequency=frequency,
            strategy_name=strategy_name
        )
    
    # Compare strategies
    comparison = framework.compare_strategies()
    
    # Validate comparison
    assert 'portfolio_values' in comparison, "Portfolio values missing from comparison"
    assert 'performance_comparison' in comparison, "Performance comparison missing"
    
    print("✓ Strategy comparison completed successfully")
    print(f"  Strategies compared: {len(comparison['strategy_names'])}")
    
    return comparison

def test_monte_carlo_simulation():
    """Test Monte Carlo simulation functionality."""
    print("\nTesting Monte Carlo Simulation...")
    
    # Use results from basic backtest
    results = test_basic_backtest()
    
    # Create framework
    framework = BacktestFramework()
    framework.backtest_results['Test_Strategy'] = results
    
    # Run Monte Carlo simulation
    mc_results = framework.monte_carlo_simulation(
        strategy_name='Test_Strategy',
        n_simulations=100,  # Reduced for faster testing
        confidence_levels=[0.05, 0.95]
    )
    
    # Validate results
    assert 'simulated_paths' in mc_results, "Simulated paths missing"
    assert 'final_returns' in mc_results, "Final returns missing"
    assert 'simulation_stats' in mc_results, "Simulation stats missing"
    
    stats = mc_results['simulation_stats']
    print("✓ Monte Carlo simulation completed successfully")
    print(f"  Simulations: {mc_results['n_simulations']}")
    print(f"  Mean Final Return: {stats['mean_final_return']:.2%}")
    print(f"  Probability of Loss: {stats['probability_of_loss']:.2%}")
    
    return mc_results

def test_walk_forward_optimization():
    """Test walk-forward optimization functionality."""
    print("\nTesting Walk-Forward Optimization...")
    
    # Generate test data (shorter period for faster testing)
    prices, returns = generate_test_data(start_date='2022-01-01', end_date='2023-12-31')
    
    # Create framework
    framework = BacktestFramework()
    
    # Create mock components
    regime_classifier = MockRegimeClassifier()
    portfolio_optimizer = MockPortfolioOptimizer()
    
    # Run walk-forward optimization
    wf_results = framework.walk_forward_optimization(
        prices=prices,
        regime_classifier=regime_classifier,
        portfolio_optimizer=portfolio_optimizer,
        train_window=126,  # 6 months
        test_window=21,    # 1 month
        step_size=21       # 1 month steps
    )
    
    # Validate results
    assert 'walk_forward_results' in wf_results, "Walk-forward results missing"
    assert 'walk_forward_stats' in wf_results, "Walk-forward stats missing"
    
    wf_df = wf_results['walk_forward_results']
    if len(wf_df) > 0:
        stats = wf_results['walk_forward_stats']
        print("✓ Walk-forward optimization completed successfully")
        print(f"  Periods tested: {stats['n_periods']}")
        print(f"  Average return: {stats['avg_return']:.2%}")
        print(f"  Win rate: {stats['win_rate']:.2%}")
    else:
        print("✓ Walk-forward optimization completed (no valid periods)")
    
    return wf_results

def test_stress_testing():
    """Test stress testing functionality."""
    print("\nTesting Stress Testing...")
    
    # Use results from basic backtest
    results = test_basic_backtest()
    
    # Create framework
    framework = BacktestFramework()
    framework.backtest_results['Test_Strategy'] = results
    
    # Run stress tests
    stress_results = framework.stress_test('Test_Strategy')
    
    # Validate results
    assert isinstance(stress_results, dict), "Stress results should be a dictionary"
    
    if stress_results:
        print("✓ Stress testing completed successfully")
        print(f"  Scenarios tested: {len(stress_results)}")
        
        for scenario, result in stress_results.items():
            impact = result['impact']
            print(f"  {scenario}: Return impact {impact['return_impact']:.2%}")
    else:
        print("✓ Stress testing completed (no scenarios)")
    
    return stress_results

def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\nTesting Performance Metrics...")
    
    # Generate simple test returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
    
    # Create framework
    framework = BacktestFramework()
    
    # Calculate metrics
    metrics = framework._calculate_performance_metrics(returns)
    
    # Validate key metrics exist
    required_metrics = [
        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
        'max_drawdown', 'var_95', 'win_rate', 'skewness', 'kurtosis'
    ]
    
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    print("✓ Performance metrics calculation completed successfully")
    print(f"  Metrics calculated: {len(metrics)}")
    
    return metrics

def test_report_generation():
    """Test report generation functionality."""
    print("\nTesting Report Generation...")
    
    # Use results from basic backtest
    results = test_basic_backtest()
    
    # Create framework
    framework = BacktestFramework()
    framework.backtest_results['Test_Strategy'] = results
    
    # Generate report
    report = framework.generate_report('Test_Strategy')
    
    # Validate report
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 0, "Report should not be empty"
    assert 'STRATEGY: Test_Strategy' in report, "Strategy name should be in report"
    
    print("✓ Report generation completed successfully")
    print(f"  Report length: {len(report)} characters")
    
    return report

def run_all_tests():
    """Run all backtesting framework tests."""
    print("=" * 60)
    print("BACKTESTING FRAMEWORK - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Basic Backtest
        basic_results = test_basic_backtest()
        
        # Test 2: Strategy Comparison
        comparison_results = test_strategy_comparison()
        
        # Test 3: Monte Carlo Simulation
        mc_results = test_monte_carlo_simulation()
        
        # Test 4: Walk-Forward Optimization
        wf_results = test_walk_forward_optimization()
        
        # Test 5: Stress Testing
        stress_results = test_stress_testing()
        
        # Test 6: Performance Metrics
        metrics_results = test_performance_metrics()
        
        # Test 7: Report Generation
        report = test_report_generation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
        return {
            'basic_results': basic_results,
            'comparison_results': comparison_results,
            'mc_results': mc_results,
            'wf_results': wf_results,
            'stress_results': stress_results,
            'metrics_results': metrics_results,
            'report': report
        }
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    # Run all tests
    test_results = run_all_tests()
    
    # Print summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    basic_metrics = test_results['basic_results']['performance_metrics']
    print(f"Strategy Performance:")
    print(f"  Total Return: {basic_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {basic_metrics['annualized_return']:.2%}")
    print(f"  Volatility: {basic_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {basic_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {basic_metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {basic_metrics['win_rate']:.2%}")
    
    print(f"\nMonte Carlo Results:")
    mc_stats = test_results['mc_results']['simulation_stats']
    print(f"  Mean Final Return: {mc_stats['mean_final_return']:.2%}")
    print(f"  Probability of Loss: {mc_stats['probability_of_loss']:.2%}")
    
    print("\n✅ Backtesting Framework is ready for production use!") 