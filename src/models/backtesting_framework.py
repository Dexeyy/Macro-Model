"""
Comprehensive Backtesting Framework for Regime-Based Portfolio Strategies

This module provides a complete backtesting framework for evaluating portfolio strategies
that use regime classification and dynamic portfolio optimization.

Key Features:
- Historical simulation with transaction costs
- Multiple rebalancing frequencies
- Regime-based strategy evaluation
- Performance metrics calculation
- Monte Carlo simulation
- Walk-forward optimization
- Stress testing capabilities
- Strategy comparison tools

Author: Macro Regime Model Project
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestFramework:
    """
    Comprehensive backtesting framework for regime-based portfolio strategies.
    
    This class provides all the tools needed to backtest portfolio strategies
    that use regime classification and dynamic optimization.
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Initialize the backtesting framework.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital for the backtest
        transaction_cost : float
            Transaction cost as a percentage of trade value
        risk_free_rate : float
            Risk-free rate for performance calculations
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        
        logger.info(f"BacktestFramework initialized with ${initial_capital:,.0f} initial capital")
    
    def run_backtest(self,
                    prices: pd.DataFrame,
                    regime_classifier,
                    portfolio_optimizer,
                    rebalance_frequency: str = 'M',
                    lookback_window: int = 252,
                    warmup_period: int = 252,
                    strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Run a comprehensive backtest of a regime-based portfolio strategy.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset price data with dates as index and assets as columns
        regime_classifier : object
            Regime classification model with classify_regime method
        portfolio_optimizer : object
            Portfolio optimization model with optimize_portfolio method
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        lookback_window : int
            Number of days to look back for regime classification
        warmup_period : int
            Number of days before starting the backtest
        strategy_name : str
            Name for this strategy
            
        Returns:
        --------
        Dict containing all backtest results
        """
        logger.info(f"Starting backtest for {strategy_name}")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Initialize results containers
        portfolio_value = pd.Series(index=returns.index, dtype=float)
        portfolio_weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        regimes = pd.Series(index=returns.index, dtype='object')
        turnover = pd.Series(index=returns.index, dtype=float)
        transaction_costs = pd.Series(index=returns.index, dtype=float)
        
        # Set initial portfolio (equal weight)
        n_assets = len(returns.columns)
        current_weights = pd.Series(1.0 / n_assets, index=returns.columns)
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(returns, rebalance_frequency)
        
        # Initialize portfolio value
        portfolio_value.iloc[0] = self.initial_capital
        portfolio_weights.iloc[0] = current_weights
        
        # Run backtest simulation
        for i, date in enumerate(returns.index):
            # Check if this is a rebalance date and we have enough history
            if date in rebalance_dates and i >= warmup_period:
                # Get historical data for regime classification
                history_start = max(0, i - lookback_window)
                historical_data = returns.iloc[history_start:i]
                
                try:
                    # Classify current regime
                    current_regime = regime_classifier.classify_regime(historical_data)
                    regimes.loc[date] = current_regime
                    
                    # Optimize portfolio for current regime
                    new_weights = portfolio_optimizer.optimize_portfolio(
                        historical_data, current_regime, current_weights
                    )
                    
                    # Calculate turnover
                    turnover.loc[date] = np.sum(np.abs(new_weights - current_weights))
                    
                    # Calculate transaction costs
                    transaction_costs.loc[date] = turnover.loc[date] * self.transaction_cost
                    
                    # Update weights
                    current_weights = new_weights
                    
                except Exception as e:
                    logger.warning(f"Error in optimization on {date}: {e}")
                    # Use previous weights
                    if i > 0:
                        regimes.loc[date] = regimes.iloc[i-1] if not pd.isna(regimes.iloc[i-1]) else 'Unknown'
                    turnover.loc[date] = 0
                    transaction_costs.loc[date] = 0
            else:
                # Use previous weights and regime
                if i > 0:
                    regimes.loc[date] = regimes.iloc[i-1] if not pd.isna(regimes.iloc[i-1]) else 'Unknown'
                turnover.loc[date] = 0
                transaction_costs.loc[date] = 0
            
            # Update portfolio weights
            portfolio_weights.loc[date] = current_weights
            
            # Calculate portfolio return for this day
            if i > 0:
                # Get previous day's weights
                prev_weights = portfolio_weights.iloc[i-1]
                
                # Calculate portfolio return (before rebalancing)
                port_return = np.sum(prev_weights * returns.iloc[i])
                
                # Apply transaction costs if this is a rebalance date
                if date in rebalance_dates and i >= warmup_period:
                    port_return -= transaction_costs.loc[date]
                
                # Update portfolio value
                portfolio_value.loc[date] = portfolio_value.iloc[i-1] * (1 + port_return)
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_returns)
        
        # Store results
        results = {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'portfolio_weights': portfolio_weights,
            'regimes': regimes,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'performance_metrics': performance_metrics,
            'strategy_name': strategy_name,
            'rebalance_frequency': rebalance_frequency,
            'lookback_window': lookback_window
        }
        
        self.backtest_results[strategy_name] = results
        logger.info(f"Backtest completed for {strategy_name}")
        
        return results
    
    def _get_rebalance_dates(self, returns: pd.DataFrame, frequency: str) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency."""
        freq_map = {
            'D': 'D',
            'W': 'W',
            'M': 'ME',  # Updated for pandas 2.0+
            'Q': 'QE',  # Updated for pandas 2.0+
            'Y': 'YE'   # Updated for pandas 2.0+
        }
        
        if frequency in freq_map:
            rebalance_dates = returns.resample(freq_map[frequency]).last().index
        else:
            # Default to monthly
            rebalance_dates = returns.resample('ME').last().index
        
        # Add the last date
        if returns.index[-1] not in rebalance_dates:
            rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([returns.index[-1]]))
        
        return rebalance_dates
    
    def _calculate_performance_metrics(self, 
                                     returns: pd.Series, 
                                     annualization_factor: int = 252) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a return series.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio return series
        annualization_factor : int
            Factor for annualizing metrics (252 for daily data)
            
        Returns:
        --------
        Dictionary of performance metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(annualization_factor)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        expected_shortfall_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        expected_shortfall_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        sortino_ratio = (annualized_return - self.risk_free_rate) / (returns[returns < 0].std() * np.sqrt(annualization_factor)) if len(returns[returns < 0]) > 0 else np.inf
        win_rate = (returns > 0).mean()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'expected_shortfall_99': expected_shortfall_99,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def compare_strategies(self, 
                          strategy_names: List[str] = None,
                          benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Compare multiple backtest strategies.
        
        Parameters:
        -----------
        strategy_names : List[str], optional
            Names of strategies to compare. If None, compares all strategies
        benchmark_returns : pd.Series, optional
            Benchmark return series for comparison
            
        Returns:
        --------
        Dictionary containing comparison results
        """
        if strategy_names is None:
            strategy_names = list(self.backtest_results.keys())
        
        if not strategy_names:
            logger.warning("No strategies available for comparison")
            return {}
        
        # Combine portfolio values
        portfolio_values = pd.DataFrame()
        for name in strategy_names:
            if name in self.backtest_results:
                results = self.backtest_results[name]
                normalized_value = results['portfolio_value'] / results['portfolio_value'].iloc[0]
                portfolio_values[name] = normalized_value
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum_return = (1 + benchmark_returns).cumprod()
            portfolio_values['Benchmark'] = benchmark_cum_return / benchmark_cum_return.iloc[0]
        
        # Compare performance metrics
        performance_comparison = pd.DataFrame()
        for name in strategy_names:
            if name in self.backtest_results:
                performance_comparison[name] = pd.Series(self.backtest_results[name]['performance_metrics'])
        
        # Add benchmark metrics if provided
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_performance_metrics(benchmark_returns)
            performance_comparison['Benchmark'] = pd.Series(benchmark_metrics)
        
        return {
            'portfolio_values': portfolio_values,
            'performance_comparison': performance_comparison,
            'strategy_names': strategy_names
        }
    
    def monte_carlo_simulation(self,
                              strategy_name: str,
                              n_simulations: int = 1000,
                              confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation on strategy returns.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to simulate
        n_simulations : int
            Number of Monte Carlo simulations
        confidence_levels : List[float]
            Confidence levels for the simulation
            
        Returns:
        --------
        Dictionary containing simulation results
        """
        if strategy_name not in self.backtest_results:
            logger.error(f"Strategy {strategy_name} not found in backtest results")
            return {}
        
        logger.info(f"Running Monte Carlo simulation for {strategy_name} with {n_simulations} simulations")
        
        returns = self.backtest_results[strategy_name]['portfolio_returns']
        
        # Bootstrap simulation
        simulated_paths = []
        final_returns = []
        
        for i in range(n_simulations):
            # Bootstrap returns
            sim_returns = np.random.choice(returns.values, size=len(returns), replace=True)
            
            # Calculate cumulative path
            cum_path = (1 + pd.Series(sim_returns)).cumprod()
            simulated_paths.append(cum_path.values)
            final_returns.append(cum_path.iloc[-1] - 1)
        
        # Calculate confidence intervals
        final_returns = np.array(final_returns)
        confidence_intervals = {}
        for level in confidence_levels:
            confidence_intervals[f'{level*100:.0f}%'] = np.percentile(final_returns, level * 100)
        
        # Calculate statistics
        simulation_stats = {
            'mean_final_return': np.mean(final_returns),
            'std_final_return': np.std(final_returns),
            'min_final_return': np.min(final_returns),
            'max_final_return': np.max(final_returns),
            'confidence_intervals': confidence_intervals,
            'probability_of_loss': (final_returns < 0).mean(),
            'expected_shortfall_5%': final_returns[final_returns <= np.percentile(final_returns, 5)].mean()
        }
        
        return {
            'simulated_paths': np.array(simulated_paths),
            'final_returns': final_returns,
            'simulation_stats': simulation_stats,
            'n_simulations': n_simulations,
            'original_returns': returns
        }
    
    def walk_forward_optimization(self,
                                 prices: pd.DataFrame,
                                 regime_classifier,
                                 portfolio_optimizer,
                                 train_window: int = 252,
                                 test_window: int = 63,
                                 step_size: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward optimization and testing.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset price data
        regime_classifier : object
            Regime classification model
        portfolio_optimizer : object
            Portfolio optimization model
        train_window : int
            Training window size in days
        test_window : int
            Testing window size in days
        step_size : int
            Step size for rolling window
            
        Returns:
        --------
        Dictionary containing walk-forward results
        """
        logger.info("Starting walk-forward optimization")
        
        returns = prices.pct_change().dropna()
        
        # Initialize results
        walk_forward_results = []
        all_predictions = pd.Series(dtype=float)
        all_actuals = pd.Series(dtype=float)
        
        start_idx = train_window
        while start_idx + test_window < len(returns):
            # Define windows
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + test_window, len(returns))
            
            # Get data
            train_data = returns.iloc[train_start:train_end]
            test_data = returns.iloc[test_start:test_end]
            
            try:
                # Train regime classifier on training data
                regime_classifier.fit(train_data)
                
                # Run backtest on test period
                test_results = self.run_backtest(
                    prices.iloc[test_start:test_end + 1],  # +1 for prices
                    regime_classifier,
                    portfolio_optimizer,
                    rebalance_frequency='W',  # Weekly for shorter test periods
                    lookback_window=min(63, len(train_data)),
                    warmup_period=0,  # No warmup for test period
                    strategy_name=f"WF_{test_start}_{test_end}"
                )
                
                # Store results
                period_result = {
                    'train_start': returns.index[train_start],
                    'train_end': returns.index[train_end - 1],
                    'test_start': returns.index[test_start],
                    'test_end': returns.index[test_end - 1],
                    'test_return': test_results['performance_metrics']['total_return'],
                    'test_sharpe': test_results['performance_metrics']['sharpe_ratio'],
                    'test_max_dd': test_results['performance_metrics']['max_drawdown']
                }
                
                walk_forward_results.append(period_result)
                
                # Collect predictions and actuals
                test_returns = test_results['portfolio_returns']
                all_predictions = pd.concat([all_predictions, test_returns])
                
                # Calculate benchmark returns for the same period
                benchmark_returns = returns.iloc[test_start:test_end].mean(axis=1)
                all_actuals = pd.concat([all_actuals, benchmark_returns])
                
            except Exception as e:
                logger.warning(f"Error in walk-forward period {test_start}-{test_end}: {e}")
                continue
            
            # Move to next period
            start_idx += step_size
        
        # Calculate overall walk-forward statistics
        wf_df = pd.DataFrame(walk_forward_results)
        
        if len(wf_df) > 0:
            wf_stats = {
                'avg_return': wf_df['test_return'].mean(),
                'avg_sharpe': wf_df['test_sharpe'].mean(),
                'avg_max_dd': wf_df['test_max_dd'].mean(),
                'win_rate': (wf_df['test_return'] > 0).mean(),
                'consistency': wf_df['test_return'].std(),
                'n_periods': len(wf_df)
            }
        else:
            wf_stats = {}
        
        return {
            'walk_forward_results': wf_df,
            'walk_forward_stats': wf_stats,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals
        }
    
    def stress_test(self,
                   strategy_name: str,
                   stress_scenarios: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Perform stress testing on a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to stress test
        stress_scenarios : Dict[str, Dict]
            Custom stress scenarios
            
        Returns:
        --------
        Dictionary containing stress test results
        """
        if strategy_name not in self.backtest_results:
            logger.error(f"Strategy {strategy_name} not found in backtest results")
            return {}
        
        logger.info(f"Running stress tests for {strategy_name}")
        
        returns = self.backtest_results[strategy_name]['portfolio_returns']
        
        # Default stress scenarios
        if stress_scenarios is None:
            stress_scenarios = {
                'Market Crash': {'shock': -0.20, 'duration': 5},
                'High Volatility': {'volatility_multiplier': 2.0, 'duration': 21},
                'Bear Market': {'trend': -0.001, 'duration': 252},
                'Flash Crash': {'shock': -0.10, 'duration': 1},
                'Extended Recession': {'trend': -0.0005, 'duration': 504}
            }
        
        stress_results = {}
        
        for scenario_name, params in stress_scenarios.items():
            try:
                # Apply stress scenario
                stressed_returns = self._apply_stress_scenario(returns, params)
                
                # Calculate stressed performance
                stressed_metrics = self._calculate_performance_metrics(stressed_returns)
                
                # Calculate impact
                original_metrics = self.backtest_results[strategy_name]['performance_metrics']
                impact = {
                    'return_impact': stressed_metrics['total_return'] - original_metrics['total_return'],
                    'sharpe_impact': stressed_metrics['sharpe_ratio'] - original_metrics['sharpe_ratio'],
                    'max_dd_impact': stressed_metrics['max_drawdown'] - original_metrics['max_drawdown']
                }
                
                stress_results[scenario_name] = {
                    'stressed_metrics': stressed_metrics,
                    'impact': impact,
                    'scenario_params': params
                }
                
            except Exception as e:
                logger.warning(f"Error in stress scenario {scenario_name}: {e}")
                continue
        
        return stress_results
    
    def _apply_stress_scenario(self, returns: pd.Series, params: Dict) -> pd.Series:
        """Apply a stress scenario to returns."""
        stressed_returns = returns.copy()
        
        if 'shock' in params:
            # Apply immediate shock
            duration = params.get('duration', 1)
            shock_dates = returns.index[:duration]
            stressed_returns.loc[shock_dates] += params['shock'] / duration
            
        if 'volatility_multiplier' in params:
            # Increase volatility
            duration = params.get('duration', len(returns))
            multiplier = params['volatility_multiplier']
            vol_dates = returns.index[:duration]
            
            # Scale returns around their mean
            mean_return = returns.loc[vol_dates].mean()
            stressed_returns.loc[vol_dates] = mean_return + (returns.loc[vol_dates] - mean_return) * multiplier
            
        if 'trend' in params:
            # Apply negative trend
            duration = params.get('duration', len(returns))
            trend = params['trend']
            trend_dates = returns.index[:duration]
            
            # Add linear trend
            for i, date in enumerate(trend_dates):
                stressed_returns.loc[date] += trend * i
        
        return stressed_returns
    
    def generate_report(self, strategy_name: str = None) -> str:
        """
        Generate a comprehensive backtest report.
        
        Parameters:
        -----------
        strategy_name : str, optional
            Strategy to report on. If None, reports on all strategies
            
        Returns:
        --------
        Formatted report string
        """
        if strategy_name and strategy_name not in self.backtest_results:
            return f"Strategy {strategy_name} not found in backtest results"
        
        strategies = [strategy_name] if strategy_name else list(self.backtest_results.keys())
        
        report = "=" * 80 + "\n"
        report += "BACKTESTING FRAMEWORK - COMPREHENSIVE REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for strat_name in strategies:
            results = self.backtest_results[strat_name]
            metrics = results['performance_metrics']
            
            report += f"STRATEGY: {strat_name}\n"
            report += "-" * 40 + "\n"
            report += f"Rebalance Frequency: {results['rebalance_frequency']}\n"
            report += f"Lookback Window: {results['lookback_window']} days\n"
            report += f"Total Return: {metrics['total_return']:.2%}\n"
            report += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            report += f"Volatility: {metrics['volatility']:.2%}\n"
            report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
            report += f"Sortino Ratio: {metrics['sortino_ratio']:.3f}\n"
            report += f"Calmar Ratio: {metrics['calmar_ratio']:.3f}\n"
            report += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            report += f"Win Rate: {metrics['win_rate']:.2%}\n"
            report += f"VaR (95%): {metrics['var_95']:.2%}\n"
            report += f"Expected Shortfall (95%): {metrics['expected_shortfall_95']:.2%}\n"
            report += f"Skewness: {metrics['skewness']:.3f}\n"
            report += f"Kurtosis: {metrics['kurtosis']:.3f}\n"
            
            # Regime analysis
            regimes = results['regimes'].dropna()
            if len(regimes) > 0:
                report += f"\nRegime Distribution:\n"
                regime_counts = regimes.value_counts()
                for regime, count in regime_counts.items():
                    report += f"  {regime}: {count} days ({count/len(regimes):.1%})\n"
            
            # Transaction costs
            total_turnover = results['turnover'].sum()
            total_costs = results['transaction_costs'].sum()
            report += f"\nTransaction Analysis:\n"
            report += f"  Total Turnover: {total_turnover:.2f}\n"
            report += f"  Total Transaction Costs: {total_costs:.4f} ({total_costs/metrics['total_return']:.2%} of returns)\n"
            
            report += "\n" + "=" * 80 + "\n\n"
        
        return report 