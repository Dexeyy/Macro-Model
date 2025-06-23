"""
Simple plotting functions for backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

def plot_backtest_results(backtest_results: Dict[str, Any], 
                         benchmark_returns: Optional[pd.Series] = None,
                         figsize: tuple = (15, 10)):
    """
    Plot comprehensive backtesting results.
    
    Parameters:
    -----------
    backtest_results : Dict
        Results from BacktestFramework.run_backtest()
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Backtest Results: {backtest_results.get('strategy_name', 'Strategy')}", fontsize=16)
    
    # Extract data
    portfolio_value = backtest_results['portfolio_value']
    portfolio_returns = backtest_results['portfolio_returns']
    regimes = backtest_results['regimes']
    weights = backtest_results['portfolio_weights']
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    normalized_value = portfolio_value / portfolio_value.iloc[0]
    ax1.plot(normalized_value.index, normalized_value, label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_cum = (1 + benchmark_returns).cumprod()
        ax1.plot(benchmark_cum.index, benchmark_cum, label='Benchmark', linewidth=2)
    
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    ax2.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
    ax2.set_title('Drawdown Analysis')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Regime Timeline
    ax3 = axes[1, 0]
    unique_regimes = regimes.dropna().unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_regimes)))
    
    for i, regime in enumerate(unique_regimes):
        regime_mask = regimes == regime
        regime_dates = regimes.index[regime_mask]
        ax3.scatter(regime_dates, [i] * len(regime_dates), 
                   c=[colors[i]], label=regime, alpha=0.7, s=10)
    
    ax3.set_title('Regime Classification Over Time')
    ax3.set_ylabel('Regime')
    ax3.set_yticks(range(len(unique_regimes)))
    ax3.set_yticklabels(unique_regimes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Portfolio Weights
    ax4 = axes[1, 1]
    weights.plot.area(ax=ax4, alpha=0.7)
    ax4.set_title('Portfolio Weights Over Time')
    ax4.set_ylabel('Weight')
    ax4.set_ylim(0, 1)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_performance_metrics(performance_metrics: Dict[str, float], figsize: tuple = (10, 6)):
    """
    Plot performance metrics as a bar chart.
    
    Parameters:
    -----------
    performance_metrics : Dict
        Dictionary of performance metrics
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select key metrics for visualization
    key_metrics = {
        'Total Return': performance_metrics.get('total_return', 0),
        'Annualized Return': performance_metrics.get('annualized_return', 0),
        'Volatility': performance_metrics.get('volatility', 0),
        'Sharpe Ratio': performance_metrics.get('sharpe_ratio', 0),
        'Max Drawdown': abs(performance_metrics.get('max_drawdown', 0)),
        'Win Rate': performance_metrics.get('win_rate', 0)
    }
    
    metrics = list(key_metrics.keys())
    values = list(key_metrics.values())
    
    # Color bars based on whether higher is better
    colors = ['green' if v >= 0 else 'red' for v in values]
    colors[4] = 'red'  # Max Drawdown (lower is better)
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_title('Performance Metrics Summary')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_regime_performance(backtest_results: Dict[str, Any], figsize: tuple = (12, 8)):
    """
    Plot performance breakdown by regime.
    
    Parameters:
    -----------
    backtest_results : Dict
        Results from BacktestFramework.run_backtest()
    figsize : tuple
        Figure size
    """
    portfolio_returns = backtest_results['portfolio_returns']
    regimes = backtest_results['regimes']
    
    # Calculate regime-based performance
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Performance by Regime', fontsize=16)
    
    regime_stats = {}
    for regime in regimes.dropna().unique():
        regime_mask = regimes == regime
        regime_returns = portfolio_returns[regime_mask]
        
        if len(regime_returns) > 0:
            regime_stats[regime] = {
                'mean_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),
                'win_rate': (regime_returns > 0).mean(),
                'count': len(regime_returns)
            }
    
    regimes_list = list(regime_stats.keys())
    
    # 1. Mean Returns
    ax1 = axes[0, 0]
    returns_data = [regime_stats[r]['mean_return'] for r in regimes_list]
    ax1.bar(regimes_list, returns_data, alpha=0.7)
    ax1.set_title('Annualized Returns by Regime')
    ax1.set_ylabel('Return')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Volatility
    ax2 = axes[0, 1]
    vol_data = [regime_stats[r]['volatility'] for r in regimes_list]
    ax2.bar(regimes_list, vol_data, alpha=0.7, color='orange')
    ax2.set_title('Volatility by Regime')
    ax2.set_ylabel('Volatility')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Win Rate
    ax3 = axes[1, 0]
    win_rates = [regime_stats[r]['win_rate'] for r in regimes_list]
    ax3.bar(regimes_list, win_rates, alpha=0.7, color='green')
    ax3.set_title('Win Rate by Regime')
    ax3.set_ylabel('Win Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Days in Regime
    ax4 = axes[1, 1]
    durations = [regime_stats[r]['count'] for r in regimes_list]
    ax4.bar(regimes_list, durations, alpha=0.7, color='purple')
    ax4.set_title('Days in Each Regime')
    ax4.set_ylabel('Days')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig 