"""
Visualization Module for Backtesting Framework

This module provides comprehensive visualization capabilities for backtesting results,
including performance charts, regime analysis, drawdown analysis, and comparison plots.

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
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class BacktestVisualizer:
    """
    Comprehensive visualization class for backtesting results.
    """
    
    def __init__(self, theme: str = 'default'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        theme : str
            Visualization theme ('default', 'dark', 'minimal')
        """
        self.theme = theme
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib and seaborn styling."""
        if self.theme == 'dark':
            plt.style.use('dark_background')
            sns.set_palette("bright")
        else:
            plt.style.use('default')
            sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_performance_summary(self, backtest_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive performance summary plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Drawdown', 'Regime Timeline', 'Weights'),
            vertical_spacing=0.1
        )
        
        # Extract data
        portfolio_value = backtest_results['portfolio_value']
        portfolio_returns = backtest_results['portfolio_returns']
        regimes = backtest_results['regimes']
        weights = backtest_results['portfolio_weights']
        
        # 1. Cumulative Returns
        normalized_value = portfolio_value / portfolio_value.iloc[0]
        fig.add_trace(
            go.Scatter(x=portfolio_value.index, y=normalized_value,
                      name='Strategy', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown * 100,
                      fill='tonexty', name='Drawdown (%)',
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=1, col=2
        )
        
        # 3. Regime Timeline
        unique_regimes = regimes.dropna().unique()
        colors = px.colors.qualitative.Set1
        for i, regime in enumerate(unique_regimes):
            regime_mask = regimes == regime
            fig.add_trace(
                go.Scatter(x=regimes.index[regime_mask], 
                          y=[i] * sum(regime_mask),
                          mode='markers', name=f'Regime: {regime}',
                          marker=dict(color=colors[i % len(colors)], size=3)),
                row=2, col=1
            )
        
        # 4. Portfolio Weights
        for i, asset in enumerate(weights.columns):
            fig.add_trace(
                go.Scatter(x=weights.index, y=weights[asset],
                          stackgroup='one', name=asset,
                          mode='none', fill='tonexty'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Performance Summary - {backtest_results.get('strategy_name', 'Strategy')}",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_strategy_comparison(self, comparison_results: Dict[str, Any]) -> go.Figure:
        """Create strategy comparison plot."""
        portfolio_values = comparison_results['portfolio_values']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, strategy in enumerate(portfolio_values.columns):
            fig.add_trace(
                go.Scatter(x=portfolio_values.index, y=portfolio_values[strategy],
                          name=strategy, line=dict(color=colors[i % len(colors)], width=2))
            )
        
        fig.update_layout(
            title="Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Normalized)",
            height=400
        )
        
        return fig
    
    def plot_regime_performance_comparison(self, 
                                         backtest_results: Dict[str, Any],
                                         figsize: Tuple[int, int] = (12, 8)) -> go.Figure:
        """
        Create regime-based performance comparison charts.
        
        Parameters:
        -----------
        backtest_results : Dict
            Results from backtest framework
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        Plotly figure object
        """
        portfolio_returns = backtest_results['portfolio_returns']
        regimes = backtest_results['regimes']
        
        # Calculate regime-based performance
        regime_performance = {}
        for regime in regimes.dropna().unique():
            regime_mask = regimes == regime
            regime_returns = portfolio_returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'mean_return': regime_returns.mean() * 252,  # Annualized
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                    'max_drawdown': ((1 + regime_returns).cumprod() / (1 + regime_returns).cumprod().cummax() - 1).min(),
                    'win_rate': (regime_returns > 0).mean(),
                    'count': len(regime_returns)
                }
        
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annualized Returns by Regime', 'Risk-Return Profile',
                          'Win Rate by Regime', 'Regime Duration'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        regimes_list = list(regime_performance.keys())
        colors = px.colors.qualitative.Set1[:len(regimes_list)]
        
        # 1. Annualized Returns
        returns_data = [regime_performance[r]['mean_return'] for r in regimes_list]
        fig.add_trace(
            go.Bar(x=regimes_list, y=returns_data, name='Annualized Return',
                   marker_color=colors),
            row=1, col=1
        )
        
        # 2. Risk-Return Profile
        vol_data = [regime_performance[r]['volatility'] for r in regimes_list]
        fig.add_trace(
            go.Scatter(x=vol_data, y=returns_data, mode='markers+text',
                      text=regimes_list, textposition="top center",
                      marker=dict(size=15, color=colors),
                      name='Risk-Return'),
            row=1, col=2
        )
        
        # 3. Win Rate
        win_rates = [regime_performance[r]['win_rate'] for r in regimes_list]
        fig.add_trace(
            go.Bar(x=regimes_list, y=win_rates, name='Win Rate',
                   marker_color=colors),
            row=2, col=1
        )
        
        # 4. Regime Duration
        durations = [regime_performance[r]['count'] for r in regimes_list]
        fig.add_trace(
            go.Bar(x=regimes_list, y=durations, name='Days in Regime',
                   marker_color=colors),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Regime-Based Performance Analysis",
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Return", row=1, col=1)
        fig.update_xaxes(title_text="Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Return", row=1, col=2)
        fig.update_xaxes(title_text="Regime", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", row=2, col=1)
        fig.update_xaxes(title_text="Regime", row=2, col=2)
        fig.update_yaxes(title_text="Days", row=2, col=2)
        
        return fig
    
    def plot_drawdown_analysis(self, 
                              backtest_results: Dict[str, Any],
                              top_n: int = 5,
                              figsize: Tuple[int, int] = (14, 8)) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Create detailed drawdown analysis with regime overlay.
        
        Parameters:
        -----------
        backtest_results : Dict
            Results from backtest framework
        top_n : int
            Number of top drawdowns to highlight
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        Tuple of (Plotly figure, DataFrame with drawdown details)
        """
        portfolio_returns = backtest_results['portfolio_returns']
        regimes = backtest_results['regimes']
        
        # Calculate drawdowns
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        
        # Find drawdown periods
        is_drawdown = drawdown < -0.001  # 0.1% threshold
        drawdown_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Get start and end dates
        start_dates = portfolio_returns.index[drawdown_start]
        end_dates = portfolio_returns.index[drawdown_end]
        
        # If we're still in a drawdown, add the last date
        if len(start_dates) > len(end_dates):
            end_dates = end_dates.append(pd.DatetimeIndex([portfolio_returns.index[-1]]))
        
        # Calculate drawdown info
        drawdown_info = []
        for i in range(min(len(start_dates), len(end_dates))):
            start_date = start_dates[i]
            end_date = end_dates[i]
            
            # Get min drawdown in this period
            period_drawdown = drawdown.loc[start_date:end_date]
            max_dd = period_drawdown.min()
            max_dd_date = period_drawdown.idxmin()
            
            # Calculate recovery
            if end_date != portfolio_returns.index[-1]:
                recovery = (end_date - max_dd_date).days
            else:
                recovery = np.nan
            
            # Get duration
            duration = (end_date - start_date).days
            
            # Add to list
            drawdown_info.append({
                'start_date': start_date,
                'end_date': end_date,
                'max_drawdown': max_dd,
                'max_drawdown_date': max_dd_date,
                'duration': duration,
                'recovery': recovery
            })
        
        # Convert to DataFrame and sort
        drawdown_df = pd.DataFrame(drawdown_info)
        if len(drawdown_df) > 0:
            drawdown_df = drawdown_df.sort_values('max_drawdown').head(top_n)
        
        # Create plot
        fig = go.Figure()
        
        # Plot drawdown line
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown * 100,
                      fill='tonexty', name='Drawdown (%)',
                      line=dict(color='red', width=2),
                      fillcolor='rgba(255,0,0,0.3)')
        )
        
        # Highlight top drawdowns
        for _, row in drawdown_df.iterrows():
            fig.add_vrect(
                x0=row['start_date'], x1=row['end_date'],
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0
            )
        
        # Add regime background
        unique_regimes = regimes.dropna().unique()
        regime_colors = {'Bull': 'rgba(0,255,0,0.1)', 'Bear': 'rgba(255,0,0,0.1)', 
                        'Neutral': 'rgba(0,0,255,0.1)', 'Unknown': 'rgba(128,128,128,0.1)'}
        
        for regime in unique_regimes:
            regime_periods = regimes[regimes == regime]
            if len(regime_periods) > 0:
                # Group consecutive periods
                regime_starts = []
                regime_ends = []
                current_start = None
                
                for i, date in enumerate(regime_periods.index):
                    if current_start is None:
                        current_start = date
                    elif i == len(regime_periods) - 1 or regime_periods.index[i+1] != date + pd.Timedelta(days=1):
                        regime_ends.append(date)
                        regime_starts.append(current_start)
                        current_start = None
                
                # Add rectangles for regime periods
                for start, end in zip(regime_starts, regime_ends):
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=regime_colors.get(regime, 'rgba(128,128,128,0.1)'),
                        opacity=0.3, layer="below", line_width=0
                    )
        
        # Update layout
        fig.update_layout(
            title="Portfolio Drawdown Analysis with Regime Overlay",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            showlegend=True
        )
        
        return fig, drawdown_df
    
    def plot_monte_carlo_results(self, 
                                mc_results: Dict[str, Any],
                                figsize: Tuple[int, int] = (14, 8)) -> go.Figure:
        """
        Visualize Monte Carlo simulation results.
        
        Parameters:
        -----------
        mc_results : Dict
            Results from monte_carlo_simulation method
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        Plotly figure object
        """
        simulated_paths = mc_results['simulated_paths']
        final_returns = mc_results['final_returns']
        simulation_stats = mc_results['simulation_stats']
        original_returns = mc_results['original_returns']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Simulated Paths', 'Final Return Distribution',
                          'Confidence Intervals', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # 1. Simulated Paths (show subset to avoid overcrowding)
        n_paths_to_show = min(100, len(simulated_paths))
        indices = np.random.choice(len(simulated_paths), n_paths_to_show, replace=False)
        
        for i in indices:
            fig.add_trace(
                go.Scatter(x=list(range(len(simulated_paths[i]))), y=simulated_paths[i],
                          line=dict(color='lightblue', width=0.5),
                          showlegend=False, opacity=0.3),
                row=1, col=1
            )
        
        # Add original path
        original_cum = (1 + original_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=list(range(len(original_cum))), y=original_cum,
                      line=dict(color='red', width=3),
                      name='Original Path'),
            row=1, col=1
        )
        
        # 2. Final Return Distribution
        fig.add_trace(
            go.Histogram(x=final_returns * 100, nbinsx=50,
                        name='Final Returns (%)', opacity=0.7),
            row=1, col=2
        )
        
        # Add confidence intervals as vertical lines
        for level, value in simulation_stats['confidence_intervals'].items():
            fig.add_vline(x=value * 100, line_dash="dash",
                         annotation_text=f"{level}: {value:.2%}",
                         row=1, col=2)
        
        # 3. Confidence Intervals
        confidence_levels = list(simulation_stats['confidence_intervals'].keys())
        confidence_values = [v * 100 for v in simulation_stats['confidence_intervals'].values()]
        
        fig.add_trace(
            go.Bar(x=confidence_levels, y=confidence_values,
                   name='Confidence Intervals',
                   marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Risk Metrics Table
        risk_metrics = {
            'Mean Final Return': f"{simulation_stats['mean_final_return']:.2%}",
            'Std Final Return': f"{simulation_stats['std_final_return']:.2%}",
            'Probability of Loss': f"{simulation_stats['probability_of_loss']:.2%}",
            'Expected Shortfall (5%)': f"{simulation_stats['expected_shortfall_5%']:.2%}",
            'Min Return': f"{simulation_stats['min_final_return']:.2%}",
            'Max Return': f"{simulation_stats['max_final_return']:.2%}"
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Risk Metric', 'Value'], fill_color='lightcoral'),
                cells=dict(values=[list(risk_metrics.keys()), list(risk_metrics.values())],
                          fill_color='white')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Monte Carlo Simulation Results ({mc_results['n_simulations']} simulations)",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   backtest_results: Dict[str, Any],
                                   comparison_results: Dict[str, Any] = None) -> go.Figure:
        """
        Create an interactive dashboard combining all visualizations.
        
        Parameters:
        -----------
        backtest_results : Dict
            Results from backtest framework
        comparison_results : Dict, optional
            Results from strategy comparison
            
        Returns:
        --------
        Plotly figure object with interactive dashboard
        """
        # This would create a comprehensive dashboard
        # For now, return the performance summary
        return self.plot_performance_summary(backtest_results) 