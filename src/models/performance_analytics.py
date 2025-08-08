"""
Performance Analytics Module

This module provides comprehensive performance analysis capabilities for portfolio
management, including regime-based attribution, drawdown analysis, and benchmark
comparison functionality.

Author: AI Assistant
Date: 2025-06-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Comprehensive performance analytics for portfolio management.
    
    This class provides methods for calculating performance metrics, analyzing
    drawdowns, performing regime-based attribution, and creating visualizations.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, annualization_factor: int = 12):
        """
        Initialize the PerformanceAnalytics module.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            annualization_factor: Days per year for annualization (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        logger.info("PerformanceAnalytics initialized")
    
    # ========================================================================
    # SUBTASK 10.1: Core Performance Metrics Calculation
    # ========================================================================
    
    def calculate_returns(self, prices: pd.DataFrame, weights: Union[pd.DataFrame, pd.Series, Dict], 
                         rebalance_frequency: str = 'M') -> pd.Series:
        """
        Calculate portfolio returns based on prices and weights.
        
        Args:
            prices: DataFrame of asset prices with datetime index
            weights: Portfolio weights (DataFrame, Series, or Dict)
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Series of portfolio returns
        """
        try:
            # Calculate asset returns
            asset_returns = prices.pct_change(fill_method=None).dropna()
            
            # Initialize portfolio returns
            portfolio_returns = pd.Series(index=asset_returns.index, dtype=float)
            
            # Get rebalance dates
            if rebalance_frequency == 'D':
                rebalance_dates = asset_returns.index
            elif rebalance_frequency == 'W':
                rebalance_dates = asset_returns.resample('W').last().index
            elif rebalance_frequency == 'M':
                rebalance_dates = asset_returns.resample('ME').last().index
            elif rebalance_frequency == 'Q':
                rebalance_dates = asset_returns.resample('QE').last().index
            elif rebalance_frequency == 'Y':
                rebalance_dates = asset_returns.resample('YE').last().index
            else:
                rebalance_dates = pd.DatetimeIndex([asset_returns.index[0]])
            
            # Ensure we have the last date
            if len(rebalance_dates) == 0 or rebalance_dates[-1] != asset_returns.index[-1]:
                if len(rebalance_dates) == 0:
                    rebalance_dates = pd.DatetimeIndex([asset_returns.index[0], asset_returns.index[-1]])
                else:
                    rebalance_dates = rebalance_dates.union(pd.DatetimeIndex([asset_returns.index[-1]]))
            
            # Handle case where we only have one rebalance date
            if len(rebalance_dates) == 1:
                rebalance_dates = pd.DatetimeIndex([asset_returns.index[0], asset_returns.index[-1]])
            
            # Calculate portfolio returns for each period
            for i in range(len(rebalance_dates) - 1):
                start_date = rebalance_dates[i]
                end_date = rebalance_dates[i + 1]
                
                # Get weights for this period
                if isinstance(weights, dict):
                    period_weights = weights.get(start_date, weights)
                elif isinstance(weights, pd.DataFrame):
                    if start_date in weights.index:
                        period_weights = weights.loc[start_date]
                    else:
                        # Use the most recent weights available
                        available_dates = weights.index[weights.index <= start_date]
                        if len(available_dates) > 0:
                            period_weights = weights.loc[available_dates[-1]]
                        else:
                            period_weights = weights.iloc[0]
                else:
                    period_weights = weights
                
                # Ensure weights align with asset returns columns
                if isinstance(period_weights, pd.Series):
                    period_weights = period_weights.reindex(asset_returns.columns, fill_value=0)
                elif isinstance(period_weights, dict):
                    period_weights = pd.Series(period_weights).reindex(asset_returns.columns, fill_value=0)
                
                # Get returns for this period
                period_returns = asset_returns.loc[start_date:end_date]
                
                # Calculate portfolio returns
                period_portfolio_returns = (period_returns * period_weights).sum(axis=1)
                
                # Add to portfolio returns
                portfolio_returns.loc[period_returns.index] = period_portfolio_returns
            
            return portfolio_returns.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    risk_free_rate: Optional[float] = None,
                                    annualization_factor: Optional[int] = None) -> Dict:
        """
        Calculate comprehensive performance metrics for a return series.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (uses instance default if None)
            annualization_factor: Annualization factor (uses instance default if None)
            
        Returns:
            Dictionary of performance metrics
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        if annualization_factor is None:
            annualization_factor = self.annualization_factor
        
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) == 0:
                return self._empty_metrics()
            
            # Basic metrics
            total_return = (1 + returns_clean).prod() - 1
            # Infer frequency: fallback to provided annualization_factor (default monthly=12)
            try:
                inferred = pd.infer_freq(returns_clean.index)
                if inferred is not None:
                    if inferred.startswith('D') or inferred.startswith('B'):
                        annualization_factor = 252
                    elif inferred.startswith('W'):
                        annualization_factor = 52
                    elif inferred.endswith('M') or inferred in ('MS', 'M', 'ME'):
                        annualization_factor = 12
                    elif inferred.startswith('Q'):
                        annualization_factor = 4
                    elif inferred.startswith('A') or inferred.startswith('Y'):
                        annualization_factor = 1
            except Exception:
                pass
            annualized_return = (1 + total_return) ** (annualization_factor / len(returns_clean)) - 1
            volatility = returns_clean.std() * np.sqrt(annualization_factor)
            
            # Risk-adjusted metrics
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            cum_returns = (1 + returns_clean).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
            
            # Additional metrics
            win_rate = (returns_clean > 0).mean()
            loss_rate = (returns_clean < 0).mean()
            avg_win = returns_clean[returns_clean > 0].mean() if (returns_clean > 0).any() else 0
            avg_loss = returns_clean[returns_clean < 0].mean() if (returns_clean < 0).any() else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            
            # Skewness and Kurtosis
            skewness = returns_clean.skew()
            kurtosis = returns_clean.kurtosis()
            
            # Value at Risk (VaR) and Expected Shortfall (ES)
            var_95 = returns_clean.quantile(0.05)
            var_99 = returns_clean.quantile(0.01)
            es_95 = returns_clean[returns_clean <= var_95].mean() if (returns_clean <= var_95).any() else var_95
            es_99 = returns_clean[returns_clean <= var_99].mean() if (returns_clean <= var_99).any() else var_99
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_clean[returns_clean < risk_free_rate/annualization_factor]
            downside_deviation = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else np.inf
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'observations': len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary for error cases."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'es_95': 0.0,
            'es_99': 0.0,
            'observations': 0
        }
    
    # ========================================================================
    # SUBTASK 10.2: Drawdown Analysis Framework
    # ========================================================================
    
    def analyze_drawdowns(self, returns: pd.Series, top_n: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Comprehensive drawdown analysis.
        
        Args:
            returns: Series of returns
            top_n: Number of top drawdowns to analyze
            
        Returns:
            Tuple of (drawdown_periods_df, drawdown_series)
        """
        try:
            returns_clean = returns.dropna()
            
            # Calculate drawdowns
            cum_returns = (1 + returns_clean).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            
            # Find drawdown periods
            is_drawdown = drawdown < 0
            drawdown_start = is_drawdown & ~is_drawdown.shift(1).fillna(False).infer_objects(copy=False)
            drawdown_end = ~is_drawdown & is_drawdown.shift(1).fillna(False).infer_objects(copy=False)
            
            # Get start and end dates
            start_dates = returns_clean.index[drawdown_start]
            end_dates = returns_clean.index[drawdown_end]
            
            # Handle case where we're still in a drawdown
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.DatetimeIndex([returns_clean.index[-1]]))
            
            # Calculate drawdown statistics
            drawdown_periods = []
            for i in range(len(start_dates)):
                start_date = start_dates[i]
                end_date = end_dates[i] if i < len(end_dates) else returns_clean.index[-1]
                
                # Get drawdown for this period
                period_drawdown = drawdown.loc[start_date:end_date]
                max_dd = period_drawdown.min()
                max_dd_date = period_drawdown.idxmin()
                
                # Calculate duration and recovery
                duration = (max_dd_date - start_date).days
                if end_date != returns_clean.index[-1]:
                    recovery = (end_date - max_dd_date).days
                    total_duration = (end_date - start_date).days
                else:
                    recovery = np.nan
                    total_duration = (returns_clean.index[-1] - start_date).days
                
                # Calculate recovery factor (how much of the drawdown was recovered)
                if end_date != returns_clean.index[-1]:
                    recovery_factor = 1.0  # Full recovery
                else:
                    current_dd = drawdown.iloc[-1]
                    recovery_factor = 1 - (current_dd / max_dd) if max_dd != 0 else 0
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'peak_date': max_dd_date,
                    'max_drawdown': max_dd,
                    'duration_to_trough': duration,
                    'recovery_days': recovery,
                    'total_duration': total_duration,
                    'recovery_factor': recovery_factor,
                    'is_recovered': end_date != returns_clean.index[-1]
                })
            
            # Convert to DataFrame and sort by severity
            drawdown_df = pd.DataFrame(drawdown_periods)
            if len(drawdown_df) > 0:
                drawdown_df = drawdown_df.sort_values('max_drawdown').head(top_n)
            
            return drawdown_df, drawdown
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def calculate_underwater_curve(self, returns: pd.Series) -> pd.Series:
        """
        Calculate the underwater curve (drawdown over time).
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of drawdown values over time
        """
        try:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            return (cum_returns / running_max) - 1
        except Exception as e:
            logger.error(f"Error calculating underwater curve: {str(e)}")
            return pd.Series()
    
    # ========================================================================
    # SUBTASK 10.3: Regime-Based Performance Attribution
    # ========================================================================
    
    def regime_performance_attribution(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """
        Analyze performance attribution by market regime.
        
        Args:
            returns: Series of portfolio returns
            regimes: Series of regime classifications
            
        Returns:
            Dictionary of regime performance statistics
        """
        try:
            # Align returns and regimes
            common_index = returns.index.intersection(regimes.index)
            returns_aligned = returns.loc[common_index]
            regimes_aligned = regimes.loc[common_index]
            
            # Calculate overall performance
            total_metrics = self.calculate_performance_metrics(returns_aligned)
            
            # Calculate performance by regime
            regime_performance = {}
            unique_regimes = regimes_aligned.unique()
            
            for regime in unique_regimes:
                regime_mask = regimes_aligned == regime
                regime_returns = returns_aligned[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_metrics = self.calculate_performance_metrics(regime_returns)
                    
                    # Calculate regime statistics
                    regime_days = regime_mask.sum()
                    regime_weight = regime_days / len(regimes_aligned)
                    
                    # Calculate contribution to total return
                    regime_total_return = regime_metrics['total_return']
                    contribution = regime_total_return * regime_weight
                    contribution_pct = contribution / total_metrics['total_return'] if total_metrics['total_return'] != 0 else 0
                    
                    # Add regime-specific information
                    regime_metrics.update({
                        'regime': regime,
                        'days': regime_days,
                        'weight': regime_weight,
                        'contribution': contribution,
                        'contribution_pct': contribution_pct,
                        'frequency': regime_weight
                    })
                    
                    regime_performance[regime] = regime_metrics
            
            # Calculate regime transition effects
            regime_transitions = self._analyze_regime_transitions(returns_aligned, regimes_aligned)
            
            return {
                'overall_performance': total_metrics,
                'regime_performance': regime_performance,
                'regime_transitions': regime_transitions,
                'analysis_period': {
                    'start_date': returns_aligned.index[0],
                    'end_date': returns_aligned.index[-1],
                    'total_days': len(returns_aligned)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in regime performance attribution: {str(e)}")
            return {}

    def build_regime_scorecard(self, returns_df: pd.DataFrame, regime_series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compute per-regime metrics across multiple label sets and save CSV.

        For each regime label set name and each asset column in returns_df, compute:
          - per-regime mean, volatility, Sharpe (monthly stats)
          - max drawdown of the asset's cumulative return within each regime slice
          - ANOVA p-value testing if monthly returns differ across regimes
        """
        try:
            from scipy import stats  # optional, but common in environments
        except Exception:
            stats = None

        results: List[Dict] = []
        for label_name, regimes in regime_series_dict.items():
            if regimes is None or regimes.empty:
                continue
            # Align on index
            idx = returns_df.index.intersection(regimes.index)
            if len(idx) == 0:
                continue
            rets = returns_df.loc[idx]
            reg = regimes.loc[idx]
            for asset in rets.columns:
                r = rets[asset].dropna()
                common = r.index.intersection(reg.index)
                r = r.loc[common]
                lab = reg.loc[common]
                if r.empty:
                    continue
                # ANOVA across regimes (if scipy available and >=2 groups)
                anova_p = np.nan
                if stats is not None:
                    groups = [r[lab == g] for g in lab.unique()]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        try:
                            f, p = stats.f_oneway(*groups)
                            anova_p = float(p)
                        except Exception:
                            anova_p = np.nan
                # Per-regime metrics
                for g in lab.unique():
                    mask = lab == g
                    rg = r[mask]
                    if rg.empty:
                        continue
                    mean = float(rg.mean())
                    vol = float(rg.std())
                    sharpe = mean / vol if vol > 0 else 0.0
                    # max drawdown within slice
                    cum = (1 + rg).cumprod()
                    mdd = float(((cum / cum.cummax()) - 1).min()) if len(cum) > 0 else 0.0
                    results.append({
                        "label_set": label_name,
                        "asset": asset,
                        "regime": g,
                        "mean": mean,
                        "vol": vol,
                        "sharpe": sharpe,
                        "max_drawdown": mdd,
                        "anova_p": anova_p,
                        "observations": int(len(rg)),
                    })

        scorecard = pd.DataFrame(results)
        try:
            import os
            out_dir = os.path.join("Output", "diagnostics")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "regime_scorecard.csv")
            scorecard.to_csv(out_path, index=False)
            logger.info("Saved regime scorecard to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to save regime scorecard: %s", exc)
        return scorecard
    
    def _analyze_regime_transitions(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """Analyze performance around regime transitions."""
        try:
            # Find regime change points
            regime_changes = regimes != regimes.shift(1)
            transition_dates = returns.index[regime_changes]
            
            if len(transition_dates) == 0:
                return {'transition_count': 0, 'avg_transition_return': 0}
            
            # Analyze returns around transitions (±5 days)
            window = 5
            transition_returns = []
            
            for date in transition_dates[1:]:  # Skip first date
                try:
                    date_idx = returns.index.get_loc(date)
                    start_idx = max(0, date_idx - window)
                    end_idx = min(len(returns), date_idx + window + 1)
                    
                    period_returns = returns.iloc[start_idx:end_idx]
                    transition_returns.extend(period_returns.values)
                except:
                    continue
            
            avg_transition_return = np.mean(transition_returns) if transition_returns else 0
            transition_volatility = np.std(transition_returns) if len(transition_returns) > 1 else 0
            
            return {
                'transition_count': len(transition_dates) - 1,
                'avg_transition_return': avg_transition_return,
                'transition_volatility': transition_volatility,
                'transition_dates': transition_dates.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {str(e)}")
            return {'transition_count': 0, 'avg_transition_return': 0}
    
    # ========================================================================
    # SUBTASK 10.4: Benchmark Comparison Methods
    # ========================================================================
    
    def compare_to_benchmark(self, portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series,
                           benchmark_name: str = "Benchmark") -> Dict:
        """
        Compare portfolio performance to benchmark.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary of comparison metrics
        """
        try:
            # Align series
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            port_aligned = portfolio_returns.loc[common_index]
            bench_aligned = benchmark_returns.loc[common_index]
            
            # Calculate individual metrics
            port_metrics = self.calculate_performance_metrics(port_aligned)
            bench_metrics = self.calculate_performance_metrics(bench_aligned)
            
            # Calculate relative metrics
            excess_returns = port_aligned - bench_aligned
            excess_metrics = self.calculate_performance_metrics(excess_returns)
            
            # Information ratio
            tracking_error = excess_returns.std() * np.sqrt(self.annualization_factor)
            information_ratio = excess_metrics['annualized_return'] / tracking_error if tracking_error > 0 else 0
            
            # Beta and Alpha
            covariance = np.cov(port_aligned, bench_aligned)[0, 1]
            benchmark_variance = bench_aligned.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            alpha = port_metrics['annualized_return'] - (self.risk_free_rate + beta * (bench_metrics['annualized_return'] - self.risk_free_rate))
            
            # Up/Down capture ratios
            up_market = bench_aligned > 0
            down_market = bench_aligned <= 0
            
            up_capture = (port_aligned[up_market].mean() / bench_aligned[up_market].mean()) if up_market.any() and bench_aligned[up_market].mean() != 0 else 1
            down_capture = (port_aligned[down_market].mean() / bench_aligned[down_market].mean()) if down_market.any() and bench_aligned[down_market].mean() != 0 else 1
            
            # Win rate against benchmark
            outperformance_rate = (excess_returns > 0).mean()
            
            return {
                'portfolio_metrics': port_metrics,
                'benchmark_metrics': bench_metrics,
                'benchmark_name': benchmark_name,
                'excess_return': excess_metrics['annualized_return'],
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'up_capture': up_capture,
                'down_capture': down_capture,
                'outperformance_rate': outperformance_rate,
                'correlation': port_aligned.corr(bench_aligned),
                'analysis_period': {
                    'start_date': common_index[0],
                    'end_date': common_index[-1],
                    'observations': len(common_index)
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmark: {str(e)}")
            return {}
    
    def multi_benchmark_comparison(self, portfolio_returns: pd.Series,
                                 benchmarks: Dict[str, pd.Series]) -> Dict:
        """
        Compare portfolio to multiple benchmarks.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmarks: Dictionary of benchmark name -> return series
            
        Returns:
            Dictionary of comparison results for each benchmark
        """
        try:
            results = {}
            for name, benchmark_returns in benchmarks.items():
                results[name] = self.compare_to_benchmark(
                    portfolio_returns, benchmark_returns, name
                )
            return results
        except Exception as e:
            logger.error(f"Error in multi-benchmark comparison: {str(e)}")
            return {}
    
    # ========================================================================
    # SUBTASK 10.5: Performance Visualization Functions
    # ========================================================================
    
    def plot_performance_summary(self, returns: pd.Series, regimes: Optional[pd.Series] = None,
                               title: str = "Portfolio Performance") -> go.Figure:
        """
        Create a comprehensive performance summary plot.
        
        Args:
            returns: Portfolio return series
            regimes: Optional regime series for background coloring
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate cumulative returns and drawdown
            cum_returns = (1 + returns).cumprod()
            drawdown = self.calculate_underwater_curve(returns)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cumulative Returns', 'Drawdown'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Plot cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Plot drawdown
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Add regime background if provided
            if regimes is not None:
                self._add_regime_background(fig, regimes, returns.index)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance summary plot: {str(e)}")
            return go.Figure()
    
    def plot_regime_performance_comparison(self, regime_attribution: Dict) -> go.Figure:
        """
        Plot performance comparison across regimes.
        
        Args:
            regime_attribution: Output from regime_performance_attribution
            
        Returns:
            Plotly figure object
        """
        try:
            regime_data = regime_attribution.get('regime_performance', {})
            
            if not regime_data:
                return go.Figure()
            
            # Extract data for plotting
            regimes = list(regime_data.keys())
            returns = [regime_data[r]['annualized_return'] for r in regimes]
            volatilities = [regime_data[r]['volatility'] for r in regimes]
            sharpe_ratios = [regime_data[r]['sharpe_ratio'] for r in regimes]
            max_drawdowns = [abs(regime_data[r]['max_drawdown']) for r in regimes]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Annualized Returns', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Returns
            fig.add_trace(
                go.Bar(x=regimes, y=returns, name='Returns', marker_color='blue'),
                row=1, col=1
            )
            
            # Volatility
            fig.add_trace(
                go.Bar(x=regimes, y=volatilities, name='Volatility', marker_color='orange'),
                row=1, col=2
            )
            
            # Sharpe Ratio
            fig.add_trace(
                go.Bar(x=regimes, y=sharpe_ratios, name='Sharpe', marker_color='green'),
                row=2, col=1
            )
            
            # Max Drawdown
            fig.add_trace(
                go.Bar(x=regimes, y=max_drawdowns, name='Max DD', marker_color='red'),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Performance Metrics by Regime",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regime performance plot: {str(e)}")
            return go.Figure()
    
    def plot_drawdown_analysis(self, returns: pd.Series, regimes: Optional[pd.Series] = None,
                             top_n: int = 5) -> go.Figure:
        """
        Plot detailed drawdown analysis.
        
        Args:
            returns: Portfolio return series
            regimes: Optional regime series
            top_n: Number of top drawdowns to highlight
            
        Returns:
            Plotly figure object
        """
        try:
            drawdown_df, drawdown_series = self.analyze_drawdowns(returns, top_n)
            
            fig = go.Figure()
            
            # Plot drawdown line
            fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=2)
                )
            )
            
            # Highlight top drawdowns
            if not drawdown_df.empty:
                for _, row in drawdown_df.iterrows():
                    fig.add_vrect(
                        x0=row['start_date'],
                        x1=row['end_date'],
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        layer="below",
                        line_width=0
                    )
            
            # Add regime background if provided
            if regimes is not None:
                self._add_regime_background(fig, regimes, returns.index)
            
            fig.update_layout(
                title="Drawdown Analysis with Top Drawdown Periods Highlighted",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=400,
                hovermode='x'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown analysis plot: {str(e)}")
            return go.Figure()
    
    def plot_benchmark_comparison(self, comparison_results: Dict) -> go.Figure:
        """
        Plot benchmark comparison results.
        
        Args:
            comparison_results: Output from compare_to_benchmark
            
        Returns:
            Plotly figure object
        """
        try:
            port_metrics = comparison_results.get('portfolio_metrics', {})
            bench_metrics = comparison_results.get('benchmark_metrics', {})
            benchmark_name = comparison_results.get('benchmark_name', 'Benchmark')
            
            # Metrics to compare
            metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
            metric_names = ['Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
            
            portfolio_values = [port_metrics.get(m, 0) for m in metrics]
            benchmark_values = [bench_metrics.get(m, 0) for m in metrics]
            
            # Make max drawdown positive for visualization
            portfolio_values[3] = abs(portfolio_values[3])
            benchmark_values[3] = abs(benchmark_values[3])
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=portfolio_values,
                    name='Portfolio',
                    marker_color='blue'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=benchmark_values,
                    name=benchmark_name,
                    marker_color='orange'
                )
            )
            
            fig.update_layout(
                title=f"Portfolio vs {benchmark_name} Comparison",
                xaxis_title="Metrics",
                yaxis_title="Value",
                barmode='group',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating benchmark comparison plot: {str(e)}")
            return go.Figure()
    
    def _add_regime_background(self, fig: go.Figure, regimes: pd.Series, 
                             date_index: pd.DatetimeIndex):
        """Add regime background coloring to a plot."""
        try:
            unique_regimes = regimes.unique()
            colors = px.colors.qualitative.Set1[:len(unique_regimes)]
            regime_colors = {regime: colors[i % len(colors)] for i, regime in enumerate(unique_regimes)}
            
            # Find regime periods
            regime_changes = regimes != regimes.shift(1)
            change_points = date_index[regimes.index.isin(date_index)][regime_changes[regimes.index.isin(date_index)]]
            
            if len(change_points) == 0:
                return
            
            # Add colored backgrounds
            current_regime = regimes.iloc[0]
            start_date = date_index[0]
            
            for change_date in change_points[1:]:
                fig.add_vrect(
                    x0=start_date,
                    x1=change_date,
                    fillcolor=regime_colors[current_regime],
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
                start_date = change_date
                current_regime = regimes.loc[change_date] if change_date in regimes.index else current_regime
            
            # Add final period
            fig.add_vrect(
                x0=start_date,
                x1=date_index[-1],
                fillcolor=regime_colors[current_regime],
                opacity=0.2,
                layer="below",
                line_width=0
            )
            
        except Exception as e:
            logger.warning(f"Could not add regime background: {str(e)}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def generate_performance_report(self, returns: pd.Series, 
                                  regimes: Optional[pd.Series] = None,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  benchmark_name: str = "Benchmark") -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            returns: Portfolio return series
            regimes: Optional regime series
            benchmark_returns: Optional benchmark return series
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.calculate_performance_metrics(returns)
            }
            
            # Add drawdown analysis
            drawdown_df, drawdown_series = self.analyze_drawdowns(returns)
            report['drawdown_analysis'] = {
                'drawdown_periods': drawdown_df.to_dict('records') if not drawdown_df.empty else [],
                'current_drawdown': drawdown_series.iloc[-1] if len(drawdown_series) > 0 else 0
            }
            
            # Add regime analysis if available
            if regimes is not None:
                report['regime_attribution'] = self.regime_performance_attribution(returns, regimes)
            
            # Add benchmark comparison if available
            if benchmark_returns is not None:
                report['benchmark_comparison'] = self.compare_to_benchmark(
                    returns, benchmark_returns, benchmark_name
                )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    def export_metrics_to_dataframe(self, metrics_dict: Dict) -> pd.DataFrame:
        """
        Export performance metrics to a pandas DataFrame.
        
        Args:
            metrics_dict: Dictionary of performance metrics
            
        Returns:
            DataFrame with metrics
        """
        try:
            # Flatten nested dictionaries
            flattened = {}
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened[f"{key}_{subkey}"] = subvalue
                else:
                    flattened[key] = value
            
            return pd.DataFrame([flattened])
            
        except Exception as e:
            logger.error(f"Error exporting metrics to DataFrame: {str(e)}")
            return pd.DataFrame()


# ========================================================================
# Quick Analysis Functions
# ========================================================================

def quick_performance_analysis(returns: pd.Series, 
                             regimes: Optional[pd.Series] = None,
                             risk_free_rate: float = 0.02) -> Dict:
    """
    Quick performance analysis function for easy access.
    
    Args:
        returns: Portfolio return series
        regimes: Optional regime series
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of analysis results
    """
    analyzer = PerformanceAnalytics(risk_free_rate=risk_free_rate)
    
    results = {
        'metrics': analyzer.calculate_performance_metrics(returns),
        'drawdown_analysis': analyzer.analyze_drawdowns(returns)[0].to_dict('records')
    }
    
    if regimes is not None:
        results['regime_attribution'] = analyzer.regime_performance_attribution(returns, regimes)
    
    return results


def compare_portfolios(portfolio_returns: Dict[str, pd.Series],
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Compare multiple portfolios and optionally a benchmark.
    
    Args:
        portfolio_returns: Dictionary of portfolio name -> return series
        benchmark_returns: Optional benchmark return series
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame comparing all portfolios
    """
    analyzer = PerformanceAnalytics(risk_free_rate=risk_free_rate)
    
    results = []
    for name, returns in portfolio_returns.items():
        metrics = analyzer.calculate_performance_metrics(returns)
        metrics['portfolio'] = name
        results.append(metrics)
    
    if benchmark_returns is not None:
        bench_metrics = analyzer.calculate_performance_metrics(benchmark_returns)
        bench_metrics['portfolio'] = 'Benchmark'
        results.append(bench_metrics)
    
    return pd.DataFrame(results).set_index('portfolio')


if __name__ == "__main__":
    # Example usage and testing
    print("Performance Analytics Module Loaded Successfully!")
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
    
    # Test the module
    analyzer = PerformanceAnalytics()
    metrics = analyzer.calculate_performance_metrics(returns)
    
    print("\nSample Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}") 