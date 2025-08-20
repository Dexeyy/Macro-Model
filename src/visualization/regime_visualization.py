"""
Comprehensive Regime Visualization Framework

This module implements a complete visualization framework for regime analysis,
portfolio performance, and interactive dashboard integration as specified in Task 6.

Features:
- Regime timeline visualizations with color-coded regimes
- Transition matrix heatmaps 
- Portfolio performance comparisons
- Interactive plotly visualizations
- Theme support and customization
- Dashboard integration layer

Author: Macro Regime Analysis System
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings

# Import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    warnings.warn("Plotly not available. Interactive visualizations will be disabled.")
    PLOTLY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeVisualization:
    """
    Comprehensive visualization framework for regime analysis and portfolio performance.
    
    This class provides various visualization methods including timeline plots,
    transition analysis, portfolio comparisons, and interactive dashboard components.
    Implements the exact specifications from Task 6.
    """

    def __init__(self, theme: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the regime visualization framework.
        
        Args:
            theme: Visualization theme ('default', 'dark', 'minimal')
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.theme = theme
        self.set_theme(theme)
        
        # Define color palettes for different regimes
        self.regime_colors = {
            'expansion': '#2E8B57',      # Sea Green
            'recession': '#DC143C',       # Crimson  
            'recovery': '#4169E1',        # Royal Blue
            'stagflation': '#FF8C00',     # Dark Orange
            'neutral': '#708090',         # Slate Gray
            'high_growth': '#32CD32',     # Lime Green
            'low_growth': '#FFD700',      # Gold
            'crisis': '#8B0000',          # Dark Red
            'stable': '#4682B4',          # Steel Blue
            'volatile': '#FF6347'         # Tomato
        }
        
        logger.info(f"RegimeVisualization initialized with theme: {theme}")

    def set_theme(self, theme: str):
        """
        Set visualization theme for consistent styling.
        
        Args:
            theme: Theme name ('default', 'dark', 'minimal')
        """
        try:
            if theme == 'default':
                plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("colorblind")
            elif theme == 'dark':
                plt.style.use('dark_background')
                sns.set_palette("bright")
            elif theme == 'minimal':
                plt.style.use('seaborn-v0_8-white')
                sns.set_palette("muted")
            else:
                logger.warning(f"Unknown theme '{theme}', using default")
                plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("colorblind")
                
            # Set common plot parameters
            plt.rcParams['figure.figsize'] = self.figsize
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['font.size'] = 10
            
            self.theme = theme
            logger.info(f"Theme set to: {theme}")
            
        except Exception as e:
            logger.error(f"Error setting theme: {e}")
            # Fallback to basic style
            plt.rcParams['figure.figsize'] = self.figsize

    def plot_regime_timeline(self, regimes: pd.Series, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot regime timeline with color-coded regimes.
        
        Args:
            regimes: Series with regime classifications and datetime index
            figsize: Figure size override
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info("Creating regime timeline visualization...")
            
            # Create a numeric mapping for regimes
            unique_regimes = regimes.unique()
            regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
            numeric_regimes = regimes.map(regime_map)
            
            # Create color map
            colors = sns.color_palette("Set1", len(unique_regimes))
            regime_colors = {regime: colors[i] for regime, i in regime_map.items()}
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot timeline using fill_between for better visual effect
            for regime in unique_regimes:
                mask = regimes == regime
                regime_periods = regimes[mask]
                
                # Get color for this regime
                color = self.regime_colors.get(regime, regime_colors[regime])
                
                # Plot filled areas for each regime period
                for date in regime_periods.index:
                    ax.fill_between([date, date], 0, 1, 
                                  color=color, alpha=0.7, label=regime if date == regime_periods.index[0] else "")
            
            # Format plot
            ax.set_yticks([])
            ax.set_ylim(0, 1)
            ax.set_title('Market Regime Timeline', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            
            # Add legend with unique entries only
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                     ncol=len(unique_regimes))
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            logger.info("Regime timeline visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regime timeline: {e}")
            raise

    def plot_regime_transitions(self, regimes: pd.Series) -> plt.Figure:
        """
        Plot regime transition matrix as a heatmap.
        
        Args:
            regimes: Series with regime classifications
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info("Creating regime transition matrix visualization...")
            
            # Create transition matrix
            transitions = pd.crosstab(
                regimes.shift(1),
                regimes,
                normalize='index'
            )
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(transitions, 
                       annot=True, 
                       cmap='YlGnBu', 
                       fmt='.2f',
                       ax=ax,
                       cbar_kws={'label': 'Transition Probability'})
            
            ax.set_title('Regime Transition Probabilities', fontsize=14, fontweight='bold')
            ax.set_xlabel('To Regime', fontsize=12)
            ax.set_ylabel('From Regime', fontsize=12)
            
            plt.tight_layout()
            
            logger.info("Regime transition matrix created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating transition matrix: {e}")
            raise

    def plot_regime_performance(self, returns: pd.DataFrame, regimes: pd.Series) -> plt.Figure:
        """
        Plot asset performance by regime.
        
        Args:
            returns: DataFrame with asset returns
            regimes: Series with regime classifications
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info("Creating regime performance visualization...")
            
            # Align returns and regimes
            common_index = returns.index.intersection(regimes.index)
            aligned_returns = returns.loc[common_index]
            aligned_regimes = regimes.loc[common_index]
            
            # Calculate performance by regime
            performance = {}
            for regime in aligned_regimes.unique():
                regime_mask = aligned_regimes == regime
                regime_returns = aligned_returns[regime_mask]
                # Annualized mean returns
                performance[regime] = regime_returns.mean() * 252
            
            # Convert to DataFrame
            performance_df = pd.DataFrame(performance)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot grouped bar chart
            performance_df.plot(kind='bar', ax=ax, alpha=0.8)
            
            ax.set_title('Annualized Asset Performance by Regime', fontsize=14, fontweight='bold')
            ax.set_xlabel('Assets', fontsize=12)
            ax.set_ylabel('Annualized Return (%)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)
            
            # Format legend
            ax.legend(title='Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            logger.info("Regime performance visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regime performance plot: {e}")
            raise

    def plot_portfolio_weights(self, portfolio_weights: pd.Series, title: str = 'Portfolio Weights') -> plt.Figure:
        """
        Plot portfolio weights as a pie chart.
        
        Args:
            portfolio_weights: Series with asset weights
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info(f"Creating portfolio weights visualization: {title}")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                portfolio_weights.values,
                labels=portfolio_weights.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("Set3", len(portfolio_weights))
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            
            logger.info("Portfolio weights visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating portfolio weights plot: {e}")
            raise

    def plot_portfolio_comparison(self, regime_portfolios: Dict[str, Dict]) -> plt.Figure:
        """
        Compare portfolio allocations across regimes.
        
        Args:
            regime_portfolios: Dictionary with regime portfolios containing 'weights'
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info("Creating portfolio comparison visualization...")
            
            # Combine weights from all regime portfolios
            weights_data = {}
            for regime, portfolio in regime_portfolios.items():
                if 'weights' in portfolio:
                    weights_data[regime] = portfolio['weights']
                else:
                    logger.warning(f"No weights found for regime: {regime}")
            
            if not weights_data:
                raise ValueError("No portfolio weights found in regime_portfolios")
            
            weights_df = pd.DataFrame(weights_data)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create grouped bar chart
            weights_df.plot(kind='bar', ax=ax, alpha=0.8)
            
            ax.set_title('Portfolio Weights by Regime', fontsize=14, fontweight='bold')
            ax.set_xlabel('Assets', fontsize=12)
            ax.set_ylabel('Weight (%)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Format legend
            ax.legend(title='Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Portfolio comparison visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating portfolio comparison plot: {e}")
            raise

    # Interactive Plotly Visualizations (Subtask 6.4)
    def create_interactive_regime_timeline(self, regimes: pd.Series, title: str = 'Interactive Regime Timeline'):
        """
        Create interactive regime timeline using Plotly.
        
        Args:
            regimes: Series with regime classifications
            title: Plot title
            
        Returns:
            plotly Figure object or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive timeline.")
            return None
            
        try:
            logger.info("Creating interactive regime timeline...")
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add traces for each regime
            for regime in regimes.unique():
                regime_data = regimes[regimes == regime]
                color = self.regime_colors.get(regime, '#808080')
                
                fig.add_trace(go.Scatter(
                    x=regime_data.index,
                    y=[regime] * len(regime_data),
                    mode='markers',
                    name=regime,
                    marker=dict(color=color, size=8),
                    hovertemplate=f'<b>{regime}</b><br>Date: %{{x}}<br><extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Regime',
                hovermode='closest',
                height=400
            )
            
            logger.info("Interactive regime timeline created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive timeline: {e}")
            return None

    def create_interactive_performance_chart(self, returns: pd.DataFrame, regimes: pd.Series, 
                                           title: str = 'Interactive Performance by Regime'):
        """
        Create interactive performance comparison using Plotly.
        
        Args:
            returns: DataFrame with asset returns
            regimes: Series with regime classifications
            title: Plot title
            
        Returns:
            plotly Figure object or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive performance chart.")
            return None
            
        try:
            logger.info("Creating interactive performance chart...")
            
            # Align data
            common_index = returns.index.intersection(regimes.index)
            aligned_returns = returns.loc[common_index]
            aligned_regimes = regimes.loc[common_index]
            
            # Calculate performance by regime
            performance_data = []
            for regime in aligned_regimes.unique():
                regime_mask = aligned_regimes == regime
                regime_returns = aligned_returns[regime_mask]
                annualized_returns = regime_returns.mean() * 252
                
                for asset in annualized_returns.index:
                    performance_data.append({
                        'Asset': asset,
                        'Regime': regime,
                        'Annualized_Return': annualized_returns[asset] * 100
                    })
            
            performance_df = pd.DataFrame(performance_data)
            
            # Create interactive bar chart
            fig = px.bar(
                performance_df,
                x='Asset',
                y='Annualized_Return',
                color='Regime',
                title=title,
                labels={'Annualized_Return': 'Annualized Return (%)'},
                barmode='group'
            )
            
            fig.update_layout(height=500)
            
            logger.info("Interactive performance chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive performance chart: {e}")
            return None

    def create_interactive_portfolio_comparison(self, regime_portfolios: Dict[str, Dict],
                                              title: str = 'Interactive Portfolio Comparison'):
        """
        Create interactive portfolio comparison using Plotly.
        
        Args:
            regime_portfolios: Dictionary with regime portfolios
            title: Plot title
            
        Returns:
            plotly Figure object or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive portfolio comparison.")
            return None
            
        try:
            logger.info("Creating interactive portfolio comparison...")
            
            # Prepare data
            portfolio_data = []
            for regime, portfolio in regime_portfolios.items():
                if 'weights' in portfolio:
                    weights = portfolio['weights']
                    for asset, weight in weights.items():
                        portfolio_data.append({
                            'Asset': asset,
                            'Regime': regime,
                            'Weight': weight * 100
                        })
            
            if not portfolio_data:
                raise ValueError("No portfolio data found")
            
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Create interactive bar chart
            fig = px.bar(
                portfolio_df,
                x='Asset',
                y='Weight',
                color='Regime',
                title=title,
                labels={'Weight': 'Weight (%)'},
                barmode='group'
            )
            
            fig.update_layout(height=500)
            
            logger.info("Interactive portfolio comparison created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive portfolio comparison: {e}")
            return None

    def create_dashboard_layout(self, regimes: pd.Series, returns: pd.DataFrame, 
                              regime_portfolios: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create a complete dashboard layout with all visualizations.
        
        Args:
            regimes: Series with regime classifications
            returns: DataFrame with asset returns
            regime_portfolios: Dictionary with regime portfolios
            
        Returns:
            Dictionary containing all visualization components
        """
        try:
            logger.info("Creating complete dashboard layout...")
            
            dashboard_components = {}
            
            # Static matplotlib visualizations
            dashboard_components['static'] = {
                'regime_timeline': self.plot_regime_timeline(regimes),
                'transition_matrix': self.plot_regime_transitions(regimes),
                'performance_chart': self.plot_regime_performance(returns, regimes),
                'portfolio_comparison': self.plot_portfolio_comparison(regime_portfolios)
            }
            
            # Interactive plotly visualizations (if available)
            if PLOTLY_AVAILABLE:
                dashboard_components['interactive'] = {
                    'regime_timeline': self.create_interactive_regime_timeline(regimes),
                    'performance_chart': self.create_interactive_performance_chart(returns, regimes),
                    'portfolio_comparison': self.create_interactive_portfolio_comparison(regime_portfolios)
                }
            else:
                dashboard_components['interactive'] = None
                logger.warning("Plotly not available. Interactive components not created.")
            
            # Summary statistics
            dashboard_components['statistics'] = {
                'total_regimes': len(regimes.unique()),
                'regime_distribution': regimes.value_counts().to_dict(),
                'total_periods': len(regimes),
                'date_range': (regimes.index.min(), regimes.index.max())
            }
            
            logger.info("Dashboard layout created successfully")
            return dashboard_components
            
        except Exception as e:
            logger.error(f"Error creating dashboard layout: {e}")
            raise

    def save_all_visualizations(self, regimes: pd.Series, returns: pd.DataFrame,
                               regime_portfolios: Dict[str, Dict], output_dir: str = 'output'):
        """
        Save all visualizations to files.
        
        Args:
            regimes: Series with regime classifications
            returns: DataFrame with asset returns  
            regime_portfolios: Dictionary with regime portfolios
            output_dir: Output directory for saved plots
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Saving all visualizations to {output_dir}...")
            
            # Save matplotlib plots
            timeline_fig = self.plot_regime_timeline(regimes)
            timeline_fig.savefig(f'{output_dir}/regime_timeline.png', dpi=300, bbox_inches='tight')
            plt.close(timeline_fig)
            
            transition_fig = self.plot_regime_transitions(regimes)
            transition_fig.savefig(f'{output_dir}/regime_transitions.png', dpi=300, bbox_inches='tight')
            plt.close(transition_fig)
            
            performance_fig = self.plot_regime_performance(returns, regimes)
            performance_fig.savefig(f'{output_dir}/regime_performance.png', dpi=300, bbox_inches='tight')
            plt.close(performance_fig)
            
            portfolio_fig = self.plot_portfolio_comparison(regime_portfolios)
            portfolio_fig.savefig(f'{output_dir}/portfolio_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(portfolio_fig)
            
            # Save interactive plots if available
            if PLOTLY_AVAILABLE:
                interactive_timeline = self.create_interactive_regime_timeline(regimes)
                if interactive_timeline:
                    interactive_timeline.write_html(f'{output_dir}/interactive_timeline.html')
                
                interactive_performance = self.create_interactive_performance_chart(returns, regimes)
                if interactive_performance:
                    interactive_performance.write_html(f'{output_dir}/interactive_performance.html')
                
                interactive_portfolio = self.create_interactive_portfolio_comparison(regime_portfolios)
                if interactive_portfolio:
                    interactive_portfolio.write_html(f'{output_dir}/interactive_portfolio.html')
            
            logger.info(f"All visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
            raise


# Utility functions for easy usage
def create_regime_visualizer(theme: str = 'default') -> RegimeVisualization:
    """
    Create a RegimeVisualization instance with specified theme.
    
    Args:
        theme: Visualization theme ('default', 'dark', 'minimal')
        
    Returns:
        RegimeVisualization instance
    """
    return RegimeVisualization(theme=theme)


def quick_regime_analysis(regimes: pd.Series, returns: pd.DataFrame = None,
                         regime_portfolios: Dict[str, Dict] = None,
                         output_dir: str = None) -> RegimeVisualization:
    """
    Perform quick regime analysis with all available visualizations.
    
    Args:
        regimes: Series with regime classifications
        returns: Optional DataFrame with asset returns
        regime_portfolios: Optional dictionary with regime portfolios
        output_dir: Optional output directory to save plots
        
    Returns:
        RegimeVisualization instance with created plots
    """
    visualizer = RegimeVisualization()
    
    # Create timeline
    visualizer.plot_regime_timeline(regimes)
    plt.show()
    
    # Create transition matrix
    visualizer.plot_regime_transitions(regimes)
    plt.show()
    
    # Create performance chart if returns provided
    if returns is not None:
        visualizer.plot_regime_performance(returns, regimes)
        plt.show()
    
    # Create portfolio comparison if portfolios provided
    if regime_portfolios is not None:
        visualizer.plot_portfolio_comparison(regime_portfolios)
        plt.show()
    
    # Save all if output directory specified
    if output_dir and returns is not None and regime_portfolios is not None:
        visualizer.save_all_visualizations(regimes, returns, regime_portfolios, output_dir)
    
    return visualizer 