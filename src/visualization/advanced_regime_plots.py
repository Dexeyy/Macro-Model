"""
Visualization Methods for Advanced Regime Models

This module provides comprehensive visualization capabilities for advanced
regime classification models including HMM, Factor Analysis, and ensemble methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedRegimeVisualizer:
    """Visualization class for advanced regime models."""
    
    def __init__(self, theme: str = 'default'):
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
        
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_hmm_regimes(self, data: pd.DataFrame, hmm_results: Dict[str, Any]) -> go.Figure:
        """Visualize HMM regimes using PCA for dimensionality reduction."""
        # Apply PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        data_scaled = (data - data.mean()) / data.std()
        pca_result = pca.fit_transform(data_scaled.dropna())
        
        # Create visualization DataFrame
        viz_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Regime': hmm_results['regime_labels']
        })
        
        # Create scatter plot
        fig = px.scatter(
            viz_df, x='PC1', y='PC2', color='Regime',
            title='HMM Regime Visualization (PCA)',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
        )
        
        # Add regime centroids
        for regime in viz_df['Regime'].unique():
            regime_data = viz_df[viz_df['Regime'] == regime]
            centroid_x = regime_data['PC1'].mean()
            centroid_y = regime_data['PC2'].mean()
            
            fig.add_trace(go.Scatter(
                x=[centroid_x], y=[centroid_y],
                mode='markers',
                marker=dict(size=15, symbol='x', color='black'),
                name=f'{regime} Center',
                showlegend=True
            ))
        
        fig.update_layout(height=600)
        return fig
    
    def plot_regime_transitions(self, regime_labels: pd.Series) -> go.Figure:
        """Plot regime transitions over time."""
        # Create numeric mapping for regimes
        unique_regimes = regime_labels.unique()
        regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
        numeric_regimes = regime_labels.map(regime_map)
        
        fig = go.Figure()
        
        # Plot regime timeline
        fig.add_trace(go.Scatter(
            x=regime_labels.index,
            y=numeric_regimes,
            mode='lines+markers',
            line=dict(shape='hv'),  # Step-like appearance
            name='Regime Sequence'
        ))
        
        # Update layout
        fig.update_layout(
            title='Regime Transitions Over Time',
            xaxis_title='Date',
            yaxis_title='Regime',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(unique_regimes))),
                ticktext=unique_regimes
            ),
            height=400
        )
        
        return fig
    
    def plot_transition_matrix(self, transition_matrix: np.ndarray, 
                              regime_names: Dict[int, str] = None) -> go.Figure:
        """Plot HMM transition probability matrix."""
        if regime_names is None:
            labels = [f'Regime {i}' for i in range(len(transition_matrix))]
        else:
            labels = [regime_names[i] for i in range(len(transition_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=np.round(transition_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            title='Regime Transition Probability Matrix',
            xaxis_title='To Regime',
            yaxis_title='From Regime',
            height=500
        )
        
        return fig
    
    def plot_regime_probabilities(self, regime_probs: pd.DataFrame) -> go.Figure:
        """Plot regime probabilities over time."""
        fig = go.Figure()
        
        for column in regime_probs.columns:
            fig.add_trace(go.Scatter(
                x=regime_probs.index,
                y=regime_probs[column],
                mode='lines',
                name=column,
                stackgroup='one'  # Creates stacked area chart
            ))
        
        fig.update_layout(
            title='Regime Probabilities Over Time',
            xaxis_title='Date',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=500
        )
        
        return fig
    
    def plot_factor_analysis(self, factor_results: Dict[str, Any]) -> go.Figure:
        """Plot factor analysis results."""
        factor_loadings = factor_results['factor_loadings']
        components = factor_results['components']
        explained_variance = factor_results['explained_variance_ratio']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Factor Loadings Over Time', 'Component Loadings',
                          'Explained Variance', 'Factor Correlation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        # 1. Factor loadings over time
        for column in factor_loadings.columns:
            fig.add_trace(
                go.Scatter(x=factor_loadings.index, y=factor_loadings[column],
                          name=column, mode='lines'),
                row=1, col=1
            )
        
        # 2. Component loadings heatmap
        fig.add_trace(
            go.Heatmap(z=components.values.T,
                      x=components.index,
                      y=components.columns,
                      colorscale='RdBu',
                      zmid=0),
            row=1, col=2
        )
        
        # 3. Explained variance
        fig.add_trace(
            go.Bar(x=list(range(1, len(explained_variance) + 1)),
                   y=explained_variance,
                   name='Explained Variance'),
            row=2, col=1
        )
        
        # 4. Factor correlation
        factor_corr = factor_loadings.corr()
        fig.add_trace(
            go.Heatmap(z=factor_corr.values,
                      x=factor_corr.columns,
                      y=factor_corr.index,
                      colorscale='RdBu',
                      zmid=0),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Factor Analysis Results',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any]) -> go.Figure:
        """Plot model comparison results."""
        models = list(comparison_results.keys())
        
        # Extract metrics
        mean_scores = [comparison_results[model]['mean_score'] for model in models]
        std_scores = [comparison_results[model]['std_score'] for model in models]
        
        fig = go.Figure()
        
        # Bar plot with error bars
        fig.add_trace(go.Bar(
            x=models,
            y=mean_scores,
            error_y=dict(type='data', array=std_scores),
            name='Cross-Validation Score'
        ))
        
        fig.update_layout(
            title='Model Comparison - Cross-Validation Scores',
            xaxis_title='Model',
            yaxis_title='Score',
            height=400
        )
        
        return fig
    
    def plot_ensemble_results(self, ensemble_results: Dict[str, Any]) -> go.Figure:
        """Plot ensemble model results."""
        base_predictions = ensemble_results['base_predictions']
        ensemble_labels = ensemble_results['ensemble_labels']
        
        # Create subplots
        fig = make_subplots(
            rows=len(base_predictions) + 1, cols=1,
            subplot_titles=list(base_predictions.keys()) + ['Ensemble'],
            vertical_spacing=0.05
        )
        
        # Plot base model predictions
        for i, (model_name, predictions) in enumerate(base_predictions.items()):
            # Convert regime labels to numeric for plotting
            unique_regimes = predictions.unique()
            regime_map = {regime: j for j, regime in enumerate(unique_regimes)}
            numeric_preds = predictions.map(regime_map)
            
            fig.add_trace(
                go.Scatter(x=predictions.index, y=numeric_preds,
                          mode='lines+markers', name=model_name,
                          line=dict(shape='hv')),
                row=i+1, col=1
            )
            
            # Update y-axis
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(len(unique_regimes))),
                ticktext=unique_regimes,
                row=i+1, col=1
            )
        
        # Plot ensemble predictions
        ensemble_unique = ensemble_labels.unique()
        ensemble_map = {regime: j for j, regime in enumerate(ensemble_unique)}
        ensemble_numeric = ensemble_labels.map(ensemble_map)
        
        fig.add_trace(
            go.Scatter(x=ensemble_labels.index, y=ensemble_numeric,
                      mode='lines+markers', name='Ensemble',
                      line=dict(shape='hv', width=3)),
            row=len(base_predictions)+1, col=1
        )
        
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(ensemble_unique))),
            ticktext=ensemble_unique,
            row=len(base_predictions)+1, col=1
        )
        
        fig.update_layout(
            title='Ensemble Model Results',
            height=200 * (len(base_predictions) + 1),
            showlegend=True
        )
        
        return fig
    
    def plot_regime_characteristics(self, data: pd.DataFrame, 
                                   regime_labels: pd.Series) -> go.Figure:
        """Plot regime characteristics (returns, volatility, etc.)."""
        # Calculate regime statistics
        regime_stats = {}
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_data = data[mask]
            
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'mean_return': regime_data.mean().mean() * 252,  # Annualized
                    'volatility': regime_data.std().mean() * np.sqrt(252),  # Annualized
                    'count': len(regime_data),
                    'duration': len(regime_data)  # Average duration would need more complex calculation
                }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Returns by Regime', 'Volatility by Regime',
                          'Regime Frequency', 'Risk-Return Profile')
        )
        
        regimes = list(regime_stats.keys())
        
        # 1. Mean returns
        returns = [regime_stats[r]['mean_return'] for r in regimes]
        fig.add_trace(
            go.Bar(x=regimes, y=returns, name='Mean Return'),
            row=1, col=1
        )
        
        # 2. Volatility
        vols = [regime_stats[r]['volatility'] for r in regimes]
        fig.add_trace(
            go.Bar(x=regimes, y=vols, name='Volatility'),
            row=1, col=2
        )
        
        # 3. Frequency
        counts = [regime_stats[r]['count'] for r in regimes]
        fig.add_trace(
            go.Bar(x=regimes, y=counts, name='Frequency'),
            row=2, col=1
        )
        
        # 4. Risk-Return scatter
        fig.add_trace(
            go.Scatter(x=vols, y=returns, mode='markers+text',
                      text=regimes, textposition="top center",
                      marker=dict(size=15), name='Risk-Return'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Regime Characteristics Analysis',
            height=600,
            showlegend=False
        )
        
        return fig 