"""
Regime Visualization Module

This module implements comprehensive visualization capabilities for regime classification
results, including timeline plots, transition matrices, heatmaps, and statistical
analyses of regime behavior.

Author: Macro Regime Analysis System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
sns.set_palette("husl")


class RegimeVisualizer:
    """
    Comprehensive visualization suite for regime classification results.
    
    This class provides various visualization methods including timeline plots,
    transition analysis, regime distribution analysis, and performance metrics.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the regime visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.color_map = {
            'expansion': '#2E8B57',      # Sea Green
            'recession': '#DC143C',       # Crimson
            'recovery': '#4169E1',        # Royal Blue
            'stagflation': '#FF8C00',     # Dark Orange
            'neutral': '#708090',         # Slate Gray
            'unknown': '#D3D3D3'          # Light Gray
        }
    
    def plot_regime_timeline(self, 
                           regimes: pd.Series, 
                           data: Optional[pd.DataFrame] = None,
                           indicators: Optional[List[str]] = None,
                           title: str = "Market Regime Timeline",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a timeline plot showing regime classifications over time.
        
        Args:
            regimes: Series with regime classifications
            data: Optional DataFrame with economic indicators
            indicators: List of indicators to plot alongside regimes
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Determine number of subplots needed
        n_plots = 1 + (len(indicators) if indicators and data is not None else 0)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * n_plots // 2))
        if n_plots == 1:
            axes = [axes]
        
        # Plot regime timeline
        ax_regime = axes[0]
        
        # Convert regimes to numeric for plotting
        unique_regimes = list(regimes.unique())
        regime_to_num = {regime: i for i, regime in enumerate(unique_regimes)}
        regime_numeric = regimes.map(regime_to_num)
        
        # Create colored timeline
        for i, (date, regime) in enumerate(regimes.items()):
            color = self.color_map.get(regime, '#808080')
            ax_regime.bar(date, 1, width=20, color=color, alpha=0.7, edgecolor='none')
        
        # Customize regime plot
        ax_regime.set_ylabel('Market Regime')
        ax_regime.set_title(title, fontsize=14, fontweight='bold')
        ax_regime.set_ylim(0, 1.2)
        ax_regime.set_yticks([])
        
        # Add legend
        legend_elements = [mpatches.Patch(color=self.color_map.get(regime, '#808080'), 
                                        label=regime.title()) 
                         for regime in unique_regimes]
        ax_regime.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Format x-axis
        ax_regime.tick_params(axis='x', rotation=45)
        
        # Plot indicators if provided
        if indicators and data is not None:
            for i, indicator in enumerate(indicators):
                if indicator in data.columns:
                    ax = axes[i + 1]
                    ax.plot(data.index, data[indicator], linewidth=2, alpha=0.8)
                    ax.set_ylabel(indicator.replace('_', ' ').title())
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline plot saved to: {save_path}")
        
        return fig
    
    def plot_regime_transitions(self, 
                              regimes: pd.Series,
                              title: str = "Regime Transition Matrix",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing transitions between regimes.
        
        Args:
            regimes: Series with regime classifications
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regimes)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(transition_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   ax=ax,
                   cbar_kws={'label': 'Number of Transitions'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transition matrix saved to: {save_path}")
        
        return fig
    
    def plot_regime_distribution(self, 
                               regimes: pd.Series,
                               title: str = "Regime Distribution",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing regime distribution.
        
        Args:
            regimes: Series with regime classifications
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Calculate regime counts
        regime_counts = regimes.value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] // 2))
        
        # Pie chart
        colors = [self.color_map.get(regime, '#808080') for regime in regime_counts.index]
        wedges, texts, autotexts = ax1.pie(regime_counts.values, 
                                          labels=regime_counts.index,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        
        ax1.set_title(f"{title} - Percentage", fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(regime_counts.index, regime_counts.values, color=colors, alpha=0.7)
        ax2.set_title(f"{title} - Count", fontweight='bold')
        ax2.set_ylabel('Number of Periods')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_regime_duration_analysis(self, 
                                    regimes: pd.Series,
                                    title: str = "Regime Duration Analysis",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and plot regime duration statistics.
        
        Args:
            regimes: Series with regime classifications
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Calculate durations
        durations = self._calculate_regime_durations(regimes)
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # Average duration by regime
        avg_durations = {regime: np.mean(duration_list) 
                        for regime, duration_list in durations.items()}
        
        regimes_list = list(avg_durations.keys())
        avg_values = list(avg_durations.values())
        colors = [self.color_map.get(regime, '#808080') for regime in regimes_list]
        
        axes[0, 0].bar(regimes_list, avg_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Average Regime Duration')
        axes[0, 0].set_ylabel('Periods')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Duration distribution (box plot)
        duration_data = []
        duration_labels = []
        for regime, duration_list in durations.items():
            duration_data.extend(duration_list)
            duration_labels.extend([regime] * len(duration_list))
        
        df_duration = pd.DataFrame({'Regime': duration_labels, 'Duration': duration_data})
        
        sns.boxplot(data=df_duration, x='Regime', y='Duration', ax=axes[0, 1])
        axes[0, 1].set_title('Duration Distribution by Regime')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Regime frequency over time
        regime_counts_over_time = regimes.resample('Q').agg(lambda x: x.mode()[0] if not x.empty else 'unknown')
        quarterly_counts = regime_counts_over_time.value_counts()
        
        axes[1, 0].bar(quarterly_counts.index, quarterly_counts.values, 
                      color=[self.color_map.get(regime, '#808080') for regime in quarterly_counts.index],
                      alpha=0.7)
        axes[1, 0].set_title('Dominant Regime by Quarter')
        axes[1, 0].set_ylabel('Number of Quarters')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Stability metrics
        stability_metrics = self._calculate_stability_metrics(regimes)
        metrics_names = list(stability_metrics.keys())
        metrics_values = list(stability_metrics.values())
        
        axes[1, 1].bar(metrics_names, metrics_values, color='skyblue', alpha=0.7)
        axes[1, 1].set_title('Regime Stability Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Duration analysis saved to: {save_path}")
        
        return fig
    
    def plot_regime_correlation_matrix(self, 
                                     regimes: pd.Series,
                                     data: pd.DataFrame,
                                     title: str = "Regime-Indicator Correlation",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation between regimes and economic indicators.
        
        Args:
            regimes: Series with regime classifications
            data: DataFrame with economic indicators
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Create regime dummy variables
        regime_dummies = pd.get_dummies(regimes, prefix='regime')
        
        # Combine with data
        combined_data = pd.concat([data, regime_dummies], axis=1)
        
        # Calculate correlation matrix
        correlation_matrix = combined_data.corr()
        
        # Extract correlations between indicators and regimes
        regime_cols = [col for col in correlation_matrix.columns if col.startswith('regime_')]
        indicator_cols = [col for col in correlation_matrix.columns if not col.startswith('regime_')]
        
        regime_indicator_corr = correlation_matrix.loc[indicator_cols, regime_cols]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(regime_indicator_corr, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Regime Type')
        ax.set_ylabel('Economic Indicators')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        return fig
    
    def create_regime_report(self, 
                           regimes: pd.Series,
                           data: Optional[pd.DataFrame] = None,
                           classifier_stats: Optional[Dict] = None,
                           output_dir: str = "regime_analysis_report") -> str:
        """
        Create a comprehensive regime analysis report with multiple visualizations.
        
        Args:
            regimes: Series with regime classifications
            data: Optional DataFrame with economic indicators
            classifier_stats: Optional statistics from classifier
            output_dir: Directory to save report files
            
        Returns:
            Path to the report directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Creating comprehensive regime analysis report...")
        
        # 1. Timeline plot
        timeline_fig = self.plot_regime_timeline(
            regimes, 
            data, 
            indicators=['gdp_growth', 'inflation'] if data is not None else None,
            title="Market Regime Timeline Analysis",
            save_path=f"{output_dir}/regime_timeline.png"
        )
        plt.close(timeline_fig)
        
        # 2. Transition matrix
        transition_fig = self.plot_regime_transitions(
            regimes,
            title="Regime Transition Analysis",
            save_path=f"{output_dir}/regime_transitions.png"
        )
        plt.close(transition_fig)
        
        # 3. Distribution analysis
        distribution_fig = self.plot_regime_distribution(
            regimes,
            title="Regime Distribution Analysis",
            save_path=f"{output_dir}/regime_distribution.png"
        )
        plt.close(distribution_fig)
        
        # 4. Duration analysis
        duration_fig = self.plot_regime_duration_analysis(
            regimes,
            title="Regime Duration & Stability Analysis",
            save_path=f"{output_dir}/regime_duration.png"
        )
        plt.close(duration_fig)
        
        # 5. Correlation analysis (if data provided)
        if data is not None:
            correlation_fig = self.plot_regime_correlation_matrix(
                regimes,
                data,
                title="Regime-Economic Indicator Correlations",
                save_path=f"{output_dir}/regime_correlations.png"
            )
            plt.close(correlation_fig)
        
        # 6. Create summary report
        self._create_text_summary(regimes, classifier_stats, f"{output_dir}/regime_summary.txt")
        
        print(f"✅ Report created successfully in: {output_dir}")
        return output_dir
    
    def _calculate_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """Calculate transition matrix between regimes."""
        unique_regimes = sorted(regimes.unique())
        transition_matrix = pd.DataFrame(0, index=unique_regimes, columns=unique_regimes)
        
        for i in range(1, len(regimes)):
            from_regime = regimes.iloc[i-1]
            to_regime = regimes.iloc[i]
            transition_matrix.loc[from_regime, to_regime] += 1
        
        return transition_matrix
    
    def _calculate_regime_durations(self, regimes: pd.Series) -> Dict[str, List[int]]:
        """Calculate duration of each regime period."""
        if regimes.empty:
            return {}
        
        durations = {}
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                # End of current regime
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(current_duration)
                
                # Start new regime
                current_regime = regimes.iloc[i]
                current_duration = 1
        
        # Add final regime duration
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(current_duration)
        
        return durations
    
    def _calculate_stability_metrics(self, regimes: pd.Series) -> Dict[str, float]:
        """Calculate various stability metrics."""
        if len(regimes) < 2:
            return {'stability': 1.0, 'volatility': 0.0}
        
        # Transition count
        transitions = sum(1 for i in range(1, len(regimes)) 
                         if regimes.iloc[i] != regimes.iloc[i-1])
        max_transitions = len(regimes) - 1
        
        stability = 1 - (transitions / max_transitions) if max_transitions > 0 else 1.0
        volatility = transitions / len(regimes)
        
        # Persistence metric (average duration)
        durations = self._calculate_regime_durations(regimes)
        all_durations = [d for duration_list in durations.values() for d in duration_list]
        persistence = np.mean(all_durations) if all_durations else 1.0
        
        return {
            'Stability': stability,
            'Volatility': volatility,
            'Persistence': persistence / len(regimes),  # Normalized
            'Diversity': len(regimes.unique()) / 6  # Normalized by max possible regimes
        }
    
    def _create_text_summary(self, 
                           regimes: pd.Series, 
                           classifier_stats: Optional[Dict], 
                           save_path: str):
        """Create a text summary of the regime analysis."""
        summary = []
        summary.append("=" * 60)
        summary.append("MARKET REGIME ANALYSIS SUMMARY REPORT")
        summary.append("=" * 60)
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Time Period: {regimes.index[0]} to {regimes.index[-1]}")
        summary.append(f"Total Periods: {len(regimes)}")
        summary.append("")
        
        # Regime distribution
        summary.append("REGIME DISTRIBUTION:")
        summary.append("-" * 30)
        regime_counts = regimes.value_counts()
        regime_percentages = (regime_counts / len(regimes) * 100).round(1)
        
        for regime, count in regime_counts.items():
            percentage = regime_percentages[regime]
            summary.append(f"{regime.upper():>12}: {count:>3} periods ({percentage:>5.1f}%)")
        
        summary.append("")
        
        # Stability metrics
        stability_metrics = self._calculate_stability_metrics(regimes)
        summary.append("STABILITY METRICS:")
        summary.append("-" * 30)
        for metric, value in stability_metrics.items():
            summary.append(f"{metric:>12}: {value:>6.3f}")
        
        summary.append("")
        
        # Transition analysis
        transition_matrix = self._calculate_transition_matrix(regimes)
        total_transitions = transition_matrix.sum().sum()
        summary.append("TRANSITION ANALYSIS:")
        summary.append("-" * 30)
        summary.append(f"Total Transitions: {total_transitions}")
        
        if total_transitions > 0:
            # Most common transitions
            transitions_flat = []
            for from_regime in transition_matrix.index:
                for to_regime in transition_matrix.columns:
                    if transition_matrix.loc[from_regime, to_regime] > 0:
                        transitions_flat.append((
                            f"{from_regime} → {to_regime}",
                            transition_matrix.loc[from_regime, to_regime]
                        ))
            
            transitions_flat.sort(key=lambda x: x[1], reverse=True)
            summary.append("Most Common Transitions:")
            for transition, count in transitions_flat[:5]:
                summary.append(f"  {transition}: {count}")
        
        summary.append("")
        
        # Duration analysis
        durations = self._calculate_regime_durations(regimes)
        summary.append("DURATION ANALYSIS:")
        summary.append("-" * 30)
        for regime, duration_list in durations.items():
            avg_duration = np.mean(duration_list)
            max_duration = max(duration_list)
            min_duration = min(duration_list)
            summary.append(f"{regime.upper():>12}: Avg={avg_duration:.1f}, Max={max_duration}, Min={min_duration}")
        
        # Add classifier stats if provided
        if classifier_stats:
            summary.append("")
            summary.append("CLASSIFIER STATISTICS:")
            summary.append("-" * 30)
            for key, value in classifier_stats.items():
                if isinstance(value, dict):
                    summary.append(f"{key.upper()}:")
                    for sub_key, sub_value in value.items():
                        summary.append(f"  {sub_key}: {sub_value}")
                else:
                    summary.append(f"{key}: {value}")
        
        summary.append("")
        summary.append("=" * 60)
        summary.append("End of Report")
        summary.append("=" * 60)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary))
        
        print(f"📝 Summary report saved to: {save_path}")


# Convenience functions
def create_quick_regime_timeline(regimes: pd.Series, 
                               title: str = "Market Regimes", 
                               save_path: Optional[str] = None) -> plt.Figure:
    """Quick function to create a regime timeline plot."""
    visualizer = RegimeVisualizer()
    return visualizer.plot_regime_timeline(regimes, title=title, save_path=save_path)


def analyze_regime_transitions(regimes: pd.Series) -> Dict[str, Any]:
    """Quick function to analyze regime transitions."""
    visualizer = RegimeVisualizer()
    transition_matrix = visualizer._calculate_transition_matrix(regimes)
    durations = visualizer._calculate_regime_durations(regimes)
    stability = visualizer._calculate_stability_metrics(regimes)
    
    return {
        'transition_matrix': transition_matrix,
        'durations': durations,
        'stability_metrics': stability,
        'total_transitions': transition_matrix.sum().sum()
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Regime Visualization System...")
    
    # Create sample regime data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    
    # Generate realistic regime sequence
    regimes_list = []
    current_regime = 'expansion'
    
    for i in range(100):
        # Add some regime switching logic
        if np.random.random() < 0.1:  # 10% chance of switching
            possible_regimes = ['expansion', 'recession', 'recovery', 'stagflation', 'neutral']
            current_regime = np.random.choice(possible_regimes)
        regimes_list.append(current_regime)
    
    sample_regimes = pd.Series(regimes_list, index=dates, name='regime')
    
    # Create sample economic data
    sample_data = pd.DataFrame({
        'gdp_growth': np.random.normal(2, 2, 100),
        'inflation': np.random.normal(3, 1.5, 100),
        'unemployment_gap': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Initialize visualizer
    visualizer = RegimeVisualizer()
    
    print(f"Sample regimes shape: {sample_regimes.shape}")
    print(f"Unique regimes: {sample_regimes.unique()}")
    
    # Test timeline plot
    timeline_fig = visualizer.plot_regime_timeline(
        sample_regimes, 
        sample_data, 
        indicators=['gdp_growth', 'inflation']
    )
    plt.show()
    
    # Test transition analysis
    transition_analysis = analyze_regime_transitions(sample_regimes)
    print(f"\nTransition Analysis:")
    print(f"Total transitions: {transition_analysis['total_transitions']}")
    print(f"Stability score: {transition_analysis['stability_metrics']['Stability']:.3f}")
    
    print("\n✅ Regime Visualization System created successfully!")
    print("🎨 Features: Timeline plots, transition matrices, duration analysis, correlation heatmaps")
