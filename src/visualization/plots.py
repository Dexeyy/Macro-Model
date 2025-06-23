import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Try to import seaborn, but handle if not available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn not available. Some visualizations may be limited.")
    SEABORN_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """Set up the plot style for consistent visualizations."""
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_regime_timeline(macro_data, regime_col, output_dir=None):
    """
    Plot the timeline of regimes.
    
    Args:
        macro_data: DataFrame with macro data and regime column
        regime_col: Name of the regime column
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Figure object
    """
    try:
        setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get unique regimes and assign colors
        regimes = macro_data[regime_col].dropna().unique()
        
        # Create a categorical plot with regimes
        if SEABORN_AVAILABLE:
            sns.scatterplot(
                data=macro_data,
                x=macro_data.index,
                y=regime_col,
                hue=regime_col,
                s=100,
                ax=ax
            )
        else:
            for i, regime in enumerate(regimes):
                regime_data = macro_data[macro_data[regime_col] == regime]
                ax.scatter(
                    regime_data.index,
                    [i] * len(regime_data),
                    label=regime,
                    s=100
                )
        
        # Set labels and title
        ax.set_title(f'Economic Regimes Timeline ({regime_col})', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Regime', fontsize=14)
        
        # Format x-axis to show years
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend(title='Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'regime_timeline_{regime_col}.png'))
            logger.info(f"Saved regime timeline plot to {output_dir}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting regime timeline: {e}")
        return None

def plot_regime_performance(regime_performance, output_dir=None):
    """
    Plot regime performance metrics.
    
    Args:
        regime_performance: DataFrame with regime performance metrics
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Figure object
    """
    try:
        setup_plot_style()
        
        if regime_performance is None or regime_performance.empty:
            logger.error("No regime performance data available for plotting")
            return None
        
        # Get the mean returns for plotting
        if isinstance(regime_performance.columns, pd.MultiIndex):
            logger.info("Extracting mean returns from MultiIndex columns")
            try:
                # Try the .xs method first
                mean_returns = regime_performance.xs('Ann_Mean_Return', axis=1, level='Metric', drop_level=True)
                logger.info("Successfully extracted mean returns using .xs")
            except Exception as e_xs:
                logger.error(f"Error using .xs: {e_xs}")
                # Fallback to direct column selection
                mean_return_cols = [(asset, 'Ann_Mean_Return') for asset in set(col[0] for col in regime_performance.columns)]
                available_cols = [col for col in mean_return_cols if col in regime_performance.columns]
                if available_cols:
                    mean_returns = regime_performance[available_cols].copy()
                    mean_returns.columns = [col[0] for col in available_cols]  # Simplify to just asset names
                    logger.info("Successfully extracted mean returns using direct selection")
                else:
                    logger.error("Could not find Ann_Mean_Return columns")
                    return None
        else:
            logger.error("regime_performance columns are not a MultiIndex")
            logger.info(f"Column structure: {type(regime_performance.columns)}")
            logger.info(f"Columns: {regime_performance.columns.tolist()}")
            # Try to extract columns ending with Ann_Mean_Return
            mean_return_cols = [col for col in regime_performance.columns if str(col).endswith('Ann_Mean_Return')]
            if mean_return_cols:
                mean_returns = regime_performance[mean_return_cols].copy()
                logger.info("Extracted columns that appear to be mean returns")
            else:
                logger.error("Could not identify mean return columns")
                return None
        
        logger.info(f"Mean returns shape: {mean_returns.shape}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        mean_returns.plot(kind='bar', ax=ax)
        plt.title(f'Mean Annualized Asset Returns by Macro Regime', fontsize=14)
        plt.ylabel('Mean Annualized Return (%)', fontsize=12)
        plt.xlabel('Macro Regime', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Assets', bbox_to_anchor=(1.03, 1), loc='upper left', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        
        # Save plot to file if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'asset_returns_by_regime.png'))
            logger.info(f"Saved regime performance plot to {output_dir}")
        
        return fig
    
    except Exception as e_plot:
        logger.error(f"Error plotting regime performance: {e_plot}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_regime_feature_distribution(macro_data, regime_col, features, output_dir=None):
    """
    Plot the distribution of features by regime.
    
    Args:
        macro_data: DataFrame with macro data and regime column
        regime_col: Name of the regime column
        features: List of features to plot
        output_dir: Directory to save the plot (optional)
        
    Returns:
        List of Figure objects
    """
    try:
        setup_plot_style()
        
        figures = []
        
        for feature in features:
            if feature not in macro_data.columns:
                logger.warning(f"Feature {feature} not found in macro_data")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if SEABORN_AVAILABLE:
                sns.boxplot(
                    data=macro_data.dropna(subset=[regime_col, feature]),
                    x=regime_col,
                    y=feature,
                    ax=ax
                )
            else:
                regimes = macro_data[regime_col].dropna().unique()
                data_to_plot = []
                
                for regime in regimes:
                    regime_data = macro_data[macro_data[regime_col] == regime][feature].dropna()
                    data_to_plot.append(regime_data.values)
                
                ax.boxplot(data_to_plot, labels=regimes)
            
            plt.title(f'Distribution of {feature} by Regime', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            figures.append(fig)
            
            # Save the plot if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'regime_feature_{feature}.png'))
                logger.info(f"Saved feature distribution plot for {feature} to {output_dir}")
        
        return figures
    
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        return []

def plot_regime_transitions(macro_data, regime_col, output_dir=None):
    """
    Plot regime transitions over time.
    
    Args:
        macro_data: DataFrame with macro data and regime column
        regime_col: Name of the regime column
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Figure object
    """
    try:
        setup_plot_style()
        
        # Create a copy of the data with just the regime column
        regime_data = macro_data[[regime_col]].copy()
        
        # Remove rows with NaN regimes
        regime_data = regime_data.dropna()
        
        # Get unique regimes
        regimes = regime_data[regime_col].unique()
        
        # Create a numeric mapping for regimes
        regime_map = {regime: i for i, regime in enumerate(regimes)}
        
        # Apply mapping
        regime_data['Regime_Numeric'] = regime_data[regime_col].map(regime_map)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(regime_data.index, regime_data['Regime_Numeric'], 'o-', linewidth=2)
        
        # Set y-ticks to regime names
        ax.set_yticks(list(regime_map.values()))
        ax.set_yticklabels(list(regime_map.keys()))
        
        # Set labels and title
        ax.set_title(f'Regime Transitions Over Time ({regime_col})', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Regime', fontsize=14)
        
        # Format x-axis to show years
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'regime_transitions_{regime_col}.png'))
            logger.info(f"Saved regime transitions plot to {output_dir}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting regime transitions: {e}")
        return None 