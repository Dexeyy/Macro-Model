import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_ensemble_comparison(macro_data, figsize=(16, 12)):
    """
    Plot a comparison of different regime classification methods and the ensemble result.
    
    Args:
        macro_data: DataFrame with multiple regime classifications and ensemble results
        figsize: Figure size as tuple (width, height)
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Define the regime columns to compare
        regime_columns = [
            'Regime_Rule_Based',
            'Regime_KMeans_Labeled',
            'Regime_HMM_Smoothed',
            'Regime_MS_Labeled',
            'Regime_DFM_Labeled',
            'Regime_Ensemble'
        ]
        
        # Check which regime columns are actually in the DataFrame
        available_columns = [col for col in regime_columns if col in macro_data.columns]
        
        if len(available_columns) < 2:
            logger.error("Not enough regime columns for comparison")
            return None
        
        # Create figure with subplots - one for each method plus one for the ensemble confidence
        n_methods = len(available_columns)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=figsize, sharex=True)
        
        # Plot each regime classification method
        for i, col in enumerate(available_columns):
            # Get regime series and drop missing values
            regime_series = macro_data[col].dropna()
            
            if not regime_series.empty:
                # Plot regimes
                regime_series.plot(ax=axes[i], marker='o', linestyle='-', markersize=3)
                
                # Set title and labels
                axes[i].set_title(f'{col}')
                axes[i].set_ylabel('Regime')
                axes[i].grid(True)
                
                # Set y-ticks to unique regime values
                unique_regimes = sorted(regime_series.unique())
                if len(unique_regimes) <= 10:  # Only set if not too many regimes
                    axes[i].set_yticks(range(len(unique_regimes)))
                    axes[i].set_yticklabels(unique_regimes)
        
        # Plot ensemble confidence in the last subplot if available
        if 'Ensemble_Confidence' in macro_data.columns:
            confidence_series = macro_data['Ensemble_Confidence'].dropna()
            
            if not confidence_series.empty:
                # Plot confidence
                confidence_series.plot(ax=axes[-1], color='green')
                
                # Set title and labels
                axes[-1].set_title('Ensemble Classification Confidence')
                axes[-1].set_ylabel('Confidence')
                axes[-1].set_xlabel('Date')
                axes[-1].grid(True)
                
                # Set y-limits
                axes[-1].set_ylim(0, 1.05)
        else:
            # If no confidence data, hide the last subplot
            axes[-1].set_visible(False)
        
        # Set overall title
        fig.suptitle('Comparison of Regime Classification Methods', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        logger.info("Successfully created ensemble comparison plot")
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting ensemble comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_regime_agreement_matrix(macro_data, figsize=(12, 10)):
    """
    Plot a heatmap showing the agreement between different regime classification methods.
    
    Args:
        macro_data: DataFrame with multiple regime classifications
        figsize: Figure size as tuple (width, height)
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Define the regime columns to compare
        regime_columns = [
            'Regime_Rule_Based',
            'Regime_KMeans_Labeled',
            'Regime_HMM_Smoothed',
            'Regime_MS_Labeled',
            'Regime_DFM_Labeled',
            'Regime_Ensemble'
        ]
        
        # Check which regime columns are actually in the DataFrame
        available_columns = [col for col in regime_columns if col in macro_data.columns]
        
        if len(available_columns) < 2:
            logger.error("Not enough regime columns for agreement matrix")
            return None
        
        # Create a DataFrame to hold agreement scores
        agreement_matrix = pd.DataFrame(index=available_columns, columns=available_columns)
        
        # Calculate agreement between each pair of methods
        for col1 in available_columns:
            for col2 in available_columns:
                # Get common indices where both methods have classifications
                common_indices = macro_data.loc[
                    macro_data[col1].notna() & macro_data[col2].notna()
                ].index
                
                if len(common_indices) > 0:
                    # Calculate percentage of matching classifications
                    matches = (macro_data.loc[common_indices, col1] == macro_data.loc[common_indices, col2]).mean()
                    agreement_matrix.loc[col1, col2] = matches
                else:
                    agreement_matrix.loc[col1, col2] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            agreement_matrix,
            annot=True,
            cmap='YlGnBu',
            fmt='.2f',
            linewidths=0.5,
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        # Set title and labels
        ax.set_title('Agreement Between Regime Classification Methods')
        
        # Adjust layout
        plt.tight_layout()
        
        logger.info("Successfully created regime agreement matrix plot")
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting regime agreement matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None 