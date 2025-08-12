"""
Main entry point for the macro-regime-model application.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import configuration
import sys
sys.path.append('config')
from config import config  # import the module config/config.py explicitly

# Import modules
from src.data.fetchers import fetch_fred_series, fetch_asset_data, create_dummy_asset_data
from src.data.processors import (
    process_macro_data, 
    calculate_returns, 
    merge_macro_and_asset_data,
    calculate_regime_performance,
    create_advanced_features
)
from src.models.regime_classifier import fit_regimes
from src.utils.helpers import get_regimes_config
from src.models.portfolio import (
    create_equal_weight_portfolio,
    create_regime_based_portfolio,
    calculate_portfolio_metrics
)
from src.visualization.plots import (
    plot_regime_timeline,
    plot_regime_performance,
    plot_regime_feature_distribution,
    plot_regime_transitions
)
from src.models.performance_analytics import PerformanceAnalytics
from src.utils.helpers import (
    validate_dataframe,
    diagnose_dataframe,
    save_data,
    load_data,
    create_output_filename
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_and_process_data():
    """Fetch and process macro and asset data."""
    logger.info("Starting data fetching and processing...")
    
    # Fetch macro data from FRED
    try:
        logger.info("Fetching macro data from FRED...")
        macro_data_raw = fetch_fred_series(config.FRED_SERIES, config.START_DATE, config.END_DATE)
        # Apply FRED-MD transformations if available
        from src.data.fetchers import apply_fredmd_tcodes
        macro_data_raw = apply_fredmd_tcodes(macro_data_raw)
        save_data(
            macro_data_raw,
            os.path.join(config.RAW_DATA_DIR, 'macro_data_raw.csv')
        )
    except Exception as e:
        logger.error(f"Error fetching macro data: {e}")
        return None, None
    
    # Process macro data
    try:
        logger.info("Processing macro data...")
        base_features = process_macro_data(macro_data_raw)
        diagnose_dataframe(base_features, "macro_features_legacy")

        # Prefer the mode specified in YAML to select RT vs RETRO artifacts produced by process_macro_data
        try:
            cfg_yaml = get_regimes_config() or {}
            run_mode = ((cfg_yaml.get("run") or {}).get("mode") or "").lower()
        except Exception:
            run_mode = ""

        chosen = None
        try:
            if run_mode in ("rt", "real-time", "realtime"):
                p = os.path.join(config.PROCESSED_DATA_DIR, 'macro_features_rt.parquet')
                if os.path.exists(p):
                    chosen = pd.read_parquet(p)
            elif run_mode in ("retro", "revised", "full"):
                p = os.path.join(config.PROCESSED_DATA_DIR, 'macro_features_retro.parquet')
                if os.path.exists(p):
                    chosen = pd.read_parquet(p)
        except Exception as _:
            chosen = None

        # Fallback to building advanced features in-process if no artifact was found
        macro_data_featured = chosen if chosen is not None else create_advanced_features(base_features, mode=run_mode or "retro")
        diagnose_dataframe(macro_data_featured, "macro_data_featured")

        # Persist a CSV view for downstream CLI/tools to read consistently
        save_data(
            macro_data_featured,
            os.path.join(config.PROCESSED_DATA_DIR, 'macro_data_featured.csv')
        )
    except Exception as e:
        logger.error(f"Error processing macro data: {e}")
        return None, None
    
    # Fetch asset data from Yahoo Finance
    try:
        logger.info("Fetching asset data from Yahoo Finance...")
        asset_prices = fetch_asset_data(
            config.ASSET_TICKERS,
            config.ASSET_START_DATE,
            config.END_DATE
        )
        
        # If fetching fails, create dummy data for testing
        if asset_prices is None or asset_prices.empty:
            logger.warning("Failed to fetch real asset data, creating dummy data...")
            asset_prices = create_dummy_asset_data(
                config.ASSET_START_DATE,
                config.END_DATE,
                list(config.ASSET_TICKERS.keys())
            )
        
        save_data(
            asset_prices,
            os.path.join(config.RAW_DATA_DIR, 'asset_prices.csv')
        )
    except Exception as e:
        logger.error(f"Error fetching asset data: {e}")
        return macro_data_featured, None
    
    # Calculate asset returns
    try:
        logger.info("Calculating asset returns...")
        asset_returns = calculate_returns(asset_prices)
        diagnose_dataframe(asset_returns, "asset_returns")
        save_data(
            asset_returns,
            os.path.join(config.PROCESSED_DATA_DIR, 'asset_returns.csv')
        )
    except Exception as e:
        logger.error(f"Error calculating asset returns: {e}")
        return macro_data_featured, None
    
    logger.info("Data fetching and processing completed successfully")
    return macro_data_featured, asset_returns

def classify_regimes(macro_data):
    """Apply v2 regime classification and return merged DataFrame (features + labels/probs)."""
    logger.info("Starting regime classification...")
    
    if macro_data is None:
        logger.error("No macro data available for regime classification")
        return None

    try:
        cfg = get_regimes_config()
        # Bundle preference from YAML unless overridden elsewhere
        bundle = (cfg.get("run") or {}).get("bundle")
        out = fit_regimes(macro_data, features=None, n_regimes=4, bundle=bundle)
        merged = macro_data.join(out, how="left")
        save_data(merged, os.path.join(config.PROCESSED_DATA_DIR, 'macro_data_with_regimes.csv'))
        logger.info("Regime classification completed")
        return merged
    except Exception as e:
        logger.error(f"Regime classification failed: {e}")
        return macro_data

def analyze_regime_performance(macro_data, asset_returns):
    """Analyze asset performance across different regimes."""
    logger.info("Starting regime performance analysis...")
    
    if macro_data is None or asset_returns is None:
        logger.error("Missing data for regime performance analysis")
        return None, None
    
    results = {}
    
    # Analyze for available regimes in the unified output
    candidate_cols = [c for c in ['Rule', 'HMM', 'KMeans', 'GMM', 'HSMM', 'Ensemble_Regime', 'Regime_Ensemble'] if c in macro_data.columns]
    for regime_col in candidate_cols:
        if regime_col not in macro_data.columns:
            logger.warning(f"Regime column {regime_col} not found in macro_data")
            continue
        
        try:
            # Merge macro regimes with asset returns
            logger.info(f"Merging macro regimes ({regime_col}) with asset returns...")
            data_for_analysis = merge_macro_and_asset_data(
                macro_data,
                asset_returns,
                regime_col
            )
            
            if data_for_analysis is None or data_for_analysis.empty:
                logger.warning(f"No data available for analysis after merging {regime_col}")
                continue
            
            # Calculate performance metrics by regime
            logger.info(f"Calculating performance metrics for {regime_col}...")
            regime_performance = calculate_regime_performance(
                data_for_analysis,
                regime_col
            )
            
            if regime_performance is None or regime_performance.empty:
                logger.warning(f"No performance metrics calculated for {regime_col}")
                continue
            
            # Save results
            save_data(
                regime_performance,
                os.path.join(config.PROCESSED_DATA_DIR, f'regime_performance_{regime_col}.csv')
            )
            
            results[regime_col] = {
                'data_for_analysis': data_for_analysis,
                'regime_performance': regime_performance
            }
            
            logger.info(f"Successfully analyzed performance for {regime_col}")
        
        except Exception as e:
            logger.error(f"Error analyzing performance for {regime_col}: {e}")
    
    logger.info("Regime performance analysis completed")

    # Build and save regime scorecard diagnostics
    try:
        if asset_returns is not None and macro_data is not None and results:
            regime_series_dict = {}
            for col in [c for c in macro_data.columns if c in ("HMM", "GMM", "KMeans", "Rule", "Regime_Ensemble")]:
                regime_series_dict[col] = macro_data[col]
            if regime_series_dict:
                analyzer = PerformanceAnalytics()
                analyzer.build_regime_scorecard(asset_returns, regime_series_dict)
    except Exception as exc:
        logger.warning(f"Failed to build regime scorecard: {exc}")
    return results

def create_visualizations(macro_data, analysis_results):
    """Create visualizations of regime data and performance."""
    logger.info("Starting visualization creation...")
    
    if macro_data is None:
        logger.error("No macro data available for visualizations")
        return
    
    # Create regime timeline plots for available columns
    for regime_col in [c for c in ['Rule', 'HMM', 'GMM', 'KMeans', 'HSMM', 'Ensemble_Regime', 'Regime_Ensemble'] if c in macro_data.columns]:
        if regime_col not in macro_data.columns:
            logger.warning(f"Regime column {regime_col} not found in macro_data")
            continue
        
        try:
            logger.info(f"Creating timeline plot for {regime_col}...")
            plot_regime_timeline(
                macro_data,
                regime_col,
                config.OUTPUT_DIR
            )
            
            logger.info(f"Creating regime transitions plot for {regime_col}...")
            plot_regime_transitions(
                macro_data,
                regime_col,
                config.OUTPUT_DIR
            )
            
            # Plot feature distributions for key features
            key_features = ['CPI_YoY', 'GDP_YoY', 'UNRATE']
            logger.info(f"Creating feature distribution plots for {regime_col}...")
            plot_regime_feature_distribution(
                macro_data,
                regime_col,
                key_features,
                config.OUTPUT_DIR
            )
        except Exception as e:
            logger.error(f"Error creating plots for {regime_col}: {e}")
    
    # Create performance plots
    if analysis_results:
        for regime_col, result in analysis_results.items():
            try:
                logger.info(f"Creating performance plot for {regime_col}...")
                plot_regime_performance(
                    result['regime_performance'],
                    config.OUTPUT_DIR
                )
            except Exception as e:
                logger.error(f"Error creating performance plot for {regime_col}: {e}")
    
    logger.info("Visualization creation completed")

def create_portfolios(analysis_results):
    """Create and evaluate portfolios based on regime analysis."""
    logger.info("Starting portfolio creation...")
    
    if not analysis_results:
        logger.error("No analysis results available for portfolio creation")
        return
    
    portfolio_results = {}
    
    for regime_col, result in analysis_results.items():
        try:
            data_for_analysis = result['data_for_analysis']
            regime_performance = result['regime_performance']
            
            # Extract asset returns
            asset_returns = data_for_analysis.drop(columns=[regime_col])
            
            # Create an equal-weight portfolio for comparison
            logger.info(f"Creating equal-weight portfolio for {regime_col}...")
            equal_weight_returns = create_equal_weight_portfolio(asset_returns)
            
            equal_weight_metrics = calculate_portfolio_metrics(equal_weight_returns)
            
            # Create a regime-based portfolio
            logger.info(f"Creating regime-based portfolio for {regime_col}...")
            regime_portfolio = create_regime_based_portfolio(
                asset_returns,
                data_for_analysis[[regime_col]],
                regime_performance
            )
            
            regime_portfolio_metrics = calculate_portfolio_metrics(
                regime_portfolio['portfolio_return']
            )
            
            # Save results
            portfolio_results[regime_col] = {
                'equal_weight': {
                    'returns': equal_weight_returns,
                    'metrics': equal_weight_metrics
                },
                'regime_based': {
                    'portfolio': regime_portfolio,
                    'metrics': regime_portfolio_metrics
                }
            }
            
            # Save portfolio returns
            save_data(
                pd.DataFrame({
                    'equal_weight': equal_weight_returns,
                    'regime_based': regime_portfolio['portfolio_return']
                }),
                os.path.join(config.PROCESSED_DATA_DIR, f'portfolio_returns_{regime_col}.csv')
            )
            
            # Compare portfolio performance
            logger.info(f"\nPortfolio comparison for {regime_col}:")
            logger.info("Equal Weight Portfolio:")
            for metric, value in equal_weight_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("\nRegime-Based Portfolio:")
            for metric, value in regime_portfolio_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Plot portfolio comparison
            try:
                plt.figure(figsize=(12, 8))
                
                # Calculate cumulative returns
                cum_equal_weight = (1 + equal_weight_returns).cumprod()
                cum_regime_based = (1 + regime_portfolio['portfolio_return']).cumprod()
                
                plt.plot(cum_equal_weight, label='Equal Weight')
                plt.plot(cum_regime_based, label='Regime-Based')
                
                plt.title(f'Portfolio Comparison ({regime_col})', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Cumulative Return', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(config.OUTPUT_DIR, f'portfolio_comparison_{regime_col}.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting portfolio comparison: {e}")
        
        except Exception as e:
            logger.error(f"Error creating portfolios for {regime_col}: {e}")
    
    logger.info("Portfolio creation completed")
    return portfolio_results

def main():
    """Main function to run the analysis pipeline."""
    logger.info("Starting macro regime analysis pipeline...")
    
    # Step 1: Fetch and process data
    macro_data, asset_returns = fetch_and_process_data()
    
    # Step 2: Classify regimes
    macro_data_with_regimes = classify_regimes(macro_data)
    # Ensure we pass a DataFrame with regime columns downstream
    if isinstance(macro_data_with_regimes, pd.Series):
        name = macro_data_with_regimes.name or "Regime"
        tmp_df = macro_data.copy()
        tmp_df[name] = macro_data_with_regimes
        macro_data_with_regimes = tmp_df
    
    # Step 3: Analyze regime performance
    analysis_results = analyze_regime_performance(macro_data_with_regimes, asset_returns)
    
    # Step 4: Create visualizations
    create_visualizations(macro_data_with_regimes, analysis_results)
    
    # Step 5: Create and evaluate portfolios
    portfolio_results = create_portfolios(analysis_results)
    
    logger.info("Macro regime analysis pipeline completed successfully")

if __name__ == "__main__":
    main() 

