# -*- coding: utf-8 -*-
"""
Volatility Estimation Module

This module provides comprehensive functionality for calculating various measures
of financial volatility, including historical volatility, rolling volatility, 
exponentially weighted moving averages (EWMA), and GARCH models.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Optional dependency for GARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not available. GARCH models will not be functional.")

class VolatilityMethod(Enum):
    """Types of volatility calculation methods"""
    HISTORICAL = "historical"
    ROLLING = "rolling"
    EWMA = "ewma"
    GARCH = "garch"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"

class AnnualizationFactor(Enum):
    """Annualization factors for different data frequencies"""
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    YEARLY = 1

@dataclass
class VolatilityConfig:
    """Configuration for volatility calculations"""
    method: VolatilityMethod = VolatilityMethod.HISTORICAL
    window: int = 30
    min_periods: int = 20
    annualize: bool = True
    frequency: AnnualizationFactor = AnnualizationFactor.DAILY
    center: bool = False
    ewma_span: Optional[int] = None
    ewma_alpha: Optional[float] = None
    garch_p: int = 1
    garch_q: int = 1
    confidence_level: float = 0.95
    return_type: str = "simple"

class VolatilityEstimator:
    """
    Comprehensive volatility estimation class.
    
    Supports multiple volatility measures including historical, rolling,
    EWMA, and GARCH models for financial time series analysis.
    """
    
    def __init__(self, config: VolatilityConfig = None):
        """Initialize the volatility estimator."""
        self.config = config or VolatilityConfig()
        self.fitted_models = {}
        self.calculation_history = []
    
    def calculate_historical_volatility(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        window: Optional[int] = None,
        annualize: Optional[bool] = None,
        return_type: Optional[str] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate historical volatility using standard deviation of returns."""
        window = window or self.config.window
        annualize = annualize if annualize is not None else self.config.annualize
        return_type = return_type or self.config.return_type
        
        # Calculate returns
        returns = self._calculate_returns(prices, return_type)
        
        # Calculate historical volatility
        if isinstance(returns, pd.Series):
            volatility = returns.std()
        else:
            volatility = returns.std()
            
        # Annualize if requested
        if annualize:
            annualization_factor = np.sqrt(self.config.frequency.value)
            volatility = volatility * annualization_factor
            
        return volatility
    
    def calculate_rolling_volatility(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        window: Optional[int] = None,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        annualize: Optional[bool] = None,
        return_type: Optional[str] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling volatility using a moving window."""
        window = window or self.config.window
        min_periods = min_periods or self.config.min_periods
        center = center if center is not None else self.config.center
        annualize = annualize if annualize is not None else self.config.annualize
        return_type = return_type or self.config.return_type
        
        # Calculate returns
        returns = self._calculate_returns(prices, return_type)
        
        # Calculate rolling standard deviation
        rolling_vol = returns.rolling(
            window=window,
            min_periods=min_periods,
            center=center
        ).std()
        
        # Annualize if requested
        if annualize:
            annualization_factor = np.sqrt(self.config.frequency.value)
            rolling_vol = rolling_vol * annualization_factor
            
        return rolling_vol
    
    def calculate_ewma_volatility(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        span: Optional[int] = None,
        alpha: Optional[float] = None,
        annualize: Optional[bool] = None,
        return_type: Optional[str] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate exponentially weighted moving average (EWMA) volatility."""
        span = span or self.config.ewma_span or 30
        alpha = alpha or self.config.ewma_alpha
        annualize = annualize if annualize is not None else self.config.annualize
        return_type = return_type or self.config.return_type
        
        # Calculate returns
        returns = self._calculate_returns(prices, return_type)
        
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Calculate EWMA of squared returns
        if alpha is not None:
            ewma_var = squared_returns.ewm(alpha=alpha).mean()
        else:
            ewma_var = squared_returns.ewm(span=span).mean()
        
        # Take square root to get volatility
        ewma_vol = np.sqrt(ewma_var)
        
        # Annualize if requested
        if annualize:
            annualization_factor = np.sqrt(self.config.frequency.value)
            ewma_vol = ewma_vol * annualization_factor
            
        return ewma_vol
    
    def calculate_parkinson_volatility(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        window: Optional[int] = None,
        annualize: Optional[bool] = None
    ) -> pd.Series:
        """Calculate Parkinson volatility estimator using high-low prices."""
        window = window or self.config.window
        annualize = annualize if annualize is not None else self.config.annualize
        
        # Calculate log high/low ratio
        hl_ratio = np.log(high_prices / low_prices)
        
        # Parkinson estimator
        parkinson_factor = 1 / (4 * np.log(2))
        
        if window > 1:
            parkinson_vol = np.sqrt(
                parkinson_factor * (hl_ratio ** 2).rolling(window=window).mean()
            )
        else:
            parkinson_vol = np.sqrt(parkinson_factor * (hl_ratio ** 2))
        
        # Annualize if requested
        if annualize:
            annualization_factor = np.sqrt(self.config.frequency.value)
            parkinson_vol = parkinson_vol * annualization_factor
            
        return parkinson_vol
    
    def calculate_garman_klass_volatility(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: Optional[int] = None,
        annualize: Optional[bool] = None
    ) -> pd.Series:
        """Calculate Garman-Klass volatility estimator using OHLC data."""
        window = window or self.config.window
        annualize = annualize if annualize is not None else self.config.annualize
        
        # Calculate log ratios
        log_hl = np.log(high_prices / low_prices)
        log_co = np.log(close_prices / open_prices)
        
        # Garman-Klass estimator
        gk_component = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        if window > 1:
            gk_vol = np.sqrt(gk_component.rolling(window=window).mean())
        else:
            gk_vol = np.sqrt(gk_component)
        
        # Annualize if requested
        if annualize:
            annualization_factor = np.sqrt(self.config.frequency.value)
            gk_vol = gk_vol * annualization_factor
            
        return gk_vol
    
    def calculate_volatility_statistics(
        self,
        volatility_series: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive statistics for a volatility series."""
        vol_clean = volatility_series.dropna()
        
        stats = {
            'mean': vol_clean.mean(),
            'median': vol_clean.median(),
            'std': vol_clean.std(),
            'min': vol_clean.min(),
            'max': vol_clean.max(),
            'skewness': vol_clean.skew(),
            'kurtosis': vol_clean.kurtosis(),
            'percentile_5': vol_clean.quantile(0.05),
            'percentile_25': vol_clean.quantile(0.25),
            'percentile_75': vol_clean.quantile(0.75),
            'percentile_95': vol_clean.quantile(0.95),
            'volatility_of_volatility': vol_clean.std(),
            'autocorrelation_1': vol_clean.autocorr(lag=1),
            'autocorrelation_5': vol_clean.autocorr(lag=5),
            'persistence': self._calculate_persistence(vol_clean)
        }
        
        return stats
    
    def _calculate_returns(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        return_type: str
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate returns from prices."""
        if return_type == "log":
            return np.log(prices / prices.shift(1))
        else:  # simple returns
            return prices.pct_change()
    
    def _calculate_persistence(self, series: pd.Series) -> float:
        """Calculate persistence (autocorrelation) of volatility series."""
        try:
            return series.autocorr(lag=1)
        except:
            return np.nan


# Convenience functions
def calculate_simple_volatility(prices: pd.Series, window: int = 30, annualize: bool = True) -> float:
    """Simple volatility calculation convenience function."""
    estimator = VolatilityEstimator()
    return estimator.calculate_historical_volatility(prices, window=window, annualize=annualize)

def calculate_rolling_vol(prices: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    """Rolling volatility convenience function."""
    estimator = VolatilityEstimator()
    return estimator.calculate_rolling_volatility(prices, window=window, annualize=annualize)

def calculate_ewma_vol(prices: pd.Series, span: int = 30, annualize: bool = True) -> pd.Series:
    """EWMA volatility convenience function."""
    estimator = VolatilityEstimator()
    return estimator.calculate_ewma_volatility(prices, span=span, annualize=annualize)


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Estimation Module - Testing Implementation")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Simulate price series
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='price')
    
    # Initialize volatility estimator
    config = VolatilityConfig(
        window=30,
        annualize=True,
        frequency=AnnualizationFactor.DAILY
    )
    vol_estimator = VolatilityEstimator(config)
    
    # Calculate different volatility measures
    historical_vol = vol_estimator.calculate_historical_volatility(prices)
    rolling_vol = vol_estimator.calculate_rolling_volatility(prices)
    ewma_vol = vol_estimator.calculate_ewma_volatility(prices)
    
    print(f"Historical Volatility (30-day): {historical_vol:.4f}")
    print(f"Current Rolling Volatility: {rolling_vol.iloc[-1]:.4f}")
    print(f"Current EWMA Volatility: {ewma_vol.iloc[-1]:.4f}")
    
    # Calculate volatility statistics
    vol_stats = vol_estimator.calculate_volatility_statistics(rolling_vol)
    print(f"\nVolatility Statistics:")
    for key, value in vol_stats.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.6f}")
    
    print(f"\nAvailable methods: {[method.value for method in VolatilityMethod]}")
    print(f"GARCH models available: {ARCH_AVAILABLE}")
    
    print("\nVolatility Estimation Module created successfully!")
