"""
Returns Calculation Module

This module provides comprehensive functionality for calculating various types of returns
from financial time series data, including simple returns, log returns, and cumulative returns.
Designed for use in macroeconomic regime analysis and portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ReturnType(Enum):
    """Types of return calculations"""
    SIMPLE = "simple"
    LOG = "log" 
    CUMULATIVE = "cumulative"
    EXCESS = "excess"

class Frequency(Enum):
    """Data frequency for annualization"""
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    YEARLY = 1

@dataclass
class ReturnConfig:
    """Configuration for return calculations"""
    return_type: ReturnType = ReturnType.SIMPLE
    periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60, 120])
    handle_missing: str = "skip"  # 'skip', 'interpolate', 'forward_fill'
    min_periods: int = 1
    annualize: bool = False
    frequency: Frequency = Frequency.DAILY
    risk_free_rate: Optional[Union[float, pd.Series]] = None
    numerical_precision: int = 8

class ReturnsCalculator:
    """
    Comprehensive returns calculator for financial time series data.
    
    Supports multiple return types with robust handling of edge cases,
    missing data, and numerical stability considerations.
    """
    
    def __init__(self, config: ReturnConfig = None):
        """
        Initialize the returns calculator.
        
        Args:
            config: Configuration object for return calculations
        """
        self.config = config or ReturnConfig()
        self._cache = {}
        
    def calculate_simple_returns(
        self, 
        prices: Union[pd.Series, pd.DataFrame], 
        periods: Union[int, List[int]] = None,
        handle_missing: str = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate simple returns: (P_t / P_{t-n}) - 1
        
        Args:
            prices: Price series or DataFrame with price columns
            periods: Number of periods for return calculation (default: [1])
            handle_missing: How to handle missing values ('skip', 'interpolate', 'forward_fill')
            
        Returns:
            Simple returns series or DataFrame
        """
        periods = periods or [1]
        if isinstance(periods, int):
            periods = [periods]
        handle_missing = handle_missing or self.config.handle_missing
        
        # Handle missing data according to strategy
        prices_cleaned = self._handle_missing_data(prices, handle_missing)
        
        if isinstance(prices_cleaned, pd.Series):
            return self._calculate_single_series_simple_returns(prices_cleaned, periods)
        else:
            return self._calculate_dataframe_simple_returns(prices_cleaned, periods)
    
    def calculate_log_returns(
        self, 
        prices: Union[pd.Series, pd.DataFrame], 
        periods: Union[int, List[int]] = None,
        handle_missing: str = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate log returns: ln(P_t / P_{t-n})
        
        Args:
            prices: Price series or DataFrame with price columns
            periods: Number of periods for return calculation (default: [1])  
            handle_missing: How to handle missing values
            
        Returns:
            Log returns series or DataFrame
        """
        periods = periods or [1]
        if isinstance(periods, int):
            periods = [periods]
        handle_missing = handle_missing or self.config.handle_missing
        
        # Handle missing data and zero/negative prices
        prices_cleaned = self._handle_missing_data(prices, handle_missing)
        prices_cleaned = self._handle_non_positive_prices(prices_cleaned)
        
        if isinstance(prices_cleaned, pd.Series):
            return self._calculate_single_series_log_returns(prices_cleaned, periods)
        else:
            return self._calculate_dataframe_log_returns(prices_cleaned, periods)
    
    def calculate_cumulative_returns(
        self, 
        returns: Union[pd.Series, pd.DataFrame],
        compound: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate cumulative returns from a return series.
        
        Args:
            returns: Return series or DataFrame
            compound: Whether to use compound returns (True) or simple sum (False)
            
        Returns:
            Cumulative returns series or DataFrame
        """
        if compound:
            # Compound returns: (1 + r1)(1 + r2)...(1 + rn) - 1
            return (1 + returns).cumprod() - 1
        else:
            # Simple cumulative sum
            return returns.cumsum()
    
    def calculate_excess_returns(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        risk_free_rate: Union[float, pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate excess returns over risk-free rate.
        
        Args:
            returns: Return series or DataFrame
            risk_free_rate: Risk-free rate (constant or time series)
            
        Returns:
            Excess returns series or DataFrame
        """
        rf_rate = risk_free_rate or self.config.risk_free_rate
        
        if rf_rate is None:
            warnings.warn("No risk-free rate provided, returning original returns")
            return returns
        
        return returns - rf_rate
    
    def calculate_rolling_returns(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        window: int,
        return_type: ReturnType = None,
        min_periods: int = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling returns over a specified window.
        
        Args:
            prices: Price series or DataFrame
            window: Rolling window size
            return_type: Type of return calculation
            min_periods: Minimum number of observations required
            
        Returns:
            Rolling returns series or DataFrame
        """
        return_type = return_type or self.config.return_type
        min_periods = min_periods or self.config.min_periods
        
        if return_type == ReturnType.SIMPLE:
            # Rolling simple return: (P_t / P_{t-window}) - 1
            return (prices / prices.shift(window)) - 1
        elif return_type == ReturnType.LOG:
            # Rolling log return: ln(P_t / P_{t-window})
            prices_clean = self._handle_non_positive_prices(prices)
            return np.log(prices_clean / prices_clean.shift(window))
        else:
            raise ValueError(f"Unsupported return type for rolling calculation: {return_type}")
    
    def annualize_returns(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        frequency: Frequency = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Annualize returns based on data frequency.
        
        Args:
            returns: Return series or DataFrame
            frequency: Data frequency for annualization
            
        Returns:
            Annualized returns
        """
        freq = frequency or self.config.frequency
        return returns * freq.value
    
    def calculate_return_statistics(
        self,
        returns: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive return statistics.
        
        Args:
            returns: Return series or DataFrame
            
        Returns:
            Dictionary with return statistics
        """
        stats = {}
        
        if isinstance(returns, pd.Series):
            returns_clean = returns.dropna()
            
            stats['count'] = len(returns_clean)
            stats['mean'] = returns_clean.mean()
            stats['std'] = returns_clean.std()
            stats['min'] = returns_clean.min()
            stats['max'] = returns_clean.max()
            stats['skew'] = returns_clean.skew()
            stats['kurtosis'] = returns_clean.kurtosis()
            stats['sharpe_ratio'] = returns_clean.mean() / returns_clean.std() if returns_clean.std() > 0 else np.nan
            
            # Percentiles
            stats['p5'] = returns_clean.quantile(0.05)
            stats['p25'] = returns_clean.quantile(0.25)
            stats['p50'] = returns_clean.quantile(0.50)
            stats['p75'] = returns_clean.quantile(0.75)
            stats['p95'] = returns_clean.quantile(0.95)
            
        else:
            # For DataFrame, calculate stats for each column
            stats = {}
            for col in returns.columns:
                stats[col] = self.calculate_return_statistics(returns[col])
        
        return stats
    
    def _handle_missing_data(
        self, 
        data: Union[pd.Series, pd.DataFrame], 
        method: str
    ) -> Union[pd.Series, pd.DataFrame]:
        """Handle missing data according to specified method."""
        if method == "skip":
            return data  # Let pandas handle NaN in calculations
        elif method == "interpolate":
            return data.interpolate(method='linear')
        elif method == "forward_fill":
            return data.ffill()
        else:
            raise ValueError(f"Unknown missing data handling method: {method}")
    
    def _handle_non_positive_prices(
        self, 
        prices: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Handle zero or negative prices for log calculations."""
        if isinstance(prices, pd.Series):
            # Replace non-positive values with small positive number
            prices_clean = prices.copy()
            non_positive_mask = prices_clean <= 0
            if non_positive_mask.any():
                min_positive = prices_clean[prices_clean > 0].min()
                replacement_value = min_positive * 1e-6 if pd.notna(min_positive) else 1e-8
                prices_clean[non_positive_mask] = replacement_value
                warnings.warn(f"Replaced {non_positive_mask.sum()} non-positive prices with {replacement_value}")
            return prices_clean
        else:
            # Handle DataFrame column by column
            prices_clean = prices.copy()
            for col in prices_clean.columns:
                prices_clean[col] = self._handle_non_positive_prices(prices_clean[col])
            return prices_clean
    
    def _calculate_single_series_simple_returns(
        self, 
        prices: pd.Series, 
        periods: List[int]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate simple returns for a single price series."""
        if len(periods) == 1:
            period = periods[0]
            return (prices / prices.shift(period)) - 1
        else:
            result = pd.DataFrame(index=prices.index)
            for period in periods:
                col_name = f"simple_return_{period}d"
                result[col_name] = (prices / prices.shift(period)) - 1
            return result
    
    def _calculate_dataframe_simple_returns(
        self, 
        prices: pd.DataFrame, 
        periods: List[int]
    ) -> pd.DataFrame:
        """Calculate simple returns for a DataFrame of prices."""
        results = {}
        
        for col in prices.columns:
            if len(periods) == 1:
                period = periods[0]
                results[f"{col}_return_{period}d"] = (prices[col] / prices[col].shift(period)) - 1
            else:
                for period in periods:
                    results[f"{col}_return_{period}d"] = (prices[col] / prices[col].shift(period)) - 1
        
        return pd.DataFrame(results, index=prices.index)
    
    def _calculate_single_series_log_returns(
        self, 
        prices: pd.Series, 
        periods: List[int]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate log returns for a single price series."""
        if len(periods) == 1:
            period = periods[0]
            return np.log(prices / prices.shift(period))
        else:
            result = pd.DataFrame(index=prices.index)
            for period in periods:
                col_name = f"log_return_{period}d"
                result[col_name] = np.log(prices / prices.shift(period))
            return result
    
    def _calculate_dataframe_log_returns(
        self, 
        prices: pd.DataFrame, 
        periods: List[int]
    ) -> pd.DataFrame:
        """Calculate log returns for a DataFrame of prices."""
        results = {}
        
        for col in prices.columns:
            if len(periods) == 1:
                period = periods[0]
                results[f"{col}_log_return_{period}d"] = np.log(prices[col] / prices[col].shift(period))
            else:
                for period in periods:
                    results[f"{col}_log_return_{period}d"] = np.log(prices[col] / prices[col].shift(period))
        
        return pd.DataFrame(results, index=prices.index)


def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    return_type: str = "simple",
    periods: Union[int, List[int]] = 1,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convenience function for calculating returns.
    
    Args:
        prices: Price data
        return_type: Type of returns ('simple', 'log', 'cumulative')
        periods: Periods for calculation
        **kwargs: Additional arguments for ReturnsCalculator
        
    Returns:
        Calculated returns
    """
    calculator = ReturnsCalculator()
    
    if return_type == "simple":
        return calculator.calculate_simple_returns(prices, periods, **kwargs)
    elif return_type == "log":
        return calculator.calculate_log_returns(prices, periods, **kwargs)
    elif return_type == "cumulative":
        # First calculate simple returns, then cumulative
        simple_returns = calculator.calculate_simple_returns(prices, periods, **kwargs)
        return calculator.calculate_cumulative_returns(simple_returns)
    else:
        raise ValueError(f"Unknown return type: {return_type}")


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    print("Returns Calculator Example")
    print("=" * 50)
    
    # Download sample data
    ticker = "SPY"
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    prices = data['Adj Close']
    
    # Initialize calculator
    config = ReturnConfig(
        periods=[1, 5, 20, 60],
        return_type=ReturnType.SIMPLE,
        handle_missing="forward_fill"
    )
    calculator = ReturnsCalculator(config)
    
    # Calculate different types of returns
    simple_returns = calculator.calculate_simple_returns(prices, [1, 5, 20])
    log_returns = calculator.calculate_log_returns(prices, [1, 5, 20])
    
    print(f"Simple Returns (last 5 rows):")
    print(simple_returns.tail())
    print(f"\nLog Returns (last 5 rows):")
    print(log_returns.tail())
    
    # Calculate statistics
    stats = calculator.calculate_return_statistics(simple_returns['simple_return_1d'])
    print(f"\nReturn Statistics for 1-day simple returns:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}") 