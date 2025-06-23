# -*- coding: utf-8 -*-
"""
Technical Indicators Library

This module provides comprehensive functionality for calculating technical analysis
indicators commonly used in financial markets. These indicators are essential for
trend analysis, momentum assessment, and market regime identification.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MovingAverageType(Enum):
    """Types of moving averages"""
    SIMPLE = "simple"
    EXPONENTIAL = "exponential"
    WEIGHTED = "weighted"
    HULL = "hull"

class SignalType(Enum):
    """Types of trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    short_period: int = 12
    long_period: int = 26
    signal_period: int = 9
    rsi_period: int = 14
    bb_period: int = 20
    bb_std_dev: float = 2.0
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    atr_period: int = 14
    adx_period: int = 14
    cci_period: int = 20
    williams_period: int = 14

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.
    
    Provides a wide range of technical analysis indicators with proper
    mathematical formulations and configurable parameters.
    """
    
    def __init__(self, config: IndicatorConfig = None):
        """Initialize the technical indicators calculator."""
        self.config = config or IndicatorConfig()
        self.calculation_cache = {}
        
    def calculate_sma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Formula: SMA = (P1 + P2 + ... + Pn) / n
        """
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int = 20, alpha: Optional[float] = None) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Formula: EMA = (Price * α) + (Previous EMA * (1 - α))
        Where α = 2 / (period + 1)
        """
        if alpha is None:
            alpha = 2 / (period + 1)
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_wma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Weighted Moving Average (WMA).
        
        Formula: WMA = (P1*1 + P2*2 + ... + Pn*n) / (1 + 2 + ... + n)
        """
        weights = np.arange(1, period + 1)
        
        def wma_calc(x):
            return np.dot(x, weights) / weights.sum()
        
        return data.rolling(window=period).apply(wma_calc, raw=True)
    
    def calculate_rsi(self, data: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Formula: RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss
        """
        period = period or self.config.rsi_period
        
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast_period: int = None, 
                      slow_period: int = None, signal_period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Formula: 
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(MACD, 9)
        Histogram = MACD Line - Signal Line
        """
        fast_period = fast_period or self.config.short_period
        slow_period = slow_period or self.config.long_period
        signal_period = signal_period or self.config.signal_period
        
        # Calculate MACD line
        ema_fast = self.calculate_ema(data, fast_period)
        ema_slow = self.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = None, 
                                 std_dev: float = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Formula:
        Middle Band = SMA(period)
        Upper Band = SMA + (std_dev * Standard Deviation)
        Lower Band = SMA - (std_dev * Standard Deviation)
        """
        period = period or self.config.bb_period
        std_dev = std_dev or self.config.bb_std_dev
        
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(data, period)
        
        # Calculate standard deviation
        rolling_std = data.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'bandwidth': (upper_band - lower_band) / middle_band * 100,
            'percent_b': (data - lower_band) / (upper_band - lower_band)
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = None, d_period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Formula:
        %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        %D = SMA(%K, period)
        """
        k_period = k_period or self.config.stoch_k_period
        d_period = d_period or self.config.stoch_d_period
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = None) -> pd.Series:
        """
        Calculate Williams %R.
        
        Formula: %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
        """
        period = period or self.config.williams_period
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = None) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Formula:
        True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        ATR = SMA(True Range, period)
        """
        period = period or self.config.atr_period
        
        # Calculate true range components
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        # Calculate true range (maximum of the three)
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = None) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Formula:
        Typical Price = (High + Low + Close) / 3
        CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
        """
        period = period or self.config.cci_period
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate SMA of typical price
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def generate_ma_crossover_signals(self, data: pd.Series, fast_period: int = 12,
                                    slow_period: int = 26) -> pd.Series:
        """Generate moving average crossover signals."""
        fast_ma = self.calculate_ema(data, fast_period)
        slow_ma = self.calculate_ema(data, slow_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy signal
        signals[fast_ma < slow_ma] = -1  # Sell signal
        
        # Only signal on crossovers (changes)
        signal_changes = signals.diff()
        crossover_signals = pd.Series(0, index=data.index)
        crossover_signals[signal_changes == 2] = 1   # Bull crossover
        crossover_signals[signal_changes == -2] = -1  # Bear crossover
        
        return crossover_signals
    
    def generate_rsi_signals(self, data: pd.Series, period: int = None,
                           oversold: float = 30, overbought: float = 70) -> pd.Series:
        """Generate RSI-based signals."""
        rsi = self.calculate_rsi(data, period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal when RSI crosses above oversold
        signals[(rsi.shift(1) <= oversold) & (rsi > oversold)] = 1
        
        # Sell signal when RSI crosses below overbought
        signals[(rsi.shift(1) >= overbought) & (rsi < overbought)] = -1
        
        return signals
    
    def calculate_multiple_indicators(self, high: pd.Series, low: pd.Series, 
                                    close: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate multiple technical indicators at once."""
        indicators = pd.DataFrame(index=close.index)
        
        # Moving averages
        indicators['sma_20'] = self.calculate_sma(close, 20)
        indicators['ema_12'] = self.calculate_ema(close, 12)
        indicators['ema_26'] = self.calculate_ema(close, 26)
        
        # Momentum indicators
        indicators['rsi'] = self.calculate_rsi(close)
        
        # MACD
        macd = self.calculate_macd(close)
        indicators['macd'] = macd['macd']
        indicators['macd_signal'] = macd['signal']
        indicators['macd_histogram'] = macd['histogram']
        
        # Bollinger Bands
        bb = self.calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb['upper']
        indicators['bb_middle'] = bb['middle']
        indicators['bb_lower'] = bb['lower']
        
        # Volatility
        indicators['atr'] = self.calculate_atr(high, low, close)
        
        # Stochastic
        stoch = self.calculate_stochastic(high, low, close)
        indicators['stoch_k'] = stoch['stoch_k']
        indicators['stoch_d'] = stoch['stoch_d']
        
        # Williams %R
        indicators['williams_r'] = self.calculate_williams_r(high, low, close)
        
        # CCI
        indicators['cci'] = self.calculate_cci(high, low, close)
        
        return indicators


# Convenience functions
def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    indicators = TechnicalIndicators()
    return indicators.calculate_rsi(data, period)

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD."""
    indicators = TechnicalIndicators()
    return indicators.calculate_macd(data, fast, slow, signal)

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    indicators = TechnicalIndicators()
    return indicators.calculate_bollinger_bands(data, period, std_dev)


# Example usage and testing
if __name__ == "__main__":
    print("Technical Indicators Library - Testing Implementation")
    print("=" * 60)
    
    # Create sample OHLC data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Simulate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    noise = np.random.normal(0, 0.005, len(dates))
    high = pd.Series(prices * (1 + np.abs(noise)), index=dates, name='high')
    low = pd.Series(prices * (1 - np.abs(noise)), index=dates, name='low')
    close = pd.Series(prices, index=dates, name='close')
    
    # Initialize technical indicators
    config = IndicatorConfig()
    tech_indicators = TechnicalIndicators(config)
    
    # Test individual indicators
    print(f"Testing individual indicators...")
    sma_20 = tech_indicators.calculate_sma(close, 20)
    ema_12 = tech_indicators.calculate_ema(close, 12)
    rsi = tech_indicators.calculate_rsi(close)
    macd = tech_indicators.calculate_macd(close)
    bb = tech_indicators.calculate_bollinger_bands(close)
    
    print(f"✓ SMA(20) current value: {sma_20.iloc[-1]:.2f}")
    print(f"✓ EMA(12) current value: {ema_12.iloc[-1]:.2f}")
    print(f"✓ RSI current value: {rsi.iloc[-1]:.2f}")
    print(f"✓ MACD current value: {macd['macd'].iloc[-1]:.4f}")
    print(f"✓ Bollinger Upper Band: {bb['upper'].iloc[-1]:.2f}")
    
    # Test multiple indicators
    all_indicators = tech_indicators.calculate_multiple_indicators(high, low, close)
    print(f"\n✓ Multiple indicators calculated: {all_indicators.shape[1]} indicators")
    print(f"✓ Data points: {all_indicators.shape[0]:,}")
    
    # Test signal generation
    ma_signals = tech_indicators.generate_ma_crossover_signals(close)
    rsi_signals = tech_indicators.generate_rsi_signals(close)
    
    print(f"\n✓ MA crossover signals generated: {ma_signals.abs().sum()} total signals")
    print(f"✓ RSI signals generated: {rsi_signals.abs().sum()} total signals")
    
    print(f"\nTechnical Indicators Library created successfully!")
    print(f"Available indicators: SMA, EMA, WMA, RSI, MACD, Bollinger Bands,")
    print(f"Stochastic, Williams %R, ATR, CCI, and signal generation functions.")
