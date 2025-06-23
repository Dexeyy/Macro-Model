"""
Enhanced Yahoo Finance Data Fetching Module

This module provides a comprehensive interface to Yahoo Finance data
with advanced features including multiple data types, time intervals,
financial indicators, robust error handling, and quota management.

Features:
- Multiple data types (prices, dividends, splits, financials, etc.)
- Various time intervals (1m, 5m, 1h, 1d, 1wk, 1mo, etc.)
- Financial indicators and metrics
- Batch operations for multiple tickers
- Comprehensive error handling and retry logic
- Rate limiting and quota management
- Detailed logging and monitoring
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
import yfinance as yf
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooInterval(Enum):
    """Yahoo Finance data interval options."""
    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    NINETY_MINUTES = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class YahooPeriod(Enum):
    """Yahoo Finance period options for historical data."""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YTD = "ytd"
    MAX = "max"


class DataType(Enum):
    """Types of data available from Yahoo Finance."""
    HISTORY = "history"
    DIVIDENDS = "dividends"
    SPLITS = "splits"
    FINANCIALS = "financials"
    BALANCE_SHEET = "balance_sheet"
    CASHFLOW = "cashflow"
    INFO = "info"
    RECOMMENDATIONS = "recommendations"
    CALENDAR = "calendar"
    OPTIONS = "options"


@dataclass
class YahooRequestParams:
    """Parameters for Yahoo Finance API requests."""
    ticker: str
    data_type: DataType = DataType.HISTORY
    period: Optional[YahooPeriod] = None
    interval: YahooInterval = YahooInterval.ONE_DAY
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    auto_adjust: bool = True
    prepost: bool = True


@dataclass
class TickerInfo:
    """Information about a ticker symbol."""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float]
    enterprise_value: Optional[float]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    enterprise_to_revenue: Optional[float]
    enterprise_to_ebitda: Optional[float]
    profit_margins: Optional[float]
    operating_margins: Optional[float]
    return_on_assets: Optional[float]
    return_on_equity: Optional[float]
    revenue: Optional[float]
    revenue_per_share: Optional[float]
    quarterly_revenue_growth: Optional[float]
    gross_profits: Optional[float]
    ebitda: Optional[float]
    net_income_to_common: Optional[float]
    trailing_eps: Optional[float]
    forward_eps: Optional[float]
    quarterly_earnings_growth: Optional[float]
    earnings_quarterly_growth: Optional[float]
    most_recent_quarter: Optional[str]
    earnings_date: Optional[str]
    dividend_rate: Optional[float]
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]
    five_year_avg_dividend_yield: Optional[float]
    beta: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    fifty_day_average: Optional[float]
    two_hundred_day_average: Optional[float]
    avg_volume: Optional[float]
    avg_volume_10days: Optional[float]
    shares_outstanding: Optional[float]
    float_shares: Optional[float]
    held_percent_insiders: Optional[float]
    held_percent_institutions: Optional[float]
    short_ratio: Optional[float]
    short_percent_of_float: Optional[float]
    book_value: Optional[float]
    price_to_book: Optional[float]
    last_fiscal_year_end: Optional[str]
    next_fiscal_year_end: Optional[str]
    most_recent_quarter: Optional[str]
    earnings_quarterly_growth: Optional[float]
    net_income_to_common: Optional[float]
    trailing_eps: Optional[float]
    forward_eps: Optional[float]
    peg_ratio: Optional[float]


class RateLimiter:
    """Thread-safe rate limiter for Yahoo Finance requests."""
    
    def __init__(self, max_calls: int = 2000, time_window: int = 3600):
        """
        Initialize rate limiter for Yahoo Finance.
        
        Args:
            max_calls: Maximum number of calls allowed in time window (default: 2000/hour)
            time_window: Time window in seconds (default: 3600 seconds = 1 hour)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0]) + 1
                if sleep_time > 0:
                    logger.info(f"Yahoo Finance rate limit reached. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    # Clean up old calls again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # Record this call
            self.calls.append(now)


class EnhancedYahooFinanceClient:
    """Enhanced Yahoo Finance client with advanced features."""
    
    def __init__(self, rate_limit_calls: int = 2000):
        """
        Initialize the enhanced Yahoo Finance client.
        
        Args:
            rate_limit_calls: Maximum API calls per hour (default: 2000)
        """
        self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, time_window=3600)
        self._ticker_cache: Dict[str, yf.Ticker] = {}
        self._info_cache: Dict[str, TickerInfo] = {}
        
        logger.info("Enhanced Yahoo Finance client initialized")
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a ticker object with caching."""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
    
    def get_ticker_info(self, symbol: str, use_cache: bool = True) -> Optional[TickerInfo]:
        """
        Get comprehensive information about a ticker.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')
            use_cache: Whether to use cached information if available
            
        Returns:
            TickerInfo object or None if not found
        """
        if use_cache and symbol in self._info_cache:
            return self._info_cache[symbol]
        
        try:
            self.rate_limiter.wait_if_needed()
            
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            if not info:
                logger.warning(f"No info available for {symbol}")
                return None
            
            ticker_info = TickerInfo(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap'),
                enterprise_value=info.get('enterpriseValue'),
                trailing_pe=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                enterprise_to_revenue=info.get('enterpriseToRevenue'),
                enterprise_to_ebitda=info.get('enterpriseToEbitda'),
                profit_margins=info.get('profitMargins'),
                operating_margins=info.get('operatingMargins'),
                return_on_assets=info.get('returnOnAssets'),
                return_on_equity=info.get('returnOnEquity'),
                revenue=info.get('totalRevenue'),
                revenue_per_share=info.get('revenuePerShare'),
                quarterly_revenue_growth=info.get('quarterlyRevenueGrowth'),
                gross_profits=info.get('grossProfits'),
                ebitda=info.get('ebitda'),
                net_income_to_common=info.get('netIncomeToCommon'),
                trailing_eps=info.get('trailingEps'),
                forward_eps=info.get('forwardEps'),
                quarterly_earnings_growth=info.get('quarterlyEarningsGrowth'),
                earnings_quarterly_growth=info.get('earningsQuarterlyGrowth'),
                most_recent_quarter=info.get('mostRecentQuarter'),
                earnings_date=info.get('earningsDate'),
                dividend_rate=info.get('dividendRate'),
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),
                five_year_avg_dividend_yield=info.get('fiveYearAvgDividendYield'),
                beta=info.get('beta'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
                fifty_day_average=info.get('fiftyDayAverage'),
                two_hundred_day_average=info.get('twoHundredDayAverage'),
                avg_volume=info.get('averageVolume'),
                avg_volume_10days=info.get('averageVolume10days'),
                shares_outstanding=info.get('sharesOutstanding'),
                float_shares=info.get('floatShares'),
                held_percent_insiders=info.get('heldPercentInsiders'),
                held_percent_institutions=info.get('heldPercentInstitutions'),
                short_ratio=info.get('shortRatio'),
                short_percent_of_float=info.get('shortPercentOfFloat'),
                book_value=info.get('bookValue'),
                last_fiscal_year_end=info.get('lastFiscalYearEnd'),
                next_fiscal_year_end=info.get('nextFiscalYearEnd')
            )
            
            if use_cache:
                self._info_cache[symbol] = ticker_info
            
            logger.info(f"Retrieved info for {symbol}: {ticker_info.name}")
            return ticker_info
            
        except Exception as e:
            logger.error(f"Error getting ticker info for {symbol}: {e}")
            return None
    
    def fetch_historical_data(self, params: YahooRequestParams, 
                            max_retries: int = 3, retry_delay: float = 1.0) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker.
        
        Args:
            params: Request parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            DataFrame with historical data or None if failed
        """
        for attempt in range(max_retries + 1):
            try:
                self.rate_limiter.wait_if_needed()
                
                ticker = self._get_ticker(params.ticker)
                
                # Build parameters for yfinance
                kwargs = {
                    'interval': params.interval.value,
                    'auto_adjust': params.auto_adjust,
                    'prepost': params.prepost
                }
                
                # Use either period or start/end dates
                if params.period:
                    kwargs['period'] = params.period.value
                else:
                    if params.start_date:
                        kwargs['start'] = params.start_date
                    if params.end_date:
                        kwargs['end'] = params.end_date
                
                # Fetch the data
                data = ticker.history(**kwargs)
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {params.ticker}: {len(data)} observations")
                    return data
                else:
                    logger.warning(f"No data returned for {params.ticker}")
                    return None
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {params.ticker}: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for {params.ticker}")
                    return None
        
        return None
    
    def fetch_dividends(self, symbol: str, period: Optional[YahooPeriod] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Fetch dividend data for a ticker.
        
        Args:
            symbol: Ticker symbol
            period: Period for data (if not using start/end dates)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Series with dividend data
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            ticker = self._get_ticker(symbol)
            
            if period:
                dividends = ticker.dividends.tail(self._period_to_days(period))
            else:
                dividends = ticker.dividends
                if start_date:
                    dividends = dividends[dividends.index >= start_date]
                if end_date:
                    dividends = dividends[dividends.index <= end_date]
            
            if not dividends.empty:
                logger.info(f"Successfully fetched dividends for {symbol}: {len(dividends)} payments")
                return dividends
            else:
                logger.warning(f"No dividend data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return None
    
    def fetch_splits(self, symbol: str, period: Optional[YahooPeriod] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Fetch stock split data for a ticker.
        
        Args:
            symbol: Ticker symbol
            period: Period for data (if not using start/end dates)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Series with split data
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            ticker = self._get_ticker(symbol)
            
            if period:
                splits = ticker.splits.tail(self._period_to_days(period))
            else:
                splits = ticker.splits
                if start_date:
                    splits = splits[splits.index >= start_date]
                if end_date:
                    splits = splits[splits.index <= end_date]
            
            if not splits.empty:
                logger.info(f"Successfully fetched splits for {symbol}: {len(splits)} splits")
                return splits
            else:
                logger.info(f"No split data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return None
    
    def fetch_financials(self, symbol: str, statement_type: str = "income") -> Optional[pd.DataFrame]:
        """
        Fetch financial statements for a ticker.
        
        Args:
            symbol: Ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cashflow')
            
        Returns:
            DataFrame with financial data
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            ticker = self._get_ticker(symbol)
            
            if statement_type.lower() == "income":
                financials = ticker.financials
            elif statement_type.lower() == "balance":
                financials = ticker.balance_sheet
            elif statement_type.lower() == "cashflow":
                financials = ticker.cashflow
            else:
                raise ValueError(f"Invalid statement_type: {statement_type}")
            
            if financials is not None and not financials.empty:
                logger.info(f"Successfully fetched {statement_type} statement for {symbol}")
                return financials
            else:
                logger.warning(f"No {statement_type} data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {statement_type} for {symbol}: {e}")
            return None
    
    def fetch_multiple_tickers(self, ticker_params: List[YahooRequestParams]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers efficiently.
        
        Args:
            ticker_params: List of request parameters for each ticker
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        
        logger.info(f"Fetching data for {len(ticker_params)} tickers...")
        
        for i, params in enumerate(ticker_params, 1):
            logger.info(f"Fetching ticker {i}/{len(ticker_params)}: {params.ticker}")
            
            data = self.fetch_historical_data(params)
            if data is not None:
                results[params.ticker] = data
            
            # Small delay to be respectful
            if i < len(ticker_params):
                time.sleep(0.1)
        
        logger.info(f"Successfully fetched {len(results)}/{len(ticker_params)} tickers")
        return results
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.
        
        Returns:
            Dictionary with rate limit information
        """
        with self.rate_limiter.lock:
            now = time.time()
            recent_calls = [call for call in self.rate_limiter.calls if now - call < 3600]
            
            return {
                "calls_in_last_hour": len(recent_calls),
                "max_calls_per_hour": self.rate_limiter.max_calls,
                "remaining_calls": max(0, self.rate_limiter.max_calls - len(recent_calls)),
                "reset_time": max(recent_calls) + 3600 if recent_calls else now
            }
    
    def _period_to_days(self, period: YahooPeriod) -> int:
        """Convert period enum to approximate number of days."""
        period_map = {
            YahooPeriod.ONE_DAY: 1,
            YahooPeriod.FIVE_DAYS: 5,
            YahooPeriod.ONE_MONTH: 30,
            YahooPeriod.THREE_MONTHS: 90,
            YahooPeriod.SIX_MONTHS: 180,
            YahooPeriod.ONE_YEAR: 365,
            YahooPeriod.TWO_YEARS: 730,
            YahooPeriod.FIVE_YEARS: 1825,
            YahooPeriod.TEN_YEARS: 3650,
            YahooPeriod.YTD: 365,
            YahooPeriod.MAX: 10000
        }
        return period_map.get(period, 365)


# Convenience functions for backward compatibility
def fetch_asset_data(asset_tickers: Dict[str, str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch multiple asset price data (backward compatible function).
    
    Args:
        asset_tickers: Dictionary mapping labels to ticker symbols
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with all asset price series
    """
    client = EnhancedYahooFinanceClient()
    
    # Create request parameters for each ticker
    ticker_params = [
        YahooRequestParams(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=YahooInterval.ONE_DAY
        )
        for ticker in asset_tickers.values()
    ]
    
    # Fetch all tickers
    results = client.fetch_multiple_tickers(ticker_params)
    
    # Process and rename according to the provided labels
    processed_results = {}
    for label, ticker in asset_tickers.items():
        if ticker in results:
            data = results[ticker]
            if 'Adj Close' in data.columns:
                processed_results[label] = data['Adj Close'].rename(label)
            elif 'Close' in data.columns:
                processed_results[label] = data['Close'].rename(label)
    
    if processed_results:
        return pd.concat(processed_results.values(), axis=1)
    else:
        logger.warning("No asset data was successfully fetched")
        return None


# Common ticker symbols for different asset classes
COMMON_TICKERS = {
    # US Indices
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow_jones": "^DJI",
    "russell_2000": "^RUT",
    "vix": "^VIX",
    
    # International Indices
    "ftse_100": "^FTSE",
    "dax": "^GDAXI",
    "nikkei": "^N225",
    "hang_seng": "^HSI",
    
    # Commodities
    "gold": "GC=F",
    "silver": "SI=F",
    "oil_wti": "CL=F",
    "oil_brent": "BZ=F",
    "natural_gas": "NG=F",
    
    # Currencies
    "usd_eur": "EURUSD=X",
    "usd_jpy": "USDJPY=X",
    "usd_gbp": "GBPUSD=X",
    "dxy": "DX-Y.NYB",
    
    # Bonds
    "us_10y": "^TNX",
    "us_2y": "^IRX",
    "us_30y": "^TYX",
    
    # Crypto
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    
    # ETFs
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "gld": "GLD",
    "tlt": "TLT"
}


if __name__ == "__main__":
    # Example usage
    client = EnhancedYahooFinanceClient()
    
    # Get ticker info
    info = client.get_ticker_info("AAPL")
    if info:
        print(f"Ticker: {info.name}")
        print(f"Sector: {info.sector}")
        print(f"Market Cap: ${info.market_cap:,.0f}" if info.market_cap else "N/A")
    
    # Fetch historical data
    params = YahooRequestParams(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval=YahooInterval.ONE_DAY
    )
    
    data = client.fetch_historical_data(params)
    if data is not None:
        print(f"Fetched AAPL data: {len(data)} observations")
        print(data.head()) 