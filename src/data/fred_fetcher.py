"""
Enhanced FRED API Data Fetching Module

This module provides a comprehensive interface to the Federal Reserve Economic Data (FRED) API
with advanced features including rate limiting, pagination handling, error recovery, and
detailed documentation of available endpoints.

Features:
- Rate limiting to respect API quotas
- Automatic pagination for large datasets
- Comprehensive error handling and recovery
- Metadata retrieval for series
- Batch operations for multiple series
- Integration with data infrastructure
- Detailed logging and monitoring
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from fredapi import Fred
import threading
from dataclasses import dataclass, asdict
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FredFrequency(Enum):
    """FRED data frequency options."""
    DAILY = "d"
    WEEKLY = "w"
    BIWEEKLY = "bw"
    MONTHLY = "m"
    QUARTERLY = "q"
    SEMIANNUAL = "sa"
    ANNUAL = "a"


class FredAggregationMethod(Enum):
    """FRED aggregation method options."""
    AVERAGE = "avg"
    SUM = "sum"
    END_OF_PERIOD = "eop"


class FredUnits(Enum):
    """FRED units transformation options."""
    LEVELS = "lin"
    CHANGE = "chg"
    CHANGE_FROM_YEAR_AGO = "ch1"
    PERCENT_CHANGE = "pch"
    PERCENT_CHANGE_FROM_YEAR_AGO = "pc1"
    COMPOUNDED_ANNUAL_RATE_OF_CHANGE = "pca"
    CONTINUOUSLY_COMPOUNDED_RATE_OF_CHANGE = "cch"
    CONTINUOUSLY_COMPOUNDED_ANNUAL_RATE_OF_CHANGE = "cca"
    NATURAL_LOG = "log"


@dataclass
class FredSeriesInfo:
    """Information about a FRED series."""
    id: str
    title: str
    observation_start: str
    observation_end: str
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment: str
    seasonal_adjustment_short: str
    last_updated: str
    popularity: int
    notes: str


@dataclass
class FredRequestParams:
    """Parameters for FRED API requests."""
    series_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: Optional[FredFrequency] = None
    aggregation_method: Optional[FredAggregationMethod] = None
    units: Optional[FredUnits] = None
    limit: int = 100000
    offset: int = 0
    sort_order: str = "asc"  # asc or desc


class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 120, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds (default: 60 seconds)
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
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    # Clean up old calls again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # Record this call
            self.calls.append(now)


class EnhancedFredClient:
    """Enhanced FRED API client with advanced features."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_calls: int = 120):
        """
        Initialize the enhanced FRED client.
        
        Args:
            api_key: FRED API key (will use environment variable if not provided)
            rate_limit_calls: Maximum API calls per minute (default: 120)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', '27c6023be325d770df1c327ab339020e')
        self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, time_window=60)
        self.fred = Fred(api_key=self.api_key)
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Cache for series metadata
        self._series_info_cache: Dict[str, FredSeriesInfo] = {}
        
        logger.info("Enhanced FRED client initialized")
    
    def get_series_info(self, series_id: str, use_cache: bool = True) -> Optional[FredSeriesInfo]:
        """
        Get detailed information about a FRED series.
        
        Args:
            series_id: FRED series identifier
            use_cache: Whether to use cached information if available
            
        Returns:
            FredSeriesInfo object or None if not found
        """
        if use_cache and series_id in self._series_info_cache:
            return self._series_info_cache[series_id]
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Use fredapi to get series info
            info = self.fred.get_series_info(series_id)
            
            series_info = FredSeriesInfo(
                id=info.get('id', series_id),
                title=info.get('title', ''),
                observation_start=info.get('observation_start', ''),
                observation_end=info.get('observation_end', ''),
                frequency=info.get('frequency', ''),
                frequency_short=info.get('frequency_short', ''),
                units=info.get('units', ''),
                units_short=info.get('units_short', ''),
                seasonal_adjustment=info.get('seasonal_adjustment', ''),
                seasonal_adjustment_short=info.get('seasonal_adjustment_short', ''),
                last_updated=info.get('last_updated', ''),
                popularity=int(info.get('popularity', 0)),
                notes=info.get('notes', '')
            )
            
            if use_cache:
                self._series_info_cache[series_id] = series_info
            
            logger.info(f"Retrieved info for series {series_id}: {series_info.title}")
            return series_info
            
        except Exception as e:
            logger.error(f"Error getting series info for {series_id}: {e}")
            return None
    
    def fetch_series(self, params: FredRequestParams, 
                    max_retries: int = 3, retry_delay: float = 1.0) -> Optional[pd.Series]:
        """
        Fetch a single FRED series with comprehensive error handling.
        
        Args:
            params: Request parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            pandas Series with the data or None if failed
        """
        for attempt in range(max_retries + 1):
            try:
                self.rate_limiter.wait_if_needed()
                
                # Build parameters for fredapi
                kwargs = {}
                if params.start_date:
                    kwargs['observation_start'] = params.start_date
                if params.end_date:
                    kwargs['observation_end'] = params.end_date
                if params.frequency:
                    kwargs['frequency'] = params.frequency.value
                if params.aggregation_method:
                    kwargs['aggregation_method'] = params.aggregation_method.value
                if params.units:
                    kwargs['units'] = params.units.value
                if params.limit and params.limit != 100000:
                    kwargs['limit'] = params.limit
                
                # Fetch the series
                series = self.fred.get_series(params.series_id, **kwargs)
                
                if series is not None and not series.empty:
                    logger.info(f"Successfully fetched {params.series_id}: {len(series)} observations")
                    return series
                else:
                    logger.warning(f"No data returned for {params.series_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {params.series_id}: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for {params.series_id}")
                    return None
        
        return None
    
    def fetch_multiple_series(self, series_params: List[FredRequestParams], 
                            max_workers: int = 5) -> Dict[str, pd.Series]:
        """
        Fetch multiple FRED series efficiently.
        
        Args:
            series_params: List of request parameters for each series
            max_workers: Maximum number of concurrent requests (limited by rate limiting)
            
        Returns:
            Dictionary mapping series IDs to pandas Series
        """
        results = {}
        
        logger.info(f"Fetching {len(series_params)} FRED series...")
        
        for i, params in enumerate(series_params, 1):
            logger.info(f"Fetching series {i}/{len(series_params)}: {params.series_id}")
            
            series = self.fetch_series(params)
            if series is not None:
                results[params.series_id] = series
            
            # Small delay to be respectful to the API
            if i < len(series_params):
                time.sleep(0.1)
        
        logger.info(f"Successfully fetched {len(results)}/{len(series_params)} series")
        return results
    
    def search_series(self, search_text: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of series information dictionaries
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            # Use requests directly for search as fredapi doesn't support all search features
            url = f"{self.base_url}/series/search"
            params = {
                'search_text': search_text,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            series_list = data.get('seriess', [])
            
            logger.info(f"Found {len(series_list)} series matching '{search_text}'")
            return series_list
            
        except Exception as e:
            logger.error(f"Error searching for series: {e}")
            return []
    
    def get_series_categories(self, series_id: str) -> List[Dict[str, Any]]:
        """
        Get categories for a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            List of category information dictionaries
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            url = f"{self.base_url}/series/categories"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            categories = data.get('categories', [])
            
            logger.info(f"Retrieved {len(categories)} categories for {series_id}")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting categories for {series_id}: {e}")
            return []
    
    def fetch_series_with_pagination(self, series_id: str, start_date: Optional[str] = None,
                                   end_date: Optional[str] = None, 
                                   page_size: int = 100000) -> Optional[pd.Series]:
        """
        Fetch a FRED series with automatic pagination for very large datasets.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            page_size: Number of observations per page
            
        Returns:
            Complete pandas Series with all data
        """
        all_data = []
        offset = 0
        
        while True:
            try:
                self.rate_limiter.wait_if_needed()
                
                # Fetch a page of data
                params = FredRequestParams(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    limit=page_size,
                    offset=offset
                )
                
                page_data = self.fetch_series(params)
                
                if page_data is None or page_data.empty:
                    break
                
                all_data.append(page_data)
                
                # If we got less than page_size, we're done
                if len(page_data) < page_size:
                    break
                
                offset += page_size
                logger.info(f"Fetched page with {len(page_data)} observations (offset: {offset})")
                
            except Exception as e:
                logger.error(f"Error in pagination for {series_id}: {e}")
                break
        
        if all_data:
            # Combine all pages
            combined_series = pd.concat(all_data).sort_index()
            # Remove any duplicates that might occur at page boundaries
            combined_series = combined_series[~combined_series.index.duplicated(keep='first')]
            
            logger.info(f"Completed pagination fetch for {series_id}: {len(combined_series)} total observations")
            return combined_series
        
        return None
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.
        
        Returns:
            Dictionary with rate limit information
        """
        with self.rate_limiter.lock:
            now = time.time()
            recent_calls = [call for call in self.rate_limiter.calls if now - call < 60]
            
            return {
                "calls_in_last_minute": len(recent_calls),
                "max_calls_per_minute": self.rate_limiter.max_calls,
                "remaining_calls": max(0, self.rate_limiter.max_calls - len(recent_calls)),
                "reset_time": max(recent_calls) + 60 if recent_calls else now
            }


# Convenience functions for backward compatibility
def initialize_fred_api() -> EnhancedFredClient:
    """Initialize enhanced FRED API client."""
    return EnhancedFredClient()


def fetch_fred_series(series_dict: Union[Dict[str, str], List[str]], 
                     start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch multiple FRED series (backward compatible function).
    
    Args:
        series_dict: Dictionary mapping labels to series IDs or list of series IDs
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with all fetched series
    """
    client = EnhancedFredClient()
    
    # Convert to dictionary if a list is provided
    if isinstance(series_dict, list):
        series_dict = {code: code for code in series_dict}
    
    # Create request parameters for each series
    series_params = [
        FredRequestParams(series_id=series_id, start_date=start_date, end_date=end_date)
        for series_id in series_dict.values()
    ]
    
    # Fetch all series
    results = client.fetch_multiple_series(series_params)
    
    # Rename series according to the provided labels
    renamed_results = {}
    for label, series_id in series_dict.items():
        if series_id in results:
            renamed_results[label] = results[series_id].rename(label)
    
    if renamed_results:
        return pd.concat(renamed_results.values(), axis=1)
    else:
        raise Exception("No data was fetched. Check API key and series codes.")


# Example usage and documentation
FRED_ENDPOINTS_DOCUMENTATION = {
    "series": {
        "description": "Get data for a specific economic data series",
        "endpoint": "/fred/series/observations",
        "parameters": {
            "series_id": "FRED series ID (required)",
            "observation_start": "Start date (YYYY-MM-DD)",
            "observation_end": "End date (YYYY-MM-DD)",
            "frequency": "Data frequency (d, w, bw, m, q, sa, a)",
            "aggregation_method": "Aggregation method (avg, sum, eop)",
            "units": "Units transformation (lin, chg, ch1, pch, pc1, pca, cch, cca, log)",
            "limit": "Maximum number of results (1-100000)",
            "offset": "Result start offset"
        }
    },
    "series/info": {
        "description": "Get metadata for a specific series",
        "endpoint": "/fred/series",
        "parameters": {
            "series_id": "FRED series ID (required)"
        }
    },
    "series/search": {
        "description": "Search for series by text",
        "endpoint": "/fred/series/search",
        "parameters": {
            "search_text": "Search terms (required)",
            "limit": "Maximum number of results (1-1000)"
        }
    },
    "series/categories": {
        "description": "Get categories for a series",
        "endpoint": "/fred/series/categories",
        "parameters": {
            "series_id": "FRED series ID (required)"
        }
    }
}

# Common FRED series for macro-economic analysis
COMMON_FRED_SERIES = {
    "gdp": "GDP",
    "unemployment": "UNRATE", 
    "inflation": "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
    "10_year_treasury": "GS10",
    "2_year_treasury": "GS2",
    "sp500": "SP500",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",
    "oil_price": "DCOILWTICO",
    "gold_price": "GOLDAMGBD228NLBM"
}


if __name__ == "__main__":
    # Example usage
    client = EnhancedFredClient()
    
    # Get series info
    info = client.get_series_info("GDP")
    if info:
        print(f"Series: {info.title}")
        print(f"Frequency: {info.frequency}")
        print(f"Units: {info.units}")
    
    # Fetch a series
    params = FredRequestParams(
        series_id="GDP",
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    series = client.fetch_series(params)
    if series is not None:
        print(f"Fetched GDP data: {len(series)} observations")
        print(series.head()) 