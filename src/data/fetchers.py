import pandas as pd
import numpy as np
import logging
import time
import os
import requests
from datetime import datetime
from fredapi import Fred
import yfinance as yf
from config import config as cfg  # access FRED_ALIASES and TCODE_MAP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_fred_api():
    """Initialize FRED API with key from environment variable."""
    try:
        fred_api_key = os.getenv('FRED_API_KEY', '27c6023be325d770df1c327ab339020e')
        fred = Fred(api_key=fred_api_key)
        logger.info("Successfully initialized FRED API")
        return fred
    except Exception as e:
        logger.error(f"Error initializing FRED API: {e}")
        raise

def _canonicalize_series_dict(series_dict: dict) -> dict:
    """Normalize input mapping to {label->code} where label is canonical FRED ID.

    Accepts either {code->desc} or {label->code}. Applies aliases from cfg.FRED_ALIASES.
    """
    out = {}
    aliases = getattr(cfg, 'FRED_ALIASES', {}) or {}
    for k, v in series_dict.items():
        # If v looks like a FRED code (uppercase letters/numbers/underscore), treat as code
        if isinstance(v, str) and v.replace('_', '').isalnum() and v.upper() == v:
            code = v
            label = k
        else:
            # Otherwise, k is the code and label; map aliases
            code = aliases.get(k, k)
            label = code
        # apply alias for label too
        label = aliases.get(label, label)
        out[label] = code
    return out


def fetch_fred_series(series_dict, start_date, end_date):
    """
    Fetch multiple series from FRED.
    
    Args:
        series_dict: Dictionary mapping labels to FRED series codes or list of FRED series codes
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with all fetched series
    """
    fred = initialize_fred_api()
    data_list = []
    
    # Convert to dictionary if a list is provided
    if isinstance(series_dict, list):
        series_dict = {code: code for code in series_dict}

    # Canonicalize mapping to {label->code}
    try:
        series_dict = _canonicalize_series_dict(series_dict)
    except Exception:
        pass
    
    for label, code in series_dict.items():
        # Per-series retry with exponential backoff for rate limits and transient errors
        max_tries = 5
        delay = 1.0
        last_exc = None
        got = None
        for attempt in range(1, max_tries + 1):
            try:
                logger.info(f"Fetching {label} ({code})... [attempt {attempt}/{max_tries}]")
                s = fred.get_series(code, observation_start=start_date, observation_end=end_date)
                if s is not None and not s.empty:
                    got = s.rename(label)
                    logger.info(f"Successfully fetched {label}")
                    break
                else:
                    msg = f"No data returned for {label} ({code})"
                    logger.warning(msg)
            except Exception as e:
                last_exc = e
                # Heuristic: backoff on rate limiting or network-ish errors
                emsg = str(e)
                if any(tok in emsg for tok in ["Too Many Requests", "Rate Limit", "timed out", "Connection", "temporarily unavailable"]):
                    time.sleep(delay)
                    delay = min(delay * 2.0, 16.0)
                    continue
                # If clearly "does not exist", don't retry further
                if "does not exist" in emsg:
                    logger.error(f"Error fetching {label} ({code}): {e}")
                    break
                # Generic retry
                time.sleep(delay)
                delay = min(delay * 2.0, 16.0)
        if got is not None:
            data_list.append(got)
        else:
            # Special-case fallback: MOVE via Stooq if FRED unavailable
            if code == "MOVE" or label == "MOVE":
                try:
                    from .fetchers import fetch_stooq_series  # local import to avoid cycles
                except Exception:
                    fetch_stooq_series = None
                if fetch_stooq_series is not None:
                    try:
                        s2 = fetch_stooq_series("^MOVE")
                        if s2 is not None and not s2.empty:
                            s2m = s2.resample('ME').last().rename('^MOVE')
                            data_list.append(s2m)
                            logger.warning("FRED MOVE unavailable; using Stooq ^MOVE fallback")
                            continue
                    except Exception as fe:
                        logger.debug("Stooq fallback for MOVE failed: %s", fe)
            if last_exc is not None:
                logger.error(f"Error fetching {label} ({code}) after retries: {last_exc}")
            else:
                logger.error(f"Failed to fetch {label} ({code}) after retries; skipping.")
    
    if not data_list:
        raise Exception("No data was fetched. Check API key and series codes.")
    
    try:
        macro_data_raw = pd.concat(data_list, axis=1)
        logger.info("Successfully created macro_data_raw")
        return macro_data_raw
    except Exception as e:
        logger.error(f"Error creating macro_data_raw: {e}")
        raise


def apply_fredmd_tcodes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply FRED-MD transformation codes to columns per cfg.TCODE_MAP.

    The transformation overwrites columns in place using canonical FRED IDs.
    """
    if df is None or df.empty:
        return df
    tmap = getattr(cfg, 'TCODE_MAP', {}) or {}
    aliases = getattr(cfg, 'FRED_ALIASES', {}) or {}

    def safe_log(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors='coerce')
        s = s.where(s > 0)
        return np.log(s)

    out = df.copy()
    for raw_key, tcode in tmap.items():
        col = raw_key
        # Map alias to actual present column name
        col = aliases.get(col, col)
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors='coerce')
        try:
            if tcode == 1:
                transformed = s
            elif tcode == 2:
                transformed = s.diff(1)
            elif tcode == 4:
                transformed = s.diff(4)
            elif tcode == 5:
                transformed = safe_log(s)
            elif tcode == 6:
                transformed = safe_log(s).diff(1)
            elif tcode == 7:
                transformed = safe_log(s).diff(1).diff(1)
            else:
                transformed = s
            out[col] = transformed
        except Exception:
            # leave original if transform fails
            pass
    return out

def fetch_stooq_series(ticker: str) -> pd.Series:
    """
    Fetch data from Stooq.
    
    Args:
        ticker: Ticker symbol for Stooq
        
    Returns:
        Series with Close prices
    """
    try:
        logger.info(f"Fetching data from Stooq for {ticker}...")
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        return df["Close"].rename(ticker)
    except Exception as e:
        logger.error(f"Error fetching data from Stooq for {ticker}: {e}")
        return pd.Series(name=ticker)

def fetch_stooq_data(stooq_tickers):
    """
    Fetch multiple series from Stooq.
    
    Args:
        stooq_tickers: Dictionary mapping labels to Stooq ticker symbols
        
    Returns:
        DataFrame with all fetched series
    """
    data_list = []
    
    for label, ticker in stooq_tickers.items():
        try:
            s = fetch_stooq_series(ticker)
            if not s.empty:
                data_list.append(s.rename(label))
                logger.info(f"Successfully fetched {label} from Stooq")
            else:
                logger.warning(f"No data returned for {label} ({ticker}) from Stooq")
        except Exception as e:
            logger.error(f"Error fetching {label} ({ticker}) from Stooq: {e}")
    
    if data_list:
        try:
            stooq_data = pd.concat(data_list, axis=1)
            logger.info("Successfully created Stooq data DataFrame")
            return stooq_data
        except Exception as e:
            logger.error(f"Error creating Stooq data DataFrame: {e}")
    
    logger.warning("No Stooq data was successfully fetched")
    return pd.DataFrame()

def fetch_ny_fed_gscpi():
    """
    Fetch NY Fed Global Supply Chain Pressure Index (GSCPI).
    
    Returns:
        DataFrame with GSCPI data
    """
    try:
        logger.info("Fetching NY Fed GSCPI data...")
        url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
        df = pd.read_excel(url, sheet_name="GSCPI", index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info("Successfully fetched NY Fed GSCPI data")
        return df
    except Exception as e:
        logger.error(f"Error fetching NY Fed GSCPI data: {e}")
        return pd.DataFrame()

def download_with_fallback(ticker, start_date, end_date):
    """
    Try multiple methods to download data from Yahoo Finance, with fallbacks.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with price data or None if failed
    """
    logger.info(f"Attempting to download {ticker} with extended timeout...")
    try:
        # First try with a longer timeout
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            interval='1d',
            timeout=30  # Extended timeout
        )
        if data is not None and not data.empty:
            return data
        
        logger.warning(f"First attempt for {ticker} failed, trying with proxy parameter...")
        # Second try with different parameters
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            interval='1d',
            proxy=None  # Try with explicit None proxy
        )
        return data
    except Exception as e:
        logger.error(f"All download attempts failed for {ticker}: {e}")
        return None

def fetch_asset_data(asset_tickers, start_date, end_date):
    """
    Fetch asset price data from Yahoo Finance.
    
    Args:
        asset_tickers: Dictionary mapping asset labels to ticker symbols
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with all asset price series
    """
    asset_data_list = []
    skipped_assets = []
    successful_assets = []
    
    for label, ticker_symbol in asset_tickers.items():
        logger.info(f"\nFetching {label} ({ticker_symbol})...")
        try:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if retry_count > 0:
                        time.sleep(2)
                    data = download_with_fallback(ticker_symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        logger.info(f"Fetched {label}: shape={data.shape}, first date={data.index.min()}, last date={data.index.max()}")
                        processed_data = process_asset_data(data, label)
                        if processed_data is not None and not processed_data.empty:
                            asset_data_list.append(processed_data)
                            successful_assets.append(label)
                            logger.info(f"Successfully processed data for {label}")
                            break
                        else:
                            logger.warning(f"Processed data for {label} is empty after cleaning.")
                            retry_count += 1
                    else:
                        logger.warning(f"No data returned for {label}")
                        retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying... (Attempt {retry_count + 1} of {max_retries})")
                except Exception as e:
                    logger.error(f"Error in attempt {retry_count + 1} for {label}: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying... (Attempt {retry_count + 1} of {max_retries})")
            if retry_count == max_retries:
                logger.error(f"Failed to fetch data for {label} after {max_retries} attempts. Skipping.")
                skipped_assets.append(label)
        except Exception as e:
            logger.error(f"Could not fetch or process data for {label} ({ticker_symbol}): {e}")
            skipped_assets.append(label)

    logger.info(f"\nSummary of asset data fetch:")
    logger.info(f"Successfully fetched: {successful_assets}")
    logger.info(f"Skipped (no data): {skipped_assets}")
    
    if asset_data_list:
        try:
            asset_prices = pd.concat(asset_data_list, axis=1)
            logger.info(f"Created asset_prices DataFrame with shape: {asset_prices.shape}")
            return asset_prices
        except Exception as e:
            logger.error(f"Error concatenating asset data: {e}")
            return None
    else:
        logger.warning("No asset data was successfully fetched")
        return None

def process_asset_data(data, label):
    """Process downloaded asset data."""
    try:
        if data is not None and not data.empty:
            logger.info(f"Processing data for {label} with columns: {data.columns.tolist()}")

            # Normalize to a 1-D Series of prices
            price_obj = None
            # Handle MultiIndex columns from yfinance when multiple tickers used
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                first_level = [lvl for lvl in ['Adj Close', 'Close'] if lvl in data.columns.get_level_values(0)]
                if first_level:
                    price_obj = data[first_level[0]]
                    logger.info(f"Using '{first_level[0]}' column for {label}")
                else:
                    # fallback: take first top-level column
                    price_obj = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0)
            else:
                if 'Adj Close' in data.columns:
                    price_obj = data['Adj Close']
                    logger.info(f"Using 'Adj Close' column for {label}")
                elif 'Close' in data.columns:
                    price_obj = data['Close']
                    logger.info(f"Using 'Close' column for {label}")
                else:
                    # If no Adj Close or Close, use the first numeric column
                    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                    if numeric_cols:
                        price_obj = data[numeric_cols[0]]
                        logger.info(f"Using '{numeric_cols[0]}' column for {label} (fallback)")
                    else:
                        logger.error(f"No numeric columns found for {label}")
                        return None

            # If still a DataFrame (e.g., single-column), reduce to Series
            if isinstance(price_obj, pd.DataFrame):
                if price_obj.shape[1] >= 1:
                    price_series = price_obj.iloc[:, 0].rename(label)
                else:
                    logger.error(f"No usable price series extracted for {label}")
                    return None
            else:
                price_series = price_obj.rename(label)

            # Check if we got valid data
            if price_series is None or price_series.empty:
                logger.error(f"Price series for {label} is None or empty")
                return None

            non_null = int(price_series.notna().sum())
            logger.info(f"Price series for {label} before cleaning: length={len(price_series)}, non-null={non_null}")
            
            # Forward fill any missing values (use .ffill() to avoid deprecation)
            price_series = price_series.ffill()
            
            # Check if we still have nulls
            null_count = price_series.isnull().sum()
            if null_count > 0:
                logger.warning(f"After forward fill, {label} still has {null_count} nulls")
                # Try backward fill as well
                price_series = price_series.bfill()
                null_count = price_series.isnull().sum()
                if null_count > 0:
                    logger.warning(f"After backward fill, {label} still has {null_count} nulls")
            
            # Drop any remaining NaN values as last resort
            price_series = price_series.dropna()
            
            if not price_series.empty:
                logger.info(f"Successfully processed price data for {label}: final length={len(price_series)}")
                return price_series.rename(label)
            else:
                logger.warning(f"No valid price data after cleaning for {label}")
                return None
        else:
            logger.warning(f"Input data for {label} is None or empty")
            return None
    except Exception as e:
        logger.error(f"Error processing data for {label}: {e}")
        return None

def create_dummy_asset_data(start_date, end_date, asset_names):
    """
    Create dummy asset data for testing when real data can't be fetched.
    
    Args:
        start_date: Start date for dummy data
        end_date: End date for dummy data
        asset_names: List of asset names to create dummy data for
        
    Returns:
        DataFrame with dummy price data
    """
    logger.warning("Creating DUMMY TEST DATA since no real data could be fetched")
    logger.warning("This is for testing purposes only - DO NOT use for real analysis!")
    
    # Generate a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a dictionary to hold the price series for each asset
    dummy_data = {}
    
    # For each asset, create a random walk price series
    for asset in asset_names:
        # Start with a base price
        base_price = 100.0
        
        # Generate random daily returns (mean=0, std=0.01)
        daily_returns = np.random.normal(0, 0.01, len(date_range))
        
        # Convert returns to price series using cumulative product
        prices = base_price * (1 + daily_returns).cumprod()
        
        # Create a pandas Series with the dates
        dummy_data[asset] = pd.Series(prices, index=date_range)
    
    # Combine all series into a DataFrame
    df = pd.DataFrame(dummy_data)
    
    return df

def update_all_data(config, output_dir=None):
    """
    Update all data sources in one function.
    
    Args:
        config: Configuration object containing FRED_SERIES, STOOQ_TICKERS, etc.
        output_dir: Directory to save raw data files (optional)
        
    Returns:
        Dictionary with all fetched data
    """
    start_date = config.START_DATE
    end_date = config.END_DATE
    asset_start_date = config.ASSET_START_DATE
    
    results = {}
    
    # Fetch FRED data
    try:
        logger.info("Fetching FRED data...")
        fred_data = fetch_fred_series(config.FRED_SERIES, start_date, end_date)
        results['fred_data'] = fred_data
        if output_dir:
            fred_path = os.path.join(output_dir, 'fred_data_raw.csv')
            fred_data.to_csv(fred_path)
            logger.info(f"Saved FRED data to {fred_path}")
    except Exception as e:
        logger.error(f"Error fetching FRED data: {e}")
    
    # Fetch Stooq data if available
    if hasattr(config, 'STOOQ_TICKERS') and config.STOOQ_TICKERS:
        try:
            logger.info("Fetching Stooq data...")
            stooq_data = fetch_stooq_data(config.STOOQ_TICKERS)
            results['stooq_data'] = stooq_data
            if output_dir and not stooq_data.empty:
                stooq_path = os.path.join(output_dir, 'stooq_data_raw.csv')
                stooq_data.to_csv(stooq_path)
                logger.info(f"Saved Stooq data to {stooq_path}")
        except Exception as e:
            logger.error(f"Error fetching Stooq data: {e}")
    
    # Fetch NY Fed GSCPI
    try:
        logger.info("Fetching NY Fed GSCPI data...")
        gscpi_data = fetch_ny_fed_gscpi()
        results['gscpi_data'] = gscpi_data
        if output_dir and not gscpi_data.empty:
            gscpi_path = os.path.join(output_dir, 'gscpi_data_raw.csv')
            gscpi_data.to_csv(gscpi_path)
            logger.info(f"Saved NY Fed GSCPI data to {gscpi_path}")
    except Exception as e:
        logger.error(f"Error fetching NY Fed GSCPI data: {e}")
    
    # Fetch asset data
    try:
        logger.info("Fetching asset data...")
        asset_data = fetch_asset_data(config.ASSET_TICKERS, asset_start_date, end_date)
        results['asset_data'] = asset_data
        if output_dir and asset_data is not None:
            asset_path = os.path.join(output_dir, 'asset_prices_raw.csv')
            asset_data.to_csv(asset_path)
            logger.info(f"Saved asset data to {asset_path}")
    except Exception as e:
        logger.error(f"Error fetching asset data: {e}")
    
    logger.info("Data update completed")
    return results 