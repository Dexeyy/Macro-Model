"""High-level data-ingest helper used by main pipeline.

get_configured_data() reads the ticker / series lists from
config/config.py and delegates the actual downloading work to the
lower-level fetch helpers already defined in *fetchers.py*.

It also performs lightweight caching: every raw pull is persisted to
`Data/raw/{series_name}.csv`.  That keeps the raw snapshots on disk so
sub-sequent executions can skip the network step if desired (the logic
for cache-busting remains upstream in the individual fetch helpers).
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from config import config  # type: ignore
from .fetchers import (
    fetch_fred_series,
    fetch_asset_data,
    fetch_stooq_data,
)

logger = logging.getLogger(__name__)

RAW_DIR = os.path.join("Data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def _cache_dataframe(df: pd.DataFrame) -> None:
    """Save each column of *df* as an individual CSV in Data/raw.

    File naming: <column>.csv  (e.g. CPI.csv, SPX.csv)
    The dataframe must have a DatetimeIndex.
    """
    if df is None or df.empty:
        return

    for col in df.columns:
        path = os.path.join(RAW_DIR, f"{col}.csv")
        try:
            df[[col]].to_csv(path, index_label="Date")
            logger.debug("Cached %s to %s", col, path)
        except Exception as exc:
            logger.warning("Failed to cache %s: %s", col, exc)


def get_configured_data(*, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch all macro / market series specified in *config*.

    Parameters
    ----------
    force : bool (default False)
        If *True* forces a fresh pull even if cached files exist.

    Returns
    -------
    dict
        Keys: ``macro``, ``assets``, ``stooq`` mapping to their
        respective DataFrames.  Missing categories (e.g. no Stooq
        tickers) are returned as empty DataFrames.
    """

    start_date: datetime = config.START_DATE  # type: ignore
    end_date: datetime = config.END_DATE  # type: ignore

    # ------------------------------------------------------------------
    # FRED macro series
    # ------------------------------------------------------------------
    fred_df: pd.DataFrame = fetch_fred_series(
        config.FRED_SERIES, start_date, end_date  # type: ignore
    )
    _cache_dataframe(fred_df)

    # ------------------------------------------------------------------
    # Asset prices (Yahoo Finance)
    # ------------------------------------------------------------------
    assets_df: pd.DataFrame = fetch_asset_data(
        config.ASSET_TICKERS, config.ASSET_START_DATE, end_date  # type: ignore
    )
    _cache_dataframe(assets_df)

    # ------------------------------------------------------------------
    # Stooq extras (MOVE, VIX3M, ...)
    # ------------------------------------------------------------------
    stooq_df: pd.DataFrame = pd.DataFrame()
    if hasattr(config, "STOOQ_TICKERS") and config.STOOQ_TICKERS:  # type: ignore
        stooq_df = fetch_stooq_data(config.STOOQ_TICKERS)  # type: ignore
        _cache_dataframe(stooq_df)

    return {
        "macro": fred_df,
        "assets": assets_df,
        "stooq": stooq_df,
    }
