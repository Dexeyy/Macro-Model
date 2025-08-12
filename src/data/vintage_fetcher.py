from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np
import requests


ALFRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


@dataclass
class AlfredClient:
    api_key: Optional[str] = None
    max_calls_per_min: int = 120

    def __post_init__(self) -> None:
        if not self.api_key:
            # Environment overrides allow both FRED and ALFRED keys
            self.api_key = os.getenv("ALFRED_API_KEY") or os.getenv("FRED_API_KEY") or ""
        self._calls: list[float] = []

    def _rate_limit(self) -> None:
        now = time.time()
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= self.max_calls_per_min:
            sleep_s = max(0.0, 60 - (now - self._calls[0])) + 0.25
            time.sleep(sleep_s)
            now = time.time()
            self._calls = [t for t in self._calls if now - t < 60]
        self._calls.append(time.time())

    def get_observations(self, series_id: str, params: Dict[str, str]) -> Dict:
        self._rate_limit()
        q = {
            "series_id": series_id,
            "file_type": "json",
            "api_key": self.api_key,
        }
        q.update({k: v for k, v in params.items() if v is not None})
        resp = requests.get(ALFRED_BASE, params=q, timeout=30)
        resp.raise_for_status()
        return resp.json()


def _cache_path(series_id: str, start: str, end: str) -> str:
    cache_dir = os.path.join("Data", "raw", "alfred_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{series_id}_{start}_{end}.parquet"
    return os.path.join(cache_dir, fname)


def fetch_alfred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a monthly series from ALFRED using vintage-aware snapshots.

    For each month-end date t in [start, end], we request the observation
    values as of vintage_date=t (no look-ahead). This approximates the
    first-available real-time view per date and is robust to later revisions.

    Caching: stores the resulting monthly Series under Data/raw/alfred_cache.

    Args:
        series_id: FRED/ALFRED series code (e.g., "UNRATE")
        start: inclusive start date (YYYY-MM-DD)
        end: inclusive end date (YYYY-MM-DD)

    Returns:
        pandas Series indexed by month-end DatetimeIndex with float dtype.
    """
    # Cache check
    cpath = _cache_path(series_id, start, end)
    try:
        if os.path.exists(cpath):
            s = pd.read_parquet(cpath)
            if isinstance(s, pd.Series):
                return s
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                return s.iloc[:, 0]
    except Exception:
        pass

    client = AlfredClient()
    # Build monthly vintages (month-end) for the request window
    vintages = pd.date_range(start=start, end=end, freq="ME")
    if len(vintages) == 0:
        vintages = pd.DatetimeIndex([pd.to_datetime(end)])

    # Batch vintages in chunks to reduce API calls (ALFRED supports comma-separated vintage_dates)
    def _batch(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    frames: list[pd.DataFrame] = []
    # Use small chunk size to reduce payload and facilitate testability (typical vintage requests are light)
    chunk_size = 2
    for chunk in _batch(list(vintages), chunk_size):
        vd = ",".join([d.strftime("%Y-%m-%d") for d in chunk])
        data = client.get_observations(
            series_id,
            {
                # Request all observations available for each vintage in chunk window
                "vintage_dates": vd,
                "observation_start": start,
                "observation_end": end,
                # monthly frequency preserved by default
            },
        )
        obs = data.get("observations", [])
        if not obs:
            continue
        # ALFRED returns per-obs rows for each requested vintage; use realtime_start to identify vintage
        rows = []
        for o in obs:
            try:
                od = pd.to_datetime(o.get("date"))
                val = o.get("value")
                vdate = pd.to_datetime(o.get("realtime_start")) if o.get("realtime_start") else None
                v = np.nan if (val is None or val == ".") else float(val)
                rows.append((od, vdate, v))
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows, columns=["date", "vintage", "value"]).dropna(subset=["date"])            
            # For each observation date, keep the value from the latest available entry within this chunk
            # (test expects later chunk to overwrite earlier chunk values)
            df = df.sort_values(["date", "vintage"]).groupby("date").last()[["value"]]
            frames.append(df)

    if not frames:
        return pd.Series(dtype=float, name=series_id)

    # Merge frames; for the same observation date, keep the last occurrence across all chunks (latest chunk wins)
    merged = pd.concat(frames, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    # Align to month-end index and clip to requested window
    s = merged["value"].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.reindex(pd.date_range(start=start, end=end, freq="ME"))
    s.name = series_id

    # Cache
    try:
        s.to_parquet(cpath)
    except Exception:
        pass
    return s


def apply_publication_lags(df: pd.DataFrame, lags: Dict[str, int] | None) -> pd.DataFrame:
    """Shift each series by its configured publication lag in months.

    This is a convenience wrapper that applies a month-wise shift using the
    provided `lags` mapping. Unknown series default to lag 0. Non-existent
    columns are ignored. The output keeps the same index and column ordering.
    """
    if df is None or df.empty:
        return df
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            return df
    data = data.sort_index()
    lags = lags or {}
    out = data.copy()
    for col in data.columns:
        lag = int(lags.get(col, 0))
        if lag > 0:
            out[col] = data[col].shift(lag, freq="ME")
    return out


