import pandas as pd
from datetime import datetime, timedelta

import src.data.fetch as fetch_mod


def dummy_df(series_name: str, days: int = 30) -> pd.DataFrame:
    idx = pd.date_range(datetime.today() - timedelta(days=days), periods=days)
    data = pd.Series(range(days), index=idx, name=series_name)
    return data.to_frame()


def test_get_configured_data(monkeypatch):
    """Smoke-test get_configured_data with mocked download helpers."""

    # Patch the three low-level fetchers to avoid network calls
    monkeypatch.setattr(
        fetch_mod, "fetch_fred_series", lambda *a, **k: dummy_df("GDP")
    )
    monkeypatch.setattr(
        fetch_mod, "fetch_asset_data", lambda *a, **k: dummy_df("SPX")
    )
    monkeypatch.setattr(
        fetch_mod, "fetch_stooq_data", lambda *a, **k: dummy_df("MOVE")
    )

    result = fetch_mod.get_configured_data()

    # We expect three dataframes back and each should have >95% non-null
    for key in ["macro", "assets", "stooq"]:
        df = result[key]
        assert not df.empty, f"{key} dataframe is empty"
        recent = df.tail(5)
        ratio = 1.0 - recent.isna().mean().mean()
        assert ratio >= 0.95, f"{key} null-ratio <95% (got {ratio:.2%})"
