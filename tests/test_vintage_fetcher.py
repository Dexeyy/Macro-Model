import os
import json
from unittest.mock import patch

import pandas as pd

from src.data.vintage_fetcher import fetch_alfred_series


def _fake_obs(dates, values):
    return {
        "observations": [
            {"date": d.strftime("%Y-%m-%d"), "value": ("." if v is None else str(v))}
            for d, v in zip(dates, values)
        ]
    }


@patch("src.data.vintage_fetcher.requests.get")
def test_fetch_alfred_series_vintage_alignment(mock_get):
    # Two chunks of vintages; ensure later chunk overwrites earlier obs for same date
    idx = pd.date_range("2020-01-31", "2020-03-31", freq="ME")
    # First response: Jan/Feb vintages; values 1, 2, 3
    first = _fake_obs(idx, [1.0, 2.0, 3.0])
    # Second response: Mar vintage; revised last value 3.5
    second = _fake_obs(idx, [1.0, 2.0, 3.5])

    class Resp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    mock_get.side_effect = [Resp(first), Resp(second)]

    s = fetch_alfred_series("TEST", start="2020-01-01", end="2020-03-31")
    assert isinstance(s, pd.Series)
    assert s.index.equals(idx)
    assert float(s.loc[pd.Timestamp("2020-03-31")]) == 3.5


