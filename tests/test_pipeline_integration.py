import pandas as pd
import numpy as np

from src.data.processors import prepare_missing
from src.utils.zscores import robust_zscore_rolling


def test_pipeline_lag_window_alignment():
    # Synthetic monthly panel with known lags and transforms
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    base = pd.DataFrame({
        "A": np.linspace(100, 200, len(idx)),            # tcode 2 (diff)
        "B": np.exp(np.linspace(4, 5, len(idx))),        # tcode 6 (dlog)
        "VIXCLS": np.random.uniform(10, 30, len(idx)),   # will map to VIX
    }, index=idx)
    tmap = {"A": 2, "B": 6}
    out = prepare_missing(base, tmap, outlier_method="hampel", zmax=6.0)
    # First valid for diff at t=2; for dlog at t=2; plus robust window 36 -> earliest non-NaN after ~min_periods
    z = robust_zscore_rolling(out["A"], window=36, min_periods=18)
    assert z.notna().sum() >= len(idx) - 36  # most should be defined after initial window
    # VIX mapped
    assert "VIX" in out.columns


