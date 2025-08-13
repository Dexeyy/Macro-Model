import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is on path for 'src' package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.zscores import robust_zscore_rolling, pick_window_minp
from src.utils.factors import build_factor as _build_factor
from src.data.processors import _harmonize_financial_condition_names


def test_robust_zscore_constant_series_returns_nan():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    s = pd.Series(5.0, index=idx)
    z = robust_zscore_rolling(s, window=24, min_periods=12)
    # All NaN because std/MAD are zero
    assert z.isna().all()


def test_robust_zscore_mad_zero_fallback_to_std():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    s = pd.Series(np.r_[np.zeros(30), np.arange(30)], index=idx).astype(float)
    z = robust_zscore_rolling(s, window=12, min_periods=6)
    # At least some finite values where variability exists
    assert np.isfinite(z.iloc[-10:]).any()


def test_pick_window_minp_runs_and_returns_valid():
    idx = pd.date_range("1995-01-31", periods=200, freq="ME")
    s = pd.Series(np.sin(np.arange(200)/8.0) + np.random.normal(0, 0.2, 200), index=idx)
    res = pick_window_minp(s)
    assert set(res.keys()) == {"window", "min_periods"}
    assert res["window"] in (36, 60, 120)


def test_build_factor_transform_priority_and_coverage_mask():
    # Create a panel with level and YoY, but YoY for one series is all NaN
    idx = pd.date_range("2000-01-31", periods=36, freq="ME")
    df = pd.DataFrame({
        "A": np.arange(36, dtype=float),
        "A_YoY": [np.nan]*36,  # should skip
        "B": np.linspace(10, 20, 36),
        "B_YoY": pd.Series(np.linspace(-0.1, 0.2, 36)).where(lambda x: x.index % 5 != 0),
    }, index=idx)

    f, used, cov = _build_factor(df, ["A", "B"], mode="RETRO", min_k=2)
    assert isinstance(f, pd.Series)
    assert len(used) >= 1
    # Coverage may be 0 early; ensure later coverage reaches at least min_k
    assert (cov >= 2).any()
    # When coverage < min_k, factor should be NaN
    low_cov_dates = cov[cov < 2].index
    if len(low_cov_dates) > 0:
        assert f.loc[low_cov_dates].isna().all()


def test_finconditions_harmonization():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    df = pd.DataFrame({
        "VIXCLS": np.random.uniform(10, 30, 12),
        "^MOVE": np.random.uniform(50, 150, 12),
        "AAA": np.random.uniform(3, 5, 12),
        "BAA": np.random.uniform(4, 7, 12),
    }, index=idx)
    out = _harmonize_financial_condition_names(df)
    assert "VIX" in out.columns
    assert "MOVE" in out.columns
    assert "CorporateBondSpread" in out.columns


