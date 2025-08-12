import numpy as np
import pandas as pd

from src.utils.zscores import robust_zscore_rolling


def test_robust_zscore_handles_zero_mad_and_std():
    # Construct a series with flat segments where MAD and std are ~0
    x = pd.Series([1.0] * 10 + [2.0] * 10)
    z = robust_zscore_rolling(x, window=6, min_periods=3)
    # The first few values should be NaN (insufficient periods)
    assert z.iloc[:2].isna().all()
    # In flat window, z should be NaN (no dispersion)
    assert np.isnan(z.iloc[5])


