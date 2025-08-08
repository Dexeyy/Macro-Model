import pandas as pd
import numpy as np

from src.models.regime_classifier import fit_regimes


def test_fit_regimes_smoke():
    # tiny synthetic features
    dates = pd.date_range("2020-01-31", periods=24, freq="M")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=len(dates)),
            "x2": rng.normal(size=len(dates)),
            "x3": rng.normal(size=len(dates)),
        },
        index=dates,
    )

    out = fit_regimes(df, features=["x1", "x2", "x3"], n_regimes=3)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df)
    expected_cols = {"Rule", "KMeans", "HMM", "Regime_Ensemble"}
    assert expected_cols.issubset(set(out.columns))

