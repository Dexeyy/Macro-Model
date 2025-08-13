import numpy as np
import pandas as pd

from src.models.regime_classifier import fit_regimes


def _make_synth(n=60, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="M")
    # Create three composite-like features with regime structure
    x1 = np.sin(np.linspace(0, 6.0, n)) + 0.1 * rng.standard_normal(n)
    x2 = (np.linspace(-1, 1, n)) + 0.1 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    df = pd.DataFrame({"F_Growth": x1, "F_Inflation": x2, "F_Liquidity": x3}, index=dates)
    return df


def test_fit_regimes_smoke_columns():
    df = _make_synth()
    out = fit_regimes(df, features=None, n_regimes=3, bundle=None)
    # Expect at least these label columns present (legacy or new *_Regime)
    cols = set(out.columns)
    assert any(c in cols for c in ("Rule", "Rule_Regime"))
    assert any(c in cols for c in ("GMM", "GMM_Regime"))
    assert any(c in cols for c in ("HMM", "HMM_Regime"))


def test_fit_regimes_non_empty_labels():
    df = _make_synth()
    out = fit_regimes(df, features=None, n_regimes=3)
    # Non-empty label series
    assert (~out.filter(regex=r"^(Rule|GMM|HMM)(_Regime)?$").isna()).any().any()


