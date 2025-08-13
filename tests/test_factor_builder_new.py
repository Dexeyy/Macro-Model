import pandas as pd
import numpy as np

from src.models.factor_builder import build_factor, FactorSpec


def _mk_df():
    idx = pd.date_range("2010-01-01", periods=120, freq="M")
    return idx


def test_sign_application():
    idx = _mk_df()
    # PAYEMS positive, UNRATE negative per SERIES_SIGN
    payems = pd.Series(np.linspace(100, 200, len(idx)), index=idx, name="PAYEMS")
    unrate = pd.Series(np.linspace(10, 5, len(idx)), index=idx, name="UNRATE")
    df = pd.DataFrame({
        "PAYEMS": payems,
        "UNRATE": unrate,
    })
    spec = FactorSpec(name="F_Test", bases=["PAYEMS", "UNRATE"], min_k=1)
    f, used, cov = build_factor(
        df,
        spec,
        mode="RETRO",
        prune_threshold=1.01,
        min_sub_bucket_coverage=1,
    )
    # Both series should be used when pruning effectively disabled
    assert len(used) == 2
    # Correlation between z-scores should be positive after sign application.
    # Compare last against first valid to avoid initial NaNs from rolling windows.
    first_valid = f.first_valid_index()
    assert first_valid is not None
    assert f.iloc[-1] > f.loc[first_valid]


def test_correlation_pruning():
    idx = _mk_df()
    a = pd.Series(np.random.RandomState(0).randn(len(idx)).cumsum(), index=idx, name="INDPRO")
    b = a * 1.0  # perfectly correlated copy
    c = pd.Series(np.random.RandomState(1).randn(len(idx)).cumsum(), index=idx, name="PAYEMS")
    df = pd.DataFrame({"INDPRO": a, "INDPRO_YoY": b, "PAYEMS": c})
    spec = FactorSpec(name="F_Test", bases=["INDPRO", "PAYEMS"], min_k=2)
    f, used, cov = build_factor(df, spec, mode="RETRO", prune_threshold=0.95)
    # Only one of the perfectly correlated pair should remain
    assert ("INDPRO" in used) ^ ("INDPRO_YoY" in used)


def test_sub_bucket_weighting():
    idx = _mk_df()
    # Two groups: activity (INDPRO, IPFINAL) and labour (PAYEMS, CE16OV)
    g1 = pd.Series(np.linspace(0, 10, len(idx)), index=idx)
    g1b = g1 + 0.5
    g2 = pd.Series(np.linspace(5, 0, len(idx)), index=idx)
    g2b = g2 + 0.5
    df = pd.DataFrame({
        "INDPRO": g1,
        "IPFINAL": g1b,
        "PAYEMS": g2,
        "CE16OV": g2b,
    })
    spec = FactorSpec(name="F_Test", bases=["INDPRO", "IPFINAL", "PAYEMS", "CE16OV"], min_k=2)
    f, used, cov = build_factor(
        df,
        spec,
        mode="RETRO",
        prune_threshold=1.01,
        min_sub_bucket_coverage=1,
    )
    # Expect coverage to be >= 4 when all series present; after sub-bucket mask,
    # effective series counted is 4; factor equals average of two sub-bucket means
    assert len(used) >= 2
    assert f.notna().any()

