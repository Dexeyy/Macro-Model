import pandas as pd
import numpy as np

from src.data.processors import merge_macro_and_asset_data, calculate_regime_performance


def test_performance_smoke():
    # synthetic macro with regime labels
    dates = pd.date_range("2020-01-31", periods=36, freq="M")
    macro = pd.DataFrame(index=dates)
    # alternating regimes
    macro["Regime_KMeans_Labeled"] = ["Regime_0" if i % 2 == 0 else "Regime_1" for i in range(len(dates))]

    # synthetic asset returns
    rng = np.random.default_rng(1)
    asset_returns = pd.DataFrame(
        {
            "Asset_A": rng.normal(0.01, 0.05, len(dates)),
            "Asset_B": rng.normal(0.005, 0.03, len(dates)),
        },
        index=dates,
    )

    data_for_analysis = merge_macro_and_asset_data(
        macro, asset_returns, regime_col="Regime_KMeans_Labeled"
    )
    assert isinstance(data_for_analysis, pd.DataFrame)
    assert not data_for_analysis.empty

    perf = calculate_regime_performance(data_for_analysis, regime_col="Regime_KMeans_Labeled")
    assert isinstance(perf, pd.DataFrame)
    # MultiIndex columns: (Asset, Metric)
    # Ensure expected metrics exist for each asset
    top_level_assets = set(perf.columns.get_level_values(0))
    assert {"Asset_A", "Asset_B"}.issubset(top_level_assets)
    metrics = set(perf.columns.get_level_values(1))
    expected_metrics = {"Ann_Mean_Return", "Ann_Std_Dev", "Months_Count", "Ann_Sharpe_Ratio"}
    assert expected_metrics.issubset(metrics)

