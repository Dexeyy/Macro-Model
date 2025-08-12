import pandas as pd

from src.data.processors import _harmonize_financial_condition_names


def test_series_mapping_vix_and_corp_spread():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    df = pd.DataFrame({
        "VIXCLS": [10, 20, 30],
        "BAA": [5.0, 5.1, 5.2],
        "AAA": [3.0, 3.1, 3.0],
    }, index=idx)
    out = _harmonize_financial_condition_names(df)
    assert "VIX" in out.columns
    assert "CorporateBondSpread" in out.columns
    # Check spread construction
    exp = (pd.Series([5.0, 5.1, 5.2], index=idx) - pd.Series([3.0, 3.1, 3.0], index=idx)).values.tolist()
    assert out["CorporateBondSpread"].round(2).values.tolist() == pd.Series(exp, index=idx).round(2).values.tolist()


