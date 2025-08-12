from __future__ import annotations

from typing import Optional

import pandas as pd


def yield_curve_slope(dgs10: pd.Series, dgs2: pd.Series) -> pd.Series:
    """Compute 10Y-2Y slope.

    Args:
        dgs10: 10-year Treasury series
        dgs2: 2-year Treasury series

    Returns:
        Series of slope with index aligned to the intersection of inputs.
    """
    idx = dgs10.index.intersection(dgs2.index)
    return (dgs10.loc[idx] - dgs2.loc[idx]).rename("YieldCurve_Slope")


def yield_curve_curvature(dgs2: pd.Series, dgs5: pd.Series, dgs10: pd.Series) -> pd.Series:
    """Compute simple curvature 2*5Y - 2Y - 10Y.

    Returns a Series aligned on the intersection of inputs.
    """
    idx = dgs2.index.intersection(dgs5.index).intersection(dgs10.index)
    return (2.0 * dgs5.loc[idx] - dgs2.loc[idx] - dgs10.loc[idx]).rename("YieldCurve_Curvature")


def real_rate_10y(dgs10: pd.Series, t10yie: pd.Series) -> pd.Series:
    """Compute 10Y real rate = nominal 10Y - inflation expectations (T10YIE)."""
    idx = dgs10.index.intersection(t10yie.index)
    return (dgs10.loc[idx] - t10yie.loc[idx]).rename("RealRate_10Y")


def growth_momentum(yoy_series: pd.Series, months: int = 3) -> pd.Series:
    """Growth momentum as the change over trailing window (e.g., 3-month diff)."""
    return yoy_series.diff(months).rename(f"{yoy_series.name}_Mom")


def fin_conditions_composite(nfci: Optional[pd.Series] = None,
                             vix: Optional[pd.Series] = None,
                             move: Optional[pd.Series] = None,
                             corp_spread: Optional[pd.Series] = None) -> pd.Series:
    """Simple composite = z-score average of available inputs.

    Inputs are standardized within-sample; output is NaN where <2 inputs are available.
    """
    cols = {}
    if nfci is not None:
        cols["NFCI"] = nfci
    if vix is not None:
        cols["VIX"] = vix
    if move is not None:
        cols["MOVE"] = move
    if corp_spread is not None:
        cols["CorporateBondSpread"] = corp_spread
    if not cols:
        return pd.Series(dtype=float, name="FinConditions_Composite")
    X = pd.DataFrame(cols)
    Z = X.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) not in (0, None) else 1.0))
    cov = Z.notna().sum(axis=1)
    out = Z.mean(axis=1).where(cov >= 2)
    return out.rename("FinConditions_Composite")


