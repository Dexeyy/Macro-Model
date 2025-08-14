import numpy as np
import pandas as pd
import pytest

from src.models.portfolio import compute_dynamic_regime_portfolio, PortfolioConstructor


def _toy_monthly_returns(n=120, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-31', periods=n, freq='ME')
    a = rng.normal(0.006, 0.02, n)
    b = rng.normal(0.003, 0.015, n)
    c = rng.normal(0.002, 0.01, n)
    return pd.DataFrame({'A': a, 'B': b, 'C': c}, index=idx)


def test_dynamic_portfolio_min_obs_minvar_fallback():
    R = _toy_monthly_returns(48)
    # Create a rare regime with < 12 obs at the end
    regimes = pd.Series(['Regime_0'] * 36 + ['Regime_1'] * 12, index=R.index)
    # Force very small sample for the last regime
    pr, wh = compute_dynamic_regime_portfolio(
        R, regimes,
        regime_window_years=3,
        rebal_freq='M',
        min_obs=36,
        mean_cov_method='shrinkage',
        transaction_cost=0.0,
    )
    assert isinstance(pr, pd.Series) and len(pr) == len(R)
    # Find last rebalance weights and ensure not NaNs and roughly diversified (>1 non-zero)
    assert len(wh) > 0
    last_w = list(wh.values())[-1]
    assert np.isfinite(last_w.values).all()
    assert last_w.sum() == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert (last_w > 1e-8).sum() >= 2


def test_tangency_uses_excess_returns():
    pc = PortfolioConstructor(risk_free_rate=0.12)  # 12% annual
    pc._last_periods_per_year = 12
    # If mu equals rf per period for all assets, excess is ~0 -> equal weights after projection
    rf_p = PortfolioConstructor.to_periodic_rf(0.12, 12)
    mu = pd.Series([rf_p, rf_p, rf_p], index=['A', 'B', 'C'])
    S = pd.DataFrame(np.eye(3), index=mu.index, columns=mu.index)
    w = pc._tangency_weights(mu, S)
    assert np.isfinite(w).all()
    assert np.allclose(w, np.ones(3) / 3, atol=1e-6)


def test_periodic_rf_conversion_and_sharpe_guardrails():
    # Annual rf 2%, monthly returns
    pc = PortfolioConstructor(risk_free_rate=0.02)
    rf_p = PortfolioConstructor.to_periodic_rf(0.02, 12)
    assert abs(rf_p - ((1 + 0.02)**(1/12) - 1)) < 1e-12
    # All risky premia negative -> with CASH included, should allocate to CASH
    idx = pd.date_range('2020-01-31', periods=12, freq='ME')
    R = pd.DataFrame({'A': np.full(12, rf_p - 0.01), 'B': np.full(12, rf_p - 0.005)}, index=idx)
    regimes = pd.Series(['R'] * len(R), index=R.index)
    pr, info = compute_dynamic_regime_portfolio(R, regimes, include_cash=True, risk_free_rate=0.02, return_diagnostics=True)
    w_hist = info['weights']
    last_w = list(w_hist.values())[-1]
    assert last_w.get('CASH', 0.0) >= 0.99
    # Without CASH, method should switch to min-variance and weights remain feasible
    pr2, info2 = compute_dynamic_regime_portfolio(R, regimes, include_cash=False, risk_free_rate=0.02, return_diagnostics=True)
    last_w2 = list(info2['weights'].values())[-1]
    assert np.isfinite(last_w2.values).all()
    assert abs(last_w2.sum() - 1.0) < 1e-6


def test_transaction_costs_zero_by_default():
    R = _toy_monthly_returns(24)
    regimes = pd.Series(['X'] * len(R), index=R.index)
    pr0, _ = compute_dynamic_regime_portfolio(R, regimes)
    pr1, _ = compute_dynamic_regime_portfolio(R, regimes, transaction_cost=0.0)
    assert np.allclose(pr0.values, pr1.values, atol=1e-12)


def test_blend_probs_changes_estimates():
    R = _toy_monthly_returns(36)
    regimes = pd.Series(['R0'] * 18 + ['R1'] * 18, index=R.index)
    # Build probabilities with low confidence in current regime around switch
    probs = pd.DataFrame(index=R.index, data={'R0': 1.0, 'R1': 0.0})
    probs.loc[regimes == 'R1', 'R0'] = 0.2
    probs.loc[regimes == 'R1', 'R1'] = 0.8
    pr_nb, _ = compute_dynamic_regime_portfolio(
        R, regimes,
        regime_window_years=5,
        rebal_freq='M',
        min_obs=24,
        blend_probs=False,
        regime_probabilities=probs,
    )
    pr_b, _ = compute_dynamic_regime_portfolio(
        R, regimes,
        regime_window_years=5,
        rebal_freq='M',
        min_obs=24,
        blend_probs=True,
        regime_probabilities=probs,
    )
    # Series should differ around the transition
    assert not np.allclose(pr_nb.values, pr_b.values, atol=1e-10)


