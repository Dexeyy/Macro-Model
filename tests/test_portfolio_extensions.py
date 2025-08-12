import numpy as np
import pandas as pd

from src.models.portfolio import PortfolioConstructor, PortfolioConstraints, OptimizationMethod


def _toy_returns(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-31', periods=n, freq='ME')
    a = rng.normal(0.005, 0.03, n)
    b = rng.normal(0.002, 0.02, n)
    return pd.DataFrame({'A': a, 'B': b}, index=idx)


def test_estimate_mean_cov_shapes():
    R = _toy_returns(120)
    pc = PortfolioConstructor()
    mu, S = pc.estimate_mean_cov(R, method='sample')
    assert mu.shape[0] == R.shape[1]
    assert S.shape == (R.shape[1], R.shape[1])
    mu2, S2 = pc.estimate_mean_cov(R, method='shrinkage')
    assert S2.shape == S.shape


def test_turnover_limit_zero_freezes_weights():
    R = _toy_returns(24)
    regimes = pd.Series(['x'] * len(R), index=R.index)
    pc = PortfolioConstructor(constraints=PortfolioConstraints(turnover_limit=0.0))
    stats = pc.calculate_regime_statistics(R, regimes)
    # First run
    res1 = pc.optimize_portfolio(stats, 'x', OptimizationMethod.SHARPE)
    # Second run with zero turnover allowed should keep weights the same
    res2 = pc.optimize_portfolio(stats, 'x', OptimizationMethod.SHARPE)
    assert np.allclose(res1.weights.values, res2.weights.values)


def test_optimize_with_probabilities_transitions_smoothly():
    R = _toy_returns(60)
    regimes = pd.Series(['a'] * 30 + ['b'] * 30, index=R.index)
    pc = PortfolioConstructor(constraints=PortfolioConstraints(turnover_limit=0.5))
    stats = pc.calculate_regime_statistics(R, regimes)
    # probabilities tilt from a->b
    probs = pd.DataFrame({'a': [1.0, 0.5, 0.0], 'b': [0.0, 0.5, 1.0]})
    r1 = pc.optimize_with_probabilities(stats, probs.iloc[[0]], method=OptimizationMethod.RISK_PARITY)
    r2 = pc.optimize_with_probabilities(stats, probs.iloc[[1]], method=OptimizationMethod.RISK_PARITY)
    r3 = pc.optimize_with_probabilities(stats, probs.iloc[[2]], method=OptimizationMethod.RISK_PARITY)
    # Turnover between consecutive should be bounded
    t12 = np.sum(np.abs(r2.weights.values - r1.weights.values))
    t23 = np.sum(np.abs(r3.weights.values - r2.weights.values))
    assert t12 <= 0.5 + 1e-6
    assert t23 <= 0.5 + 1e-6


