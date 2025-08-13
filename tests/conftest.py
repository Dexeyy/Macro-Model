import pytest
import numpy as np
import pandas as pd

from src.models.performance_analytics import PerformanceAnalytics
from src.models.portfolio import PortfolioConstructor


@pytest.fixture
def analyzer():
    return PerformanceAnalytics(risk_free_rate=0.02, annualization_factor=252)


@pytest.fixture
def portfolio_returns():
    idx = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0005, 0.02, len(idx)), index=idx, name='portfolio')


@pytest.fixture
def prices_df():
    idx = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(1)
    rets_a = np.random.normal(0.0006, 0.02, len(idx))
    rets_b = np.random.normal(0.0004, 0.018, len(idx))
    pa = (1 + pd.Series(rets_a, index=idx)).cumprod() * 100
    pb = (1 + pd.Series(rets_b, index=idx)).cumprod() * 100
    return pd.DataFrame({'A': pa, 'B': pb}, index=idx)


@pytest.fixture
def weights():
    return pd.Series({'A': 0.6, 'B': 0.4})


@pytest.fixture
def regime_series(portfolio_returns):
    # simple regimes alternating monthly
    idx = portfolio_returns.index
    months = idx.to_period('M')
    vals = (months.factorize()[0] % 2)
    return pd.Series(np.where(vals == 0, 'bull', 'bear'), index=idx)


@pytest.fixture
def benchmark_returns(portfolio_returns):
    np.random.seed(2)
    noise = np.random.normal(0, 0.005, len(portfolio_returns))
    return (portfolio_returns * 0.8 + noise).rename('benchmark')


# Fixtures for portfolio construction tests
@pytest.fixture
def constructor():
    return PortfolioConstructor()


@pytest.fixture
def returns_df():
    idx = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    a = np.random.normal(0.0004, 0.02, len(idx))
    b = np.random.normal(0.0003, 0.018, len(idx))
    c = np.random.normal(0.0002, 0.015, len(idx))
    d = np.random.normal(0.0005, 0.025, len(idx))
    e = np.random.normal(0.0004, 0.02, len(idx))
    return pd.DataFrame({'STOCK_A': a, 'STOCK_B': b, 'BOND_A': c, 'COMMODITY_A': d, 'REIT_A': e}, index=idx)


@pytest.fixture
def regime_stats(constructor, returns_df):
    # Build a regime series aligned to returns_df for tests that need it
    idx = returns_df.index
    months = idx.to_period('M')
    vals = (months.factorize()[0] % 2)
    regimes = pd.Series(np.where(vals == 0, 'bull', 'bear'), index=idx)
    return constructor.calculate_regime_statistics(returns_df, regimes)


