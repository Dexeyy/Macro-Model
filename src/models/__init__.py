# Import key classes and functions from submodules for easier access
from .regime_classifier import (
    RuleBasedRegimeClassifier,
    RegimeType,
    RegimeRule,
    OperatorType,
    RegimeConfig,
    create_simple_regime_classifier,
    classify_regimes_from_data
)

from .portfolio import (
    create_equal_weight_portfolio,
    create_regime_based_portfolio,
    calculate_portfolio_metrics,
    optimize_portfolio_weights
)

from .performance_analytics import (
    PerformanceAnalytics,
    quick_performance_analysis,
    compare_portfolios
)

from .performance_analytics import (
    PerformanceAnalytics,
    quick_performance_analysis,
    compare_portfolios
) 