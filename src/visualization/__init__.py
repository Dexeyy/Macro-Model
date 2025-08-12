# Import visualization functions for easier access
from .plots import (
    plot_regime_timeline,
    plot_regime_performance,
    plot_regime_feature_distribution,
    plot_regime_transitions
)

try:
    from .ensemble_plots import (
        plot_ensemble_comparison,
        plot_regime_agreement_matrix
    )
except Exception:  # optional dependency; provide no-op fallbacks for tests
    def plot_ensemble_comparison(*args, **kwargs):  # type: ignore
        return None

    def plot_regime_agreement_matrix(*args, **kwargs):  # type: ignore
        return None

from .regime_plots import (
    RegimeVisualizer,
    create_quick_regime_timeline,
    analyze_regime_transitions
) 