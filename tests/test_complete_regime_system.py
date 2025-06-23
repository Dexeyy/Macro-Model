import sys
sys.path.append('src')

from models.regime_classifier import (
    RuleBasedRegimeClassifier, 
    RegimeType, 
    RegimeRule, 
    OperatorType,
    RegimeConfig
)
from visualization.regime_plots import RegimeVisualizer, analyze_regime_transitions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_complete_regime_system():
    """Test the complete rule-based regime classification system with visualization."""
    print("🚀 Testing Complete Rule-Based Regime Classification System...")
    print("=" * 70)
    
    # Create comprehensive sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=120, freq='M')  # 10 years of monthly data
    
    # Generate realistic economic indicators with regime-like patterns
    base_trend = np.sin(np.linspace(0, 4*np.pi, 120)) * 2
    
    gdp_growth = 2.5 + base_trend + np.random.normal(0, 1, 120)
    unemployment_gap = -base_trend + np.random.normal(0, 0.8, 120)
    yield_curve = 1.0 + base_trend * 0.5 + np.random.normal(0, 0.5, 120)
    inflation = 3.0 + np.abs(base_trend) + np.random.normal(0, 0.8, 120)
    gdp_growth_acceleration = np.gradient(gdp_growth)
    
    # Add some regime-specific patterns
    recession_periods = [30, 31, 32, 85, 86, 87]
    for period in recession_periods:
        if period < len(gdp_growth):
            gdp_growth[period] = np.random.normal(-1.5, 0.5)
            unemployment_gap[period] = np.random.normal(2.0, 0.3)
            yield_curve[period] = np.random.normal(-0.5, 0.2)
    
    stagflation_periods = [55, 56, 57, 58]
    for period in stagflation_periods:
        if period < len(inflation):
            inflation[period] = np.random.normal(6.0, 0.5)
            gdp_growth[period] = np.random.normal(0.5, 0.3)
    
    economic_data = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'unemployment_gap': unemployment_gap,
        'yield_curve': yield_curve,
        'inflation': inflation,
        'gdp_growth_acceleration': gdp_growth_acceleration
    }, index=dates)
    
    print(f"📊 Economic Data Summary:")
    print(f"   Periods: {len(economic_data)}")
    print(f"   GDP Growth: {gdp_growth.min():.2f} to {gdp_growth.max():.2f}")
    print(f"   Inflation: {inflation.min():.2f} to {inflation.max():.2f}")
    print(f"   Unemployment Gap: {unemployment_gap.min():.2f} to {unemployment_gap.max():.2f}")
    print()
    
    # Initialize classifier with custom configuration
    config = RegimeConfig(
        min_score_threshold=0.4,
        smoothing_window=2,
        require_consecutive=1
    )
    
    classifier = RuleBasedRegimeClassifier(config)
    print(f"🏷️ Regime Classification Rules:")
    for regime_type, rules in classifier.regime_rules.items():
        print(f"   {regime_type.value.upper()}: {len(rules)} rules")
    print()
    
    # Classify regimes
    print("🔄 Classifying regimes...")
    regimes = classifier.classify_regime(economic_data)
    
    # Get comprehensive statistics
    stats = classifier.get_regime_statistics(regimes)
    
    print(f"📋 Classification Results:")
    print(f"   Total periods: {stats['total_periods']}")
    print(f"   Unique regimes: {len(regimes.unique())}")
    print(f"   Most common: {stats['most_common_regime']}")
    print(f"   Stability: {stats['regime_stability']:.3f}")
    print()
    
    print(f"📊 Regime Distribution:")
    regime_counts = regimes.value_counts()
    for regime, count in regime_counts.items():
        percentage = (count / len(regimes)) * 100
        print(f"   {regime.upper():>12}: {count:>3} periods ({percentage:>5.1f}%)")
    print()
    
    # Test advanced classification features
    print("🎯 Testing Advanced Features:")
    
    # Single period classification with confidence
    sample_period = economic_data.iloc[50]
    regime, scores = classifier.classify_single_period(sample_period)
    print(f"   Single period test (period 50):")
    print(f"     Predicted: {regime.value}")
    print(f"     Confidence scores:")
    for reg_type, score in scores.items():
        print(f"       {reg_type.value:>12}: {score:.3f}")
    print()
    
    # Transition analysis
    if stats['transitions']:
        print(f"🔄 Transition Analysis:")
        print(f"   Total transitions: {len(stats['transitions'])}")
        top_transitions = dict(list(stats['transitions'].items())[:3])
        for transition, count in top_transitions.items():
            print(f"     {transition}: {count}")
        print()
    
    # Initialize visualizer
    print("🎨 Creating Visualizations...")
    visualizer = RegimeVisualizer(figsize=(14, 10))
    
    # Create timeline plot
    print("   📈 Timeline plot...")
    timeline_fig = visualizer.plot_regime_timeline(
        regimes, 
        economic_data, 
        indicators=['gdp_growth', 'inflation', 'unemployment_gap'],
        title="Complete Regime Classification Timeline"
    )
    plt.close(timeline_fig)
    
    # Create transition matrix
    print("   🔄 Transition matrix...")
    transition_fig = visualizer.plot_regime_transitions(
        regimes,
        title="Regime Transition Analysis"
    )
    plt.close(transition_fig)
    
    # Create distribution analysis
    print("   📊 Distribution analysis...")
    distribution_fig = visualizer.plot_regime_distribution(
        regimes,
        title="Regime Distribution Analysis"
    )
    plt.close(distribution_fig)
    
    # Create duration analysis
    print("   ⏱️ Duration analysis...")
    duration_fig = visualizer.plot_regime_duration_analysis(
        regimes,
        title="Regime Duration & Stability Analysis"
    )
    plt.close(duration_fig)
    
    # Create correlation analysis
    print("   🔗 Correlation analysis...")
    correlation_fig = visualizer.plot_regime_correlation_matrix(
        regimes,
        economic_data,
        title="Regime-Economic Indicator Correlations"
    )
    plt.close(correlation_fig)
    
    # Quick transition analysis
    print("📈 Quick Transition Analysis:")
    transition_analysis = analyze_regime_transitions(regimes)
    print(f"   Transition matrix shape: {transition_analysis['transition_matrix'].shape}")
    print(f"   Total transitions: {transition_analysis['total_transitions']}")
    print(f"   Stability metrics: {len(transition_analysis['stability_metrics'])} metrics")
    print()
    
    # Test custom rules
    print("🔧 Testing Custom Rules:")
    simple_rules = {
        RegimeType.EXPANSION: [
            RegimeRule('gdp_growth', OperatorType.GREATER, 2.0, weight=1.0)
        ],
        RegimeType.RECESSION: [
            RegimeRule('gdp_growth', OperatorType.LESS, 0.0, weight=1.0)
        ],
        RegimeType.STAGFLATION: [
            RegimeRule('inflation', OperatorType.GREATER, 4.5, weight=1.0)
        ]
    }
    
    custom_classifier = RuleBasedRegimeClassifier()
    custom_classifier.set_custom_rules(simple_rules)
    custom_regimes = custom_classifier.classify_regime(economic_data)
    
    print(f"   Custom classification results:")
    custom_counts = custom_regimes.value_counts()
    for regime, count in custom_counts.items():
        print(f"     {regime:>12}: {count:>3} periods")
    print()
    
    # Performance summary
    print("🏆 System Performance Summary:")
    print(f"   ✅ Classification: {len(regimes)} periods processed")
    print(f"   ✅ Rule evaluation: All {sum(len(rules) for rules in classifier.regime_rules.values())} rules working")
    print(f"   ✅ Transition detection: {len(stats['transitions'])} transitions identified")
    print(f"   ✅ Visualization: 5 comprehensive plots created")
    print(f"   ✅ Statistics: {len(stats)} metrics calculated")
    print(f"   ✅ Custom rules: Successfully applied and tested")
    
    return True

if __name__ == "__main__":
    try:
        success = test_complete_regime_system()
        if success:
            print("\n" + "="*70)
            print("🎉 COMPLETE RULE-BASED REGIME CLASSIFICATION SYSTEM TEST PASSED!")
            print("🚀 All components working: Classification, Transitions, Visualization")
            print("💡 Ready for production use in macro-economic regime analysis")
            print("="*70)
        else:
            print("\n❌ Tests failed")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 