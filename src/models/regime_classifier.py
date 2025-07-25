"""
Rule-Based Regime Classification System

This module implements a sophisticated rule-based regime classifier that uses
predefined economic indicators and thresholds to classify market regimes.
The system supports multiple regime types (expansion, recession, recovery, 
stagflation) with customizable rules and evaluation logic.

Author: Macro Regime Analysis System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Enumeration of supported market regime types."""
    EXPANSION = "expansion"
    RECESSION = "recession"
    RECOVERY = "recovery"
    STAGFLATION = "stagflation"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class OperatorType(Enum):
    """Enumeration of supported comparison operators."""
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


@dataclass
class RegimeRule:
    """
    Configuration for a single regime rule.
    
    Attributes:
        indicator: Name of the economic indicator
        operator: Comparison operator
        threshold: Threshold value for comparison
        weight: Weight for this rule in regime scoring (0-1)
        required: Whether this rule must be satisfied
    """
    indicator: str
    operator: OperatorType
    threshold: float
    weight: float = 1.0
    required: bool = False
    
    def __post_init__(self):
        """Validate rule configuration."""
        if not 0 <= self.weight <= 1:
            raise ValueError("Rule weight must be between 0 and 1")
        if isinstance(self.operator, str):
            self.operator = OperatorType(self.operator)


@dataclass
class RegimeConfig:
    """
    Configuration for regime classification.
    
    Attributes:
        min_score_threshold: Minimum score to classify a regime
        default_regime: Default regime when no rules are satisfied
        smoothing_window: Window for smoothing regime transitions
        require_consecutive: Number of consecutive periods required
    """
    min_score_threshold: float = 0.5
    default_regime: RegimeType = RegimeType.NEUTRAL
    smoothing_window: int = 3
    require_consecutive: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.min_score_threshold <= 1:
            raise ValueError("min_score_threshold must be between 0 and 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if self.require_consecutive < 1:
            raise ValueError("require_consecutive must be >= 1")


class RuleBasedRegimeClassifier:
    """
    Rule-based regime classification system.
    
    This classifier uses predefined economic indicator thresholds to identify
    different market regimes. It supports customizable rules, weighted scoring,
    and various evaluation methods.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize the regime classifier.
        
        Args:
            config: Configuration for regime classification
        """
        self.config = config or RegimeConfig()
        self.regime_rules = self._initialize_default_rules()
        self.classification_history = []
        
        logger.info("RuleBasedRegimeClassifier initialized")
    
    def _initialize_default_rules(self) -> Dict[RegimeType, List[RegimeRule]]:
        """Initialize default regime classification rules."""
        rules = {
            RegimeType.EXPANSION: [
                RegimeRule('gdp_growth', OperatorType.GREATER, 2.5, weight=1.0),
                RegimeRule('unemployment_gap', OperatorType.LESS, 0, weight=0.8),
                RegimeRule('yield_curve', OperatorType.GREATER, 0.5, weight=0.7),
                RegimeRule('inflation', OperatorType.LESS, 4.0, weight=0.6)
            ],
            RegimeType.RECESSION: [
                RegimeRule('gdp_growth', OperatorType.LESS, 0, weight=1.0, required=True),
                RegimeRule('unemployment_gap', OperatorType.GREATER, 1.0, weight=0.9),
                RegimeRule('yield_curve', OperatorType.LESS, 0, weight=0.8),
                RegimeRule('inflation', OperatorType.LESS, 2.0, weight=0.5)
            ],
            RegimeType.RECOVERY: [
                RegimeRule('gdp_growth', OperatorType.GREATER, 0, weight=1.0),
                RegimeRule('gdp_growth_acceleration', OperatorType.GREATER, 0, weight=0.9),
                RegimeRule('unemployment_gap', OperatorType.GREATER, 0, weight=0.7),
                RegimeRule('yield_curve', OperatorType.GREATER, 0, weight=0.6)
            ],
            RegimeType.STAGFLATION: [
                RegimeRule('inflation', OperatorType.GREATER, 4.0, weight=1.0, required=True),
                RegimeRule('gdp_growth', OperatorType.LESS, 1.5, weight=0.9),
                RegimeRule('unemployment_gap', OperatorType.GREATER, 0, weight=0.8),
                RegimeRule('yield_curve', OperatorType.LESS, 1.0, weight=0.5)
            ]
        }
        
        logger.info(f"Initialized default rules for {len(rules)} regime types")
        return rules
    
    def set_custom_rules(self, custom_rules: Dict[RegimeType, List[RegimeRule]]):
        """
        Set custom regime classification rules.
        
        Args:
            custom_rules: Dictionary mapping regime types to rule lists
        """
        # Validate custom rules
        for regime_type, rules in custom_rules.items():
            if not isinstance(regime_type, RegimeType):
                raise ValueError(f"Invalid regime type: {regime_type}")
            if not isinstance(rules, list):
                raise ValueError(f"Rules for {regime_type} must be a list")
            for rule in rules:
                if not isinstance(rule, RegimeRule):
                    raise ValueError(f"Invalid rule type: {type(rule)}")
        
        self.regime_rules = custom_rules
        logger.info(f"Set custom rules for {len(custom_rules)} regime types")
    
    def add_regime_rule(self, regime_type: RegimeType, rule: RegimeRule):
        """
        Add a rule to an existing regime type.
        
        Args:
            regime_type: Target regime type
            rule: Rule to add
        """
        if regime_type not in self.regime_rules:
            self.regime_rules[regime_type] = []
        
        self.regime_rules[regime_type].append(rule)
        logger.info(f"Added rule for {regime_type.value}: {rule.indicator}")
    
    def _evaluate_rule(self, data_point: pd.Series, rule: RegimeRule) -> bool:
        """
        Evaluate if a data point satisfies a specific rule.
        
        Args:
            data_point: Row of economic indicators
            rule: Rule to evaluate
            
        Returns:
            Boolean indicating if rule is satisfied
        """
        if rule.indicator not in data_point:
            warnings.warn(f"Indicator '{rule.indicator}' not found in data")
            return False
        
        value = data_point[rule.indicator]
        
        # Handle NaN values
        if pd.isna(value):
            return False
        
        threshold = rule.threshold
        
        if rule.operator == OperatorType.GREATER:
            return value > threshold
        elif rule.operator == OperatorType.LESS:
            return value < threshold
        elif rule.operator == OperatorType.GREATER_EQUAL:
            return value >= threshold
        elif rule.operator == OperatorType.LESS_EQUAL:
            return value <= threshold
        elif rule.operator == OperatorType.EQUAL:
            return abs(value - threshold) < 1e-10
        elif rule.operator == OperatorType.NOT_EQUAL:
            return abs(value - threshold) >= 1e-10
        
        return False
    
    def _calculate_regime_score(self, data_point: pd.Series, 
                               rules: List[RegimeRule]) -> float:
        """
        Calculate regime score based on rule satisfaction.
        
        Args:
            data_point: Row of economic indicators
            rules: List of rules for the regime
            
        Returns:
            Weighted score between 0 and 1
        """
        if not rules:
            return 0.0
        
        satisfied_weight = 0.0
        total_weight = 0.0
        required_satisfied = True
        
        for rule in rules:
            rule_satisfied = self._evaluate_rule(data_point, rule)
            
            # Check required rules
            if rule.required and not rule_satisfied:
                required_satisfied = False
            
            if rule_satisfied:
                satisfied_weight += rule.weight
            total_weight += rule.weight
        
        # If required rules are not satisfied, return 0
        if not required_satisfied:
            return 0.0
        
        # Calculate weighted score
        if total_weight == 0:
            return 0.0
        
        return satisfied_weight / total_weight
    
    def classify_single_period(self, data_point: pd.Series) -> Tuple[RegimeType, Dict[RegimeType, float]]:
        """
        Classify regime for a single time period.
        
        Args:
            data_point: Row of economic indicators
            
        Returns:
            Tuple of (predicted regime, regime scores)
        """
        regime_scores = {}
        
        # Calculate scores for each regime
        for regime_type, rules in self.regime_rules.items():
            score = self._calculate_regime_score(data_point, rules)
            regime_scores[regime_type] = score
        
        # Find regime with highest score
        if not regime_scores:
            return self.config.default_regime, regime_scores
        
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        # Check if best score meets threshold
        if best_regime[1] >= self.config.min_score_threshold:
            return best_regime[0], regime_scores
        else:
            return self.config.default_regime, regime_scores
    
    def classify_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify market regimes for time series data.
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            Series with regime classifications
        """
        if data.empty:
            return pd.Series(dtype='object')
        
        regimes = pd.Series(index=data.index, dtype='object', name='regime')
        scores_history = []
        
        logger.info(f"Classifying regimes for {len(data)} periods")
        
        # Classify each period
        for date, row in data.iterrows():
            regime, scores = self.classify_single_period(row)
            regimes[date] = regime.value
            scores_history.append(scores)
        
        # Apply smoothing if configured
        if self.config.smoothing_window > 1:
            regimes = self._smooth_regime_transitions(regimes)
        
        # Apply consecutive period requirement
        if self.config.require_consecutive > 1:
            regimes = self._enforce_consecutive_periods(regimes)
        
        # Store classification history
        self.classification_history = scores_history
        
        logger.info(f"Classification complete. Unique regimes: {regimes.unique()}")
        return regimes
    
    def _smooth_regime_transitions(self, regimes: pd.Series) -> pd.Series:
        """
        Smooth regime transitions using a rolling window.
        
        Args:
            regimes: Raw regime classifications
            
        Returns:
            Smoothed regime classifications
        """
        smoothed = regimes.copy()
        window = self.config.smoothing_window
        
        for i in range(window, len(regimes)):
            window_regimes = regimes.iloc[i-window+1:i+1]
            # Use most common regime in window
            mode_regime = window_regimes.mode()
            if len(mode_regime) > 0:
                smoothed.iloc[i] = mode_regime.iloc[0]
        
        return smoothed
    
    def _enforce_consecutive_periods(self, regimes: pd.Series) -> pd.Series:
        """
        Enforce minimum consecutive period requirement.
        
        Args:
            regimes: Regime classifications
            
        Returns:
            Filtered regime classifications
        """
        if self.config.require_consecutive <= 1:
            return regimes
        
        filtered = regimes.copy()
        required = self.config.require_consecutive
        
        i = 0
        while i < len(regimes):
            current_regime = regimes.iloc[i]
            count = 1
            
            # Count consecutive occurrences
            j = i + 1
            while j < len(regimes) and regimes.iloc[j] == current_regime:
                count += 1
                j += 1
            
            # If not enough consecutive periods, mark as default
            if count < required:
                filtered.iloc[i:j] = self.config.default_regime.value
            
            i = j
        
        return filtered
    
    def get_regime_statistics(self, regimes: pd.Series) -> Dict[str, Any]:
        """
        Calculate statistics for regime classifications.
        
        Args:
            regimes: Regime classifications
            
        Returns:
            Dictionary with regime statistics
        """
        if regimes.empty:
            return {}
        
        # Count occurrences
        regime_counts = regimes.value_counts()
        total_periods = len(regimes)
        
        # Calculate percentages
        regime_percentages = (regime_counts / total_periods * 100).round(2)
        
        # Find transitions
        transitions = self._count_regime_transitions(regimes)
        
        # Calculate average duration
        average_durations = self._calculate_average_durations(regimes)
        
        return {
            'total_periods': total_periods,
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': regime_percentages.to_dict(),
            'transitions': transitions,
            'average_durations': average_durations,
            'most_common_regime': regime_counts.index[0] if len(regime_counts) > 0 else None,
            'regime_stability': self._calculate_stability_metric(regimes)
        }
    
    def _count_regime_transitions(self, regimes: pd.Series) -> Dict[str, int]:
        """Count transitions between regimes."""
        if len(regimes) < 2:
            return {}
        
        transitions = {}
        for i in range(1, len(regimes)):
            prev_regime = regimes.iloc[i-1]
            curr_regime = regimes.iloc[i]
            
            if prev_regime != curr_regime:
                transition = f"{prev_regime} -> {curr_regime}"
                transitions[transition] = transitions.get(transition, 0) + 1
        
        return transitions
    
    def _calculate_average_durations(self, regimes: pd.Series) -> Dict[str, float]:
        """Calculate average duration for each regime."""
        if regimes.empty:
            return {}
        
        durations = {}
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                # End of current regime
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(current_duration)
                
                # Start new regime
                current_regime = regimes.iloc[i]
                current_duration = 1
        
        # Add final regime duration
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(current_duration)
        
        # Calculate averages
        avg_durations = {}
        for regime, duration_list in durations.items():
            avg_durations[regime] = np.mean(duration_list)
        
        return avg_durations
    
    def _calculate_stability_metric(self, regimes: pd.Series) -> float:
        """Calculate regime stability metric (0-1, higher is more stable)."""
        if len(regimes) < 2:
            return 1.0
        
        transitions = sum(1 for i in range(1, len(regimes)) 
                         if regimes.iloc[i] != regimes.iloc[i-1])
        max_transitions = len(regimes) - 1
        
        return 1 - (transitions / max_transitions) if max_transitions > 0 else 1.0
    
    def get_classification_confidence(self, period_index: int = -1) -> Dict[RegimeType, float]:
        """
        Get classification confidence scores for a specific period.
        
        Args:
            period_index: Index of period (-1 for latest)
            
        Returns:
            Dictionary with confidence scores for each regime
        """
        if not self.classification_history:
            return {}
        
        if period_index < 0:
            period_index = len(self.classification_history) + period_index
        
        if 0 <= period_index < len(self.classification_history):
            return self.classification_history[period_index]
        
        return {}


# Convenience functions
def create_simple_regime_classifier(
    expansion_gdp: float = 2.5,
    recession_gdp: float = 0.0,
    stagflation_inflation: float = 4.0
) -> RuleBasedRegimeClassifier:
    """
    Create a simple regime classifier with basic rules.
    
    Args:
        expansion_gdp: GDP growth threshold for expansion
        recession_gdp: GDP growth threshold for recession
        stagflation_inflation: Inflation threshold for stagflation
        
    Returns:
        Configured regime classifier
    """
    classifier = RuleBasedRegimeClassifier()
    
    # Simplified rules
    simple_rules = {
        RegimeType.EXPANSION: [
            RegimeRule('gdp_growth', OperatorType.GREATER, expansion_gdp, weight=1.0)
        ],
        RegimeType.RECESSION: [
            RegimeRule('gdp_growth', OperatorType.LESS, recession_gdp, weight=1.0)
        ],
        RegimeType.STAGFLATION: [
            RegimeRule('inflation', OperatorType.GREATER, stagflation_inflation, weight=1.0)
        ]
    }
    
    classifier.set_custom_rules(simple_rules)
    return classifier


def classify_regimes_from_data(
    data: pd.DataFrame,
    custom_rules: Optional[Dict[RegimeType, List[RegimeRule]]] = None,
    config: Optional[RegimeConfig] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convenience function to classify regimes from data.
    
    Args:
        data: DataFrame with economic indicators
        custom_rules: Custom classification rules
        config: Regime classification configuration
        
    Returns:
        Tuple of (regime classifications, statistics)
    """
    classifier = RuleBasedRegimeClassifier(config)
    
    if custom_rules:
        classifier.set_custom_rules(custom_rules)
    
    regimes = classifier.classify_regime(data)
    statistics = classifier.get_regime_statistics(regimes)
    
    return regimes, statistics


if __name__ == "__main__":
    # Example usage
    print("Testing Rule-Based Regime Classification System...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=100, freq='M')
    
    # Generate realistic economic indicators
    gdp_base = 2.0 + np.random.normal(0, 1.5, 100)
    gdp_growth = np.cumsum(gdp_base) / 10
    unemployment_gap = np.random.normal(0, 1, 100)
    yield_curve = np.random.normal(1, 0.5, 100)
    inflation = 2.5 + np.random.normal(0, 1, 100)
    
    sample_data = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'unemployment_gap': unemployment_gap,
        'yield_curve': yield_curve,
        'inflation': inflation,
        'gdp_growth_acceleration': np.gradient(gdp_growth)
    }, index=dates)
    
    # Initialize classifier
    classifier = RuleBasedRegimeClassifier()
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Available indicators: {list(sample_data.columns)}")
    
    # Classify regimes
    regimes = classifier.classify_regime(sample_data)
    
    print(f"\nRegime classifications:")
    print(regimes.value_counts())
    
    # Get statistics
    stats = classifier.get_regime_statistics(regimes)
    
    print(f"\nRegime Statistics:")
    print(f"Total periods: {stats['total_periods']}")
    print(f"Regime percentages: {stats['regime_percentages']}")
    print(f"Most common regime: {stats['most_common_regime']}")
    print(f"Regime stability: {stats['regime_stability']:.3f}")
    
    if stats['transitions']:
        print(f"Top transitions: {dict(list(stats['transitions'].items())[:3])}")
    
    print(f"\nRule-Based Regime Classification System created successfully!")
    print(f"Available regime types: {[r.value for r in RegimeType]}")
    print(f"Supports custom rules, weighted scoring, and transition analysis.")
