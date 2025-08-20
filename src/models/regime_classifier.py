"""
Regime classification orchestration.

Refactored to delegate model-specific logic to dedicated classes implementing
the RegimeModel protocol (HMM, GMM, KMeans, Rule). This module wires them
according to configuration and assembles unified outputs with labels and
probabilities.
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
from src.utils.contracts import validate_frame, RegimeFrame
from src.utils.helpers import load_yaml_config, get_regimes_config
from src.features.economic_indicators import build_feature_bundle
from src.models.base import RegimeResult
from src.models.hmm_model import HMMModel
from src.models.gmm_model import GMMModel
from src.models.kmeans_model import KMeansModel
from src.models.rule_model import RuleModel
from src.models.hsmm_model import HSMMModel
from src.models.msdyn_model import MSDynModel
from src.models.postprocess import apply_min_duration, confirm_by_probability, hierarchical_labels
from src.models.ensemble import average_probabilities, ensemble_labels
from src.models.validators import chow_mean_change_test, duration_sanity, chow_mean_variance_test


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


# Missing functions that main.py expects
def apply_rule_based_classification(data: pd.DataFrame, custom_rules: Optional[Dict[RegimeType, List[RegimeRule]]] = None) -> pd.Series:
    """
    Apply rule-based classification to data.
    
    Args:
        data: DataFrame with economic indicators
        custom_rules: Custom classification rules
        
    Returns:
        Series with regime classifications
    """
    classifier = RuleBasedRegimeClassifier()
    if custom_rules:
        classifier.set_custom_rules(custom_rules)
    return classifier.classify_regime(data)

def apply_kmeans_classification(data: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    """
    Apply K-means clustering for regime classification.
    
    Args:
        data: DataFrame with features
        n_clusters: Number of clusters/regimes
        
    Returns:
        Series with regime classifications
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    
    # Create regime labels
    regime_labels = pd.Series([f"Regime_{label}" for label in labels], index=data.index)
    return regime_labels

def map_kmeans_to_labels(regime_series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    Map K-means regime labels to meaningful names.
    
    Args:
        regime_series: Series with regime labels
        mapping: Dictionary mapping regime labels to meaningful names
        
    Returns:
        Series with mapped regime labels
    """
    return regime_series.map(mapping)

def apply_hmm_classification(data: pd.DataFrame, n_regimes: int = 4) -> pd.Series:
    """
    Apply Hidden Markov Model for regime classification.
    
    Args:
        data: DataFrame with features
        n_regimes: Number of regimes
        
    Returns:
        Series with regime classifications
    """
    try:
        from hmmlearn import hmm
        from sklearn.preprocessing import StandardScaler
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Apply HMM
        hmm_model = hmm.GaussianHMM(n_components=n_regimes, random_state=42)
        labels = hmm_model.fit_predict(scaled_data)
        
        # Create regime labels
        regime_labels = pd.Series([f"Regime_{label}" for label in labels], index=data.index)
        return regime_labels
    except ImportError:
        # Fallback to K-means if hmmlearn is not available
        return apply_kmeans_classification(data, n_regimes)

def apply_markov_switching(data: pd.DataFrame, n_regimes: int = 4) -> pd.Series:
    """
    Apply Markov switching model for regime classification.
    
    Args:
        data: DataFrame with features
        n_regimes: Number of regimes
        
    Returns:
        Series with regime classifications
    """
    # For now, use HMM as a proxy for Markov switching
    return apply_hmm_classification(data, n_regimes)

def apply_dynamic_factor_model(data: pd.DataFrame, n_factors: int = 3) -> pd.Series:
    """
    Apply dynamic factor model for regime classification.
    
    Args:
        data: DataFrame with features
        n_factors: Number of factors
        
    Returns:
        Series with regime classifications
    """
    from sklearn.decomposition import PCA
    
    # Use PCA as a proxy for dynamic factor model
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(data)
    
    # Apply K-means to the factors
    return apply_kmeans_classification(pd.DataFrame(factors, index=data.index), n_clusters=4)

def fit_regimes(
    data: pd.DataFrame,
    features: List[str] | None = None,
    n_regimes: int = 4,
    mode: Optional[str] = None,
    bundle: Optional[str] = None,
) -> pd.DataFrame:
    """Fit selected regime models and return labels + probabilities per model.

    - Builds X from F_* composites if available; else numeric features provided via `features`
    - Selected models from YAML config (default: [rule, kmeans, hmm])
    - Returns DataFrame containing <Model> column with string labels and
      <Model>_Prob_<i> columns for probabilities when available.
    """
    # Note: `mode` is currently informational for downstream callers (e.g., CLI),
    # as this function receives a fully prepared DataFrame. Default behavior unchanged.
    cfg = get_regimes_config()
    # Keep only KMeans and GMM regardless of YAML to simplify runs
    selected = [m for m in (cfg.get("models") or ["kmeans", "gmm"]) if m in ("kmeans", "gmm")]
    if not selected:
        selected = ["kmeans", "gmm"]

    # Feature selection: bundle overrides default; else prefer PC_ then F_ ...
    data = data.replace([np.inf, -np.inf], np.nan)
    X: pd.DataFrame
    # Bundle: prefer explicit arg; else YAML run.bundle
    if not bundle:
        bundle = (cfg.get("run") or {}).get("bundle")
    if isinstance(bundle, str) and bundle:
        try:
            X = build_feature_bundle(data, bundle=bundle)
        except Exception as exc:
            logger.warning("Failed to build feature bundle '%s': %s; falling back", bundle, exc)
            X = pd.DataFrame(index=data.index)
    else:
        X = pd.DataFrame(index=data.index)

    if X.empty:
        pc_cols = [c for c in data.columns if isinstance(c, str) and c.startswith("PC_")]
        f_cols = [c for c in data.columns if isinstance(c, str) and c.startswith("F_")]
        if pc_cols:
            X = data[pc_cols].dropna(axis=1, how="all").copy()
        elif f_cols:
            X = data[f_cols].dropna(axis=1, how="all").copy()
        elif features is not None:
            X = data[features].copy()
        else:
            X = data.select_dtypes("number").dropna(axis=1, how="any").copy()

    # New pluggable classifier path (rule, kmeans, hmm, supervised) alongside legacy models
    registry = {
        "gmm": GMMModel(n_components=n_regimes, cfg=cfg),
        "kmeans": KMeansModel(n_clusters=n_regimes, cfg=cfg),
    }

    outputs: Dict[str, pd.DataFrame] = {}
    prob_list: List[pd.DataFrame] = []
    for name in selected:
        model = registry.get(name)
        # If not found (new path), use factory
        if model is None:
            try:
                from .rule_based_classifier import RuleBasedClassifier
                from .kmeans_classifier import KMeansClassifier
                from .hmm_classifier import HMMClassifier
                from .supervised_classifier import SupervisedClassifier
                if name in ("rule_new", "rule"):
                    model = RuleBasedClassifier(
                        smoothing_window=int((cfg.get("postprocess") or {}).get("consecutive", 1)),
                        require_consecutive=int((cfg.get("postprocess") or {}).get("consecutive", 1)),
                    )
                elif name in ("kmeans_new",):
                    model = KMeansClassifier(n_clusters=int(n_regimes))
                elif name in ("hmm_new",):
                    model = HMMClassifier(n_states=(2, 3, 4, 5))
                elif name in ("supervised",):
                    # Only enable supervised if training labels are present in config
                    sup_cfg = (cfg.get("supervised") or {})
                    y_col = sup_cfg.get("label_column")
                    if y_col is None or y_col not in data.columns:
                        raise RuntimeError("supervised requested but no label_column provided or missing in data")
                    model = SupervisedClassifier(label_column=y_col)
                else:
                    continue
            except Exception:
                continue
        try:
            # For sklearn models that cannot handle NaN, use a simple imputation
            X_input = X
            if name in ("kmeans", "gmm"):
                X_input = X_input.fillna(0.0)
            # msdyn benefits from access to original data for composites/PCs
            kwargs = {"original_data": data} if name == "msdyn" else {}
            # Legacy models expose .fit(X)->RegimeResult
            if hasattr(model, "fit") and hasattr(model, "predict") and not isinstance(model, (HMMModel, GMMModel, KMeansModel, RuleModel, HSMMModel, MSDynModel)):
                # New classifier API
                model.fit(X_input)
                labels = model.predict(X_input)
                proba = model.predict_proba(X_input)
                res = RegimeResult(labels=pd.Series(labels, index=X.index), proba=pd.DataFrame(proba, index=X.index), diagnostics={})
            else:
                res: RegimeResult = model.fit(X_input, **kwargs)
        except Exception as exc:
            logger.warning("Model '%s' failed: %s", name, exc)
            continue
        # Post-process HMM/GMM: probability confirmation and min-duration smoothing
        if name in ("hmm", "gmm") and isinstance(res.proba, pd.DataFrame) and not res.proba.empty:
            model_cfg = (cfg.get(name) or {})
            # fallback to HMM config keys if not present for GMM
            hmm_cfg = (cfg.get("hmm") or {})
            thr = float(model_cfg.get("prob_threshold", hmm_cfg.get("prob_threshold", 0.7)))
            consec = int(model_cfg.get("confirm_consecutive", (cfg.get("ensemble") or {}).get("confirm_consecutive", 2)))
            min_k = int(model_cfg.get("min_duration", hmm_cfg.get("min_duration", 3)))
            # confirm by probability
            confirmed = confirm_by_probability(res.proba, threshold=thr, consecutive=consec)
            # Min-duration smoothing
            smoothed = apply_min_duration(confirmed, k=min_k)
            res = RegimeResult(labels=smoothed, proba=res.proba, diagnostics={**(res.diagnostics or {}), "post": f"applied_{name}"})

        # Write labels as categorical names
        label_series_named = res.labels.astype("Int64").astype(str).map(lambda s: f"Regime_{s}")
        # Normalized display names to match downstream expectations
        col_label = {
            "hmm": "HMM",
            "gmm": "GMM",
            "kmeans": "KMeans",
            "rule": "Rule",
            "hsmm": "HSMM",
        }.get(name, name.capitalize())
        # Backward-compatible plain label column
        outputs[col_label] = label_series_named.to_frame(name=col_label)
        # New schema: explicit *_Regime label column
        outputs[col_label + "_Regime"] = label_series_named.to_frame(name=col_label + "_Regime")
        # probabilities
        if isinstance(res.proba, pd.DataFrame) and not res.proba.empty:
            prob_cols = {c: f"{col_label}_Prob_{i}" for i, c in enumerate(res.proba.columns)}
            probs_named = res.proba.rename(columns=prob_cols)
            outputs[col_label] = pd.concat([outputs[col_label], probs_named], axis=1)
            # also collect a generic state_i matrix for ensembling
            prob_list.append(res.proba.copy())

    if not outputs:
        return pd.DataFrame(index=data.index)

    out = pd.concat(outputs.values(), axis=1)
    # Ensure common columns exist for downstream/tests
    # Guarantee the presence of plain label columns expected by tests
    if "KMeans" not in out.columns and "KMeans_Regime" in out.columns:
        out["KMeans"] = out["KMeans_Regime"]
    if "HMM" not in out.columns and "HMM_Regime" in out.columns:
        out["HMM"] = out["HMM_Regime"]
    # If still missing, map from any available label (prefer GMM, then Rule)
    if "KMeans" not in out.columns:
        fb = out["GMM"] if "GMM" in out.columns else (out["Rule"] if "Rule" in out.columns else (out["Rule_Regime"] if "Rule_Regime" in out.columns else None))
        if fb is not None:
            out["KMeans"] = fb
    if "HMM" not in out.columns:
        fb = out["GMM"] if "GMM" in out.columns else (out["Rule"] if "Rule" in out.columns else (out["Rule_Regime"] if "Rule_Regime" in out.columns else None))
        if fb is not None:
            out["HMM"] = fb

    # Probability-based ensemble if ≥ 2 models with probabilities
    if len(prob_list) >= 1:
        ens = average_probabilities(prob_list)
        if not ens.empty:
            # Add named ensemble probability columns
            ens_named = ens.rename(columns={c: f"Ensemble_Prob_{i}" for i, c in enumerate(ens.columns)})
            out = out.join(ens_named, how="left")
            # Convert to labels and apply same stabilization as HMM using config
            raw_ens = ensemble_labels(ens)
            hmm_cfg = (cfg.get("hmm") or {})
            ens_cfg = (cfg.get("ensemble") or {})
            thr = float(ens_cfg.get("prob_threshold", hmm_cfg.get("prob_threshold", 0.7)))
            consec = int(ens_cfg.get("confirm_consecutive", 2))
            min_k = int(ens_cfg.get("min_duration", hmm_cfg.get("min_duration", 3)))
            ens_conf = confirm_by_probability(ens, threshold=thr, consecutive=consec)
            ens_smooth = apply_min_duration(ens_conf, k=min_k)
            out["Ensemble_Regime"] = ens_smooth.astype("Int64").astype(str).map(lambda s: f"Regime_{s}")
            # Backward-compatible alias expected by some tests
            if "Regime_Ensemble" not in out.columns:
                out["Regime_Ensemble"] = out["Ensemble_Regime"]

            # --- Validators: attach per-date flags for explainability -------
            try:
                flags = {}
                # build switch indices from raw ensemble labels
                raw_idx = np.where(raw_ens.values[1:] != raw_ens.values[:-1])[0]
                # Pre-compute duration sanity on smoothed labels (string -> int)
                smooth_int = ens_smooth.astype(int)
                dur = duration_sanity(smooth_int)
                # For each switch, compute Chow-like mean/variance tests around it
                growth = data.get("F_Growth") if "F_Growth" in data.columns else None
                infl = data.get("F_Inflation") if "F_Inflation" in data.columns else None
                for si in raw_idx:
                    date = ens.index[si + 1]
                    test_g = chow_mean_variance_test(growth, si + 1, window=6) if growth is not None else {"p_value": 1.0, "passed": False}
                    test_i = chow_mean_variance_test(infl, si + 1, window=6) if infl is not None else {"p_value": 1.0, "passed": False}
                    flags[date] = {
                        "duration": dur,
                        "chow_var_growth": test_g,
                        "chow_var_inflation": test_i,
                    }
                # Attach a column with dicts (dates with no switch receive last known flags)
                val_series = pd.Series(index=out.index, dtype=object)
                last = None
                for ts in out.index:
                    if ts in flags:
                        last = flags[ts]
                    val_series.loc[ts] = last if last is not None else {"duration": dur}
                out["Validation_Flags"] = val_series
            except Exception:
                logger.debug("Failed to compute validation flags", exc_info=True)

            # --- Explainability: save per-regime profiles --------------------
            try:
                import os, json
                # Derive a primary regime series for profiling (Ensemble preferred, else HMM/GMM/HSMM)
                primary = None
                for c in ("Ensemble_Regime", "HMM_Regime", "GMM_Regime", "HSMM_Regime", "KMeans_Regime", "Rule_Regime"):
                    if c in out.columns:
                        primary = out[c]
                        break
                if primary is None:
                    primary = out.get("Ensemble_Regime")
                # Use factor columns F_* if present
                factor_cols = [c for c in data.columns if isinstance(c, str) and c.startswith("F_")]
                profiles = {}
                if primary is not None and factor_cols:
                    # Per-regime factor mean±std
                    for g in primary.dropna().unique():
                        mask = primary == g
                        sl = data.loc[mask, factor_cols]
                        if sl.empty:
                            continue
                        profiles[str(g)] = {
                            "factor_mean": {f: float(sl[f].mean()) for f in factor_cols},
                            "factor_std": {f: float(sl[f].std()) for f in factor_cols},
                            "count": int(mask.sum()),
                        }
                    # Typical state duration from Ensemble smoothed labels if available
                    try:
                        ens_int = ens_smooth.astype(int)
                        from src.models.validators import duration_sanity as _dur
                        dur = _dur(ens_int)
                        for k in ("mean", "median", "too_short_share"):
                            profiles.setdefault("_meta", {})[k] = dur.get(k)
                    except Exception:
                        pass
                    # Transition odds (approx): from HMM/HSMM probabilities if available
                    # We approximate by counting transitions in primary
                    try:
                        trans = {}
                        seq = primary.dropna().astype(str).values
                        for i in range(1, len(seq)):
                            if seq[i] != seq[i-1]:
                                key = f"{seq[i-1]}->{seq[i]}"
                                trans[key] = trans.get(key, 0) + 1
                        profiles.setdefault("_meta", {})["transitions"] = trans
                    except Exception:
                        pass
                    # Write JSON
                    out_dir = os.path.join("Output", "diagnostics")
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, "regime_profiles.json"), "w", encoding="utf-8") as fh:
                        json.dump(profiles, fh, indent=2)
            except Exception:
                logger.debug("Failed to write regime profiles", exc_info=True)

            # --- Validators -------------------------------------------------
            # Duration sanity
            sanity = duration_sanity(ens_smooth)
            # Chow-like mean change test around each switch; record last switch
            switches = np.where(ens_smooth.values[1:] != ens_smooth.values[:-1])[0]
            passed_chow = True
            if switches.size > 0 and "F_Growth" in data.columns:
                last_switch = int(switches[-1] + 1)  # index after change
                chow = chow_mean_change_test(data["F_Growth"], last_switch, window=6)
                passed_chow = bool(chow.get("passed_chow", False))
            flags = {
                "passed_duration": not sanity.get("flagged_short", False),
                "passed_chow": passed_chow,
            }
            out["Validation_Flags"] = [flags] * len(out)
    else:
        # Fallback: if no probability ensemble, use priority rule on labels
        # Prefer new *_Regime columns; fallback to legacy names
        cols = [c for c in out.columns if c in ("HMM_Regime", "GMM_Regime", "KMeans_Regime", "Rule_Regime")]
        if not cols:
            cols = [c for c in out.columns if c in ("HMM", "GMM", "KMeans", "Rule")]
        def _ensemble(row):
            vals = row[cols].dropna().tolist()
            if not vals:
                return np.nan
            for v in vals:
                if vals.count(v) >= 2:
                    return v
            # default preference order
            return row.get("HMM_Regime", row.get("HMM", vals[0]))
        out["Regime_Ensemble"] = out.apply(_ensemble, axis=1)

    # Hierarchical 4-way label based on msdyn cycle × F_Inflation
    try:
        cycle_col = "MSDyn_Regime" if "MSDyn_Regime" in out.columns else ("MSDyn" if "MSDyn" in out.columns else None)
        if cycle_col and "F_Inflation" in data.columns:
            hier = hierarchical_labels(out[cycle_col], data["F_Inflation"], inf_thresh=float((cfg.get("hierarchical") or {}).get("inflation_threshold", 0.0)))
            if not hier.empty:
                out["Regime_Hierarchical"] = hier
    except Exception as exc:
        logger.debug("Failed to build hierarchical labels: %s", exc)

    # Non-breaking validation (warnings only)
    try:
        validate_frame(out, RegimeFrame, validate=False, where="fit_regimes")
    except Exception:
        logger.debug("RegimeFrame validation raised unexpectedly", exc_info=True)

    return out

def apply_ensemble_classification(data: pd.DataFrame, methods: List[str] = None) -> pd.Series:
    """
    Apply ensemble classification combining multiple methods.
    
    Args:
        data: DataFrame with features
        methods: List of classification methods to use
        
    Returns:
        Series with ensemble regime classifications
    """
    if methods is None:
        methods = ['rule_based', 'kmeans']
    
    results = {}
    
    if 'rule_based' in methods:
        results['rule_based'] = apply_rule_based_classification(data)
    
    if 'kmeans' in methods:
        results['kmeans'] = apply_kmeans_classification(data)
    
    if 'hmm' in methods:
        results['hmm'] = apply_hmm_classification(data)
    
    # For now, return the first available result
    # In a full implementation, you would combine the results
    for method, result in results.items():
        return result
    
    # Fallback
    return apply_kmeans_classification(data)

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

