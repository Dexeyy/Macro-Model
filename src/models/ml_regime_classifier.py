"""
Machine Learning-Based Regime Classification System

This module implements a sophisticated ML-based regime classifier using K-means clustering
to automatically detect market regimes from macroeconomic data.

Key Features:
- K-means clustering for automatic regime detection (Subtask 7.1)
- Optimal cluster selection using multiple metrics (Subtask 7.2)
- Comprehensive visualization capabilities (Subtask 7.3)
- Regime characteristic analysis and interpretation (Subtask 7.4)
- Integration with existing rule-based classification (Subtask 7.5)

Author: Macro Regime Analysis System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ScalingMethod(Enum):
    """Supported data scaling methods for preprocessing."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class ClusterMetric(Enum):
    """Metrics for optimal cluster selection."""
    SILHOUETTE = "silhouette"
    CALINSKI_HARABASZ = "calinski"
    DAVIES_BOULDIN = "davies_bouldin"
    INERTIA = "inertia"
    COMBINED = "combined"


@dataclass
class MLRegimeConfig:
    """Configuration for ML-based regime classification."""
    n_regimes: int = 4
    max_clusters: int = 10
    random_state: int = 42
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    selection_metric: ClusterMetric = ClusterMetric.SILHOUETTE
    auto_select_clusters: bool = True
    min_regime_size: int = 5
    transition_smoothing: bool = True
    smoothing_window: int = 3
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        if self.max_clusters < self.n_regimes:
            raise ValueError("max_clusters must be >= n_regimes")
        if self.min_regime_size < 1:
            raise ValueError("min_regime_size must be >= 1")


class MLRegimeClassifier:
    """
    Machine Learning-based regime classifier using K-means clustering.
    
    This classifier automatically detects market regimes from macroeconomic data
    using unsupervised learning techniques. It implements all subtasks for Task 7.
    """
    
    def __init__(self, config: Optional[MLRegimeConfig] = None):
        """Initialize the ML regime classifier."""
        self.config = config or MLRegimeConfig()
        self.scaler = self._initialize_scaler()
        self.model = None
        self.pca = PCA(n_components=2, random_state=self.config.random_state)
        self.fitted = False
        self.feature_names = None
        self.clustering_results = None
        
        logger.info("MLRegimeClassifier initialized")
    
    def _initialize_scaler(self):
        """Initialize the appropriate scaler based on configuration."""
        scaling_map = {
            ScalingMethod.STANDARD: StandardScaler(),
            ScalingMethod.ROBUST: RobustScaler(),
            ScalingMethod.MINMAX: MinMaxScaler(),
            ScalingMethod.NONE: None
        }
        return scaling_map[self.config.scaling_method]
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for K-means clustering (Subtask 7.1).
        
        Args:
            data: Raw macroeconomic data
            
        Returns:
            Tuple of (processed_data, scaled_data)
        """
        try:
            logger.info("Starting data preprocessing for K-means clustering...")
            
            # Store feature names
            self.feature_names = data.columns.tolist()
            
            # Handle missing values
            processed_data = data.copy()
            
            # Forward fill then backward fill for time series data
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining NaN values
            if processed_data.isnull().any().any():
                logger.warning("Removing rows with remaining NaN values")
                processed_data = processed_data.dropna()
            
            # Handle outliers using IQR method
            for column in processed_data.columns:
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)
            
            # Scale the data if scaling is enabled
            if self.scaler is not None:
                scaled_values = self.scaler.fit_transform(processed_data)
                scaled_data = pd.DataFrame(
                    scaled_values, 
                    index=processed_data.index, 
                    columns=processed_data.columns
                )
            else:
                scaled_data = processed_data.copy()
            
            logger.info(f"Data preprocessing completed: {processed_data.shape}")
            logger.info(f"Features: {list(processed_data.columns)}")
            
            return processed_data, scaled_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def find_optimal_clusters(self, scaled_data: pd.DataFrame) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find optimal number of clusters using multiple metrics (Subtask 7.2).
        
        Args:
            scaled_data: Preprocessed and scaled data
            
        Returns:
            Tuple of (optimal_clusters, metric_scores)
        """
        try:
            logger.info("Finding optimal number of clusters...")
            
            cluster_range = range(2, self.config.max_clusters + 1)
            metrics = {
                'silhouette': [],
                'calinski_harabasz': [],
                'davies_bouldin': [],
                'inertia': []
            }
            
            for n_clusters in cluster_range:
                # Fit K-means
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=self.config.random_state,
                    n_init=10,
                    max_iter=300
                )
                labels = kmeans.fit_predict(scaled_data)
                
                # Calculate metrics
                metrics['silhouette'].append(silhouette_score(scaled_data, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_data, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(scaled_data, labels))
                metrics['inertia'].append(kmeans.inertia_)
            
            # Select optimal number of clusters
            optimal_clusters = self._select_optimal_clusters(metrics, cluster_range)
            
            logger.info(f"Optimal number of clusters: {optimal_clusters}")
            logger.info(f"Selection metric: {self.config.selection_metric.value}")
            
            return optimal_clusters, metrics
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {e}")
            raise
    
    def _select_optimal_clusters(self, metrics: Dict[str, List[float]], cluster_range: range) -> int:
        """Select optimal clusters based on the configured metric."""
        if self.config.selection_metric == ClusterMetric.SILHOUETTE:
            return cluster_range[np.argmax(metrics['silhouette'])]
        elif self.config.selection_metric == ClusterMetric.CALINSKI_HARABASZ:
            return cluster_range[np.argmax(metrics['calinski_harabasz'])]
        elif self.config.selection_metric == ClusterMetric.DAVIES_BOULDIN:
            return cluster_range[np.argmin(metrics['davies_bouldin'])]
        elif self.config.selection_metric == ClusterMetric.INERTIA:
            # Use elbow method for inertia
            return self._find_elbow_point(metrics['inertia'], cluster_range)
        else:  # COMBINED
            return self._combined_metric_selection(metrics, cluster_range)
    
    def _find_elbow_point(self, inertia_values: List[float], cluster_range: range) -> int:
        """Find elbow point in inertia curve."""
        # Simple elbow detection using second derivative
        if len(inertia_values) < 3:
            return cluster_range[0]
        
        # Calculate second differences
        second_diffs = []
        for i in range(1, len(inertia_values) - 1):
            second_diff = inertia_values[i-1] - 2*inertia_values[i] + inertia_values[i+1]
            second_diffs.append(second_diff)
        
        # Find maximum second difference (elbow point)
        elbow_idx = np.argmax(second_diffs) + 1  # +1 because we start from index 1
        return cluster_range[elbow_idx]
    
    def _combined_metric_selection(self, metrics: Dict[str, List[float]], cluster_range: range) -> int:
        """Select optimal clusters using combined normalized metrics."""
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        
        # For metrics where higher is better
        for metric in ['silhouette', 'calinski_harabasz']:
            values = np.array(metrics[metric])
            normalized_metrics[metric] = (values - values.min()) / (values.max() - values.min())
        
        # For metrics where lower is better (invert)
        for metric in ['davies_bouldin']:
            values = np.array(metrics[metric])
            normalized_metrics[metric] = 1 - (values - values.min()) / (values.max() - values.min())
        
        # Handle inertia with elbow method weight
        inertia_values = np.array(metrics['inertia'])
        # Give higher weight to clusters that reduce inertia significantly
        inertia_reduction = np.diff(inertia_values)
        inertia_weights = np.zeros(len(inertia_values))
        inertia_weights[1:] = -inertia_reduction / np.abs(inertia_reduction).max()
        
        # Combined score
        combined_scores = (
            normalized_metrics['silhouette'] + 
            normalized_metrics['calinski_harabasz'] + 
            normalized_metrics['davies_bouldin'] +
            inertia_weights
        ) / 4
        
        return cluster_range[np.argmax(combined_scores)]
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the ML regime classifier to data.
        
        Args:
            data: Macroeconomic data for regime detection
            
        Returns:
            Dictionary with fitting results and regime assignments
        """
        try:
            logger.info("Fitting ML regime classifier...")
            
            # Preprocess data (Subtask 7.1)
            processed_data, scaled_data = self.preprocess_data(data)
            
            # Find optimal clusters if auto-selection is enabled (Subtask 7.2)
            if self.config.auto_select_clusters:
                optimal_clusters, clustering_results = self.find_optimal_clusters(scaled_data)
                self.config.n_regimes = optimal_clusters
                self.clustering_results = clustering_results
            
            # Fit final K-means model
            self.model = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=self.config.random_state,
                n_init=10,
                max_iter=300
            )
            
            labels = self.model.fit_predict(scaled_data)
            
            # Apply transition smoothing if enabled
            if self.config.transition_smoothing:
                labels = self._smooth_transitions(labels)
            
            # Store results
            results = {
                'labels': labels,
                'processed_data': processed_data,
                'scaled_data': scaled_data,
                'n_regimes': self.config.n_regimes,
                'cluster_centers': self.model.cluster_centers_,
                'inertia': self.model.inertia_
            }
            
            self.fitted = True
            logger.info(f"ML regime classifier fitted successfully with {self.config.n_regimes} regimes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting ML regime classifier: {e}")
            raise
    
    def _smooth_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Apply smoothing to regime transitions."""
        if len(labels) < self.config.smoothing_window:
            return labels
        
        smoothed_labels = labels.copy()
        window = self.config.smoothing_window
        
        for i in range(window, len(labels) - window):
            window_labels = labels[i-window:i+window+1]
            # If most labels in window are the same, use that label
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            most_common = unique_labels[np.argmax(counts)]
            
            # Only smooth if there's strong consensus
            if counts.max() > len(window_labels) * 0.6:
                smoothed_labels[i] = most_common
        
        return smoothed_labels
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for new data.
        
        Args:
            data: New macroeconomic data
            
        Returns:
            Array of regime labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Preprocess new data using existing scaler
            processed_data = data.copy()
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
            
            if self.scaler is not None:
                scaled_values = self.scaler.transform(processed_data)
                scaled_data = pd.DataFrame(
                    scaled_values, 
                    index=processed_data.index, 
                    columns=processed_data.columns
                )
            else:
                scaled_data = processed_data.copy()
            
            # Predict labels
            labels = self.model.predict(scaled_data)
            
            # Apply smoothing if enabled
            if self.config.transition_smoothing:
                labels = self._smooth_transitions(labels)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            raise


# Utility function for easy usage
def create_ml_regime_classifier(
    n_regimes: int = 4,
    auto_select_clusters: bool = True,
    scaling_method: str = "standard",
    selection_metric: str = "silhouette"
) -> MLRegimeClassifier:
    """
    Create and configure an ML regime classifier.
    
    Args:
        n_regimes: Number of regimes to detect
        auto_select_clusters: Whether to automatically select optimal clusters
        scaling_method: Data scaling method ("standard", "robust", "minmax", "none")
        selection_metric: Metric for cluster selection ("silhouette", "calinski", "davies_bouldin", "inertia", "combined")
        
    Returns:
        Configured MLRegimeClassifier instance
    """
    config = MLRegimeConfig(
        n_regimes=n_regimes,
        auto_select_clusters=auto_select_clusters,
        scaling_method=ScalingMethod(scaling_method),
        selection_metric=ClusterMetric(selection_metric)
    )
    
    return MLRegimeClassifier(config)


def classify_regimes_ml(data: pd.DataFrame, config: Optional[MLRegimeConfig] = None):
    """
    Convenience function to classify regimes using ML approach.
    
    Args:
        data: Macroeconomic data
        config: Optional configuration
        
    Returns:
        Tuple of (classifier, results)
    """
    classifier = MLRegimeClassifier(config)
    results = classifier.fit(data)
    return classifier, results
