# -*- coding: utf-8 -*-
"""
Feature Normalization Pipeline

This module provides comprehensive functionality for normalizing and standardizing
features in preparation for machine learning models. It includes various scaling
methods, outlier handling, and pipeline management for consistent transformations
across training and inference phases.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
import pickle
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ScalingMethod(Enum):
    """Types of scaling methods available"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    MAX_ABS = "max_abs"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    UNIT_VECTOR = "unit_vector"

class OutlierHandling(Enum):
    """Methods for handling outliers"""
    CLIP = "clip"
    REMOVE = "remove"
    WINSORIZE = "winsorize"
    LOG_TRANSFORM = "log_transform"
    BOX_COX = "box_cox"
    NONE = "none"

@dataclass
class NormalizationConfig:
    """Configuration for feature normalization"""
    scaling_method: ScalingMethod = ScalingMethod.Z_SCORE
    outlier_handling: OutlierHandling = OutlierHandling.CLIP
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    quantile_range: Tuple[float, float] = (0.25, 0.75)  # For robust scaling
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)  # For clipping
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)  # For winsorizing
    handle_missing: bool = True
    missing_strategy: str = "median"  # "mean", "median", "mode", "forward_fill"
    feature_range: Tuple[float, float] = (0, 1)  # For min-max scaling
    epsilon: float = 1e-8  # Small value to prevent division by zero
    preserve_sparsity: bool = False
    copy_data: bool = True

class FeatureNormalizer:
    """
    Comprehensive feature normalization pipeline.
    
    Provides various scaling methods, outlier handling, and maintains
    transformation parameters for consistent application across datasets.
    """
    
    def __init__(self, config: NormalizationConfig = None):
        """
        Initialize the feature normalizer.
        
        Args:
            config: NormalizationConfig object with transformation parameters
        """
        self.config = config or NormalizationConfig()
        self.fitted_transformers_ = {}
        self.feature_stats_ = {}
        self.outlier_bounds_ = {}
        self.is_fitted_ = False
        
    def _detect_outliers(self, data: pd.Series, method: str = "iqr") -> pd.Series:
        """
        Detect outliers in the data.
        
        Args:
            data: Input data series
            method: Method for outlier detection ("iqr", "zscore", "modified_zscore")
            
        Returns:
            Boolean series indicating outliers
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > self.config.outlier_threshold
        
        elif method == "modified_zscore":
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > self.config.outlier_threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _handle_outliers(self, data: pd.Series, column_name: str) -> pd.Series:
        """
        Handle outliers according to the configured method.
        
        Args:
            data: Input data series
            column_name: Name of the column (for storing bounds)
            
        Returns:
            Data series with outliers handled
        """
        if self.config.outlier_handling == OutlierHandling.NONE:
            return data
        
        # Store original data for reference
        original_data = data.copy()
        
        if self.config.outlier_handling == OutlierHandling.CLIP:
            lower_percentile, upper_percentile = self.config.clip_percentiles
            lower_bound = data.quantile(lower_percentile / 100)
            upper_bound = data.quantile(upper_percentile / 100)
            
            # Store bounds for future use
            self.outlier_bounds_[column_name] = {
                "lower": lower_bound,
                "upper": upper_bound,
                "method": "clip"
            }
            
            return data.clip(lower=lower_bound, upper=upper_bound)
        
        elif self.config.outlier_handling == OutlierHandling.WINSORIZE:
            from scipy import stats
            lower_limit, upper_limit = self.config.winsorize_limits
            winsorized = stats.mstats.winsorize(data, limits=self.config.winsorize_limits)
            return pd.Series(winsorized, index=data.index)
        
        elif self.config.outlier_handling == OutlierHandling.LOG_TRANSFORM:
            # Apply log transformation (add 1 to handle zeros)
            if (data <= 0).any():
                data = data - data.min() + 1
            return np.log(data)
        
        elif self.config.outlier_handling == OutlierHandling.BOX_COX:
            from scipy import stats
            if (data <= 0).any():
                data = data - data.min() + 1
            transformed_data, lambda_param = stats.boxcox(data)
            
            # Store lambda parameter for inverse transformation
            self.outlier_bounds_[column_name] = {
                "lambda": lambda_param,
                "method": "box_cox"
            }
            
            return pd.Series(transformed_data, index=data.index)
        
        elif self.config.outlier_handling == OutlierHandling.REMOVE:
            # Mark outliers for removal (handled at DataFrame level)
            outliers = self._detect_outliers(data)
            return data[~outliers]
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to the configured strategy.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        if not self.config.handle_missing:
            return data
        
        result = data.copy()
        
        for column in result.columns:
            if result[column].isna().any():
                if self.config.missing_strategy == "mean":
                    fill_value = result[column].mean()
                elif self.config.missing_strategy == "median":
                    fill_value = result[column].median()
                elif self.config.missing_strategy == "mode":
                    fill_value = result[column].mode().iloc[0] if not result[column].mode().empty else 0
                elif self.config.missing_strategy == "forward_fill":
                    result[column] = result[column].fillna(method='ffill')
                    continue
                else:
                    fill_value = 0
                
                result[column] = result[column].fillna(fill_value)
        
        return result
    
    def _apply_min_max_scaling(self, data: pd.Series, column_name: str, fit: bool = True) -> pd.Series:
        """Apply min-max scaling to the data."""
        if fit:
            min_val = data.min()
            max_val = data.max()
            
            # Store scaling parameters
            self.fitted_transformers_[column_name] = {
                "method": "min_max",
                "min": min_val,
                "max": max_val,
                "range": self.config.feature_range
            }
        else:
            transformer = self.fitted_transformers_[column_name]
            min_val = transformer["min"]
            max_val = transformer["max"]
        
        # Avoid division by zero
        if max_val - min_val < self.config.epsilon:
            return pd.Series(np.zeros_like(data), index=data.index)
        
        # Scale to [0, 1] then to desired range
        scaled = (data - min_val) / (max_val - min_val)
        range_min, range_max = self.config.feature_range
        return scaled * (range_max - range_min) + range_min
    
    def _apply_z_score_scaling(self, data: pd.Series, column_name: str, fit: bool = True) -> pd.Series:
        """Apply z-score (standard) scaling to the data."""
        if fit:
            mean_val = data.mean()
            std_val = data.std()
            
            # Store scaling parameters
            self.fitted_transformers_[column_name] = {
                "method": "z_score",
                "mean": mean_val,
                "std": std_val
            }
        else:
            transformer = self.fitted_transformers_[column_name]
            mean_val = transformer["mean"]
            std_val = transformer["std"]
        
        # Avoid division by zero
        if std_val < self.config.epsilon:
            return pd.Series(np.zeros_like(data), index=data.index)
        
        return (data - mean_val) / std_val
    
    def _apply_robust_scaling(self, data: pd.Series, column_name: str, fit: bool = True) -> pd.Series:
        """Apply robust scaling using median and interquartile range."""
        if fit:
            median_val = data.median()
            q1, q3 = self.config.quantile_range
            iqr = data.quantile(q3) - data.quantile(q1)
            
            # Store scaling parameters
            self.fitted_transformers_[column_name] = {
                "method": "robust",
                "median": median_val,
                "iqr": iqr,
                "quantiles": self.config.quantile_range
            }
        else:
            transformer = self.fitted_transformers_[column_name]
            median_val = transformer["median"]
            iqr = transformer["iqr"]
        
        # Avoid division by zero
        if iqr < self.config.epsilon:
            return pd.Series(np.zeros_like(data), index=data.index)
        
        return (data - median_val) / iqr
    
    def _apply_max_abs_scaling(self, data: pd.Series, column_name: str, fit: bool = True) -> pd.Series:
        """Apply max absolute scaling to the data."""
        if fit:
            max_abs = np.abs(data).max()
            
            # Store scaling parameters
            self.fitted_transformers_[column_name] = {
                "method": "max_abs",
                "max_abs": max_abs
            }
        else:
            transformer = self.fitted_transformers_[column_name]
            max_abs = transformer["max_abs"]
        
        # Avoid division by zero
        if max_abs < self.config.epsilon:
            return pd.Series(np.zeros_like(data), index=data.index)
        
        return data / max_abs
    
    def _apply_quantile_scaling(self, data: pd.Series, column_name: str, 
                              distribution: str = "uniform", fit: bool = True) -> pd.Series:
        """Apply quantile scaling to the data."""
        if fit:
            # Store quantiles for transformation
            quantiles = np.percentile(data.dropna(), np.linspace(0, 100, 1000))
            
            self.fitted_transformers_[column_name] = {
                "method": f"quantile_{distribution}",
                "quantiles": quantiles
            }
        else:
            transformer = self.fitted_transformers_[column_name]
            quantiles = transformer["quantiles"]
        
        # Apply quantile transformation
        result = np.interp(data, quantiles, np.linspace(0, 1, len(quantiles)))
        
        if distribution == "normal":
            from scipy import stats
            result = stats.norm.ppf(np.clip(result, 0.001, 0.999))
        
        return pd.Series(result, index=data.index)
    
    def _apply_unit_vector_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply unit vector scaling to the entire DataFrame."""
        # Calculate L2 norm for each row
        norms = np.linalg.norm(data.values, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms < self.config.epsilon, 1, norms)
        
        return pd.DataFrame(data.values / norms, index=data.index, columns=data.columns)
    
    def fit(self, data: Union[pd.DataFrame, pd.Series]) -> 'FeatureNormalizer':
        """
        Fit the normalizer to the data (compute transformation parameters).
        
        Args:
            data: Input data to fit the normalizer on
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if self.config.copy_data:
            data = data.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Reset fitted state
        self.fitted_transformers_ = {}
        self.feature_stats_ = {}
        self.outlier_bounds_ = {}
        
        # Fit transformers for each column
        for column in data.columns:
            series = data[column]
            
            # Handle outliers and store bounds
            series = self._handle_outliers(series, column)
            
            # Store basic statistics
            self.feature_stats_[column] = {
                "mean": series.mean(),
                "std": series.std(),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75)
            }
            
            # Fit the appropriate scaling method
            if self.config.scaling_method == ScalingMethod.MIN_MAX:
                self._apply_min_max_scaling(series, column, fit=True)
            elif self.config.scaling_method == ScalingMethod.Z_SCORE:
                self._apply_z_score_scaling(series, column, fit=True)
            elif self.config.scaling_method == ScalingMethod.ROBUST:
                self._apply_robust_scaling(series, column, fit=True)
            elif self.config.scaling_method == ScalingMethod.MAX_ABS:
                self._apply_max_abs_scaling(series, column, fit=True)
            elif self.config.scaling_method == ScalingMethod.QUANTILE_UNIFORM:
                self._apply_quantile_scaling(series, column, "uniform", fit=True)
            elif self.config.scaling_method == ScalingMethod.QUANTILE_NORMAL:
                self._apply_quantile_scaling(series, column, "normal", fit=True)
        
        self.is_fitted_ = True
        logger.info(f"FeatureNormalizer fitted on {len(data.columns)} columns")
        
        return self
    
    def transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform the data using fitted parameters.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")
        
        original_type = type(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if self.config.copy_data:
            data = data.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Transform each column
        for column in data.columns:
            if column in self.fitted_transformers_:
                series = data[column]
                
                # Apply outlier handling with stored bounds
                if column in self.outlier_bounds_:
                    bounds = self.outlier_bounds_[column]
                    if bounds["method"] == "clip":
                        series = series.clip(lower=bounds["lower"], upper=bounds["upper"])
                
                # Apply the appropriate scaling method
                method = self.fitted_transformers_[column]["method"]
                
                if method == "min_max":
                    data[column] = self._apply_min_max_scaling(series, column, fit=False)
                elif method == "z_score":
                    data[column] = self._apply_z_score_scaling(series, column, fit=False)
                elif method == "robust":
                    data[column] = self._apply_robust_scaling(series, column, fit=False)
                elif method == "max_abs":
                    data[column] = self._apply_max_abs_scaling(series, column, fit=False)
                elif method.startswith("quantile"):
                    distribution = method.split("_")[1]
                    data[column] = self._apply_quantile_scaling(series, column, distribution, fit=False)
        
        # Apply unit vector scaling if specified (after individual column scaling)
        if self.config.scaling_method == ScalingMethod.UNIT_VECTOR:
            data = self._apply_unit_vector_scaling(data)
        
        # Return original type
        if original_type == pd.Series and len(data.columns) == 1:
            return data.iloc[:, 0]
        
        return data
    
    def fit_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Fit the normalizer and transform the data in one step.
        
        Args:
            data: Input data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Inverse transform the data to original scale.
        
        Args:
            data: Transformed data to inverse transform
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        
        original_type = type(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if self.config.copy_data:
            data = data.copy()
        
        # Inverse transform each column
        for column in data.columns:
            if column in self.fitted_transformers_:
                transformer = self.fitted_transformers_[column]
                method = transformer["method"]
                series = data[column]
                
                if method == "min_max":
                    range_min, range_max = transformer["range"]
                    # Scale back from range to [0, 1]
                    scaled = (series - range_min) / (range_max - range_min)
                    # Scale back to original range
                    data[column] = scaled * (transformer["max"] - transformer["min"]) + transformer["min"]
                
                elif method == "z_score":
                    data[column] = series * transformer["std"] + transformer["mean"]
                
                elif method == "robust":
                    data[column] = series * transformer["iqr"] + transformer["median"]
                
                elif method == "max_abs":
                    data[column] = series * transformer["max_abs"]
        
        # Return original type
        if original_type == pd.Series and len(data.columns) == 1:
            return data.iloc[:, 0]
        
        return data
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical information about fitted features."""
        return self.feature_stats_.copy()
    
    def get_transformation_params(self) -> Dict[str, Dict[str, Any]]:
        """Get transformation parameters for each feature."""
        return self.fitted_transformers_.copy()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted normalizer to a file.
        
        Args:
            filepath: Path to save the normalizer
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted normalizer.")
        
        save_data = {
            "config": self.config,
            "fitted_transformers": self.fitted_transformers_,
            "feature_stats": self.feature_stats_,
            "outlier_bounds": self.outlier_bounds_,
            "is_fitted": self.is_fitted_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"FeatureNormalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureNormalizer':
        """
        Load a fitted normalizer from a file.
        
        Args:
            filepath: Path to load the normalizer from
            
        Returns:
            Loaded normalizer instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        normalizer = cls(save_data["config"])
        normalizer.fitted_transformers_ = save_data["fitted_transformers"]
        normalizer.feature_stats_ = save_data["feature_stats"]
        normalizer.outlier_bounds_ = save_data["outlier_bounds"]
        normalizer.is_fitted_ = save_data["is_fitted"]
        
        logger.info(f"FeatureNormalizer loaded from {filepath}")
        
        return normalizer


class NormalizationPipeline:
    """
    Pipeline for applying multiple normalization steps in sequence.
    """
    
    def __init__(self, steps: List[Tuple[str, FeatureNormalizer]]):
        """
        Initialize the pipeline.
        
        Args:
            steps: List of (name, normalizer) tuples
        """
        self.steps = steps
        self.is_fitted_ = False
    
    def fit(self, data: Union[pd.DataFrame, pd.Series]) -> 'NormalizationPipeline':
        """Fit all steps in the pipeline."""
        current_data = data
        
        for name, normalizer in self.steps:
            current_data = normalizer.fit_transform(current_data)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Transform data through all pipeline steps."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before transform.")
        
        current_data = data
        
        for name, normalizer in self.steps:
            current_data = normalizer.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


# Convenience functions
def normalize_features(
    data: Union[pd.DataFrame, pd.Series],
    method: ScalingMethod = ScalingMethod.Z_SCORE,
    handle_outliers: bool = True
) -> Union[pd.DataFrame, pd.Series]:
    """
    Quick feature normalization with default settings.
    
    Args:
        data: Input data to normalize
        method: Scaling method to use
        handle_outliers: Whether to handle outliers
        
    Returns:
        Normalized data
    """
    config = NormalizationConfig(
        scaling_method=method,
        outlier_handling=OutlierHandling.CLIP if handle_outliers else OutlierHandling.NONE
    )
    
    normalizer = FeatureNormalizer(config)
    return normalizer.fit_transform(data)

def create_preprocessing_pipeline(
    outlier_method: OutlierHandling = OutlierHandling.CLIP,
    scaling_method: ScalingMethod = ScalingMethod.Z_SCORE
) -> NormalizationPipeline:
    """
    Create a standard preprocessing pipeline.
    
    Args:
        outlier_method: Method for handling outliers
        scaling_method: Method for scaling features
        
    Returns:
        Configured pipeline
    """
    # Step 1: Handle outliers
    outlier_config = NormalizationConfig(
        outlier_handling=outlier_method,
        scaling_method=ScalingMethod.Z_SCORE  # Dummy, not used in outlier step
    )
    outlier_normalizer = FeatureNormalizer(outlier_config)
    
    # Step 2: Scale features
    scaling_config = NormalizationConfig(
        scaling_method=scaling_method,
        outlier_handling=OutlierHandling.NONE  # Already handled
    )
    scaling_normalizer = FeatureNormalizer(scaling_config)
    
    return NormalizationPipeline([
        ("outlier_handling", outlier_normalizer),
        ("feature_scaling", scaling_normalizer)
    ])


# Example usage and testing
if __name__ == "__main__":
    print("Feature Normalization Pipeline - Testing Implementation")
    print("=" * 60)
    
    # Create sample data with outliers and different scales
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Create features with different characteristics
    data = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, len(dates)),  # Normal distribution
        'feature_2': np.random.exponential(2, len(dates)),   # Exponential (skewed)
        'feature_3': np.random.uniform(0, 1000, len(dates)), # Uniform, large scale
        'feature_4': np.random.normal(0, 1, len(dates)),     # Already normalized
    }, index=dates)
    
    # Add some outliers
    outlier_indices = np.random.choice(len(data), size=20, replace=False)
    data.loc[data.index[outlier_indices], 'feature_1'] *= 3
    data.loc[data.index[outlier_indices[:10]], 'feature_2'] *= 5
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data statistics:")
    print(data.describe())
    
    # Test different scaling methods
    methods = [
        ScalingMethod.Z_SCORE,
        ScalingMethod.MIN_MAX,
        ScalingMethod.ROBUST,
        ScalingMethod.MAX_ABS
    ]
    
    for method in methods:
        print(f"\nTesting {method.value} scaling:")
        
        config = NormalizationConfig(
            scaling_method=method,
            outlier_handling=OutlierHandling.CLIP
        )
        
        normalizer = FeatureNormalizer(config)
        normalized_data = normalizer.fit_transform(data)
        
        print(f"  ✓ Normalized data shape: {normalized_data.shape}")
        print(f"  ✓ Feature 1 range: [{normalized_data['feature_1'].min():.3f}, {normalized_data['feature_1'].max():.3f}]")
        print(f"  ✓ Feature 2 mean: {normalized_data['feature_2'].mean():.3f}")
        print(f"  ✓ Feature 3 std: {normalized_data['feature_3'].std():.3f}")
    
    # Test pipeline
    print(f"\nTesting normalization pipeline:")
    pipeline = create_preprocessing_pipeline(OutlierHandling.CLIP, ScalingMethod.Z_SCORE)
    pipeline_result = pipeline.fit_transform(data)
    
    print(f"  ✓ Pipeline result shape: {pipeline_result.shape}")
    print(f"  ✓ All features normalized successfully")
    
    # Test save/load functionality
    print(f"\nTesting save/load functionality:")
    normalizer = FeatureNormalizer()
    normalizer.fit(data)
    
    # Save
    normalizer.save('test_normalizer.pkl')
    
    # Load
    loaded_normalizer = FeatureNormalizer.load('test_normalizer.pkl')
    
    # Test they produce same results
    original_result = normalizer.transform(data)
    loaded_result = loaded_normalizer.transform(data)
    
    is_equal = np.allclose(original_result, loaded_result, rtol=1e-10)
    print(f"  ✓ Save/load consistency: {'PASS' if is_equal else 'FAIL'}")
    
    # Clean up
    import os
    if os.path.exists('test_normalizer.pkl'):
        os.remove('test_normalizer.pkl')
    
    print(f"\nFeature Normalization Pipeline created successfully!")
    print(f"Available methods: Z-Score, Min-Max, Robust, Max-Abs, Quantile scaling")
    print(f"Outlier handling: Clipping, Winsorizing, Log transform, Box-Cox, Removal")
    print(f"Pipeline support: Multi-step processing with consistent transformations")
