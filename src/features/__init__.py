"""
Feature Engineering Module

This module provides comprehensive functionality for calculating and transforming
raw financial and economic data into relevant features for macroeconomic regime
analysis and portfolio optimization.

Components:
- Returns Calculation: Simple, log, and cumulative returns
- Volatility Estimation: Historical, rolling, EWMA, and GARCH volatility models
- Economic Indicators: Derived macroeconomic measures
- Technical Indicators: Moving averages, RSI, MACD, Bollinger Bands, stochastic oscillators
- Feature Normalization: Z-score, min-max, robust scaling with outlier handling and pipelines
- Feature Store Integration: Storage, versioning, and lineage tracking for features
"""

from .returns_calculator import (
    ReturnType,
    Frequency,
    ReturnConfig,
    ReturnsCalculator,
    calculate_returns
)

# Economic indicators temporarily disabled due to encoding issues
# from .economic_indicators import (
#     EconomicIndicators,
#     IndicatorCategory,
#     IndicatorDefinition,
#     calculate_yield_curve_slope,
#     calculate_real_rate,
#     calculate_unemployment_gap
# )

from .volatility_estimator import (
    VolatilityMethod,
    AnnualizationFactor,
    VolatilityConfig,
    VolatilityEstimator,
    calculate_simple_volatility,
    calculate_rolling_vol,
    calculate_ewma_vol
)

from .technical_indicators import (
    MovingAverageType,
    SignalType,
    IndicatorConfig,
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

from .feature_normalizer import (
    ScalingMethod,
    OutlierHandling,
    NormalizationConfig,
    FeatureNormalizer,
    NormalizationPipeline,
    normalize_features,
    create_preprocessing_pipeline
)

from .feature_store import (
    FeatureType,
    FeatureStatus,
    StorageBackend,
    FeatureMetadata,
    FeatureQuery,
    FeatureStoreConfig,
    FeatureStore,
    create_feature_store,
    create_file_feature_store,
    create_memory_feature_store
)

__all__ = [
    # Returns calculator exports
    'ReturnType',
    'Frequency', 
    'ReturnConfig',
    'ReturnsCalculator',
    'calculate_returns',
    
    # Economic indicators exports (temporarily disabled)
    # 'EconomicIndicators',
    # 'IndicatorCategory', 
    # 'IndicatorDefinition',
    # 'calculate_yield_curve_slope',
    # 'calculate_real_rate',
    # 'calculate_unemployment_gap',
    
    # Volatility estimator exports
    'VolatilityMethod',
    'AnnualizationFactor',
    'VolatilityConfig',
    'VolatilityEstimator',
    'calculate_simple_volatility',
    'calculate_rolling_vol',
    'calculate_ewma_vol',
    
    # Technical indicators exports
    'MovingAverageType',
    'SignalType',
    'IndicatorConfig',
    'TechnicalIndicators',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    
    # Feature normalization exports
    'ScalingMethod',
    'OutlierHandling',
    'NormalizationConfig',
    'FeatureNormalizer',
    'NormalizationPipeline',
    'normalize_features',
    'create_preprocessing_pipeline',
    
    # Feature store integration exports
    'FeatureType',
    'FeatureStatus',
    'StorageBackend',
    'FeatureMetadata',
    'FeatureQuery',
    'FeatureStoreConfig',
    'FeatureStore',
    'create_feature_store',
    'create_file_feature_store',
    'create_memory_feature_store'
]

__version__ = "1.0.0" 