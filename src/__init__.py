"""
Macro-Economic Regime Analysis Platform

This package provides comprehensive functionality for macroeconomic regime analysis
and portfolio optimization, including data fetching, feature engineering, modeling,
and visualization components.

Modules:
- data: Data fetching, processing, validation, and storage
- features: Feature engineering and transformation
- models: Machine learning models for regime analysis
- utils: Utility functions and helpers
- visualization: Plotting and visualization tools
"""

# Data modules
try:
    from . import data
except ImportError:
    pass

# Feature engineering modules  
try:
    from . import features
except ImportError:
    pass

# Model modules
try:
    from . import models
except ImportError:
    pass

# Utility modules
try:
    from . import utils
except ImportError:
    pass

# Visualization modules
try:
    from . import visualization
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Macro-Economic Regime Analysis Team"

# Key exports for easy access
__all__ = [
    'data',
    'features', 
    'models',
    'utils',
    'visualization'
] 