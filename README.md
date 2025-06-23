# Macro Regime Analysis Model

A Python framework for analyzing macroeconomic regimes and their impact on asset performance.

## Overview

This project provides tools to:
1. Fetch and process macroeconomic data from FRED
2. Fetch asset price data from Yahoo Finance
3. Classify economic regimes using rule-based and machine learning approaches
4. Analyze asset performance across different regimes
5. Create regime-based investment portfolios
6. Generate advanced macroeconomic features for enhanced regime detection

## Project Structure

```
macro-regime-model/
├── data/                      # Data storage
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Processed datasets
├── src/                       # Source code
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── fetchers.py        # Data fetching modules (FRED, Yahoo, etc.)
│   │   └── processors.py      # Data processing functions
│   ├── models/                # Analysis models
│   │   ├── __init__.py
│   │   ├── regime_classifier.py  # Regime classification models
│   │   └── portfolio.py       # Portfolio construction
│   ├── visualization/         # Visualization tools
│   │   ├── __init__.py
│   │   └── plots.py           # Plotting functions
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks/                 # Jupyter notebooks for exploration
├── output/                    # Output files (charts, reports)
├── config.py                  # Configuration settings
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/macro-regime-model.git
cd macro-regime-model
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables (optional):
Create a `.env` file in the root directory with:
```
FRED_API_KEY=your_fred_api_key
```

## Usage

### Running the full pipeline

```python
python main.py
```

### Using specific components

```python
from src.data.fetchers import fetch_fred_series
from src.models.regime_classifier import apply_rule_based_classification
from src.visualization.plots import plot_regime_timeline

# Fetch data
macro_data = fetch_fred_series(...)

# Classify regimes
macro_data = apply_rule_based_classification(macro_data)

# Visualize
plot_regime_timeline(macro_data, 'Regime_Rule_Based')
```

## Economic Regimes

The framework identifies several key macroeconomic regimes:

1. **Recession**: Negative growth, high unemployment, often with inverted yield curve. Typically favors defensive assets like Treasuries and cash.

2. **Stagflation**: Low/negative growth with high inflation. Typically favors commodities, TIPS, and real assets.

3. **Overheating**: Strong growth with high inflation, late-cycle expansion. Typically favors commodities, value stocks, and short-duration assets.

4. **Expansion**: Moderate growth with moderate inflation, "goldilocks" environment. Typically favors equities, corporate bonds, and cyclical sectors.

5. **Recovery**: Positive growth with low inflation, early-cycle expansion. Typically favors growth stocks, credit, and cyclical sectors.

6. **Slowdown/Disinflation**: Low growth with falling inflation, pre-recession phase. Typically favors long-duration assets like Treasuries and defensive sectors.

## Advanced Feature Engineering

The framework includes sophisticated feature engineering to enhance regime detection:

### Yield Curve Dynamics
- **YieldCurve_Slope**: Difference between 10Y and 2Y Treasury yields
  - Positive: Normal yield curve (expansion)
  - Negative: Inverted yield curve (recession signal)
- **YieldCurve_Curvature**: Measures the non-linearity of the yield curve
  - High positive: Steep in middle, flat at ends (mid-cycle)
  - Negative: Humped yield curve (late cycle)
- **YieldCurve_Slope_Mom**: Rate of change in yield curve slope
  - Positive: Steepening yield curve (early expansion)
  - Negative: Flattening yield curve (late cycle)

### Inflation Expectations and Real Rates
- **RealRate_10Y**: Nominal yield minus inflation expectations
  - High positive: Restrictive monetary policy
  - Negative: Accommodative policy, often during crises
- **RealRate_10Y_Mom**: Change in real rates
  - Rising: Tightening financial conditions
  - Falling: Easing financial conditions

### Financial Conditions
- **FinConditions_Composite**: Standardized average of financial stress indicators
  - Positive: Tight financial conditions (stress)
  - Negative: Easy financial conditions (complacency)

### Growth Momentum
- **GDP_YoY_Mom**, **INDPRO_YoY_Mom**, **NFP_YoY_Mom**: Change in growth metrics
  - Positive: Accelerating growth (early/mid expansion)
  - Negative: Decelerating growth (late cycle/contraction)

### Liquidity Measures
- **M2_YoY**: Year-over-year change in money supply
  - High: Expansionary monetary policy
  - Low/Negative: Contractionary monetary policy
- **RealM2_Growth**: Money supply growth adjusted for inflation
  - High: Expansionary in real terms
  - Low/Negative: Contractionary in real terms

## Extending the Framework

### Adding New Data Sources

Add new fetcher functions in `src/data/fetchers.py`.

### Implementing New Regime Classification Methods

Add new classification methods in `src/models/regime_classifier.py`.

### Creating Custom Portfolio Strategies

Add new portfolio construction methods in `src/models/portfolio.py`.

### Adding New Advanced Features

Extend the `create_advanced_features` function in `src/data/processors.py` to include additional derived features.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 