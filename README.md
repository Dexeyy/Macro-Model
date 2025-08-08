# Macro Regime Analysis Model

A Python framework for analyzing macroeconomic regimes and their impact on asset performance.

## Overview

This project provides tools to:
- Fetch and process macroeconomic data (FRED) and asset prices (Yahoo Finance)
- Engineer macro features and theme composites (F_Growth, F_Inflation, F_Liquidity, F_CreditRisk, F_Housing, F_External)
- Fit multiple regime models (Rule, KMeans, HMM, optional GMM) and compute probabilities
- Build probability-based ensemble regimes with stability post-processing
- Analyze performance by regime and export diagnostics
- Generate an Excel dashboard with clean, category-specific pages

## Architecture

```
Data (FRED, Yahoo) ──> Processing (resample, YoY/MoM, MAs, z-scores)
                         └─> Theme Composites (F_*) & optional PCA (PC_*)
                                      │
                                      ▼
                              Regime Models (Rule, KMeans, HMM, GMM)
                                      │          └─ per‑model probabilities
                                      ▼
                              Ensemble (avg probs + confirmation + min‑duration)
                                      │
                                      ▼
                        Performance Analytics (per‑regime metrics, scorecard)
                                      │
                                      ▼
                         Excel Dashboard (theme pages + probabilities strip)
```

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

### Quickstart (CLI)

Install CLI helper:
```
pip install typer[all]
```

Run the full pipeline (equivalent to `python main.py`):
```
python cli.py run full
```

Fit regimes and save merged output:
```
python cli.py regimes fit --models hmm gmm rule --use-pca false --n-regimes 4
```

Build the Excel dashboard with category pages:
```
python cli.py excel build \
  --template src/excel/Macro Reg Template.xlsm \
  --out Output/Macro_Reg_Report_with_category_dashboards.xlsm
```

### Using main.py
```
python main.py
```
This runs: fetch/process → fit regimes → analyze performance → visuals → portfolios.

## Configuration

Primary YAML config (optional): `config/regimes.yaml`
```
models: [rule, kmeans, hmm]
hmm:
  n_states_range: [2, 6]
  covariance_type: full
  min_duration: 3
  prob_threshold: 0.7
ensemble:
  confirm_consecutive: 2
themes:
  growth: []
  inflation: []
  liquidity: []
  credit_risk: []
  housing: []
  external: []
# Optional flag (either top-level or themes.use_pca):
use_pca: false
```
Notes:
- If the YAML file is missing, defaults are used and behavior is unchanged.
- `use_pca: true` enables `PC_*` factor features in addition to `F_*`.
- HMM min-duration and probability threshold stabilize labels; ensemble uses the same settings.

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

## Output Artifacts

Generated under `/Output` (and `/Data/processed`):
- Processed data
  - `Data/processed/macro_data_featured.csv`
  - `Data/processed/macro_features.parquet` (raw + engineered + F_* + optional PC_*)
  - `Data/processed/macro_data_with_regimes.csv` (labels + probabilities + Validation_Flags)
- Performance diagnostics
  - `Output/diagnostics/regime_scorecard.csv` (per‑regime mean/vol/Sharpe, max DD, ANOVA p‑value)
- Charts and Excel
  - `Output/excel_charts/*.png` (per‑theme images and dashboard plots)
  - `Output/Macro_Reg_Report_with_category_dashboards.xlsm` (theme pages, tables at J3, probability strip)
  - (Legacy) `Output/Macro_Reg_Report.xlsx` if using earlier workflow