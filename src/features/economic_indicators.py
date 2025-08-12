# -*- coding: utf-8 -*-
"""
Economic Indicators Module

This module provides comprehensive functionality for calculating and analyzing
economic indicators commonly used in macro-regime analysis and financial modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# -------------------- Feature bundle registry ----------------------------
# Logical bundle names mapped to economic concepts. Concrete column mapping
# is resolved by build_feature_bundle based on what's available in df.
LEADING: List[str] = [
    "curve_slope",
    "breakevens",
    "credit_spreads",
    "PMI",
    "new_orders",
    "consumer_expect",
]

COINCIDENT: List[str] = [
    "IP_YoY",
    "Payrolls",
    "CPI_YoY",
    "FinancialConditions",
    "Unemployment",
    "Housing_starts",
]

def _resolve_bundle_columns(df: pd.DataFrame, token: str) -> List[str]:
    """Return concrete column names for a logical token, computing simple
    proxies when needed. Unknown tokens return an empty list.
    """
    cols: List[str] = []
    t = token.lower()

    def has(c: str) -> bool:
        return c in df.columns

    if t == "curve_slope":
        if has("YieldCurve_Slope"):
            cols.append("YieldCurve_Slope")
        elif has("DGS10") and has("DGS2"):
            slope = (df["DGS10"] - df["DGS2"]).rename("YieldCurve_Slope_tmp")
            cols.append(slope.name)
            df[slope.name] = slope

    elif t == "breakevens":
        for c in ("T10YIE", "T5YIFR"):
            if has(c):
                cols.append(c)
                break

    elif t == "credit_spreads":
        if has("credit_spread"):
            cols.append("credit_spread")
        elif has("BAA") and has("AAA"):
            cs = (df["BAA"] - df["AAA"]).rename("credit_spread_tmp")
            cols.append(cs.name)
            df[cs.name] = cs

    elif t == "pmi":
        for c in ("PMI", "PMICOMPOSITE", "UMCSENT"):
            if has(c):
                cols.append(c)
                break

    elif t == "new_orders":
        for c in ("ANDENO", "AMDMNO", "ACOGNO"):
            if has(c):
                cols.append(c)
                break

    elif t == "consumer_expect":
        if has("UMCSENT"):
            cols.append("UMCSENT")

    elif t == "ip_yoy":
        if has("INDPRO_YoY"):
            cols.append("INDPRO_YoY")
        elif has("INDPRO"):
            series = (df["INDPRO"].pct_change(12) * 100).rename("INDPRO_YoY_tmp")
            cols.append(series.name)
            df[series.name] = series

    elif t == "payrolls":
        if has("PAYEMS"):
            cols.append("PAYEMS")

    elif t == "cpi_yoy":
        if has("CPI_YoY"):
            cols.append("CPI_YoY")
        elif has("CPIAUCSL"):
            series = (df["CPIAUCSL"].pct_change(12) * 100).rename("CPI_YoY_tmp")
            cols.append(series.name)
            df[series.name] = series

    elif t == "financialconditions":
        if has("FinConditions_Composite"):
            cols.append("FinConditions_Composite")
        elif has("VIXCLS"):
            cols.append("VIXCLS")

    elif t == "unemployment":
        if has("UNRATE"):
            cols.append("UNRATE")

    elif t == "housing_starts":
        if has("HOUST"):
            cols.append("HOUST")

    return cols

def build_feature_bundle(df: pd.DataFrame, bundle: str = "coincident") -> pd.DataFrame:
    """Construct a feature matrix for the requested bundle.

    - bundle="coincident": use only coincident signals
    - bundle="coincident_plus_leading": union of coincident and leading tokens
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    b = (bundle or "").lower()
    tokens: List[str]
    if b == "coincident_plus_leading":
        tokens = COINCIDENT + LEADING
    else:
        tokens = COINCIDENT

    working = df.copy()
    selected_cols: List[str] = []
    for t in tokens:
        selected_cols.extend(_resolve_bundle_columns(working, t))

    unique_cols = []
    seen = set()
    for c in selected_cols:
        if c not in seen and c in working.columns and pd.api.types.is_numeric_dtype(working[c]):
            unique_cols.append(c)
            seen.add(c)

    out = working[unique_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out

class IndicatorCategory(Enum):
    """Categories of economic indicators"""
    GROWTH = "growth"
    INFLATION = "inflation"
    LABOR = "labor"
    MONETARY = "monetary"
    FINANCIAL = "financial"
    COMPOSITE = "composite"

@dataclass
class IndicatorDefinition:
    """Definition of an economic indicator"""
    name: str
    category: IndicatorCategory
    description: str
    calculation_method: str
    frequency: str = "monthly"
    units: str = "percent"

class EconomicIndicators:
    """
    Comprehensive economic indicators calculator.
    
    Provides a wide range of economic indicators with proper
    mathematical formulations and configurable parameters.
    """
    
    def __init__(self):
        """Initialize the economic indicators calculator."""
        self.indicators = {}
        self._initialize_indicators()
        
    def _initialize_indicators(self):
        """Initialize standard economic indicators."""
        # FRED series mappings
        self.fred_series = {
            'GDPC1': 'Real GDP',
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate',
            'FEDFUNDS': 'Federal Funds Rate',
            'DGS10': '10-Year Treasury Rate',
            'DGS2': '2-Year Treasury Rate',
            'DGS3MO': '3-Month Treasury Rate',
            'NROU': 'Natural Rate of Unemployment',
            'PAYEMS': 'Total Nonfarm Payrolls',
            'INDPRO': 'Industrial Production',
            'PCE': 'Personal Consumption Expenditures',
            'M2SL': 'M2 Money Supply',
            'DEXUSEU': 'US/Euro Exchange Rate',
            'DGS30': '30-Year Treasury Rate',
            'BAA': 'BAA Corporate Bond Yield',
            'AAA': 'AAA Corporate Bond Yield'
        }
        
    def calculate_yield_curve_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate yield curve indicators.
        
        Args:
            data: DataFrame with Treasury yield data
            
        Returns:
            DataFrame with yield curve indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Yield curve slope (10Y - 2Y)
        if 'DGS10' in data.columns and 'DGS2' in data.columns:
            indicators['yield_curve_slope'] = data['DGS10'] - data['DGS2']
        
        # Yield curve slope (10Y - 3M)
        if 'DGS10' in data.columns and 'DGS3MO' in data.columns:
            indicators['yield_curve_slope_3m'] = data['DGS10'] - data['DGS3MO']
        
        # Yield curve curvature
        if 'DGS2' in data.columns and 'DGS10' in data.columns and 'DGS3MO' in data.columns:
            indicators['yield_curve_curvature'] = 2 * data['DGS2'] - data['DGS3MO'] - data['DGS10']
        
        # Term premium approximation
        if 'DGS10' in data.columns and 'DGS2' in data.columns:
            indicators['term_premium'] = (data['DGS10'] - data['DGS2']) / 8  # Simplified approximation
        
        return indicators
    
    def calculate_labor_market_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate labor market indicators.
        
        Args:
            data: DataFrame with labor market data
            
        Returns:
            DataFrame with labor market indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Unemployment gap
        if 'UNRATE' in data.columns and 'NROU' in data.columns:
            indicators['unemployment_gap'] = data['UNRATE'] - data['NROU']
        
        # Employment growth
        if 'PAYEMS' in data.columns:
            indicators['employment_growth'] = data['PAYEMS'].pct_change(12) * 100
        
        # Labor force participation rate (if available)
        if 'CIVPART' in data.columns:
            indicators['labor_force_participation'] = data['CIVPART']
        
        # Wage growth (if available)
        if 'AHETPI' in data.columns:
            indicators['wage_growth'] = data['AHETPI'].pct_change(12) * 100
        
        return indicators
    
    def calculate_inflation_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inflation indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with inflation indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # CPI inflation
        if 'CPIAUCSL' in data.columns:
            indicators['cpi_inflation'] = data['CPIAUCSL'].pct_change(12) * 100
        
        # Core CPI inflation (if available)
        if 'CPILFESL' in data.columns:
            indicators['core_cpi_inflation'] = data['CPILFESL'].pct_change(12) * 100
        
        # PCE inflation
        if 'PCE' in data.columns:
            indicators['pce_inflation'] = data['PCE'].pct_change(12) * 100
        
        # Inflation expectations (if available)
        if 'CPIAUCSL' in data.columns:
            # Simple moving average as proxy for expectations
            indicators['inflation_expectations'] = data['CPIAUCSL'].pct_change(12).rolling(12).mean() * 100
        
        return indicators
    
    def calculate_growth_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate economic growth indicators.
        
        Args:
            data: DataFrame with growth data
            
        Returns:
            DataFrame with growth indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # GDP growth
        if 'GDPC1' in data.columns:
            indicators['gdp_growth'] = data['GDPC1'].pct_change(4) * 100
        
        # Industrial production growth
        if 'INDPRO' in data.columns:
            indicators['industrial_production_growth'] = data['INDPRO'].pct_change(12) * 100
        
        # Consumption growth
        if 'PCE' in data.columns:
            indicators['consumption_growth'] = data['PCE'].pct_change(12) * 100
        
        # Money supply growth
        if 'M2SL' in data.columns:
            indicators['money_supply_growth'] = data['M2SL'].pct_change(12) * 100
        
        return indicators
    
    def calculate_monetary_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monetary policy indicators.
        
        Args:
            data: DataFrame with monetary data
            
        Returns:
            DataFrame with monetary indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Real interest rate
        if 'FEDFUNDS' in data.columns and 'CPIAUCSL' in data.columns:
            inflation = data['CPIAUCSL'].pct_change(12) * 100
            indicators['real_interest_rate'] = data['FEDFUNDS'] - inflation
        
        # Monetary policy stance
        if 'FEDFUNDS' in data.columns:
            # Compare to neutral rate (simplified)
            neutral_rate = 2.5  # Simplified assumption
            indicators['monetary_policy_stance'] = data['FEDFUNDS'] - neutral_rate
        
        # Money multiplier (if available)
        if 'M2SL' in data.columns and 'BOGMBASE' in data.columns:
            indicators['money_multiplier'] = data['M2SL'] / data['BOGMBASE']
        
        return indicators
    
    def calculate_financial_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate financial market indicators.
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            DataFrame with financial indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Credit spread
        if 'BAA' in data.columns and 'AAA' in data.columns:
            indicators['credit_spread'] = data['BAA'] - data['AAA']
        
        # Term spread
        if 'DGS10' in data.columns and 'DGS2' in data.columns:
            indicators['term_spread'] = data['DGS10'] - data['DGS2']
        
        # Exchange rate (if available)
        if 'DEXUSEU' in data.columns:
            indicators['exchange_rate'] = data['DEXUSEU']
        
        return indicators
    
    def calculate_composite_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite economic indicators.
        
        Args:
            data: DataFrame with economic data
            
        Returns:
            DataFrame with composite indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Misery index (inflation + unemployment)
        if 'CPIAUCSL' in data.columns and 'UNRATE' in data.columns:
            inflation = data['CPIAUCSL'].pct_change(12) * 100
            indicators['misery_index'] = inflation + data['UNRATE']
        
        # Economic stress index (simplified)
        if 'UNRATE' in data.columns and 'CPIAUCSL' in data.columns:
            unemployment_weight = 0.4
            inflation_weight = 0.6
            inflation = data['CPIAUCSL'].pct_change(12) * 100
            indicators['economic_stress_index'] = (
                unemployment_weight * data['UNRATE'] + 
                inflation_weight * inflation
            )
        
        return indicators

def create_advanced_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features from macro data.
    
    Args:
        macro_data: DataFrame with basic macro indicators
        
    Returns:
        DataFrame with advanced features added
    """
    # Create a copy to avoid modifying original data
    advanced_data = macro_data.copy()
    
    # Initialize economic indicators calculator
    econ_indicators = EconomicIndicators()
    
    # Add yield curve indicators if we have Treasury data
    if any(col in advanced_data.columns for col in ['DGS10', 'DGS2', 'DGS3MO']):
        yield_indicators = econ_indicators.calculate_yield_curve_indicators(advanced_data)
        advanced_data = pd.concat([advanced_data, yield_indicators], axis=1)
    
    # Add labor market indicators if we have unemployment data
    if any(col in advanced_data.columns for col in ['UNRATE', 'NROU', 'PAYEMS']):
        labor_indicators = econ_indicators.calculate_labor_market_indicators(advanced_data)
        advanced_data = pd.concat([advanced_data, labor_indicators], axis=1)
    
    # Add inflation indicators
    inflation_indicators = econ_indicators.calculate_inflation_indicators(advanced_data)
    if not inflation_indicators.empty:
        advanced_data = pd.concat([advanced_data, inflation_indicators], axis=1)
    
    # Add growth indicators
    growth_indicators = econ_indicators.calculate_growth_indicators(advanced_data)
    if not growth_indicators.empty:
        advanced_data = pd.concat([advanced_data, growth_indicators], axis=1)
    
    # Add monetary indicators
    monetary_indicators = econ_indicators.calculate_monetary_indicators(advanced_data)
    if not monetary_indicators.empty:
        advanced_data = pd.concat([advanced_data, monetary_indicators], axis=1)
    
    # Add financial indicators
    financial_indicators = econ_indicators.calculate_financial_indicators(advanced_data)
    if not financial_indicators.empty:
        advanced_data = pd.concat([advanced_data, financial_indicators], axis=1)
    
    # Add composite indicators
    composite_indicators = econ_indicators.calculate_composite_indicators(advanced_data)
    if not composite_indicators.empty:
        advanced_data = pd.concat([advanced_data, composite_indicators], axis=1)
    
    # Add momentum indicators for key series
    for col in advanced_data.columns:
        if col in ['GDPC1', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS']:
            # Add momentum (rate of change)
            advanced_data[f'{col}_momentum'] = advanced_data[col].pct_change(12)
            # Add acceleration (change in momentum)
            advanced_data[f'{col}_acceleration'] = advanced_data[f'{col}_momentum'].diff()
    
    # Add volatility indicators
    for col in advanced_data.columns:
        if col in ['GDPC1', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS']:
            # Rolling volatility
            advanced_data[f'{col}_volatility'] = advanced_data[col].rolling(window=12).std()
    
    # Add trend indicators
    for col in advanced_data.columns:
        if col in ['GDPC1', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS']:
            # Trend strength (linear regression slope)
            def calculate_trend(series, window=12):
                if len(series) < window:
                    return np.nan
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                return slope
            
            advanced_data[f'{col}_trend'] = advanced_data[col].rolling(window=12).apply(calculate_trend, raw=True)
    
    # Clean up any infinite or NaN values
    advanced_data = advanced_data.replace([np.inf, -np.inf], np.nan)
    advanced_data = advanced_data.fillna(method='ffill').fillna(method='bfill')
    
    return advanced_data


def build_theme_composites(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Build per-theme composite factors from a mapping of theme -> source columns.

    For each theme list of columns:
      - Compute z-score for each source series using a 10-year rolling window (120 months)
        when possible; fall back to full-sample z-score when there is insufficient history.
      - Equal-weight average the z-scored series to produce a composite time series.

    Output column names:
      F_Growth, F_Inflation, F_Liquidity, F_CreditRisk, F_Housing, F_External

    Any theme with no valid inputs is skipped.
    """
    composites = pd.DataFrame(index=df.index)

    def _zscore_series(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        valid = s.dropna()
        if valid.size >= 120:
            mean = s.rolling(120, min_periods=12).mean()
            std = s.rolling(120, min_periods=12).std(ddof=0)
            z = (s - mean) / std.replace(0.0, np.nan)
        else:
            mu = valid.mean()
            sd = valid.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                z = s * np.nan
            else:
                z = (s - mu) / sd
        return z

    # Normalize keys to expected canonical names
    key_map = {
        "growth": "F_Growth",
        "inflation": "F_Inflation",
        "liquidity": "F_Liquidity",
        "credit_risk": "F_CreditRisk",
        "creditrisk": "F_CreditRisk",
        "housing": "F_Housing",
        "external": "F_External",
    }

    for theme_key, cols in mapping.items():
        if not cols:
            continue
        tgt = key_map.get(str(theme_key).lower())
        if not tgt:
            # allow direct target name like "F_Growth"
            tgt = str(theme_key)
        # Filter to existing numeric columns
        valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            continue
        z_cols = [_zscore_series(df[c]) for c in valid_cols]
        if not z_cols:
            continue
        z_mat = pd.concat(z_cols, axis=1)
        composites[tgt] = z_mat.mean(axis=1)

    return composites

# Convenience functions
def calculate_yield_curve_slope(data: pd.DataFrame) -> pd.Series:
    """Calculate yield curve slope (10Y - 2Y)."""
    if 'DGS10' in data.columns and 'DGS2' in data.columns:
        return data['DGS10'] - data['DGS2']
    return pd.Series(dtype=float)

def calculate_unemployment_gap(data: pd.DataFrame) -> pd.Series:
    """Calculate unemployment gap (actual - natural rate)."""
    if 'UNRATE' in data.columns and 'NROU' in data.columns:
        return data['UNRATE'] - data['NROU']
    return pd.Series(dtype=float)

def calculate_real_interest_rate(data: pd.DataFrame) -> pd.Series:
    """Calculate real interest rate (nominal - inflation)."""
    if 'FEDFUNDS' in data.columns and 'CPIAUCSL' in data.columns:
        inflation = data['CPIAUCSL'].pct_change(12) * 100
        return data['FEDFUNDS'] - inflation
    return pd.Series(dtype=float)

def calculate_misery_index(data: pd.DataFrame) -> pd.Series:
    """Calculate misery index (inflation + unemployment)."""
    if 'CPIAUCSL' in data.columns and 'UNRATE' in data.columns:
        inflation = data['CPIAUCSL'].pct_change(12) * 100
        return inflation + data['UNRATE']
    return pd.Series(dtype=float)

if __name__ == "__main__":
    print("Economic Indicators Module - Testing Implementation")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Generate realistic economic data
    sample_data = pd.DataFrame({
        'GDPC1': 100 + np.cumsum(np.random.normal(0.5, 0.3, len(dates))),
        'CPIAUCSL': 250 + np.cumsum(np.random.normal(0.2, 0.1, len(dates))),
        'UNRATE': 5 + np.random.normal(0, 1, len(dates)),
        'FEDFUNDS': 2 + np.random.normal(0, 0.5, len(dates)),
        'DGS10': 3 + np.random.normal(0, 0.3, len(dates)),
        'DGS2': 2 + np.random.normal(0, 0.2, len(dates)),
        'NROU': 4.5 + np.random.normal(0, 0.1, len(dates))
    }, index=dates)
    
    # Initialize economic indicators
    econ_indicators = EconomicIndicators()
    
    # Test individual indicator calculations
    print(f"Testing individual indicators...")
    yield_indicators = econ_indicators.calculate_yield_curve_indicators(sample_data)
    labor_indicators = econ_indicators.calculate_labor_market_indicators(sample_data)
    inflation_indicators = econ_indicators.calculate_inflation_indicators(sample_data)
    
    print(f"✓ Yield curve indicators: {yield_indicators.shape[1]} indicators")
    print(f"✓ Labor market indicators: {labor_indicators.shape[1]} indicators")
    print(f"✓ Inflation indicators: {inflation_indicators.shape[1]} indicators")
    
    # Test advanced features
    advanced_features = create_advanced_features(sample_data)
    print(f"\n✓ Advanced features created: {advanced_features.shape[1]} total features")
    print(f"✓ Original features: {sample_data.shape[1]} features")
    print(f"✓ New features: {advanced_features.shape[1] - sample_data.shape[1]} features")
    
    # Test convenience functions
    print(f"\n✓ Yield curve slope: {calculate_yield_curve_slope(sample_data).iloc[-1]:.2f}")
    print(f"✓ Unemployment gap: {calculate_unemployment_gap(sample_data).iloc[-1]:.2f}")
    print(f"✓ Real interest rate: {calculate_real_interest_rate(sample_data).iloc[-1]:.2f}")
    print(f"✓ Misery index: {calculate_misery_index(sample_data).iloc[-1]:.2f}")
    
    print(f"\nEconomic Indicators Module created successfully!")
    print(f"Available indicators: Yield curve, Labor market, Inflation, Growth,")
    print(f"Monetary, Financial, and Composite indicators with advanced feature engineering.")
