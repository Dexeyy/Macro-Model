"""
Configuration settings for the macro-regime-model project.
"""
import os
from datetime import datetime

# API Keys
FRED_API_KEY = os.getenv('FRED_API_KEY', '27c6023be325d770df1c327ab339020e')

# Data settings
START_DATE = datetime(1990, 1, 1)
END_DATE = datetime.today()
ASSET_START_DATE = datetime(2005, 1, 1)

# FRED series to fetch
FRED_SERIES = {
    # Inflation Indicators
    'CPI': 'CPIAUCSL',          # Consumer Price Index for All Urban Consumers
    'CoreCPI': 'CPILFESL',      # Core Consumer Price Index (excluding food and energy)
    'PPI': 'PPIACO',            # Producer Price Index for All Commodities
    
    # Growth Indicators
    'GDP': 'GDPC1',             # Real Gross Domestic Product
    'RealPotentialGDP': 'GDPPOT', # Real Potential Gross Domestic Product
    'INDPRO': 'INDPRO',         # Industrial Production Index
    'RetailSales': 'RSAFS',     # Retail Sales
    
    # Labor Market
    'UNRATE': 'UNRATE',         # Unemployment Rate
    'NFP': 'PAYEMS',            # Total Nonfarm Payrolls
    'WageGrowth': 'CES0500000003', # Average Hourly Earnings of All Employees
    
    # Interest Rates & Monetary Policy
    'FEDFUNDS': 'FEDFUNDS',     # Federal Funds Effective Rate
    'DGS10': 'DGS10',           # 10-Year Treasury Constant Maturity Rate
    'DGS2': 'DGS2',             # 2-Year Treasury Constant Maturity Rate
    'DGS5': 'DGS5',             # 5-Year Treasury Constant Maturity Rate
    'T10Y2Y': 'T10Y2Y',         # 10-Year Treasury Minus 2-Year Treasury Yield
    
    # Sentiment & Financial Conditions
    'UMCSENT': 'UMCSENT',       # University of Michigan Consumer Sentiment
    'VIX': 'VIXCLS',            # CBOE Volatility Index (VIX)
    'CorporateBondSpread': 'BAMLH0A0HYM2' # ICE BofA US High Yield Index Option-Adjusted Spread
}

# Add new FRED IDs
FRED_SERIES.update({
    "T5YIFR": "T5YIFR",  # 5-Year Forward Inflation Expectation Rate
    "T10YIE": "T10YIE",  # 10-Year Breakeven Inflation Rate (Market-Based Inflation Expectations)
    "NFCI": "NFCI",      # Chicago Fed National Financial Conditions Index (Stress Indicator)
    "M2SL": "M2SL",      # M2 Money Stock (Measure of Money Supply)
    "ICSA": "ICSA",      # Initial Jobless Claims (Weekly Unemployment Insurance Claims)
    "HOUST": "HOUST",    # Housing Starts (New Residential Construction)
    "DCOILWTICO": "DCOILWTICO",  # WTI Crude Oil Price (Commodity Price Indicator)
    "MOVE": "MOVE"       # Merrill Lynch Option Volatility Estimate (Bond Market Volatility)
})

# Asset tickers
ASSET_TICKERS = {
    'SPX': '^GSPC',        # S&P 500 Index
    'QQQ': 'QQQ',          # Nasdaq 100 ETF
    'GLD': 'GLD',          # Gold ETF
    'TLT': 'TLT',          # 20+ Year Treasury Bond ETF
    'BTC': 'BTC-USD',      # Bitcoin
    'IEF': 'IEF',          # 7-10 Year Treasury Bond ETF
    'SHY': 'SHY'           # 1-3 Year Treasury Bond ETF
}

# Stooq tickers
STOOQ_TICKERS = {"MOVE": "^MOVE", "VIX3M": "^VIX3M"}

# Clustering features
CLUSTER_FEATURES = [
    'CPI_YoY',
    'GDP_YoY',
    'UNRATE',
    'FEDFUNDS',
    'DGS10',
    'INDPRO_YoY',
    'UMCSENT',
    'YieldCurve_Slope',
    'FinConditions_Composite'
]

# Feature groups
FEATURE_GROUPS = {
    # Economic growth indicators - measure the pace of economic expansion/contraction
    "Growth": ["GDPC1", "PAYEMS", "INDPRO", "GDP_YoY", "GDP_YoY_Mom"],
    
    # Inflation indicators - measure price pressures in the economy
    "Inflation": ["CPIAUCSL", "CPI_YoY"],
    
    # Liquidity indicators - measure monetary policy stance and credit conditions
    "Liquidity": ["FEDFUNDS", "DGS10", "DGS2"],
    
    # Risk indicators - measure market volatility and financial stress
    "Risk": ["VIX", "CorporateBondSpread", "FinConditions_Composite"],
    
    # Yield curve indicators - measure term structure and recession probability
    "YieldCurve": ["DGS2", "DGS5", "DGS10", "YieldCurve_Slope", "YieldCurve_Curvature", "YieldCurve_Slope_Mom"]
}

# Rule-based classification thresholds
THRESHOLDS = {
    'STRONG_GROWTH_GDP': 2.5,
    'MODERATE_GROWTH_GDP': 1.0,
    'WEAK_GROWTH_GDP': 0.0,
    'RECESSION_GDP': -0.5,
    'HIGH_INFLATION_CPI': 3.5,
    'MODERATE_INFLATION_CPI': 2.0,
    'LOW_INFLATION_CPI': 1.0,
    'STRONG_LABOR_UNRATE': 4.0,
    'WEAK_LABOR_UNRATE': 5.5
}

# File paths
DATA_DIR = 'Data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = 'Output'

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True) 