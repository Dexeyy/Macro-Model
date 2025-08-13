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
ASSET_START_DATE = datetime(1990, 1, 1)

# FRED series to fetch (key = FRED series ID, value = descriptive name)
FRED_SERIES = {
    # Growth & Labour
    'RPI': 'Personal Income',
    'W875RX1': 'Real personal income ex transfer receipts',
    'DPCERA3M086SBEA': 'Real personal consumption expenditures, 3m rate',
    'CMRMTSPLx': 'Real Mfg and trade industries sales (alias x → CMRMTSPL)',
    'RETAILx': 'Retail and Food Services Sales (alias x → RSAFS)',
    'INDPRO': 'Industrial Production Index',
    'IPFPNSS': 'IP: Final products and nonindustrial supplies',
    'IPFINAL': 'IP: Final products',
    'IPCONGD': 'IP: Consumer goods',
    'IPDCONGD': 'IP: Durable consumer goods',
    'IPNCONGD': 'IP: Nondurable consumer goods',
    'IPBUSEQ': 'IP: Business equipment',
    'IPMAT': 'IP: Materials',
    'IPDMAT': 'IP: Durable materials',
    'IPNMAT': 'IP: Nondurable materials',
    'IPMANSICS': 'IP: Manufacturing (SIC)',
    'IPB51222S': 'IP: Residential utilities',
    'IPFUELS': 'IP: Fuels',
    'CUMFNS': 'Capacity utilization: Manufacturing',
    'HWI': 'Help-Wanted Index for United States',
    'HWIURATIO': 'Help-Wanted/No. Unemployed Ratio',
    'CLF16OV': 'Civilian Labor Force',
    'CE16OV': 'Civilian Employment',
    'UNRATE': 'Unemployment Rate',
    'UEMPMEAN': 'Average Duration of Unemployment',
    'UEMPLT5': 'Less than 5 Weeks Unemployed',
    'UEMP5TO14': '5-14 Weeks Unemployed',
    'UEMP15OV': '15+ Weeks Unemployed',
    'UEMP15T26': '15-26 Weeks Unemployed',
    'UEMP27OV': '27+ Weeks Unemployed',
    'CLAIMSx': 'Initial claims (alias x → ICSA)',
    'PAYEMS': 'All Employees: Total nonfarm',
    'USGOOD': 'All Employees: Goods-Producing',
    'CES1021000001': 'Avg Hrly Earn: Construction',
    'USCONS': 'All Employees: Construction',
    'MANEMP': 'All Employees: Manufacturing',
    'DMANEMP': 'All Employees: Durable goods',
    'NDMANEMP': 'All Employees: Nondurable goods',
    'SRVPRD': 'All Employees: Service-Providing',
    'USTPU': 'All Employees: Trade, Transportation, Utilities',
    'USWTRADE': 'All Employees: Wholesale Trade',
    'USTRADE': 'All Employees: Retail Trade',
    'USFIRE': 'All Employees: Financial Activities',
    'USGOVT': 'All Employees: Government',
    'CES0600000007': 'Avg Hrly Earn: Manufacturing',
    'AWOTMAN': 'Avg Overtime Hrs: Manufacturing',
    'AWHMAN': 'Avg Weekly Hrs: Manufacturing',
    'ACOGNO': 'New Orders: Consumer Goods',
    'AMDMNOx': 'New Orders: Durable Goods (alias x → AMDMNO)',
    'ANDENOx': 'New Orders: Nondefense Capital Goods (alias x → ANDENO)',
    'AMDMUOx': 'Unfilled Orders: Durable Goods (alias x → AMDMUO)',
    'BUSINVx': 'Total Business Inventories (alias x → BUSINV)',
    'ISRATIOx': 'Inventory/Sales Ratio (alias x → ISRATIO)',
    'UMCSENTx': 'Consumer Sentiment (alias x → UMCSENT)',
    'CES0600000008': 'Avg Wkly Hrs: Manufacturing production',
    'CES2000000008': 'Avg Wkly Hrs: Construction production',
    'CES3000000008': 'Avg Wkly Hrs: Manufacturing, nondurable',

    # Inflation & Liquidity
    'CONSPI': 'Consumption price index (proxy)',
    'WPSFD49207': 'PPI: Finished consumer goods',
    'WPSFD49502': 'PPI: Finished energy goods',
    'WPSID61': 'PPI: Intermediate materials',
    'WPSID62': 'PPI: Intermediate supplies',
    'PPICMM': 'PPI: Crude materials',
    'CPIAUCSL': 'CPI: All Urban Consumers',
    'CPIAPPSL': 'CPI: Apparel',
    'CPITRNSL': 'CPI: Transportation',
    'CPIMEDSL': 'CPI: Medical Care',
    'CUSR0000SAC': 'CPI: Commodities less food & energy',
    'CUSR0000SAD': 'CPI: Durables',
    'CUSR0000SAS': 'CPI: Services',
    'CPIULFSL': 'CPI: All Items Less Food',
    'CUSR0000SA0L2': 'CPI: Less Shelter',
    'CUSR0000SA0L5': 'CPI: Less energy services',
    'PCEPI': 'PCE Price Index',
    'DDURRG3M086SBEA': 'Real PCE: Durable goods (3m rate)',
    'DNDGRG3M086SBEA': 'Real PCE: Nondurable goods (3m rate)',
    'DSERRG3M086SBEA': 'Real PCE: Services (3m rate)',
    'M1SL': 'M1 Money Stock',
    'M2SL': 'M2 Money Stock',
    'M2REAL': 'Real M2 Money Stock',
    'BOGMBASE': 'Monetary Base; Not Adjusted for Seasonal Variation',
    'TOTRESNS': 'Total Reserves of Depository Institutions',
    'NONBORRES': 'Nonborrowed Reserves',

    # Credit & Risk
    'BUSLOANS': 'Commercial and Industrial Loans',
    'REALLN': 'Real Estate Loans',
    'NONREVSL': 'Nonrevolving consumer credit',
    'SP500': 'S&P 500 Index',
    'SPDIVOR': 'S&P Dividend Yield (proxy)',
    'CAPE': 'Shiller CAPE Ratio (proxy)',
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'CP3M': '3-Month Commercial Paper Rate',
    'TB3MS': '3-Month Treasury Bill: Secondary Market Rate',
    'TB6MS': '6-Month Treasury Bill: Secondary Market Rate',
    'GS1': '1-Year Treasury Constant Maturity Rate',
    'GS5': '5-Year Treasury Constant Maturity Rate',
    'GS10': '10-Year Treasury Constant Maturity Rate',
    'AAA': 'Moody’s Seasoned Aaa Corporate Bond Yield',
    'BAA': 'Moody’s Seasoned Baa Corporate Bond Yield',
    'COMPAPFF': 'Commercial Paper-FF spread (proxy)',
    'TB3SMFFM': '3-Month T-bill minus Fed Funds',
    'TB6SMFFM': '6-Month T-bill minus Fed Funds',
    'T1YFFM': '1-Year minus Fed Funds',
    'T5YFFM': '5-Year minus Fed Funds',
    'T10YFFM': '10-Year minus Fed Funds',
    'AAAFFM': 'Aaa minus Fed Funds',
    'BAAFFM': 'Baa minus Fed Funds',
    'VIXCLS': 'CBOE Volatility Index (VIX)',

    # Housing
    'HOUST': 'Housing Starts: Total',
    'HOUSTNE': 'Housing Starts: Northeast',
    'HOUSTMW': 'Housing Starts: Midwest',
    'HOUSTS': 'Housing Starts: South',
    'HOUSTW': 'Housing Starts: West',
    'PERMIT': 'Building Permits: Total',
    'PERMITNE': 'Building Permits: Northeast',
    'PERMITMW': 'Building Permits: Midwest',
    'PERMITS': 'Building Permits: South',
    'PERMITW': 'Building Permits: West',

    # FX & Commodities
    'TWEXAFEGSMTH': 'Trade Weighted U.S. Dollar Index: Advanced For Economies (Monthly)',
    'EXSZUS': 'Swiss Franc to U.S. Dollar',
    'EXJPUS': 'Yen to U.S. Dollar',
    'EXUSUK': 'U.S. Dollar to U.K. Pound',
    'EXCAUS': 'Canadian Dollar to U.S. Dollar',
    'DCOILWTICO': 'Crude Oil Prices: WTI (NYMEX)'
}

# Additional macro series
FRED_SERIES.update({
    "T5YIFR": "5-Year Forward Inflation Expectation Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "ICSA": "Initial Jobless Claims",
    "MOVE": "Merrill Lynch Option Volatility Estimate",
})

# Alias map for FRED-MD styled codes to actual FRED IDs
FRED_ALIASES = {
    'CMRMTSPLx': 'CMRMTSPL',
    'RETAILx': 'RSAFS',
    'CLAIMSx': 'ICSA',
    'AMDMNOx': 'AMDMNO',
    'ANDENOx': 'ANDENO',
    'AMDMUOx': 'AMDMUO',
    'BUSINVx': 'BUSINV',
    'ISRATIOx': 'ISRATIO',
    'UMCSENTx': 'UMCSENT',
    'CP3Mx': 'CP3M',
    'VIXCLSx': 'VIXCLS',
    'TWEXAFEGSMTHx': 'TWEXAFEGSMTH',
    'EXSZUSx': 'EXSZUS',
    'EXJPUSx': 'EXJPUS',
    'EXUSUKx': 'EXUSUK',
    'EXCAUSx': 'EXCAUS',
    'OILPRICEx': 'DCOILWTICO',
    'COMPAPFFx': 'COMPAPFF',
}

# Transformation code map (FRED-MD tcode): 1=level, 2=diff, 4=diff4, 5=log, 6=diff(log), 7=diff2(log)
TCODE_MAP = {
    # Growth & Labour
    'RPI': 5, 'W875RX1': 5, 'DPCERA3M086SBEA': 5, 'CMRMTSPL': 5, 'RSAFS': 5,
    'INDPRO': 5, 'IPFPNSS': 5, 'IPFINAL': 5, 'IPCONGD': 5, 'IPDCONGD': 5, 'IPNCONGD': 5,
    'IPBUSEQ': 5, 'IPMAT': 5, 'IPDMAT': 5, 'IPNMAT': 5, 'IPMANSICS': 5, 'IPB51222S': 5,
    'IPFUELS': 5, 'CUMFNS': 2, 'HWI': 2, 'HWIURATIO': 2, 'CLF16OV': 5, 'CE16OV': 5,
    'UNRATE': 2, 'UEMPMEAN': 2, 'UEMPLT5': 5, 'UEMP5TO14': 5, 'UEMP15OV': 5,
    'UEMP15T26': 5, 'UEMP27OV': 5, 'ICSA': 5, 'PAYEMS': 5, 'USGOOD': 5, 'CES1021000001': 5,
    'USCONS': 5, 'MANEMP': 5, 'DMANEMP': 5, 'NDMANEMP': 5, 'SRVPRD': 5, 'USTPU': 5,
    'USWTRADE': 5, 'USTRADE': 5, 'USFIRE': 5, 'USGOVT': 1, 'CES0600000007': 2,
    'AWOTMAN': 1, 'AWHMAN': 4, 'ACOGNO': 5, 'AMDMNO': 5, 'ANDENO': 5, 'AMDMUO': 5,
    'BUSINV': 2, 'ISRATIO': 6, 'UMCSENT': 2, 'CES0600000008': 6, 'CES2000000008': 6,
    'CES3000000008': 6,

    # Inflation & Liquidity
    'CONSPI': 5, 'WPSFD49207': 6, 'WPSFD49502': 6, 'WPSID61': 6, 'WPSID62': 6, 'PPICMM': 6,
    'CPIAUCSL': 6, 'CPIAPPSL': 6, 'CPITRNSL': 6, 'CPIMEDSL': 6, 'CUSR0000SAC': 6, 'CUSR0000SAD': 6,
    'CUSR0000SAS': 6, 'CPIULFSL': 6, 'CUSR0000SA0L2': 6, 'CUSR0000SA0L5': 6, 'PCEPI': 6,
    'DDURRG3M086SBEA': 6, 'DNDGRG3M086SBEA': 6, 'DSERRG3M086SBEA': 6, 'M1SL': 6,
    'M2SL': 5, 'M2REAL': 6, 'BOGMBASE': 6, 'TOTRESNS': 7, 'NONBORRES': 6,

    # Credit & Risk
    'BUSLOANS': 6, 'REALLN': 6, 'NONREVSL': 2, 'SP500': 5, 'SPDIVOR': 2, 'CAPE': 5,
    'FEDFUNDS': 2, 'CP3M': 2, 'TB3MS': 2, 'TB6MS': 2, 'GS1': 2, 'GS5': 2, 'GS10': 2,
    'AAA': 2, 'BAA': 2, 'COMPAPFF': 1, 'TB3SMFFM': 1, 'TB6SMFFM': 1, 'T1YFFM': 1,
    'T5YFFM': 1, 'T10YFFM': 1, 'AAAFFM': 1, 'BAAFFM': 1, 'VIXCLS': 1,

    # Housing
    'HOUST': 4, 'HOUSTNE': 4, 'HOUSTMW': 4, 'HOUSTS': 4, 'HOUSTW': 4,
    'PERMIT': 4, 'PERMITNE': 4, 'PERMITMW': 4, 'PERMITS': 4, 'PERMITW': 4,

    # FX & Commodities
    'TWEXAFEGSMTH': 5, 'EXSZUS': 5, 'EXJPUS': 5, 'EXUSUK': 5, 'EXCAUS': 5, 'DCOILWTICO': 6,
}

# Asset tickers
ASSET_TICKERS = {
    'SPX': '^GSPC',         # S&P 500 Index (index)
    'NDX': '^NDX',          # Nasdaq-100 Index (index)
    'GOLD': 'GC=F',         # COMEX Gold Futures (Yahoo Finance)
    'US30Y_BOND_FUT': 'ZB=F', # 30Y Treasury Bond Futures (price)
    'US10Y_NOTE_FUT': 'ZN=F', # 10Y Treasury Note Futures (price)
    'US5Y_NOTE_FUT': 'ZF=F',  # 5Y Treasury Note Futures (price)
    'BTC': 'BTC-USD'        # Bitcoin spot in USD (spot)
}

# Stooq tickers
STOOQ_TICKERS = {"MOVE": "^MOVE", "VIX3M": "^VIX3M"}

# Clustering features (expanded, cross-theme, low-collinearity set)
# Mix of engineered features and transformed core series
CLUSTER_FEATURES = [
    # Inflation & Liquidity
    'CPI_YoY',           # inflation
    'PCEPI',             # price index (tcode applied)
    'M2SL',              # money supply (tcode applied)

    # Growth & Labour
    'GDP_YoY',           # growth
    'INDPRO_YoY',        # production cycle
    'UNRATE',            # labor slack/tightness
    'PAYEMS',            # employment level (tcode applied)

    # Rates & Curve
    'FEDFUNDS',          # policy rate
    'DGS10',             # long rate
    'YieldCurve_Slope',  # term-structure signal

    # Risk/Financial conditions
    'VIXCLS',                    # market volatility (tcode applied)
    'FinConditions_Composite',   # stress/tightness

    # Housing
    'HOUST',             # housing starts (tcode applied)

    # FX & Commodities
    'TWEXAFEGSMTH',      # dollar index (tcode applied)
    'DCOILWTICO',        # crude oil (tcode applied)

    # Sentiment
    'UMCSENT'            # consumer sentiment (tcode applied)
]

# Feature groups (by themes for Excel mapping)
FEATURE_GROUPS = {
    # Growth & Labour
    "Growth": [
        'RPI','W875RX1','DPCERA3M086SBEA','CMRMTSPL','RSAFS','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD',
        'IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222S','IPFUELS','CUMFNS','HWI','HWIURATIO','CLF16OV','CE16OV',
        'UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','ICSA','PAYEMS','USGOOD','CES1021000001',
        'USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007',
        'AWOTMAN','AWHMAN','ACOGNO','AMDMNO','ANDENO','AMDMUO','BUSINV','ISRATIO','UMCSENT','CES0600000008','CES2000000008',
        'CES3000000008'
    ],

    # Inflation & Liquidity
    "Inflation": [
        'CONSPI','WPSFD49207','WPSFD49502','WPSID61','WPSID62','PPICMM','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL',
        'CUSR0000SAC','CUSR0000SAD','CUSR0000SAS','CPIULFSL','CUSR0000SA0L2','CUSR0000SA0L5','PCEPI',
        'DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA','M1SL','M2SL','M2REAL','BOGMBASE','TOTRESNS','NONBORRES'
    ],

    # Credit & Risk
    "Risk": [
        'BUSLOANS','REALLN','NONREVSL','SP500','SPDIVOR','CAPE','FEDFUNDS','CP3M','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA',
        'COMPAPFF','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','VIXCLS'
    ],

    # Yield curve indicators
    "YieldCurve": ["DGS2","DGS5","DGS10","T10Y2Y","YieldCurve_Slope","YieldCurve_Curvature","YieldCurve_Slope_Mom"],

    # Housing
    "Housing": ['HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW'],

    # FX & Commodities
    "FX": ['TWEXAFEGSMTH','EXSZUS','EXJPUS','EXUSUK','EXCAUS','DCOILWTICO']
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

# ---------------------------------------------------------------------------
# Factor construction configuration
# ---------------------------------------------------------------------------

# Orientation of series: +1 means “higher is better” for growth/risk-on,
# -1 means “higher is worse”. Any series not listed here implicitly uses +1.
# Note: Keys should refer to the base series code (before transform suffixes
# like _YoY, _MoM, etc.).
SERIES_SIGN = {
    # Labour market (bad when higher)
    "UNRATE": -1,
    "UEMPMEAN": -1,
    "UEMPLT5": -1,
    "UEMP5TO14": -1,
    "UEMP15OV": -1,
    "UEMP15T26": -1,
    "UEMP27OV": -1,
    "ICSA": -1,

    # Risk/volatility (bad when higher)
    "VIXCLS": -1,
    "MOVE": -1,

    # Interest rates and spreads (higher yields/tightening as risk-off)
    "FEDFUNDS": -1,
    "TB3MS": -1,
    "TB6MS": -1,
    "GS1": -1,
    "GS5": -1,
    "GS10": -1,
    "AAA": -1,
    "BAA": -1,
    "TB3SMFFM": -1,
    "TB6SMFFM": -1,
    "T1YFFM": -1,
    "T5YFFM": -1,
    "T10YFFM": -1,
    "AAAFFM": -1,
    "BAAFFM": -1,
    "COMPAPFF": -1,

    # Dollar (strong USD often a headwind globally)
    "TWEXAFEGSMTH": -1,
}

# Sub-buckets within each theme to avoid overweighting very dense categories.
# Provide representative lists; projects can extend these over time.
SUB_BUCKETS = {
    "Growth": {
        "activity": [
            "INDPRO", "IPFINAL", "IPCONGD", "IPBUSEQ", "CMRMTSPL", "RSAFS", "RPI", "W875RX1",
        ],
        "labour": [
            "PAYEMS", "CE16OV", "USGOOD", "MANEMP", "AWHMAN", "AWOTMAN",
            "UNRATE", "UEMPMEAN", "ICSA",
        ],
        "sentiment": ["UMCSENT"],
        "orders": ["ACOGNO", "AMDMNO", "ANDENO", "AMDMUO"],
    },
    "Inflation": {
        "prices": [
            "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAD",
            "CUSR0000SAS", "PCEPI", "PPICMM", "WPSFD49207", "WPSFD49502",
        ],
        "money": ["M1SL", "M2SL", "M2REAL", "BOGMBASE", "TOTRESNS", "NONBORRES"],
    },
    "Risk": {
        "spreads": ["AAA", "BAA", "COMPAPFF", "AAAFFM", "BAAFFM"],
        "rates": ["FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "T1YFFM", "T5YFFM", "T10YFFM"],
        "equities": ["SP500", "CAPE", "SPDIVOR"],
        "vol_liquidity": ["VIXCLS", "MOVE", "NFCI"],
    },
    "Housing": {
        "construction": ["HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW"],
        "permits": ["PERMIT", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW"],
    },
    "FX": {
        "usd": ["TWEXAFEGSMTH"],
        "commodities": ["DCOILWTICO"],
    },
}

# Correlation pruning threshold and minimum per-sub-bucket coverage
PRUNE_CORR_THRESHOLD = 0.95
MIN_SUB_BUCKET_COVERAGE = 2

# ---------------------------------------------------------------------------
# Dynamic regime portfolio settings
# ---------------------------------------------------------------------------

# Lookback window (years) for estimating regime mean/cov
REGIME_WINDOW_YEARS = 15

# Rebalance frequency: 'M' for monthly, 'Q' for quarterly
REBAL_FREQ = 'Q'

# Per-trade transaction cost (as return, e.g., 0.0005 = 5 bps)
TRANSACTION_COST = 0.0

# If True and regime probabilities exist, blend estimates via probabilities
PROBABILITY_BLENDING = True