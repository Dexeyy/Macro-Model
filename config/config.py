"""
Updated configuration settings for the macro‑regime‑model project.

This configuration file expands the list of FRED series to include the full
set of macroeconomic indicators from the FRED‑MD data set (as provided by
the user) and assigns a transformation code (t‑code) to each series.
Transformation codes follow the FRED‑MD convention: 1 = level, 2 = first
difference, 4 = fourth difference, 5 = natural log, 6 = first difference of
the log, and 7 = second difference of the log.  These codes indicate how
each raw series should be pre‑processed before modelling.

In addition to macro series, the file specifies tickers for asset price
series.  Where possible, actual index prices are used instead of ETF
proxies (e.g., the S&P 500 index rather than the SPY ETF).  These tickers
are compatible with the Yahoo Finance downloader used elsewhere in the
project.

The file retains other settings from the original config, such as date
ranges, clustering feature names, rule‑based thresholds and directory
definitions, to ensure compatibility with existing pipeline code.

To use this configuration, import the module and access the dictionaries
directly.  For example:

    from config_updated import FRED_SERIES, TCODE_MAP, ASSET_TICKERS
    from config_updated import CLUSTER_FEATURES, THRESHOLDS

"""

import os
from datetime import datetime
try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:
    # Graceful fallback – code can still run, but .env won’t be read.
    def load_dotenv(_path: str | None = None):  # type: ignore
        import warnings
        warnings.warn("python-dotenv not installed; environment variables will not be loaded from .env", RuntimeWarning)

# Load environment variables from config/.env
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# ---------------------------------------------------------------------------
# API Keys
#
# The FRED API key can be set via the environment variable FRED_API_KEY.  A
# default is provided for convenience, but you should override it with your
# own key if available.  See https://fred.stlouisfed.org/docs/api/fred/ for
# details on obtaining an API key.
# ---------------------------------------------------------------------------
# Date range for data pulls
#
# These constants define the start and end dates for data collection.  The
# asset start date allows asset price series to have a shorter history than
# macro series if desired.
# ---------------------------------------------------------------------------
START_DATE = datetime(1960, 1, 1)
END_DATE = datetime.today()
ASSET_START_DATE = datetime(2005, 1, 1)

# ---------------------------------------------------------------------------
# FRED series definitions
#
# FRED_SERIES maps each FRED series ID to a descriptive name.  Including
# descriptive names makes logs and column labels more interpretable.  The
# keys correspond to FRED series identifiers (e.g., "RPI", "UNRATE").
# The values are human‑readable descriptions of those series.  Feel free to
# edit or extend this dictionary as needed for your analysis.
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Core inflation and growth indicators from the original config
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "CPILFESL": "Core Consumer Price Index (excluding food and energy)",
    "PPIACO": "Producer Price Index for All Commodities",
    "GDPC1": "Real Gross Domestic Product",
    "GDPPOT": "Real Potential Gross Domestic Product",
    "INDPRO": "Industrial Production Index",
    "RSAFS": "Retail Sales",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls",
    "CES0500000003": "Average Hourly Earnings of All Employees",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "DGS2": "2‑Year Treasury Constant Maturity Rate",
    "DGS5": "5‑Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10‑Year Treasury Minus 2‑Year Treasury Yield",
    "UMCSENT": "University of Michigan Consumer Sentiment Index",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option‑Adjusted Spread",

    # Additional macro and financial stress indicators added in the original config
    "T5YIFR": "5‑Year Forward Inflation Expectation Rate",
    "T10YIE": "10‑Year Breakeven Inflation Rate (market‑based inflation expectations)",
    "NFCI": "Chicago Fed National Financial Conditions Index (stress indicator)",
    "M2SL": "M2 Money Stock",
    "ICSA": "Initial Jobless Claims",
    "HOUST": "Housing Starts: Total New Privately Owned Housing Units Started",
    "DCOILWTICO": "WTI Crude Oil Price",
    "MOVE": "Merrill Lynch Option Volatility Estimate (MOVE Index)",

    # -------------------------------------------------------------------------
    # Extended FRED‑MD series (as provided by the user)
    # Each code is mapped to a descriptive name.  Transformation codes for
    # these series are provided separately in the TCODE_MAP dictionary below.
    # -------------------------------------------------------------------------
    "RPI": "Real Personal Income",
    "W875RX1": "Real personal income excluding current transfer receipts",
    "DPCERA3M086SBEA": "Real personal consumption expenditures (chain‑type quantity index)",
    "CMRMTSPLx": "Real Manufacturing and Trade Industries Sales",
    "RETAILx": "Advance Retail Sales: Retail Trade",
    "IPFPNSS": "Industrial Production: Final Products and Nonindustrial Supplies",
    "IPFINAL": "Industrial Production: Final Products",
    "IPCONGD": "Industrial Production: Consumer Goods",
    "IPDCONGD": "Industrial Production: Durable Consumer Goods",
    "IPNCONGD": "Industrial Production: Nondurable Consumer Goods",
    "IPBUSEQ": "Industrial Production: Equipment: Business Equipment",
    "IPMAT": "Industrial Production: Materials",
    "IPDMAT": "Industrial Production: Durable Goods Materials",
    "IPNMAT": "Industrial Production: Nondurable Goods Materials",
    "IPMANSICS": "Industrial Production: Manufacturing (SIC)",
    "IPB51222S": "Industrial Production: Nondurable Energy Consumer Goods: Residential Utilities",
    "IPFUELS": "Industrial Production: Nondurable Energy Consumer Goods: Fuels",
    "CUMFNS": "Capacity Utilization: Manufacturing (SIC)",
    "HWI": "Help‑Wanted Index for United States",
    "HWIURATIO": "Ratio of Help Wanted to Number Unemployed",
    "CLF16OV": "Civilian Labor Force Level",
    "CE16OV": "Employment Level",
    "UEMPMEAN": "Average Weeks Unemployed",
    "UEMPLT5": "Number Unemployed for Less Than 5 Weeks",
    "UEMP5TO14": "Number Unemployed for 5–14 Weeks",
    "UEMP15OV": "Number Unemployed for 15 Weeks & over",
    "UEMP15T26": "Number Unemployed for 15–26 Weeks",
    "UEMP27OV": "Number Unemployed for 27 Weeks & over",
    "CLAIMSx": "Initial Claims",
    "USGOOD": "All Employees: Goods‑Producing Industries",
    "CES1021000001": "All Employees: Mining and Logging, Logging",
    "USCONS": "All Employees: Construction",
    "MANEMP": "All Employees: Manufacturing",
    "DMANEMP": "All Employees: Durable Manufacturing",
    "NDMANEMP": "All Employees: Nondurable Manufacturing",
    "SRVPRD": "All Employees: Private Service‑Providing Industries",
    "USTPU": "All Employees: Trade, Transportation, and Utilities",
    "USWTRADE": "All Employees: Wholesale Trade",
    "USTRADE": "All Employees: Retail Trade",
    "USFIRE": "All Employees: Financial Activities",
    "USGOVT": "All Employees: Government",
    "CES0600000007": "Average Weekly Hours of Production and Nonsupervisory Employees, Goods‑Producing",
    "AWOTMAN": "Average Weekly Overtime Hours of Production and Nonsupervisory Employees, Manufacturing",
    "AWHMAN": "Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing",
    "HOUSTNE": "Housing Starts in Northeast Census Region",
    "HOUSTMW": "Housing Starts in Midwest Census Region",
    "HOUSTS": "Housing Starts in South Census Region",
    "HOUSTW": "Housing Starts in West Census Region",
    "PERMIT": "New Privately‑Owned Housing Units Authorized in Permit‑Issuing Places: Total Units",
    "PERMITNE": "New Privately‑Owned Housing Units Authorized in Permit‑Issuing Places in Northeast Census Region",
    "PERMITMW": "New Privately‑Owned Housing Units Authorized in Permit‑Issuing Places in Midwest Census Region",
    "PERMITS": "New Privately‑Owned Housing Units Authorized in Permit‑Issuing Places in South Census Region",
    "PERMITW": "New Privately‑Owned Housing Units Authorized in Permit‑Issuing Places in West Census Region",
    "ACOGNO": "New Orders for Consumer Goods",
    "AMDMNOx": "New Orders for Durable Goods",
    "ANDENOx": "New Orders for Nondefense Capital Goods Excluding Aircraft",
    "AMDMUOx": "New Orders for Durable Goods Excluding Transportation",
    "BUSINVx": "Total Business Inventories",
    "ISRATIOx": "Total Business: Inventories to Sales Ratio",
    "M1SL": "M1 Money Stock",
    "M2REAL": "Real M2 Money Stock",
    "BOGMBASE": "Monetary Base; Total",
    "TOTRESNS": "Total Reserves of Depository Institutions, Not Seasonally Adjusted",
    "NONBORRES": "Reserves of Depository Institutions, Nonborrowed",
    "BUSLOANS": "Commercial and Industrial Loans, All Commercial Banks",
    "REALLN": "Real Estate Loans, All Commercial Banks",
    "NONREVSL": "Total Nonrevolving Credit Outstanding",
    "CONSPI": "Personal Consumption Expenditures: Chain‑type Price Index",
    "SP500": "S&P 500 Index (level)",
    "SP_DIV_YIELD": "S&P 500 Dividend Yield",
    "SP_PE_RATIO": "S&P 500 Price‑to‑Earnings Ratio",
    "CP3Mx": "3‑Month Commercial Paper Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "TB6MS": "6‑Month Treasury Bill Secondary Market Rate",
    "GS1": "1‑Year Treasury Constant Maturity Rate",
    "GS5": "5‑Year Treasury Constant Maturity Rate",
    "AAA": "Moody's Seasoned AAA Corporate Bond Yield",
    "BAA": "Moody's Seasoned BAA Corporate Bond Yield",
    "COMPAPFFx": "Commercial Paper Spreads",
    "TB3SMFFM": "3‑Month Treasury Bill Rate minus Federal Funds Rate",
    "TB6SMFFM": "6‑Month Treasury Bill Rate minus Federal Funds Rate",
    "T1YFFM": "1‑Year Treasury Constant Maturity Rate minus Federal Funds Rate",
    "T5YFFM": "5‑Year Treasury Constant Maturity Rate minus Federal Funds Rate",
    "T10YFFM": "10‑Year Treasury Constant Maturity Rate minus Federal Funds Rate",
    "AAAFFM": "Moody's Seasoned AAA Corporate Bond Yield minus Federal Funds Rate",
    "BAAFFM": "Moody's Seasoned BAA Corporate Bond Yield minus Federal Funds Rate",
    "TWEXAFEGSMTHx": "Trade Weighted U.S. Dollar Index: Advanced Foreign Economies",
    "EXSZUSx": "Switzerland / U.S. Foreign Exchange Rate",
    "EXJPUSx": "Japan / U.S. Foreign Exchange Rate",
    "EXUSUKx": "U.S. / U.K. Foreign Exchange Rate",
    "EXCAUSx": "Canada / U.S. Foreign Exchange Rate",
    "WPSFD49207": "Producer Price Index by Commodity: Finished Goods",
    "WPSFD49502": "Producer Price Index by Commodity: Finished Consumer Foods",
    "WPSID61": "Producer Price Index by Commodity: Intermediate Materials, Supplies and Components",
    "WPSID62": "Producer Price Index by Commodity: Crude Materials for Further Processing",
    "OILPRICEx": "Crude Oil Prices",
    "PPICMM": "Producer Price Index by Commodity: Metals and Metal Products",
    "CPIAPPSL": "Consumer Price Index for All Urban Consumers: Apparel",
    "CPITRNSL": "Consumer Price Index for All Urban Consumers: Transportation",
    "CPIMEDSL": "Consumer Price Index for All Urban Consumers: Medical Care",
    "CUSR0000SAC": "Consumer Price Index for All Urban Consumers: Commodities",
    "CUSR0000SAD": "Consumer Price Index for All Urban Consumers: Durables",
    "CUSR0000SAS": "Consumer Price Index for All Urban Consumers: Services",
    "CPIULFSL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
    "CUSR0000SA0L2": "Consumer Price Index for All Urban Consumers: All Items Less Shelter",
    "CUSR0000SA0L5": "Consumer Price Index for All Urban Consumers: All Items Less Medical Care",
    "PCEPI": "Personal Consumption Expenditures: Chain‑type Price Index",
    "DDURRG3M086SBEA": "Personal Consumption Expenditures: Durable Goods",
    "DNDGRG3M086SBEA": "Personal Consumption Expenditures: Nondurable Goods",
    "DSERRG3M086SBEA": "Personal Consumption Expenditures: Services",
    "CES0600000008": "Average Hourly Earnings of Production and Nonsupervisory Employees, Goods‑Producing",
    "CES2000000008": "Average Hourly Earnings of Production and Nonsupervisory Employees, Construction",
    "CES3000000008": "Average Hourly Earnings of Production and Nonsupervisory Employees, Manufacturing",
    "UMCSENTx": "University of Michigan: Consumer Sentiment (alternative)",
    "DTCOLNVHFNM": "No information found (placeholder)",
    "DTCTHFNM": "No information found (placeholder)",
    "INVEST": "No information found (placeholder)",
    "VIXCLSx": "CBOE Volatility Index: VIX (alternative)"
}

# ---------------------------------------------------------------------------
# Transformation codes
#
# The TCODE_MAP dictionary assigns a transformation code to each FRED series ID.
# The codes should align with the FRED‑MD conventions used in the research
# literature.  These values guide the data processing functions on how to
# transform each series before analysis.  If a series is not present in this
# dictionary, it will default to a level transformation (t‑code 1).
# ---------------------------------------------------------------------------
TCODE_MAP = {
    "RPI": 5,
    "W875RX1": 5,
    "DPCERA3M086SBEA": 5,
    "CMRMTSPLx": 5,
    "RETAILx": 5,
    "INDPRO": 5,
    "IPFPNSS": 5,
    "IPFINAL": 5,
    "IPCONGD": 5,
    "IPDCONGD": 5,
    "IPNCONGD": 5,
    "IPBUSEQ": 5,
    "IPMAT": 5,
    "IPDMAT": 5,
    "IPNMAT": 5,
    "IPMANSICS": 5,
    "IPB51222S": 5,
    "IPFUELS": 5,
    "CUMFNS": 2,
    "HWI": 2,
    "HWIURATIO": 2,
    "CLF16OV": 5,
    "CE16OV": 5,
    "UNRATE": 2,
    "UEMPMEAN": 2,
    "UEMPLT5": 5,
    "UEMP5TO14": 5,
    "UEMP15OV": 5,
    "UEMP15T26": 5,
    "UEMP27OV": 5,
    "CLAIMSx": 5,
    "PAYEMS": 5,
    "USGOOD": 5,
    "CES1021000001": 5,
    "USCONS": 5,
    "MANEMP": 5,
    "DMANEMP": 5,
    "NDMANEMP": 5,
    "SRVPRD": 5,
    "USTPU": 5,
    "USWTRADE": 5,
    "USTRADE": 5,
    "USFIRE": 5,
    "USGOVT": 1,
    "CES0600000007": 2,
    "AWOTMAN": 1,
    "AWHMAN": 4,
    "HOUST": 4,
    "HOUSTNE": 4,
    "HOUSTMW": 4,
    "HOUSTS": 4,
    "HOUSTW": 4,
    "PERMIT": 4,
    "PERMITNE": 4,
    "PERMITMW": 4,
    "PERMITS": 4,
    "PERMITW": 4,
    "ACOGNO": 5,
    "AMDMNOx": 5,
    "ANDENOx": 5,
    "AMDMUOx": 5,
    "BUSINVx": 2,
    "ISRATIOx": 6,
    "M1SL": 6,
    "M2SL": 5,
    "M2REAL": 6,
    "BOGMBASE": 6,
    "TOTRESNS": 7,
    "NONBORRES": 6,
    "BUSLOANS": 6,
    "REALLN": 6,
    "NONREVSL": 2,
    "CONSPI": 5,
    "SP500": 2,
    "SP_DIV_YIELD": 5,
    "SP_PE_RATIO": 2,
    "FEDFUNDS": 2,
    "CP3Mx": 2,
    "TB3MS": 2,
    "TB6MS": 2,
    "GS1": 2,
    "GS5": 2,
    "GS10": 2,
    "AAA": 2,
    "BAA": 1,
    "COMPAPFFx": 1,
    "TB3SMFFM": 1,
    "TB6SMFFM": 1,
    "T1YFFM": 1,
    "T5YFFM": 1,
    "T10YFFM": 1,
    "AAAFFM": 1,
    "BAAFFM": 5,
    "TWEXAFEGSMTHx": 5,
    "EXSZUSx": 5,
    "EXJPUSx": 5,
    "EXUSUKx": 5,
    "EXCAUSx": 6,
    "WPSFD49207": 6,
    "WPSFD49502": 6,
    "WPSID61": 6,
    "WPSID62": 6,
    "OILPRICEx": 6,
    "PPICMM": 6,
    "CPIAUCSL": 6,
    "CPIAPPSL": 6,
    "CPITRNSL": 6,
    "CPIMEDSL": 6,
    "CUSR0000SAC": 6,
    "CUSR0000SAD": 6,
    "CUSR0000SAS": 6,
    "CPIULFSL": 6,
    "CUSR0000SA0L2": 6,
    "CUSR0000SA0L5": 6,
    "PCEPI": 6,
    "DDURRG3M086SBEA": 6,
    "DNDGRG3M086SBEA": 6,
    "DSERRG3M086SBEA": 6,
    "CES0600000008": 6,
    "CES2000000008": 6,
    "CES3000000008": 2,
    "UMCSENTx": 6,
    "DTCOLNVHFNM": 6,
    "DTCTHFNM": 6,
    "INVEST": 1,
    "VIXCLSx": 1
}

# ---------------------------------------------------------------------------
# Asset tickers for index prices
#
# To fetch actual index levels (rather than ETF proxies), we use Yahoo Finance
# index symbols.  These symbols begin with a caret (^) for major indices.  If
# a particular index series is not available, you may need to adjust the
# symbols according to your data provider.
# ---------------------------------------------------------------------------
ASSET_TICKERS = {
    "SPX": "^GSPC",      # S&P 500 index level (spot price)
    "NDX": "^NDX",      # NASDAQ‑100 index level
    "GOLD": "GC=F",     # COMEX gold futures price (as a proxy for gold spot)
    "BTC": "BTC-USD",    # Bitcoin price in USD
    "US10Y": "^TNX",    # 10‑year US Treasury yield index
    "US2Y": "^FVX"       # 5‑year US Treasury yield index (proxy for 2‑year yield)
}

# ---------------------------------------------------------------------------
# Stooq tickers
#
# Stooq provides certain financial indices (e.g., MOVE) that may not be
# available from other sources.  These tickers can be used with the Stooq
# downloader functions defined in the data fetching module.
# ---------------------------------------------------------------------------
STOOQ_TICKERS = {
    "MOVE": "^MOVE",
    "VIX3M": "^VIX3M"
}

# ---------------------------------------------------------------------------
# Clustering features
#
# These features are used when performing clustering or regime classification
# in the pipeline.  The names refer to processed columns in the merged data
# set (e.g., CPI_YoY is the year‑over‑year change in CPI).  You can adjust
# this list based on which indicators you want to include in the model.
# ---------------------------------------------------------------------------
CLUSTER_FEATURES = [
    "CPI_YoY",
    "GDP_YoY",
    "UNRATE",
    "FEDFUNDS",
    "DGS10",
    "INDPRO_YoY",
    "UMCSENT",
    "YieldCurve_Slope",
    "FinConditions_Composite"
]

# ---------------------------------------------------------------------------
# Rule‑based classification thresholds
#
# The thresholds below define cut‑offs used in the rule‑based regime
# classifier (see src/models/regime_classifier.py for details).  They can be
# tuned based on the historical distribution of your data or specific
# forecasting requirements.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "STRONG_GROWTH_GDP": 2.5,
    "MODERATE_GROWTH_GDP": 1.0,
    "WEAK_GROWTH_GDP": 0.0,
    "RECESSION_GDP": -0.5,
    "HIGH_INFLATION_CPI": 3.5,
    "MODERATE_INFLATION_CPI": 2.0,
    "LOW_INFLATION_CPI": 1.0,
    "STRONG_LABOR_UNRATE": 4.0,
    "WEAK_LABOR_UNRATE": 5.5
}

# ---------------------------------------------------------------------------
# File paths
#
# These directories determine where raw, processed and output files are stored.
# They follow the same structure as the original configuration.  The
# os.makedirs calls ensure that directories exist when the module is loaded.
# ---------------------------------------------------------------------------
DATA_DIR = "Data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = "Output"

# Ensure required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)