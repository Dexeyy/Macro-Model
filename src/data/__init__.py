# Import key functions from submodules for easier access
from .fetchers import (
    fetch_fred_series,
    fetch_asset_data,
    create_dummy_asset_data
)

from .processors import (
    process_macro_data,
    calculate_returns,
    merge_macro_and_asset_data
)

# Import the new data infrastructure system
from .infrastructure import (
    DataInfrastructure,
    ProcessedDataCache,
    get_data_infrastructure
)

# Import the feature store
from .feature_store import (
    FeatureStore
)

# Import the metadata tracker
from .metadata_tracker import (
    MetadataTracker,
    DataType,
    OperationType,
    DataLineageNode,
    DataTransformation,
    create_metadata_tracker
)

# Import the enhanced FRED fetcher
from .fred_fetcher import (
    EnhancedFredClient,
    FredRequestParams,
    FredFrequency,
    FredAggregationMethod,
    FredUnits,
    FredSeriesInfo,
    RateLimiter,
    COMMON_FRED_SERIES,
    FRED_ENDPOINTS_DOCUMENTATION
)

# Import the enhanced Yahoo Finance fetcher
from .yahoo_finance_fetcher import (
    EnhancedYahooFinanceClient,
    YahooRequestParams,
    YahooInterval,
    YahooPeriod,
    TickerInfo,
    COMMON_TICKERS
)

# Import the Data Cleaning and Normalization framework
from .data_cleaner import (
    DataCleaner,
    CleaningConfig,
    DataSourceConfig,
    MissingValueStrategy,
    OutlierMethod,
    OutlierTreatment,
    NormalizationMethod,
    create_fred_config,
    create_yahoo_finance_config,
    create_sample_cleaning_pipeline
)

# Import the Data Validation System
from .data_validator import (
    DataValidator,
    ValidationReport,
    ValidationIssue,
    FieldSchema,
    DataSchema,
    ValidationSeverity,
    ValidationType,
    DataType,
    create_financial_data_schema,
    create_economic_indicator_schema
) 