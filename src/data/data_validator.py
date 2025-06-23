"""
Comprehensive Data Validation System

This module provides a robust data validation framework for macro-economic data
analysis pipelines. It ensures data quality and integrity through multiple
validation layers including schema validation, range checks, format validation,
and consistency checks across related fields.

Key Features:
- Schema validation for different data sources
- Numerical range and boundary checks
- Date and categorical format validation
- Cross-field consistency validation
- Comprehensive reporting with severity levels
- Extensible validation rule system
- Integration with data cleaning pipeline
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationType(Enum):
    """Types of validation checks"""
    SCHEMA = "schema"
    RANGE = "range"
    FORMAT = "format"
    CONSISTENCY = "consistency"
    CUSTOM = "custom"

class DataType(Enum):
    """Supported data types for validation"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"

@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    validation_type: ValidationType
    severity: ValidationSeverity
    field: str
    message: str
    value: Any = None
    expected: Any = None
    row_index: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'validation_type': self.validation_type.value,
            'severity': self.severity.value,
            'field': self.field,
            'message': self.message,
            'value': str(self.value) if self.value is not None else None,
            'expected': str(self.expected) if self.expected is not None else None,
            'row_index': self.row_index,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    source_name: str
    validation_timestamp: datetime
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    passed_validations: int = 0
    failed_validations: int = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue"""
        self.issues.append(issue)
        self.failed_validations += 1
    
    def add_success(self):
        """Increment successful validation count"""
        self.passed_validations += 1
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_type(self, validation_type: ValidationType) -> List[ValidationIssue]:
        """Get issues filtered by validation type"""
        return [issue for issue in self.issues if issue.validation_type == validation_type]
    
    def get_summary(self) -> Dict:
        """Get validation summary"""
        severity_counts = {}
        type_counts = {}
        
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len(self.get_issues_by_severity(severity))
        
        for val_type in ValidationType:
            type_counts[val_type.value] = len(self.get_issues_by_type(val_type))
        
        return {
            'source_name': self.source_name,
            'timestamp': self.validation_timestamp.isoformat(),
            'data_shape': {'rows': self.total_rows, 'columns': self.total_columns},
            'validation_counts': {
                'passed': self.passed_validations,
                'failed': self.failed_validations,
                'total': self.passed_validations + self.failed_validations
            },
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'success_rate': self.passed_validations / (self.passed_validations + self.failed_validations) * 100 if (self.passed_validations + self.failed_validations) > 0 else 0
        }
    
    def to_dict(self) -> Dict:
        """Convert full report to dictionary"""
        return {
            'summary': self.get_summary(),
            'issues': [issue.to_dict() for issue in self.issues]
        }

@dataclass
class FieldSchema:
    """Schema definition for a single field"""
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    date_format: Optional[str] = None
    description: Optional[str] = None

@dataclass
class DataSchema:
    """Complete schema definition for a data source"""
    name: str
    description: str
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    consistency_rules: List[Dict] = field(default_factory=list)
    
    def add_field(self, field_schema: FieldSchema):
        """Add a field to the schema"""
        self.fields[field_schema.name] = field_schema
        if field_schema.required:
            self.required_fields.append(field_schema.name)

class DataValidator:
    """
    Main data validation class
    
    Provides comprehensive data validation capabilities with support for
    multiple validation types, severity levels, and detailed reporting.
    """
    
    def __init__(self, schemas_dir: Optional[str] = None):
        """
        Initialize the DataValidator
        
        Args:
            schemas_dir: Directory containing schema definition files
        """
        self.schemas: Dict[str, DataSchema] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_history: List[ValidationReport] = []
        
        if schemas_dir:
            self.load_schemas_from_directory(schemas_dir)
        
        # Load default schemas
        self._load_default_schemas()
        
        logger.info("DataValidator initialized")
    
    def _load_default_schemas(self):
        """Load default schemas for common data sources"""
        # FRED data schema
        fred_schema = DataSchema(
            name="FRED",
            description="Federal Reserve Economic Data validation schema"
        )
        
        fred_schema.add_field(FieldSchema(
            name="date",
            data_type=DataType.DATETIME,
            required=True,
            nullable=False,
            description="Observation date"
        ))
        
        fred_schema.add_field(FieldSchema(
            name="value",
            data_type=DataType.FLOAT,
            required=True,
            nullable=True,
            description="Economic indicator value"
        ))
        
        # Add common FRED series validations
        fred_schema.add_field(FieldSchema(
            name="UNRATE",
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=25.0,
            description="Unemployment rate (percentage)"
        ))
        
        fred_schema.add_field(FieldSchema(
            name="CPIAUCSL",
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=1000.0,
            description="Consumer Price Index"
        ))
        
        fred_schema.add_field(FieldSchema(
            name="GDP",
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=50000.0,
            description="Gross Domestic Product (billions)"
        ))
        
        self.register_schema(fred_schema)
        
        # Yahoo Finance data schema
        yahoo_schema = DataSchema(
            name="YahooFinance",
            description="Yahoo Finance market data validation schema"
        )
        
        for field_name in ["Open", "High", "Low", "Close", "AdjClose"]:
            yahoo_schema.add_field(FieldSchema(
                name=field_name,
                data_type=DataType.FLOAT,
                required=True,
                nullable=False,
                min_value=0.0,
                description=f"Stock {field_name.lower()} price"
            ))
        
        yahoo_schema.add_field(FieldSchema(
            name="Volume",
            data_type=DataType.INTEGER,
            required=True,
            nullable=False,
            min_value=0,
            max_value=1e12,
            description="Trading volume"
        ))
        
        # Add consistency rules for Yahoo Finance data
        yahoo_schema.consistency_rules = [
            {
                "name": "OHLC_consistency",
                "description": "High should be >= Low, Open, Close",
                "rule": "High >= Low and High >= Open and High >= Close"
            },
            {
                "name": "Low_consistency", 
                "description": "Low should be <= High, Open, Close",
                "rule": "Low <= High and Low <= Open and Low <= Close"
            }
        ]
        
        self.register_schema(yahoo_schema)
        
        logger.info("Loaded default schemas for FRED and Yahoo Finance")
    
    def register_schema(self, schema: DataSchema):
        """Register a data schema"""
        self.schemas[schema.name] = schema
        logger.info(f"Registered schema: {schema.name}")
    
    def register_custom_validator(self, name: str, validator_func: Callable):
        """Register a custom validation function"""
        self.custom_validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    def validate_data(self, 
                     data: pd.DataFrame, 
                     schema_name: Optional[str] = None,
                     source_name: Optional[str] = None) -> ValidationReport:
        """
        Main validation method
        
        Args:
            data: DataFrame to validate
            schema_name: Name of schema to use for validation
            source_name: Name of data source for reporting
            
        Returns:
            ValidationReport with all validation results
        """
        if source_name is None:
            source_name = schema_name or "Unknown"
        
        report = ValidationReport(
            source_name=source_name,
            validation_timestamp=datetime.now(),
            total_rows=len(data),
            total_columns=len(data.columns)
        )
        
        logger.info(f"Starting validation for {source_name}: {len(data)} rows, {len(data.columns)} columns")
        
        try:
            # Schema validation
            if schema_name and schema_name in self.schemas:
                self._validate_schema(data, self.schemas[schema_name], report)
            else:
                logger.warning(f"No schema found for {schema_name}, performing basic validation")
                self._validate_basic(data, report)
            
            # Always perform general data quality checks
            self._validate_data_quality(data, report)
            
            # Consistency checks if schema available
            if schema_name and schema_name in self.schemas:
                self._validate_consistency(data, self.schemas[schema_name], report)
            
            logger.info(f"Validation completed: {report.passed_validations} passed, {report.failed_validations} failed")
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                validation_type=ValidationType.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                field="SYSTEM",
                message=f"Validation system error: {str(e)}"
            ))
            logger.error(f"Validation system error: {e}")
        
        finally:
            self.validation_history.append(report)
        
        return report
    
    def _validate_schema(self, data: pd.DataFrame, schema: DataSchema, report: ValidationReport):
        """Validate data against schema definition"""
        # Check required fields
        for required_field in schema.required_fields:
            if required_field not in data.columns:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    field=required_field,
                    message=f"Required field '{required_field}' is missing"
                ))
            else:
                report.add_success()
        
        # Validate each field
        for field_name, field_schema in schema.fields.items():
            if field_name in data.columns:
                self._validate_field(data[field_name], field_schema, report)
    
    def _validate_field(self, series: pd.Series, field_schema: FieldSchema, report: ValidationReport):
        """Validate a single field against its schema"""
        field_name = field_schema.name
        
        # Check for required non-null values
        if field_schema.required and not field_schema.nullable:
            null_count = series.isnull().sum()
            if null_count > 0:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    field=field_name,
                    message=f"Field '{field_name}' has {null_count} null values but nulls are not allowed"
                ))
            else:
                report.add_success()
        
        # Data type validation
        if field_schema.data_type == DataType.FLOAT:
            non_numeric = pd.to_numeric(series, errors='coerce').isnull() & series.notnull()
            if non_numeric.any():
                invalid_rows = series.index[non_numeric].tolist()
                for row_idx in invalid_rows[:5]:  # Report first 5 issues
                    report.add_issue(ValidationIssue(
                        validation_type=ValidationType.FORMAT,
                        severity=ValidationSeverity.ERROR,
                        field=field_name,
                        message=f"Invalid numeric value",
                        value=series.iloc[row_idx],
                        expected="numeric",
                        row_index=row_idx
                    ))
            else:
                report.add_success()
        
        elif field_schema.data_type == DataType.INTEGER:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                is_integer = (numeric_series == numeric_series.astype(int, errors='ignore'))
                non_integer = ~is_integer & series.notnull()
                if non_integer.any():
                    invalid_rows = series.index[non_integer].tolist()
                    for row_idx in invalid_rows[:5]:
                        report.add_issue(ValidationIssue(
                            validation_type=ValidationType.FORMAT,
                            severity=ValidationSeverity.ERROR,
                            field=field_name,
                            message=f"Invalid integer value",
                            value=series.iloc[row_idx],
                            expected="integer",
                            row_index=row_idx
                        ))
                else:
                    report.add_success()
            except Exception:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.FORMAT,
                    severity=ValidationSeverity.ERROR,
                    field=field_name,
                    message=f"Could not validate integer format"
                ))
        
        elif field_schema.data_type == DataType.DATETIME:
            try:
                datetime_series = pd.to_datetime(series, errors='coerce')
                invalid_dates = datetime_series.isnull() & series.notnull()
                if invalid_dates.any():
                    invalid_rows = series.index[invalid_dates].tolist()
                    for row_idx in invalid_rows[:5]:
                        report.add_issue(ValidationIssue(
                            validation_type=ValidationType.FORMAT,
                            severity=ValidationSeverity.ERROR,
                            field=field_name,
                            message=f"Invalid datetime format",
                            value=series.iloc[row_idx],
                            expected="datetime",
                            row_index=row_idx
                        ))
                else:
                    report.add_success()
            except Exception:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.FORMAT,
                    severity=ValidationSeverity.ERROR,
                    field=field_name,
                    message=f"Could not validate datetime format"
                ))
        
        # Range validation for numeric fields
        if field_schema.min_value is not None or field_schema.max_value is not None:
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            if field_schema.min_value is not None:
                below_min = numeric_series < field_schema.min_value
                below_min_count = below_min.sum()
                if below_min_count > 0:
                    report.add_issue(ValidationIssue(
                        validation_type=ValidationType.RANGE,
                        severity=ValidationSeverity.WARNING,
                        field=field_name,
                        message=f"{below_min_count} values below minimum",
                        expected=f">= {field_schema.min_value}"
                    ))
                else:
                    report.add_success()
            
            if field_schema.max_value is not None:
                above_max = numeric_series > field_schema.max_value
                above_max_count = above_max.sum()
                if above_max_count > 0:
                    report.add_issue(ValidationIssue(
                        validation_type=ValidationType.RANGE,
                        severity=ValidationSeverity.WARNING,
                        field=field_name,
                        message=f"{above_max_count} values above maximum",
                        expected=f"<= {field_schema.max_value}"
                    ))
                else:
                    report.add_success()
        
        # Pattern validation for string fields
        if field_schema.pattern:
            try:
                pattern_matches = series.astype(str).str.match(field_schema.pattern, na=False)
                invalid_pattern = ~pattern_matches & series.notnull()
                if invalid_pattern.any():
                    invalid_count = invalid_pattern.sum()
                    report.add_issue(ValidationIssue(
                        validation_type=ValidationType.FORMAT,
                        severity=ValidationSeverity.WARNING,
                        field=field_name,
                        message=f"{invalid_count} values don't match required pattern",
                        expected=field_schema.pattern
                    ))
                else:
                    report.add_success()
            except Exception as e:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.FORMAT,
                    severity=ValidationSeverity.ERROR,
                    field=field_name,
                    message=f"Pattern validation error: {str(e)}"
                ))
        
        # Allowed values validation
        if field_schema.allowed_values:
            invalid_values = ~series.isin(field_schema.allowed_values) & series.notnull()
            if invalid_values.any():
                invalid_count = invalid_values.sum()
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.FORMAT,
                    severity=ValidationSeverity.ERROR,
                    field=field_name,
                    message=f"{invalid_count} values not in allowed list",
                    expected=str(field_schema.allowed_values)
                ))
            else:
                report.add_success()
    
    def _validate_basic(self, data: pd.DataFrame, report: ValidationReport):
        """Perform basic validation when no schema is available"""
        # Check for completely empty columns
        for col in data.columns:
            if data[col].isnull().all():
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    field=col,
                    message=f"Column '{col}' is completely empty"
                ))
            else:
                report.add_success()
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            report.add_issue(ValidationIssue(
                validation_type=ValidationType.CONSISTENCY,
                severity=ValidationSeverity.INFO,
                field="ENTIRE_ROW",
                message=f"Found {duplicate_count} duplicate rows"
            ))
        else:
            report.add_success()
    
    def _validate_data_quality(self, data: pd.DataFrame, report: ValidationReport):
        """Perform general data quality checks"""
        # Check missing value patterns
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        
        for col, pct in missing_percentage.items():
            if pct > 50:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    field=col,
                    message=f"High missing value rate: {pct:.1f}%"
                ))
            elif pct > 0:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.INFO,
                    field=col,
                    message=f"Missing values: {pct:.1f}%"
                ))
            else:
                report.add_success()
        
        # Check for infinite values in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            inf_count = np.isinf(data[col]).sum()
            if inf_count > 0:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.RANGE,
                    severity=ValidationSeverity.ERROR,
                    field=col,
                    message=f"Found {inf_count} infinite values"
                ))
            else:
                report.add_success()
    
    def _validate_consistency(self, data: pd.DataFrame, schema: DataSchema, report: ValidationReport):
        """Validate consistency rules"""
        for rule in schema.consistency_rules:
            try:
                rule_name = rule.get('name', 'unnamed_rule')
                rule_expression = rule.get('rule', '')
                
                # Simple rule evaluation for common patterns
                if schema.name == "YahooFinance":
                    if rule_name == "OHLC_consistency" and all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
                        violations = ~((data['High'] >= data['Low']) & 
                                     (data['High'] >= data['Open']) & 
                                     (data['High'] >= data['Close']))
                        violation_count = violations.sum()
                        
                        if violation_count > 0:
                            report.add_issue(ValidationIssue(
                                validation_type=ValidationType.CONSISTENCY,
                                severity=ValidationSeverity.ERROR,
                                field="OHLC",
                                message=f"{violation_count} rows violate OHLC consistency (High should be >= Low, Open, Close)"
                            ))
                        else:
                            report.add_success()
                    
                    elif rule_name == "Low_consistency" and all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
                        violations = ~((data['Low'] <= data['High']) & 
                                     (data['Low'] <= data['Open']) & 
                                     (data['Low'] <= data['Close']))
                        violation_count = violations.sum()
                        
                        if violation_count > 0:
                            report.add_issue(ValidationIssue(
                                validation_type=ValidationType.CONSISTENCY,
                                severity=ValidationSeverity.ERROR,
                                field="OHLC",
                                message=f"{violation_count} rows violate Low consistency (Low should be <= High, Open, Close)"
                            ))
                        else:
                            report.add_success()
                
            except Exception as e:
                report.add_issue(ValidationIssue(
                    validation_type=ValidationType.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    field="CONSISTENCY_RULE",
                    message=f"Error evaluating consistency rule '{rule_name}': {str(e)}"
                ))
    
    def save_report(self, report: ValidationReport, filepath: str):
        """Save validation report to file"""
        try:
            report_dict = report.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    def load_schemas_from_directory(self, schemas_dir: str):
        """Load schema definitions from JSON files in directory"""
        try:
            schemas_path = Path(schemas_dir)
            if not schemas_path.exists():
                logger.warning(f"Schemas directory {schemas_dir} does not exist")
                return
            
            for schema_file in schemas_path.glob("*.json"):
                try:
                    with open(schema_file, 'r') as f:
                        schema_dict = json.load(f)
                    
                    # Convert dictionary to DataSchema object
                    schema = self._dict_to_schema(schema_dict)
                    self.register_schema(schema)
                    
                    logger.info(f"Loaded schema from {schema_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading schema from {schema_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading schemas from directory: {e}")
    
    def _dict_to_schema(self, schema_dict: Dict) -> DataSchema:
        """Convert dictionary to DataSchema object"""
        schema = DataSchema(
            name=schema_dict['name'],
            description=schema_dict.get('description', ''),
            consistency_rules=schema_dict.get('consistency_rules', [])
        )
        
        for field_dict in schema_dict.get('fields', []):
            field_schema = FieldSchema(
                name=field_dict['name'],
                data_type=DataType(field_dict['data_type']),
                required=field_dict.get('required', True),
                nullable=field_dict.get('nullable', True),
                min_value=field_dict.get('min_value'),
                max_value=field_dict.get('max_value'),
                min_length=field_dict.get('min_length'),
                max_length=field_dict.get('max_length'),
                pattern=field_dict.get('pattern'),
                allowed_values=field_dict.get('allowed_values'),
                date_format=field_dict.get('date_format'),
                description=field_dict.get('description')
            )
            schema.add_field(field_schema)
        
        return schema
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation history"""
        if not self.validation_history:
            return {"message": "No validation history available"}
        
        total_validations = len(self.validation_history)
        total_issues = sum(len(report.issues) for report in self.validation_history)
        
        severity_totals = {severity.value: 0 for severity in ValidationSeverity}
        type_totals = {val_type.value: 0 for val_type in ValidationType}
        
        for report in self.validation_history:
            for issue in report.issues:
                severity_totals[issue.severity.value] += 1
                type_totals[issue.validation_type.value] += 1
        
        return {
            'total_validation_runs': total_validations,
            'total_issues_found': total_issues,
            'severity_breakdown': severity_totals,
            'type_breakdown': type_totals,
            'latest_validation': self.validation_history[-1].get_summary() if self.validation_history else None
        }

# Utility functions for creating common schemas

def create_financial_data_schema() -> DataSchema:
    """Create a schema for general financial market data"""
    schema = DataSchema(
        name="FinancialData",
        description="General financial market data validation schema"
    )
    
    schema.add_field(FieldSchema(
        name="date",
        data_type=DataType.DATETIME,
        required=True,
        nullable=False
    ))
    
    schema.add_field(FieldSchema(
        name="symbol",
        data_type=DataType.STRING,
        required=True,
        nullable=False,
        pattern=r'^[A-Z]{1,5}$'
    ))
    
    for price_field in ["open", "high", "low", "close", "adjusted_close"]:
        schema.add_field(FieldSchema(
            name=price_field,
            data_type=DataType.FLOAT,
            required=True,
            nullable=False,
            min_value=0.0
        ))
    
    schema.add_field(FieldSchema(
        name="volume",
        data_type=DataType.INTEGER,
        required=True,
        nullable=False,
        min_value=0
    ))
    
    return schema

def create_economic_indicator_schema() -> DataSchema:
    """Create a schema for economic indicator data"""
    schema = DataSchema(
        name="EconomicIndicator",
        description="Economic indicator data validation schema"
    )
    
    schema.add_field(FieldSchema(
        name="date",
        data_type=DataType.DATETIME,
        required=True,
        nullable=False
    ))
    
    schema.add_field(FieldSchema(
        name="indicator_code",
        data_type=DataType.STRING,
        required=True,
        nullable=False
    ))
    
    schema.add_field(FieldSchema(
        name="value",
        data_type=DataType.FLOAT,
        required=True,
        nullable=True
    ))
    
    schema.add_field(FieldSchema(
        name="unit",
        data_type=DataType.STRING,
        required=False,
        nullable=True
    ))
    
    return schema

if __name__ == "__main__":
    # Example usage
    print("Data Validation System")
    print("=====================")
    
    # Create sample data with validation issues
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'Open': [100, 101, 102, 'invalid', 104, 105, 106, 107, 108, 109],
        'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'AdjClose': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    # Introduce some validation issues
    sample_data.loc[5, 'High'] = 95  # Violation: High < Low
    sample_data.loc[7, 'Volume'] = -100  # Violation: Negative volume
    
    # Create validator and validate
    validator = DataValidator()
    report = validator.validate_data(sample_data, schema_name="YahooFinance")
    
    print(f"Validation Results:")
    print(f"- Passed: {report.passed_validations}")
    print(f"- Failed: {report.failed_validations}")
    print(f"- Issues found: {len(report.issues)}")
    
    print("\nIssues by severity:")
    for severity in ValidationSeverity:
        issues = report.get_issues_by_severity(severity)
        if issues:
            print(f"- {severity.value}: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"  * {issue.field}: {issue.message}")
    
    print(f"\nValidation Summary:")
    summary = report.get_summary()
    print(f"Success Rate: {summary['success_rate']:.1f}%")