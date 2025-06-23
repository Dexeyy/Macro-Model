"""
Test suite for the Data Validation System

This test suite validates the comprehensive data validation framework
including schema validation, range checks, format validation, consistency
checks, and reporting functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os
import json
import tempfile

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data.data_validator import (
    DataValidator, ValidationReport, ValidationIssue, FieldSchema, DataSchema,
    ValidationSeverity, ValidationType, DataType,
    create_financial_data_schema, create_economic_indicator_schema
)

def create_sample_data_with_validation_issues():
    """Create sample data with various validation issues for testing"""
    np.random.seed(42)  # For reproducible results
    
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=20),
        'Open': [100, 101, 102, 'invalid', 104, np.nan, 106, 107, 108, 109, 
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
        'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
               105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
        'AdjClose': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]
    })
    
    # Introduce specific validation issues
    data.loc[5, 'High'] = 95   # Violation: High < Low (should be >= 100)
    data.loc[7, 'Volume'] = -100  # Violation: Negative volume
    data.loc[10, 'Low'] = 125  # Violation: Low > High (should be <= 115)
    data.loc[12, 'date'] = 'invalid_date'  # Invalid date format
    data.loc[15, 'Volume'] = 1e15  # Volume too high (exceeds max)
    
    return data

def create_fred_sample_data():
    """Create sample FRED data for testing"""
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=15, freq='M'),
        'UNRATE': [3.5, 3.6, 3.8, 4.4, 14.7, 13.3, 11.1, 10.2, 8.4, 7.9, 
                  6.7, 6.3, 6.0, 5.4, 4.6],  # Unemployment rates
        'CPIAUCSL': [258.8, 258.7, 258.1, 256.4, 256.7, 259.1, 259.9, 260.3,
                    260.4, 261.6, 263.2, 265.0, 266.6, 269.2, 271.7],  # CPI values
        'GDP': [21734.0, 19173.0, 19520.1, 20049.8, 20612.2, 21060.5, 21494.7,
               21821.0, 22038.2, 22199.2, 22741.5, 23315.4, 24002.8, 24661.0, 25035.2]
    })
    
    # Introduce some validation issues
    data.loc[3, 'UNRATE'] = -1.0  # Invalid: negative unemployment rate
    data.loc[7, 'CPIAUCSL'] = 1500.0  # Warning: very high CPI value
    data.loc[10, 'GDP'] = -5000.0  # Invalid: negative GDP
    
    return data

def test_basic_initialization():
    """Test basic DataValidator initialization"""
    print("Testing basic DataValidator initialization...")
    
    # Test default initialization
    validator = DataValidator()
    assert len(validator.schemas) >= 2  # Should have FRED and YahooFinance schemas
    assert 'FRED' in validator.schemas
    assert 'YahooFinance' in validator.schemas
    assert len(validator.validation_history) == 0
    
    print("✓ Basic initialization tests passed")

def test_schema_registration():
    """Test schema registration functionality"""
    print("Testing schema registration...")
    
    validator = DataValidator()
    initial_count = len(validator.schemas)
    
    # Create and register a custom schema
    custom_schema = DataSchema(
        name="CustomTest",
        description="Test schema for validation"
    )
    custom_schema.add_field(FieldSchema(
        name="test_field",
        data_type=DataType.STRING,
        required=True,
        nullable=False
    ))
    
    validator.register_schema(custom_schema)
    
    assert len(validator.schemas) == initial_count + 1
    assert 'CustomTest' in validator.schemas
    assert validator.schemas['CustomTest'].fields['test_field'].required == True
    
    print("✓ Schema registration tests passed")

def test_yahoo_finance_validation():
    """Test validation with Yahoo Finance schema"""
    print("Testing Yahoo Finance validation...")
    
    data = create_sample_data_with_validation_issues()
    validator = DataValidator()
    
    report = validator.validate_data(data, schema_name="YahooFinance")
    
    # Should have validation issues
    assert len(report.issues) > 0
    assert report.failed_validations > 0
    assert report.source_name == "YahooFinance"
    
    # Check for specific issue types
    format_issues = report.get_issues_by_type(ValidationType.FORMAT)
    range_issues = report.get_issues_by_type(ValidationType.RANGE)
    consistency_issues = report.get_issues_by_type(ValidationType.CONSISTENCY)
    
    assert len(format_issues) > 0  # Should find format issues (invalid Open value)
    assert len(range_issues) > 0   # Should find range issues (negative volume)
    assert len(consistency_issues) > 0  # Should find OHLC consistency issues
    
    # Check severity levels
    error_issues = report.get_issues_by_severity(ValidationSeverity.ERROR)
    warning_issues = report.get_issues_by_severity(ValidationSeverity.WARNING)
    
    assert len(error_issues) > 0
    
    print(f"  Found {len(report.issues)} total issues:")
    print(f"  - Format: {len(format_issues)}")
    print(f"  - Range: {len(range_issues)}")
    print(f"  - Consistency: {len(consistency_issues)}")
    print(f"  - Errors: {len(error_issues)}")
    print(f"  - Warnings: {len(warning_issues)}")
    
    print("✓ Yahoo Finance validation tests passed")

def test_fred_validation():
    """Test validation with FRED schema"""
    print("Testing FRED validation...")
    
    data = create_fred_sample_data()
    validator = DataValidator()
    
    report = validator.validate_data(data, schema_name="FRED")
    
    # Should detect range violations
    range_issues = report.get_issues_by_type(ValidationType.RANGE)
    assert len(range_issues) > 0  # Should find negative unemployment rate, etc.
    
    # Check specific field validations
    unrate_issues = [issue for issue in report.issues if 'UNRATE' in issue.field]
    gdp_issues = [issue for issue in report.issues if 'GDP' in issue.field]
    
    assert len(unrate_issues) > 0  # Should find negative unemployment rate
    assert len(gdp_issues) > 0     # Should find negative GDP
    
    print(f"  Found {len(report.issues)} issues in FRED data:")
    print(f"  - UNRATE issues: {len(unrate_issues)}")
    print(f"  - GDP issues: {len(gdp_issues)}")
    print(f"  - Range issues: {len(range_issues)}")
    
    print("✓ FRED validation tests passed")

def test_validation_severity_levels():
    """Test different validation severity levels"""
    print("Testing validation severity levels...")
    
    validator = DataValidator()
    
    # Create data with different severity issues
    data = pd.DataFrame({
        'Open': [100, 101, 'critical_error', 103],  # Format error (ERROR)
        'High': [105, 106, 107, 50000],             # Range warning (high value)
        'Low': [95, 96, 97, 98],
        'Close': [102, 103, 104, 105],
        'AdjClose': [102, 103, 104, 105],
        'Volume': [1000, 1100, np.inf, 1300]       # Infinite value (ERROR)
    })
    
    report = validator.validate_data(data, schema_name="YahooFinance")
    
    # Check different severity levels
    critical_issues = report.get_issues_by_severity(ValidationSeverity.CRITICAL)
    error_issues = report.get_issues_by_severity(ValidationSeverity.ERROR)
    warning_issues = report.get_issues_by_severity(ValidationSeverity.WARNING)
    info_issues = report.get_issues_by_severity(ValidationSeverity.INFO)
    
    # Should have errors for format and infinite values
    assert len(error_issues) > 0
    
    print(f"  Severity breakdown:")
    print(f"  - Critical: {len(critical_issues)}")
    print(f"  - Error: {len(error_issues)}")
    print(f"  - Warning: {len(warning_issues)}")
    print(f"  - Info: {len(info_issues)}")
    
    print("✓ Validation severity tests passed")

def test_consistency_validation():
    """Test cross-field consistency validation"""
    print("Testing consistency validation...")
    
    validator = DataValidator()
    
    # Create data with OHLC consistency issues
    data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 90, 107, 108, 109],    # Row 1: High < Open (inconsistent)
        'Low': [95, 96, 110, 98, 99],        # Row 2: Low > High (inconsistent)
        'Close': [102, 103, 104, 105, 106],
        'AdjClose': [102, 103, 104, 105, 106],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    report = validator.validate_data(data, schema_name="YahooFinance")
    
    # Should find consistency issues
    consistency_issues = report.get_issues_by_type(ValidationType.CONSISTENCY)
    assert len(consistency_issues) > 0
    
    # Check specific consistency rules
    ohlc_issues = [issue for issue in consistency_issues if 'OHLC' in issue.field]
    assert len(ohlc_issues) > 0
    
    print(f"  Found {len(consistency_issues)} consistency issues")
    print(f"  - OHLC consistency: {len(ohlc_issues)}")
    
    print("✓ Consistency validation tests passed")

def test_basic_validation_fallback():
    """Test basic validation when no schema is available"""
    print("Testing basic validation fallback...")
    
    validator = DataValidator()
    
    # Create data with basic issues
    data = pd.DataFrame({
        'column1': [1, 2, 3, 4, 5],
        'column2': [np.nan, np.nan, np.nan, np.nan, np.nan],  # Completely empty
        'column3': [10, 20, 10, 20, 10],  # Has duplicates
        'column4': [1.5, 2.5, np.inf, 4.5, 5.5]  # Has infinite values
    })
    
    # Add duplicate rows
    data = pd.concat([data, data.iloc[[1, 2]]], ignore_index=True)
    
    report = validator.validate_data(data, schema_name="UnknownSchema")
    
    # Should perform basic validation
    assert len(report.issues) > 0
    
    # Check for basic validation issues
    empty_column_issues = [issue for issue in report.issues if 'empty' in issue.message.lower()]
    duplicate_issues = [issue for issue in report.issues if 'duplicate' in issue.message.lower()]
    infinite_issues = [issue for issue in report.issues if 'infinite' in issue.message.lower()]
    
    assert len(empty_column_issues) > 0
    assert len(duplicate_issues) > 0
    assert len(infinite_issues) > 0
    
    print(f"  Basic validation found {len(report.issues)} issues:")
    print(f"  - Empty columns: {len(empty_column_issues)}")
    print(f"  - Duplicates: {len(duplicate_issues)}")
    print(f"  - Infinite values: {len(infinite_issues)}")
    
    print("✓ Basic validation fallback tests passed")

def test_validation_report_functionality():
    """Test validation report features"""
    print("Testing validation report functionality...")
    
    validator = DataValidator()
    data = create_sample_data_with_validation_issues()
    
    report = validator.validate_data(data, schema_name="YahooFinance")
    
    # Test report summary
    summary = report.get_summary()
    assert 'source_name' in summary
    assert 'validation_counts' in summary
    assert 'severity_breakdown' in summary
    assert 'type_breakdown' in summary
    assert 'success_rate' in summary
    
    # Test report conversion to dict
    report_dict = report.to_dict()
    assert 'summary' in report_dict
    assert 'issues' in report_dict
    
    # Test filtering methods
    for severity in ValidationSeverity:
        filtered_issues = report.get_issues_by_severity(severity)
        assert isinstance(filtered_issues, list)
    
    for val_type in ValidationType:
        filtered_issues = report.get_issues_by_type(val_type)
        assert isinstance(filtered_issues, list)
    
    print(f"  Report summary generated successfully")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    print(f"  Total validations: {summary['validation_counts']['total']}")
    
    print("✓ Validation report tests passed")

def test_custom_schema_creation():
    """Test creating and using custom schemas"""
    print("Testing custom schema creation...")
    
    # Create a custom schema for portfolio data
    portfolio_schema = DataSchema(
        name="Portfolio",
        description="Portfolio holdings validation schema"
    )
    
    portfolio_schema.add_field(FieldSchema(
        name="symbol",
        data_type=DataType.STRING,
        required=True,
        nullable=False,
        pattern=r'^[A-Z]{1,5}$'
    ))
    
    portfolio_schema.add_field(FieldSchema(
        name="weight",
        data_type=DataType.FLOAT,
        required=True,
        nullable=False,
        min_value=0.0,
        max_value=1.0
    ))
    
    portfolio_schema.add_field(FieldSchema(
        name="sector",
        data_type=DataType.STRING,
        required=True,
        allowed_values=["Technology", "Healthcare", "Finance", "Energy"]
    ))
    
    # Register and test the schema
    validator = DataValidator()
    validator.register_schema(portfolio_schema)
    
    # Create test data
    test_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'invalid_symbol', 'GOOGL'],
        'weight': [0.3, 0.25, 1.5, 0.2],  # One weight > 1.0
        'sector': ['Technology', 'Technology', 'InvalidSector', 'Technology']
    })
    
    report = validator.validate_data(test_data, schema_name="Portfolio")
    
    # Should find validation issues
    assert len(report.issues) > 0
    
    # Check specific issues
    pattern_issues = [issue for issue in report.issues if 'pattern' in issue.message.lower()]
    range_issues = [issue for issue in report.issues if 'maximum' in issue.message.lower()]
    allowed_values_issues = [issue for issue in report.issues if 'allowed' in issue.message.lower()]
    
    assert len(pattern_issues) > 0      # Invalid symbol format
    assert len(range_issues) > 0        # Weight > 1.0
    assert len(allowed_values_issues) > 0  # Invalid sector
    
    print(f"  Custom schema validation found {len(report.issues)} issues:")
    print(f"  - Pattern violations: {len(pattern_issues)}")
    print(f"  - Range violations: {len(range_issues)}")
    print(f"  - Allowed values violations: {len(allowed_values_issues)}")
    
    print("✓ Custom schema tests passed")

def test_validation_history():
    """Test validation history tracking"""
    print("Testing validation history tracking...")
    
    validator = DataValidator()
    
    # Perform multiple validations
    data1 = create_sample_data_with_validation_issues()
    data2 = create_fred_sample_data()
    
    report1 = validator.validate_data(data1, schema_name="YahooFinance")
    report2 = validator.validate_data(data2, schema_name="FRED")
    
    # Check history
    assert len(validator.validation_history) == 2
    assert validator.validation_history[0] == report1
    assert validator.validation_history[1] == report2
    
    # Test validation summary
    summary = validator.get_validation_summary()
    assert summary['total_validation_runs'] == 2
    assert summary['total_issues_found'] > 0
    assert 'severity_breakdown' in summary
    assert 'type_breakdown' in summary
    assert 'latest_validation' in summary
    
    print(f"  Tracked {len(validator.validation_history)} validation runs")
    print(f"  Total issues across all runs: {summary['total_issues_found']}")
    
    print("✓ Validation history tests passed")

def test_report_saving_and_loading():
    """Test saving validation reports to files"""
    print("Testing report saving functionality...")
    
    validator = DataValidator()
    data = create_sample_data_with_validation_issues()
    report = validator.validate_data(data, schema_name="YahooFinance")
    
    # Save report to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        validator.save_report(report, tmp_path)
        
        # Verify file was created and contains expected data
        assert os.path.exists(tmp_path)
        
        with open(tmp_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'summary' in saved_data
        assert 'issues' in saved_data
        assert saved_data['summary']['source_name'] == 'YahooFinance'
        assert len(saved_data['issues']) > 0
        
        print(f"  Successfully saved report to {tmp_path}")
        print(f"  Report contains {len(saved_data['issues'])} issues")
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    print("✓ Report saving tests passed")

def test_utility_schema_functions():
    """Test utility functions for creating common schemas"""
    print("Testing utility schema functions...")
    
    # Test financial data schema
    financial_schema = create_financial_data_schema()
    assert financial_schema.name == "FinancialData"
    assert 'date' in financial_schema.fields
    assert 'symbol' in financial_schema.fields
    assert 'volume' in financial_schema.fields
    
    # Test economic indicator schema
    econ_schema = create_economic_indicator_schema()
    assert econ_schema.name == "EconomicIndicator"
    assert 'date' in econ_schema.fields
    assert 'indicator_code' in econ_schema.fields
    assert 'value' in econ_schema.fields
    
    print("  ✓ Financial data schema created successfully")
    print("  ✓ Economic indicator schema created successfully")
    
    print("✓ Utility schema function tests passed")

def run_all_tests():
    """Run all validation tests"""
    print("Running Data Validation System Tests")
    print("=" * 50)
    
    try:
        test_basic_initialization()
        print()
        
        test_schema_registration()
        print()
        
        test_yahoo_finance_validation()
        print()
        
        test_fred_validation()
        print()
        
        test_validation_severity_levels()
        print()
        
        test_consistency_validation()
        print()
        
        test_basic_validation_fallback()
        print()
        
        test_validation_report_functionality()
        print()
        
        test_custom_schema_creation()
        print()
        
        test_validation_history()
        print()
        
        test_report_saving_and_loading()
        print()
        
        test_utility_schema_functions()
        print()
        
        print("=" * 50)
        print("🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("Data Validation System is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)