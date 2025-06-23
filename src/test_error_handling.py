"""
Test suite for the Error Handling and Recovery Framework

This test suite validates the comprehensive error handling framework
including custom exceptions, retry mechanisms, logging, alerting,
and graceful degradation capabilities.
"""

import time
import threading
import tempfile
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data.error_handling import (
    # Exception classes
    DataPipelineError, DataFetchError, NetworkError, AuthenticationError,
    RateLimitError, DataProcessingError, DataValidationError, 
    ConfigurationError, SystemError,
    
    # Enums
    ErrorSeverity, ErrorCategory, RecoveryAction,
    
    # Core classes
    RetryConfig, RetryManager, ErrorContext, ErrorContextManager,
    ErrorLogger, AlertConfig, AlertManager, GracefulDegradationManager,
    ErrorHandler,
    
    # Decorators and functions
    retry_with_backoff, with_error_handling, log_errors,
    get_recovery_procedures, ERROR_CODES,
    
    # Global instances
    error_context, error_logger, alert_manager, degradation_manager, error_handler
)

def test_custom_exceptions():
    """Test custom exception hierarchy and functionality"""
    print("Testing custom exception hierarchy...")
    
    # Test base DataPipelineError
    base_error = DataPipelineError(
        message="Test error",
        error_code="TEST_001",
        category=ErrorCategory.DATA_FETCH,
        severity=ErrorSeverity.HIGH,
        context={"test_key": "test_value"}
    )
    
    assert base_error.message == "Test error"
    assert base_error.error_code == "TEST_001"
    assert base_error.category == ErrorCategory.DATA_FETCH
    assert base_error.severity == ErrorSeverity.HIGH
    assert base_error.context["test_key"] == "test_value"
    assert base_error.timestamp is not None
    
    # Test error dictionary conversion
    error_dict = base_error.to_dict()
    assert error_dict['error_type'] == 'DataPipelineError'
    assert error_dict['message'] == "Test error"
    assert error_dict['category'] == 'data_fetch'
    assert error_dict['severity'] == 'high'
    
    # Test specialized exceptions
    fetch_error = DataFetchError("Fetch failed", source="FRED")
    assert fetch_error.category == ErrorCategory.DATA_FETCH
    assert fetch_error.context['source'] == "FRED"
    
    network_error = NetworkError("Connection failed", status_code=500)
    assert network_error.category == ErrorCategory.NETWORK
    assert network_error.context['status_code'] == 500
    
    auth_error = AuthenticationError("Invalid credentials")
    assert auth_error.category == ErrorCategory.AUTHENTICATION
    assert auth_error.severity == ErrorSeverity.HIGH
    
    rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=60)
    assert rate_limit_error.category == ErrorCategory.RATE_LIMIT
    assert rate_limit_error.context['retry_after'] == 60
    
    processing_error = DataProcessingError("Processing failed", step="normalization")
    assert processing_error.category == ErrorCategory.DATA_PROCESSING
    assert processing_error.context['processing_step'] == "normalization"
    
    validation_error = DataValidationError("Invalid data", field="price", validation_type="range")
    assert validation_error.category == ErrorCategory.DATA_VALIDATION
    assert validation_error.context['field'] == "price"
    assert validation_error.context['validation_type'] == "range"
    
    config_error = ConfigurationError("Missing config", config_key="api_key")
    assert config_error.category == ErrorCategory.CONFIGURATION
    assert config_error.severity == ErrorSeverity.HIGH
    assert config_error.context['config_key'] == "api_key"
    
    system_error = SystemError("System failure")
    assert system_error.category == ErrorCategory.SYSTEM
    assert system_error.severity == ErrorSeverity.CRITICAL
    
    print("✓ Custom exception tests passed")

def test_retry_mechanism():
    """Test retry logic with exponential backoff"""
    print("Testing retry mechanism...")
    
    # Test RetryConfig
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        exponential_base=2.0,
        jitter=False
    )
    
    retry_manager = RetryManager(config)
    
    # Test delay calculation
    delay1 = retry_manager.calculate_delay(1)
    delay2 = retry_manager.calculate_delay(2)
    delay3 = retry_manager.calculate_delay(3)
    
    assert delay1 == 0.1
    assert delay2 == 0.2
    assert delay3 == 0.4
    
    # Test should_retry logic
    network_error = NetworkError("Network failure")
    auth_error = AuthenticationError("Auth failure")
    
    assert retry_manager.should_retry(network_error, 1) == True
    assert retry_manager.should_retry(network_error, 3) == False  # Max attempts reached
    assert retry_manager.should_retry(auth_error, 1) == False    # Auth errors don't retry
    
    # Test successful retry operation
    attempt_count = [0]
    def sometimes_failing_operation():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise NetworkError("Temporary failure")
        return "Success!"
    
    result = retry_manager.retry_operation(sometimes_failing_operation, "test_op")
    assert result == "Success!"
    assert attempt_count[0] == 3
    
    # Test complete failure
    def always_failing_operation():
        raise NetworkError("Always fails")
    
    try:
        retry_manager.retry_operation(always_failing_operation, "fail_op")
        assert False, "Should have raised exception"
    except NetworkError:
        pass  # Expected
    
    print("✓ Retry mechanism tests passed")

def test_retry_decorator():
    """Test retry decorator functionality"""
    print("Testing retry decorator...")
    
    attempt_count = [0]
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def decorated_function():
        attempt_count[0] += 1
        if attempt_count[0] < 2:
            raise DataFetchError("Temporary failure")
        return "Decorated success!"
    
    result = decorated_function()
    assert result == "Decorated success!"
    assert attempt_count[0] == 2
    
    print("✓ Retry decorator tests passed")

def test_error_context():
    """Test error context management"""
    print("Testing error context management...")
    
    context_manager = ErrorContextManager()
    
    # Test setting and getting context
    test_context = ErrorContext(
        operation="test_operation",
        component="test_component",
        user_id="user123",
        session_id="session456",
        additional_context={"request_type": "data_fetch"}
    )
    
    context_manager.set_context(test_context)
    retrieved_context = context_manager.get_context()
    
    assert retrieved_context.operation == "test_operation"
    assert retrieved_context.component == "test_component"
    assert retrieved_context.user_id == "user123"
    assert retrieved_context.additional_context["request_type"] == "data_fetch"
    
    # Test context update
    context_manager.update_context(new_key="new_value")
    updated_context = context_manager.get_context()
    assert updated_context.additional_context["new_key"] == "new_value"
    
    # Test context clearing
    context_manager.clear_context()
    cleared_context = context_manager.get_context()
    assert cleared_context is None
    
    # Test context dictionary conversion
    context_dict = test_context.to_dict()
    assert context_dict['operation'] == "test_operation"
    assert context_dict['component'] == "test_component"
    assert context_dict['user_id'] == "user123"
    assert context_dict['request_type'] == "data_fetch"
    
    print("✓ Error context tests passed")

def test_error_logger():
    """Test centralized error logging"""
    print("Testing error logging...")
    
    logger = ErrorLogger("test_logger")
    
    # Create test errors
    error1 = DataFetchError("First error", source="FRED")
    error2 = NetworkError("Network error", status_code=500)
    error3 = DataFetchError("Second fetch error", source="Yahoo")
    
    # Log errors
    logger.log_error(error1, {"additional": "context1"})
    logger.log_error(error2)
    logger.log_error(error3)
    
    # Test statistics
    stats = logger.get_error_statistics()
    assert stats['total_errors'] == 3
    assert 'data_fetch:DataFetchError' in stats['error_counts_by_type']
    assert 'network:NetworkError' in stats['error_counts_by_type']
    assert stats['error_counts_by_type']['data_fetch:DataFetchError'] == 2
    assert stats['error_counts_by_type']['network:NetworkError'] == 1
    
    # Test recent errors
    assert len(logger.recent_errors) == 3
    assert logger.recent_errors[0]['message'] == "First error"
    assert logger.recent_errors[0]['context']['additional'] == "context1"
    
    print("✓ Error logging tests passed")

def test_alert_manager():
    """Test alerting system"""
    print("Testing alert manager...")
    
    # Test with disabled alerts
    config = AlertConfig(enabled=False)
    alert_mgr = AlertManager(config)
    
    error = DataPipelineError("Test error", severity=ErrorSeverity.CRITICAL)
    
    # Should not alert when disabled
    assert not alert_mgr.should_alert(error)
    
    # Test with enabled alerts
    config = AlertConfig(
        enabled=True,
        email_enabled=False,  # Don't actually send emails in tests
        alert_threshold={
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 2
        },
        cooldown_period=1  # Short cooldown for testing
    )
    
    alert_mgr = AlertManager(config)
    
    critical_error = DataPipelineError("Critical error", severity=ErrorSeverity.CRITICAL, error_code="CRIT_001")
    high_error = DataPipelineError("High error", severity=ErrorSeverity.HIGH, error_code="HIGH_001")
    
    # First critical error should trigger alert
    assert alert_mgr.should_alert(critical_error)
    
    # Same error immediately should not (cooldown)
    assert not alert_mgr.should_alert(critical_error)
    
    # First high error should not trigger (threshold = 2)
    assert not alert_mgr.should_alert(high_error)
    
    # Second high error should trigger
    assert alert_mgr.should_alert(high_error)
    
    # Test alert sending (without actual email)
    with patch.object(alert_mgr, '_send_email_alert') as mock_send:
        alert_mgr.send_alert(critical_error)
        # Small delay to allow background processing
        time.sleep(0.1)
    
    print("✓ Alert manager tests passed")

def test_graceful_degradation():
    """Test graceful degradation manager"""
    print("Testing graceful degradation...")
    
    degradation_mgr = GracefulDegradationManager()
    
    # Register components
    degradation_mgr.register_component("critical_service", is_critical=True)
    degradation_mgr.register_component("optional_service", is_critical=False)
    
    # Test initial healthy state
    assert degradation_mgr.get_component_status("critical_service") == True
    assert degradation_mgr.get_component_status("optional_service") == True
    
    # Test error accumulation
    for i in range(3):
        degradation_mgr.mark_component_error("critical_service", Exception("Error"))
    
    # Should still be healthy (threshold is 5)
    assert degradation_mgr.get_component_status("critical_service") == True
    
    # Add more errors to exceed threshold
    for i in range(3):
        degradation_mgr.mark_component_error("critical_service", Exception("Error"))
    
    # Should now be unhealthy
    assert degradation_mgr.get_component_status("critical_service") == False
    
    # Test recovery
    for i in range(4):
        degradation_mgr.mark_component_success("critical_service")
    
    # Should be healthy again
    assert degradation_mgr.get_component_status("critical_service") == True
    
    # Test fallback functionality
    def primary_function():
        return "primary_result"
    
    def fallback_function():
        return "fallback_result"
    
    degradation_mgr.register_fallback("test_component", fallback_function)
    
    # Should use primary when healthy
    result = degradation_mgr.execute_with_fallback("test_component", primary_function)
    assert result == "primary_result"
    
    # Make component unhealthy
    for i in range(6):
        degradation_mgr.mark_component_error("test_component", Exception("Error"))
    
    # Should use fallback when unhealthy
    result = degradation_mgr.execute_with_fallback("test_component", primary_function)
    assert result == "fallback_result"
    
    print("✓ Graceful degradation tests passed")

def test_comprehensive_error_handler():
    """Test the main ErrorHandler class"""
    print("Testing comprehensive error handler...")
    
    # Create handler with test configurations
    retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
    alert_config = AlertConfig(enabled=False)  # Disable alerts for testing
    
    handler = ErrorHandler(retry_config, alert_config)
    
    # Test error handling
    test_exception = ValueError("Test error")
    context = ErrorContext(operation="test_op", component="test_component")
    
    handled_error = handler.handle_error(test_exception, context, "test_component")
    
    assert isinstance(handled_error, DataPipelineError)
    assert handled_error.original_error == test_exception
    assert handled_error.message == "Test error"
    
    # Test operation execution with error handling
    attempt_count = [0]
    
    def test_operation():
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            raise DataFetchError("Temporary failure")
        return "Success after retry"
    
    result = handler.execute_with_error_handling(
        test_operation,
        context,
        "test_component",
        use_retry=True
    )
    
    assert result == "Success after retry"
    assert attempt_count[0] == 2
    
    print("✓ Comprehensive error handler tests passed")

def test_error_handling_decorators():
    """Test error handling decorators"""
    print("Testing error handling decorators...")
    
    # Test with_error_handling decorator
    @with_error_handling(component_name="test_component", use_retry=False)
    def decorated_function(should_fail=False):
        if should_fail:
            raise DataFetchError("Decorated function error")
        return "Decorated success"
    
    # Test successful execution
    result = decorated_function(should_fail=False)
    assert result == "Decorated success"
    
    # Test error handling
    try:
        decorated_function(should_fail=True)
        assert False, "Should have raised exception"
    except DataPipelineError:
        pass  # Expected
    
    # Test log_errors decorator
    error_logged = [False]
    
    @log_errors
    def logging_function(should_fail=False):
        if should_fail:
            raise ValueError("Logging test error")
        return "Logging success"
    
    # Mock the error handler to verify logging
    original_handle_error = error_handler.handle_error
    def mock_handle_error(*args, **kwargs):
        error_logged[0] = True
        return original_handle_error(*args, **kwargs)
    
    error_handler.handle_error = mock_handle_error
    
    try:
        logging_function(should_fail=True)
        assert False, "Should have raised exception"
    except ValueError:
        pass  # Expected
    
    assert error_logged[0], "Error should have been logged"
    
    # Restore original method
    error_handler.handle_error = original_handle_error
    
    print("✓ Error handling decorator tests passed")

def test_recovery_procedures():
    """Test error recovery documentation"""
    print("Testing recovery procedures...")
    
    # Test known error codes
    network_recovery = get_recovery_procedures("NetworkError")
    assert network_recovery['description'] == 'Network connectivity issues'
    assert len(network_recovery['recovery_actions']) > 0
    assert 'Check internet connection' in network_recovery['recovery_actions']
    
    auth_recovery = get_recovery_procedures("AuthenticationError")
    assert auth_recovery['description'] == 'Authentication or authorization failures'
    assert 'Check API credentials' in auth_recovery['recovery_actions']
    
    # Test unknown error code
    unknown_recovery = get_recovery_procedures("UnknownError")
    assert unknown_recovery['description'] == 'Unknown error type'
    assert 'Check logs for more details' in unknown_recovery['recovery_actions']
    
    # Test all defined error codes
    for error_code in ERROR_CODES:
        recovery = get_recovery_procedures(error_code)
        assert 'description' in recovery
        assert 'recovery_actions' in recovery
        assert len(recovery['recovery_actions']) > 0
    
    print("✓ Recovery procedures tests passed")

def test_thread_safety():
    """Test thread safety of error handling components"""
    print("Testing thread safety...")
    
    results = []
    errors = []
    
    def worker_function(worker_id):
        try:
            # Set different contexts in different threads
            context = ErrorContext(
                operation=f"worker_{worker_id}",
                component="thread_test",
                additional_context={"worker_id": worker_id}
            )
            
            error_context.set_context(context)
            
            # Create and log errors
            error = DataFetchError(f"Error from worker {worker_id}")
            error_logger.log_error(error)
            
            # Get context to verify thread isolation
            retrieved_context = error_context.get_context()
            results.append({
                'worker_id': worker_id,
                'context_operation': retrieved_context.operation,
                'context_worker_id': retrieved_context.additional_context['worker_id']
            })
            
        except Exception as e:
            errors.append(e)
        finally:
            error_context.clear_context()
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Unexpected errors: {errors}"
    assert len(results) == 5
    
    # Verify thread isolation
    for result in results:
        assert result['context_operation'] == f"worker_{result['worker_id']}"
        assert result['context_worker_id'] == result['worker_id']
    
    print("✓ Thread safety tests passed")

def test_performance_monitoring():
    """Test performance and monitoring capabilities"""
    print("Testing performance monitoring...")
    
    logger = ErrorLogger("performance_test")
    
    # Generate many errors to test performance
    start_time = time.time()
    
    for i in range(100):
        error = DataFetchError(f"Performance test error {i}", source="test")
        logger.log_error(error)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should handle 100 errors quickly (under 1 second)
    assert duration < 1.0, f"Error logging too slow: {duration}s"
    
    # Test statistics
    stats = logger.get_error_statistics()
    assert stats['total_errors'] == 100
    assert len(logger.recent_errors) == 100
    
    # Test memory management (recent errors cap)
    logger.max_recent_errors = 50
    
    for i in range(25):
        error = DataFetchError(f"Additional error {i}")
        logger.log_error(error)
    
    # Should cap at max_recent_errors
    assert len(logger.recent_errors) == 50
    
    print(f"✓ Performance monitoring tests passed (100 errors in {duration:.3f}s)")

def run_all_tests():
    """Run all error handling tests"""
    print("Running Error Handling and Recovery Framework Tests")
    print("=" * 60)
    
    try:
        test_custom_exceptions()
        print()
        
        test_retry_mechanism()
        print()
        
        test_retry_decorator()
        print()
        
        test_error_context()
        print()
        
        test_error_logger()
        print()
        
        test_alert_manager()
        print()
        
        test_graceful_degradation()
        print()
        
        test_comprehensive_error_handler()
        print()
        
        test_error_handling_decorators()
        print()
        
        test_recovery_procedures()
        print()
        
        test_thread_safety()
        print()
        
        test_performance_monitoring()
        print()
        
        print("=" * 60)
        print("🎉 ALL ERROR HANDLING TESTS PASSED! 🎉")
        print("Error Handling and Recovery Framework is working correctly.")
        
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