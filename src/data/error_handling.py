"""
Comprehensive Error Handling and Recovery Framework

This module provides a robust error handling system for the macro-economic data
pipeline. It includes custom exceptions, retry mechanisms, centralized logging,
alerting systems, and graceful degradation capabilities.

Key Features:
- Custom exception hierarchy for different error types
- Retry mechanisms with exponential backoff
- Centralized error logging with context
- Alerting system for critical failures
- Graceful degradation options
- Error recovery procedures
- Performance monitoring and metrics
"""

import logging
import time
import traceback
import functools
import threading
import queue
import smtplib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import asyncio
import warnings

# Email imports - handle Python 3.13 compatibility
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    # Fallback for older Python versions or different configurations
    try:
        from email.MIMEText import MIMEText
        from email.MIMEMultipart import MIMEMultipart
    except ImportError:
        # If email modules are not available, provide dummy classes
        class MIMEText:
            def __init__(self, *args, **kwargs): pass
        class MIMEMultipart:
            def __init__(self, *args, **kwargs): pass

# Configure logging
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors"""
    DATA_FETCH = "data_fetch"
    DATA_PROCESSING = "data_processing"
    DATA_VALIDATION = "data_validation"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Types of recovery actions"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    ALERT = "alert"
    CACHE = "cache"

# Custom Exception Hierarchy

class DataPipelineError(Exception):
    """Base exception for all data pipeline errors"""
    
    def __init__(self, message: str, error_code: str = None, 
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict = None, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = traceback.format_exc() if original_error else None
    
    def to_dict(self) -> Dict:
        """Convert error to dictionary for logging/serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None,
            'traceback': self.traceback_str
        }

class DataFetchError(DataPipelineError):
    """Errors related to data fetching"""
    
    def __init__(self, message: str, source: str = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA_FETCH)
        kwargs.setdefault('context', {}).update({'source': source} if source else {})
        super().__init__(message, **kwargs)

class NetworkError(DataFetchError):
    """Network-related errors"""
    
    def __init__(self, message: str, status_code: int = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('context', {}).update({'status_code': status_code} if status_code else {})
        super().__init__(message, **kwargs)

class AuthenticationError(DataFetchError):
    """Authentication-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

class RateLimitError(DataFetchError):
    """Rate limiting errors"""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RATE_LIMIT)
        kwargs.setdefault('context', {}).update({'retry_after': retry_after} if retry_after else {})
        super().__init__(message, **kwargs)

class DataProcessingError(DataPipelineError):
    """Errors related to data processing"""
    
    def __init__(self, message: str, step: str = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA_PROCESSING)
        kwargs.setdefault('context', {}).update({'processing_step': step} if step else {})
        super().__init__(message, **kwargs)

class DataValidationError(DataPipelineError):
    """Errors related to data validation"""
    
    def __init__(self, message: str, field: str = None, validation_type: str = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA_VALIDATION)
        context_update = {}
        if field:
            context_update['field'] = field
        if validation_type:
            context_update['validation_type'] = validation_type
        kwargs.setdefault('context', {}).update(context_update)
        super().__init__(message, **kwargs)

class ConfigurationError(DataPipelineError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('context', {}).update({'config_key': config_key} if config_key else {})
        super().__init__(message, **kwargs)

class SystemError(DataPipelineError):
    """System-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)

# Retry Mechanism

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: Tuple[Type[Exception], ...] = (DataFetchError, NetworkError, RateLimitError)
    stop_on_exceptions: Tuple[Type[Exception], ...] = (AuthenticationError, ConfigurationError)

class RetryManager:
    """Manages retry logic with exponential backoff"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.attempt_counts = {}
        self.last_errors = {}
    
    def calculate_delay(self, attempt: int, base_delay: float = None) -> float:
        """Calculate delay for retry attempt with exponential backoff"""
        base = base_delay or self.config.base_delay
        delay = min(
            base * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried"""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception type should stop retries
        if any(isinstance(exception, exc_type) for exc_type in self.config.stop_on_exceptions):
            return False
        
        # Check if exception type allows retries
        if any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions):
            return True
        
        # Default: don't retry unknown exceptions
        return False
    
    def retry_operation(self, operation: Callable, operation_id: str = None, 
                       context: Dict = None) -> Any:
        """Execute operation with retry logic"""
        op_id = operation_id or f"{operation.__name__}_{id(operation)}"
        attempt = 0
        last_exception = None
        
        while attempt < self.config.max_attempts:
            attempt += 1
            
            try:
                result = operation()
                # Reset counters on success
                if op_id in self.attempt_counts:
                    del self.attempt_counts[op_id]
                if op_id in self.last_errors:
                    del self.last_errors[op_id]
                return result
                
            except Exception as e:
                last_exception = e
                self.attempt_counts[op_id] = attempt
                self.last_errors[op_id] = e
                
                if not self.should_retry(e, attempt):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    
                    # Special handling for rate limit errors
                    if isinstance(e, RateLimitError) and hasattr(e, 'context') and e.context.get('retry_after'):
                        delay = max(delay, e.context['retry_after'])
                    
                    logger.warning(f"Attempt {attempt} failed for {op_id}: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {self.config.max_attempts} attempts failed for {op_id}")
        raise last_exception

def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, 
                      retry_on: Tuple[Type[Exception], ...] = None):
    """Decorator for adding retry functionality to functions"""
    retry_on = retry_on or (DataFetchError, NetworkError, RateLimitError)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                retry_on_exceptions=retry_on
            )
            retry_manager = RetryManager(config)
            return retry_manager.retry_operation(
                lambda: func(*args, **kwargs),
                operation_id=func.__name__
            )
        return wrapper
    return decorator

# Error Context Manager

@dataclass
class ErrorContext:
    """Context information for error tracking"""
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'component': self.component,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            **self.additional_context
        }

class ErrorContextManager:
    """Thread-local error context management"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_context(self, context: ErrorContext):
        """Set error context for current thread"""
        self._local.context = context
    
    def get_context(self) -> Optional[ErrorContext]:
        """Get current error context"""
        return getattr(self._local, 'context', None)
    
    def clear_context(self):
        """Clear current error context"""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def update_context(self, **kwargs):
        """Update current context with additional information"""
        current = self.get_context()
        if current:
            current.additional_context.update(kwargs)

# Global context manager instance
error_context = ErrorContextManager()

# Centralized Error Logger

class ErrorLogger:
    """Centralized error logging with context"""
    
    def __init__(self, logger_name: str = "data_pipeline_errors"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def log_error(self, error: DataPipelineError, additional_context: Dict = None):
        """Log error with full context"""
        error_dict = error.to_dict()
        
        # Add thread-local context
        context = error_context.get_context()
        if context:
            error_dict['context'].update(context.to_dict())
        
        # Add additional context
        if additional_context:
            error_dict['context'].update(additional_context)
        
        # Update error counts
        error_key = f"{error.category.value}:{error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Store recent errors
        self.recent_errors.append(error_dict)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
        
        # Log based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra={'error_data': error_dict})
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra={'error_data': error_dict})
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra={'error_data': error_dict})
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra={'error_data': error_dict})
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors_count': len(self.recent_errors),
            'most_common_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }

# Global error logger instance
error_logger = ErrorLogger()

# Alerting System

@dataclass
class AlertConfig:
    """Configuration for alerting system"""
    enabled: bool = True
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    alert_threshold: Dict[ErrorSeverity, int] = field(default_factory=lambda: {
        ErrorSeverity.CRITICAL: 1,
        ErrorSeverity.HIGH: 3,
        ErrorSeverity.MEDIUM: 10,
        ErrorSeverity.LOW: 50
    })
    cooldown_period: int = 300  # 5 minutes

class AlertManager:
    """Manages alerting for critical errors"""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alert_counts = {}
        self.last_alert_times = {}
        self.alert_queue = queue.Queue()
        
        if self.config.enabled:
            self._start_alert_processor()
    
    def _start_alert_processor(self):
        """Start background thread for processing alerts"""
        def process_alerts():
            while True:
                try:
                    alert_data = self.alert_queue.get(timeout=1)
                    if alert_data is None:  # Shutdown signal
                        break
                    self._send_alert(alert_data)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
        
        self.alert_thread = threading.Thread(target=process_alerts, daemon=True)
        self.alert_thread.start()
    
    def should_alert(self, error: DataPipelineError) -> bool:
        """Determine if an alert should be sent"""
        if not self.config.enabled:
            return False
        
        severity = error.severity
        threshold = self.config.alert_threshold.get(severity, float('inf'))
        
        # Check count threshold
        error_key = f"{error.category.value}:{error.error_code}"
        count = self.alert_counts.get(error_key, 0) + 1
        self.alert_counts[error_key] = count
        
        if count < threshold:
            return False
        
        # Check cooldown period
        now = time.time()
        last_alert = self.last_alert_times.get(error_key, 0)
        
        if now - last_alert < self.config.cooldown_period:
            return False
        
        self.last_alert_times[error_key] = now
        return True
    
    def send_alert(self, error: DataPipelineError, additional_context: Dict = None):
        """Queue an alert for sending"""
        if self.should_alert(error):
            alert_data = {
                'error': error.to_dict(),
                'additional_context': additional_context or {},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'alert_count': self.alert_counts.get(f"{error.category.value}:{error.error_code}", 0)
            }
            self.alert_queue.put(alert_data)
    
    def _send_alert(self, alert_data: Dict):
        """Send actual alert"""
        try:
            if self.config.email_enabled:
                self._send_email_alert(alert_data)
            
            # Could add other alert channels here (Slack, PagerDuty, etc.)
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert"""
        if not self.config.email_recipients:
            return
        
        error_data = alert_data['error']
        subject = f"Data Pipeline Alert - {error_data['severity'].upper()}: {error_data['error_code']}"
        
        body = f"""
        A {error_data['severity']} severity error has occurred in the data pipeline.
        
        Error Details:
        - Type: {error_data['error_type']}
        - Message: {error_data['message']}
        - Category: {error_data['category']}
        - Timestamp: {error_data['timestamp']}
        - Alert Count: {alert_data['alert_count']}
        
        Context:
        {json.dumps(error_data['context'], indent=2)}
        
        Additional Context:
        {json.dumps(alert_data['additional_context'], indent=2)}
        
        Please investigate and take appropriate action.
        """
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_host, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            logger.info(f"Alert email sent for {error_data['error_code']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

# Global alert manager instance
alert_manager = AlertManager()

# Graceful Degradation

class GracefulDegradationManager:
    """Manages graceful degradation when components fail"""
    
    def __init__(self):
        self.component_status = {}
        self.fallback_functions = {}
        self.circuit_breakers = {}
    
    def register_component(self, component_name: str, is_critical: bool = True):
        """Register a component for monitoring"""
        self.component_status[component_name] = {
            'is_critical': is_critical,
            'is_healthy': True,
            'error_count': 0,
            'last_error': None,
            'last_success': datetime.now(timezone.utc)
        }
    
    def register_fallback(self, component_name: str, fallback_function: Callable):
        """Register a fallback function for a component"""
        self.fallback_functions[component_name] = fallback_function
    
    def mark_component_error(self, component_name: str, error: Exception):
        """Mark a component as having an error"""
        if component_name not in self.component_status:
            self.register_component(component_name)
        
        status = self.component_status[component_name]
        status['error_count'] += 1
        status['last_error'] = error
        
        # Mark as unhealthy if too many errors
        if status['error_count'] >= 5:
            status['is_healthy'] = False
            logger.warning(f"Component {component_name} marked as unhealthy")
    
    def mark_component_success(self, component_name: str):
        """Mark a component as successful"""
        if component_name not in self.component_status:
            self.register_component(component_name)
        
        status = self.component_status[component_name]
        status['error_count'] = max(0, status['error_count'] - 1)
        status['last_success'] = datetime.now(timezone.utc)
        
        # Mark as healthy if errors reduced
        if status['error_count'] <= 2:
            status['is_healthy'] = True
    
    def get_component_status(self, component_name: str) -> bool:
        """Check if component is healthy"""
        return self.component_status.get(component_name, {}).get('is_healthy', True)
    
    def execute_with_fallback(self, component_name: str, primary_function: Callable, 
                            *args, **kwargs) -> Any:
        """Execute function with fallback if component is unhealthy"""
        if self.get_component_status(component_name):
            try:
                result = primary_function(*args, **kwargs)
                self.mark_component_success(component_name)
                return result
            except Exception as e:
                self.mark_component_error(component_name, e)
                
                # If component is now unhealthy and we have a fallback
                if (not self.get_component_status(component_name) and 
                    component_name in self.fallback_functions):
                    logger.warning(f"Using fallback for component {component_name}")
                    return self.fallback_functions[component_name](*args, **kwargs)
                
                # Re-raise if no fallback
                raise
        else:
            # Component already unhealthy, use fallback if available
            if component_name in self.fallback_functions:
                logger.info(f"Component {component_name} unhealthy, using fallback")
                return self.fallback_functions[component_name](*args, **kwargs)
            else:
                raise SystemError(f"Component {component_name} is unhealthy and no fallback available")

# Global degradation manager instance
degradation_manager = GracefulDegradationManager()

# Comprehensive Error Handler

class ErrorHandler:
    """Main error handling coordinator"""
    
    def __init__(self, 
                 retry_config: RetryConfig = None,
                 alert_config: AlertConfig = None):
        self.retry_manager = RetryManager(retry_config)
        self.alert_manager = AlertManager(alert_config)
        self.degradation_manager = GracefulDegradationManager()
        self.error_logger = error_logger
    
    def handle_error(self, error: Exception, context: ErrorContext = None, 
                    component_name: str = None) -> DataPipelineError:
        """Central error handling method"""
        
        # Convert to DataPipelineError if needed
        if isinstance(error, DataPipelineError):
            pipeline_error = error
        else:
            pipeline_error = DataPipelineError(
                message=str(error),
                original_error=error,
                context=context.to_dict() if context else {}
            )
        
        # Set context if provided
        if context:
            error_context.set_context(context)
        
        # Log the error
        self.error_logger.log_error(pipeline_error)
        
        # Mark component error if specified
        if component_name:
            self.degradation_manager.mark_component_error(component_name, error)
        
        # Send alert if necessary
        self.alert_manager.send_alert(pipeline_error)
        
        return pipeline_error
    
    def execute_with_error_handling(self, operation: Callable, 
                                  operation_context: ErrorContext,
                                  component_name: str = None,
                                  use_retry: bool = True) -> Any:
        """Execute operation with full error handling"""
        error_context.set_context(operation_context)
        
        try:
            if use_retry:
                return self.retry_manager.retry_operation(operation, operation_context.operation)
            else:
                return operation()
                
        except Exception as e:
            handled_error = self.handle_error(e, operation_context, component_name)
            raise handled_error
        finally:
            error_context.clear_context()

# Global error handler instance
error_handler = ErrorHandler()

# Utility Functions and Decorators

def with_error_handling(component_name: str = None, use_retry: bool = True, 
                       operation_name: str = None):
    """Decorator to add comprehensive error handling to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            context = ErrorContext(
                operation=op_name,
                component=component_name or func.__module__
            )
            
            return error_handler.execute_with_error_handling(
                lambda: func(*args, **kwargs),
                context,
                component_name,
                use_retry
            )
        return wrapper
    return decorator

def log_errors(func):
    """Simple decorator to log errors without handling them"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = ErrorContext(
                operation=func.__name__,
                component=func.__module__
            )
            error_handler.handle_error(e, context)
            raise
    return wrapper

# Error Recovery Documentation

ERROR_CODES = {
    'NetworkError': {
        'description': 'Network connectivity issues',
        'recovery_actions': [
            'Check internet connection',
            'Verify API endpoint availability',
            'Check firewall settings',
            'Wait and retry with exponential backoff'
        ]
    },
    'AuthenticationError': {
        'description': 'Authentication or authorization failures',
        'recovery_actions': [
            'Check API credentials',
            'Verify API key validity',
            'Check account permissions',
            'Regenerate API keys if needed'
        ]
    },
    'RateLimitError': {
        'description': 'API rate limit exceeded',
        'recovery_actions': [
            'Wait for rate limit reset',
            'Implement exponential backoff',
            'Consider upgrading API plan',
            'Optimize request frequency'
        ]
    },
    'DataValidationError': {
        'description': 'Data quality or format issues',
        'recovery_actions': [
            'Check data source for changes',
            'Update validation rules',
            'Implement data cleaning',
            'Use fallback data sources'
        ]
    },
    'DataProcessingError': {
        'description': 'Errors during data transformation',
        'recovery_actions': [
            'Check processing logic',
            'Verify input data format',
            'Implement graceful degradation',
            'Use cached data if available'
        ]
    },
    'ConfigurationError': {
        'description': 'Configuration or setup issues',
        'recovery_actions': [
            'Verify configuration files',
            'Check environment variables',
            'Validate configuration schema',
            'Reset to default configuration'
        ]
    }
}

def get_recovery_procedures(error_code: str) -> Dict:
    """Get recovery procedures for a specific error code"""
    return ERROR_CODES.get(error_code, {
        'description': 'Unknown error type',
        'recovery_actions': [
            'Check logs for more details',
            'Contact system administrator',
            'Restart affected components'
        ]
    })

if __name__ == "__main__":
    # Example usage and testing
    print("Error Handling and Recovery Framework")
    print("=" * 40)
    
    # Test custom exceptions
    try:
        raise DataFetchError("Test fetch error", source="FRED")
    except DataPipelineError as e:
        print(f"Caught error: {e.error_code}")
        print(f"Context: {e.context}")
    
    # Test retry mechanism
    @retry_with_backoff(max_attempts=3, base_delay=0.1)
    def failing_function():
        print("Attempting operation...")
        raise NetworkError("Simulated network failure")
    
    try:
        failing_function()
    except Exception as e:
        print(f"Final failure: {e}")
    
    # Test error handling decorator
    @with_error_handling(component_name="test_component")
    def test_function():
        return "Success!"
    
    result = test_function()
    print(f"Function result: {result}")
    
    print("\nError statistics:")
    stats = error_logger.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print(f"Error types: {list(stats['error_counts_by_type'].keys())}")