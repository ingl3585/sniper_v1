"""
Advanced logging configuration for the trading system.
Provides centralized, structured, and performance-aware logging.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TradingContext:
    """Trading-specific context for structured logging."""
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    order_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None


class TradingFormatter(logging.Formatter):
    """Custom formatter with trading context and performance metrics."""
    
    def __init__(self, include_context: bool = True, structured: bool = False):
        self.include_context = include_context
        self.structured = structured
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp in ISO format
        record.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Extract trading context if available
        trading_context = getattr(record, 'trading_context', None)
        
        if self.structured:
            return self._format_structured(record, trading_context)
        else:
            return self._format_human_readable(record, trading_context)
    
    def _format_structured(self, record: logging.LogRecord, context: Optional[TradingContext]) -> str:
        """Format as structured JSON for machine processing."""
        log_data = {
            'timestamp': record.timestamp,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add trading context
        if context and self.include_context:
            log_data['trading_context'] = asdict(context)
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_data['performance'] = {'duration_ms': record.duration_ms}
        
        return json.dumps(log_data, default=str)
    
    def _format_human_readable(self, record: logging.LogRecord, context: Optional[TradingContext]) -> str:
        """Format for human-readable console output."""
        # Use simple timestamp format
        timestamp = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00')).strftime('%H:%M:%S')
        
        # Clean message of problematic unicode characters
        message = record.getMessage().encode('ascii', 'replace').decode('ascii')
        
        # Simple format: timestamp level message
        base_format = f"[{timestamp}] {record.levelname}: {message}"
        
        # Add trading context only for important events
        if context and self.include_context:
            # Only show symbol and strategy for trading-specific logs
            important_context = {}
            if context.symbol:
                important_context['symbol'] = context.symbol
            if context.strategy and context.strategy != 'None':
                important_context['strategy'] = context.strategy
            
            if important_context:
                context_str = ', '.join([f"{k}={v}" for k, v in important_context.items()])
                base_format += f" ({context_str})"
        
        # Add performance info
        if hasattr(record, 'duration_ms'):
            base_format += f" [{record.duration_ms:.1f}ms]"
        
        # Add exception info
        if record.exc_info:
            base_format += f"\n{self.formatException(record.exc_info)}"
        
        return base_format


class TradingLogger:
    """Enhanced logger with trading-specific functionality."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context = TradingContext()
    
    def set_context(self, **kwargs):
        """Set trading context for all subsequent log messages."""
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
    
    def clear_context(self):
        """Clear all trading context."""
        self._context = TradingContext()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with trading context."""
        extra = kwargs.get('extra', {})
        extra['trading_context'] = self._context
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, **context):
        """Log performance metrics."""
        extra = {'duration_ms': duration_ms, 'trading_context': self._context}
        if context:
            temp_context = TradingContext(**{**asdict(self._context), **context})
            extra['trading_context'] = temp_context
        
        self.logger.info(f"PERF: {operation} completed", extra=extra)
    
    def trade_signal(self, signal_type: str, symbol: str, **details):
        """Log trading signals with context."""
        temp_context = TradingContext(**{**asdict(self._context), 'symbol': symbol, 'component': 'signal'})
        extra = {'trading_context': temp_context}
        
        details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        self.info(f"SIGNAL: {signal_type} for {symbol} | {details_str}", extra=extra)
    
    def trade_execution(self, action: str, symbol: str, quantity: float, price: float, order_id: str = None):
        """Log trade execution events."""
        temp_context = TradingContext(
            **{**asdict(self._context), 'symbol': symbol, 'order_id': order_id, 'component': 'execution'}
        )
        extra = {'trading_context': temp_context}
        
        self.info(f"TRADE: {action} {quantity} {symbol} @ ${price:.4f}", extra=extra)
    
    def risk_event(self, event_type: str, symbol: str, **details):
        """Log risk management events."""
        temp_context = TradingContext(**{**asdict(self._context), 'symbol': symbol, 'component': 'risk'})
        extra = {'trading_context': temp_context}
        
        details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        self.warning(f"RISK: {event_type} for {symbol} | {details_str}", extra=extra)


class LoggingConfig:
    """Centralized logging configuration for the trading system."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 10,
                 structured_logging: bool = False,
                 console_logging: bool = True):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.structured_logging = structured_logging
        self.console_logging = console_logging
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging system."""
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root level
        root_logger.setLevel(self.log_level)
        
        # Create formatters
        console_formatter = TradingFormatter(structured=False)
        file_formatter = TradingFormatter(structured=self.structured_logging)
        
        # Console handler
        if self.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Main application log file
        main_file_handler = self._create_rotating_handler(
            self.log_dir / "trading_system.log", file_formatter
        )
        root_logger.addHandler(main_file_handler)
        
        # Separate log files for different components
        self._setup_component_loggers(file_formatter)
    
    def _create_rotating_handler(self, filename: Path, formatter: logging.Formatter) -> logging.Handler:
        """Create a rotating file handler."""
        handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=self.max_file_size, backupCount=self.backup_count
        )
        handler.setLevel(self.log_level)
        handler.setFormatter(formatter)
        return handler
    
    def _setup_component_loggers(self, formatter: logging.Formatter):
        """Setup separate loggers for different components."""
        components = {
            'trades': ['execution', 'risk', 'signal'],
            'strategies': ['strategies'],
            'data': ['data', 'providers', 'parsers'],
            'system': ['core', 'services', 'connection'],
            'performance': ['performance', 'PERF']
        }
        
        for log_file, prefixes in components.items():
            handler = self._create_rotating_handler(
                self.log_dir / f"{log_file}.log", formatter
            )
            
            for prefix in prefixes:
                logger = logging.getLogger(prefix)
                logger.addHandler(handler)
    
    @staticmethod
    def get_logger(name: str) -> TradingLogger:
        """Get a trading logger instance."""
        return TradingLogger(name)


def setup_logging(config_dict: Optional[Dict[str, Any]] = None) -> LoggingConfig:
    """
    Setup logging configuration from environment variables or config dict.
    
    Environment variables:
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_DIR: Directory for log files
        LOG_STRUCTURED: Enable structured JSON logging (true/false)
        LOG_CONSOLE: Enable console logging (true/false)
    """
    if config_dict is None:
        config_dict = {}
    
    # Get configuration from environment with defaults
    log_config = {
        'log_level': os.getenv('LOG_LEVEL', config_dict.get('log_level', 'INFO')),
        'log_dir': os.getenv('LOG_DIR', config_dict.get('log_dir', 'logs')),
        'structured_logging': os.getenv('LOG_STRUCTURED', 
                                      str(config_dict.get('structured_logging', False))).lower() == 'true',
        'console_logging': os.getenv('LOG_CONSOLE', 
                                   str(config_dict.get('console_logging', True))).lower() == 'true',
    }
    
    return LoggingConfig(**log_config)


# Convenience function for getting loggers
def get_logger(name: str) -> TradingLogger:
    """Get a trading logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        TradingLogger instance with set_context support
    """
    # Ensure basic logging is configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Return TradingLogger which has set_context method
    return TradingLogger(name)