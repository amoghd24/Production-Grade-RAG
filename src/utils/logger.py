"""
Logging utilities for the Second Brain AI Assistant.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config.settings import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (will be saved in logs directory)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper())
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure logs directory exists
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        file_path = settings.LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file name
    
    Returns:
        Logger instance
    """
    return setup_logger(name, log_file=log_file)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"{self.__class__.__module__}.{class_name}")
        return self._logger


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def create_session_log_file() -> str:
    """Create a unique log file name for the current session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}.log"


# Default application logger
app_logger = get_logger("second_brain", "app.log")


if __name__ == "__main__":
    # Test logging setup
    test_logger = get_logger(__name__, "test.log")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print("✅ Logging test completed. Check logs directory.") 