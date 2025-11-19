"""
Tracing utilities for logging and observability.

This module provides utilities for logging errors and tracing application behavior.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with optional context.
    
    Args:
        error: The exception to log
        context: Optional dictionary with additional context
    """
    error_msg = f"Error: {str(error)}"
    
    if context:
        error_msg += f" | Context: {context}"
    
    logger.error(error_msg, exc_info=True)


def log_info(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an info message with optional context.
    
    Args:
        message: The message to log
        context: Optional dictionary with additional context
    """
    if context:
        message += f" | Context: {context}"
    
    logger.info(message)


def log_warning(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a warning message with optional context.
    
    Args:
        message: The message to log
        context: Optional dictionary with additional context
    """
    if context:
        message += f" | Context: {context}"
    
    logger.warning(message)
