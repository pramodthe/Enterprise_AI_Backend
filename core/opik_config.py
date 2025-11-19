"""
Opik configuration for tracing and observability.

This module provides configuration for Opik tracing integration.
"""
import os
from typing import Dict, Optional
import uuid


def is_tracing_enabled() -> bool:
    """
    Check if tracing is enabled.
    
    Returns:
        True if tracing is enabled, False otherwise
    """
    return os.getenv("ENABLE_TRACING", "false").lower() == "true"


def get_opik_metadata() -> Dict[str, str]:
    """
    Get Opik metadata for tracing.
    
    Returns:
        Dictionary containing metadata for Opik tracing
    """
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "service": "enterprise-ai-assistant",
        "version": "1.0.0"
    }


def get_session_id() -> str:
    """
    Generate or retrieve a session ID for tracing.
    
    Returns:
        Session ID string
    """
    return str(uuid.uuid4())
