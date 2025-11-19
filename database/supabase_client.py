"""
Supabase client singleton for the application.

This module provides a centralized Supabase client instance
that can be used throughout the application.
"""
from typing import Optional
from supabase import create_client, Client
import logging

from backend.core.config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Singleton Supabase client manager."""
    
    _instance: Optional[Client] = None
    _initialized: bool = False
    
    @classmethod
    def get_client(cls) -> Optional[Client]:
        """
        Get or create the Supabase client instance.
        
        Returns:
            Supabase client instance, or None if credentials are not configured
        """
        if cls._initialized:
            return cls._instance
        
        # Check if Supabase is configured
        if not settings.supabase_url or not settings.supabase_key:
            logger.warning(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
            )
            cls._initialized = True
            return None
        
        try:
            cls._instance = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
            cls._initialized = True
            logger.info("Supabase client initialized successfully")
            return cls._instance
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            cls._initialized = True
            return None
    
    @classmethod
    def reset(cls):
        """Reset the client instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


def get_supabase_client() -> Optional[Client]:
    """
    Convenience function to get the Supabase client.
    
    Returns:
        Supabase client instance, or None if not configured
    """
    return SupabaseClient.get_client()
