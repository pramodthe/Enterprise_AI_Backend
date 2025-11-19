"""
Database module for Enterprise AI Assistant Platform.

This module provides Supabase integration for persistent storage of:
- User accounts and profiles
- Chat sessions and conversation history
- Agent performance metrics
- Document metadata

Quick Start:
    from backend.database.supabase_client import get_supabase_client
    from backend.database.operations import db_ops
    
    # Get client
    client = get_supabase_client()
    
    # Use high-level operations
    db_ops.create_user("user123", "john_doe", "john@example.com")
    stats = db_ops.get_agent_statistics(days=7)

For detailed setup instructions, see:
    backend/database/SUPABASE_SETUP.md
"""

from backend.database.supabase_client import get_supabase_client, SupabaseClient
from backend.database.operations import DatabaseOperations, db_ops

__all__ = [
    'get_supabase_client',
    'SupabaseClient',
    'DatabaseOperations',
    'db_ops'
]
