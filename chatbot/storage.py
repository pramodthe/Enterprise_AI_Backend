"""
Storage backend implementations for session persistence.

This module provides abstract and concrete implementations for storing
and retrieving session data, including in-memory and Redis backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime, timedelta

from backend.chatbot.models import Session

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """
    Abstract base class for session storage backends.
    
    Defines the interface that all storage implementations must follow.
    """
    
    @abstractmethod
    def save_session(self, session_id: str, session_data: Dict) -> None:
        """
        Save session data to storage.
        
        Args:
            session_id: Unique identifier for the session
            session_data: Dictionary containing session data
        """
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load session data from storage.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary containing session data, or None if not found
        """
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from storage.
        
        Args:
            session_id: Unique identifier for the session
        """
        pass
    
    @abstractmethod
    def list_sessions(self, filters: Optional[Dict] = None) -> List[str]:
        """
        List session IDs matching optional filters.
        
        Args:
            filters: Optional dictionary of filter criteria
            
        Returns:
            List of session IDs
        """
        pass


class InMemoryStorageBackend(StorageBackend):
    """
    In-memory storage backend for development and testing.
    
    Stores sessions in a dictionary. Data is lost when the process terminates.
    """
    
    def __init__(self):
        """Initialize the in-memory storage."""
        self._sessions: Dict[str, Dict] = {}
        logger.info("Initialized InMemoryStorageBackend")
    
    def save_session(self, session_id: str, session_data: Dict) -> None:
        """
        Save session data to memory.
        
        Args:
            session_id: Unique identifier for the session
            session_data: Dictionary containing session data
        """
        self._sessions[session_id] = session_data
        logger.debug(f"Saved session {session_id} to memory")
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load session data from memory.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary containing session data, or None if not found
        """
        session_data = self._sessions.get(session_id)
        if session_data:
            logger.debug(f"Loaded session {session_id} from memory")
        else:
            logger.debug(f"Session {session_id} not found in memory")
        return session_data
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from memory.
        
        Args:
            session_id: Unique identifier for the session
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session {session_id} from memory")
        else:
            logger.debug(f"Session {session_id} not found for deletion")
    
    def list_sessions(self, filters: Optional[Dict] = None) -> List[str]:
        """
        List session IDs, optionally filtered.
        
        Args:
            filters: Optional dictionary of filter criteria
                    Supported filters:
                    - 'user_id': Filter by user ID
                    - 'is_expired': Filter by expiration status
            
        Returns:
            List of session IDs matching the filters
        """
        session_ids = list(self._sessions.keys())
        
        if filters:
            filtered_ids = []
            for session_id in session_ids:
                session_data = self._sessions[session_id]
                
                # Apply user_id filter
                if 'user_id' in filters:
                    if session_data.get('user_id') != filters['user_id']:
                        continue
                
                # Apply is_expired filter
                if 'is_expired' in filters:
                    if session_data.get('is_expired') != filters['is_expired']:
                        continue
                
                filtered_ids.append(session_id)
            
            return filtered_ids
        
        return session_ids


class RedisStorageBackend(StorageBackend):
    """
    Redis storage backend for production use.
    
    Stores sessions in Redis with automatic expiration support.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 2592000):  # 30 days in seconds
        """
        Initialize the Redis storage backend.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live for sessions in seconds (default: 30 days)
        """
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._default_ttl = default_ttl
            # Test connection
            self._redis.ping()
            logger.info(f"Initialized RedisStorageBackend with URL: {redis_url}")
        except ImportError:
            raise ImportError(
                "Redis package is required for RedisStorageBackend. "
                "Install it with: pip install redis"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"session:{session_id}"
    
    def save_session(self, session_id: str, session_data: Dict) -> None:
        """
        Save session data to Redis.
        
        Args:
            session_id: Unique identifier for the session
            session_data: Dictionary containing session data
        """
        key = self._get_key(session_id)
        serialized_data = json.dumps(session_data)
        
        # Save with TTL
        self._redis.setex(key, self._default_ttl, serialized_data)
        logger.debug(f"Saved session {session_id} to Redis with TTL {self._default_ttl}s")
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load session data from Redis.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary containing session data, or None if not found
        """
        key = self._get_key(session_id)
        serialized_data = self._redis.get(key)
        
        if serialized_data:
            logger.debug(f"Loaded session {session_id} from Redis")
            return json.loads(serialized_data)
        else:
            logger.debug(f"Session {session_id} not found in Redis")
            return None
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from Redis.
        
        Args:
            session_id: Unique identifier for the session
        """
        key = self._get_key(session_id)
        deleted = self._redis.delete(key)
        if deleted:
            logger.debug(f"Deleted session {session_id} from Redis")
        else:
            logger.debug(f"Session {session_id} not found for deletion")
    
    def list_sessions(self, filters: Optional[Dict] = None) -> List[str]:
        """
        List session IDs from Redis, optionally filtered.
        
        Args:
            filters: Optional dictionary of filter criteria
                    Supported filters:
                    - 'user_id': Filter by user ID
                    - 'is_expired': Filter by expiration status
            
        Returns:
            List of session IDs matching the filters
        """
        # Get all session keys
        pattern = "session:*"
        keys = self._redis.keys(pattern)
        
        # Extract session IDs from keys
        session_ids = [key.replace("session:", "") for key in keys]
        
        # Apply filters if provided
        if filters:
            filtered_ids = []
            for session_id in session_ids:
                session_data = self.load_session(session_id)
                if not session_data:
                    continue
                
                # Apply user_id filter
                if 'user_id' in filters:
                    if session_data.get('user_id') != filters['user_id']:
                        continue
                
                # Apply is_expired filter
                if 'is_expired' in filters:
                    if session_data.get('is_expired') != filters['is_expired']:
                        continue
                
                filtered_ids.append(session_id)
            
            return filtered_ids
        
        return session_ids


class SupabaseStorageBackend(StorageBackend):
    """
    Supabase storage backend for production use.
    
    Stores sessions in Supabase PostgreSQL with full relational capabilities.
    Supports users, sessions, and messages with proper relationships.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the Supabase storage backend.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (anon or service role)
        """
        try:
            from supabase import create_client, Client
            self._client: Client = create_client(supabase_url, supabase_key)
            logger.info(f"Initialized SupabaseStorageBackend with URL: {supabase_url}")
        except ImportError:
            raise ImportError(
                "Supabase package is required for SupabaseStorageBackend. "
                "Install it with: pip install supabase"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    def save_session(self, session_id: str, session_data: Dict) -> None:
        """
        Save session data to Supabase.
        
        Args:
            session_id: Unique identifier for the session
            session_data: Dictionary containing session data
        """
        try:
            # Prepare session record
            session_record = {
                "session_id": session_id,
                "user_id": session_data.get("user_id"),
                "created_at": session_data.get("created_at"),
                "updated_at": session_data.get("updated_at"),
                "metadata": session_data.get("metadata", {}),
                "is_expired": session_data.get("is_expired", False)
            }
            
            # Upsert session (insert or update)
            self._client.table("sessions").upsert(session_record).execute()
            
            # Save conversation history as separate message records
            conversation_history = session_data.get("conversation_history", [])
            if conversation_history:
                # Delete existing messages for this session
                self._client.table("messages").delete().eq("session_id", session_id).execute()
                
                # Insert new messages
                message_records = []
                for idx, msg in enumerate(conversation_history):
                    message_records.append({
                        "session_id": session_id,
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "timestamp": msg.get("timestamp"),
                        "metadata": msg.get("metadata"),
                        "agent_used": msg.get("agent_used"),
                        "sequence_number": idx
                    })
                
                if message_records:
                    self._client.table("messages").insert(message_records).execute()
            
            logger.debug(f"Saved session {session_id} to Supabase")
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load session data from Supabase.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary containing session data, or None if not found
        """
        try:
            # Load session record
            response = self._client.table("sessions").select("*").eq("session_id", session_id).execute()
            
            if not response.data:
                logger.debug(f"Session {session_id} not found in Supabase")
                return None
            
            session_record = response.data[0]
            
            # Load conversation history
            messages_response = self._client.table("messages").select("*").eq(
                "session_id", session_id
            ).order("sequence_number").execute()
            
            conversation_history = []
            for msg in messages_response.data:
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "metadata": msg.get("metadata"),
                    "agent_used": msg.get("agent_used")
                })
            
            # Reconstruct session data
            session_data = {
                "session_id": session_record["session_id"],
                "user_id": session_record.get("user_id"),
                "created_at": session_record["created_at"],
                "updated_at": session_record["updated_at"],
                "conversation_history": conversation_history,
                "metadata": session_record.get("metadata", {}),
                "is_expired": session_record.get("is_expired", False)
            }
            
            logger.debug(f"Loaded session {session_id} from Supabase")
            return session_data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from Supabase.
        
        Args:
            session_id: Unique identifier for the session
        """
        try:
            # Delete messages first (foreign key constraint)
            self._client.table("messages").delete().eq("session_id", session_id).execute()
            
            # Delete session
            self._client.table("sessions").delete().eq("session_id", session_id).execute()
            
            logger.debug(f"Deleted session {session_id} from Supabase")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise
    
    def list_sessions(self, filters: Optional[Dict] = None) -> List[str]:
        """
        List session IDs from Supabase, optionally filtered.
        
        Args:
            filters: Optional dictionary of filter criteria
                    Supported filters:
                    - 'user_id': Filter by user ID
                    - 'is_expired': Filter by expiration status
            
        Returns:
            List of session IDs matching the filters
        """
        try:
            query = self._client.table("sessions").select("session_id")
            
            # Apply filters
            if filters:
                if 'user_id' in filters:
                    query = query.eq("user_id", filters['user_id'])
                if 'is_expired' in filters:
                    query = query.eq("is_expired", filters['is_expired'])
            
            response = query.execute()
            session_ids = [record["session_id"] for record in response.data]
            
            return session_ids
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def save_user(self, user_id: str, user_data: Dict) -> None:
        """
        Save user data to Supabase.
        
        Args:
            user_id: Unique identifier for the user
            user_data: Dictionary containing user data
        """
        try:
            user_record = {
                "user_id": user_id,
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "full_name": user_data.get("full_name"),
                "disabled": user_data.get("disabled", False),
                "metadata": user_data.get("metadata", {})
            }
            
            self._client.table("users").upsert(user_record).execute()
            logger.debug(f"Saved user {user_id} to Supabase")
        except Exception as e:
            logger.error(f"Failed to save user {user_id}: {e}")
            raise
    
    def load_user(self, user_id: str) -> Optional[Dict]:
        """
        Load user data from Supabase.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing user data, or None if not found
        """
        try:
            response = self._client.table("users").select("*").eq("user_id", user_id).execute()
            
            if not response.data:
                logger.debug(f"User {user_id} not found in Supabase")
                return None
            
            logger.debug(f"Loaded user {user_id} from Supabase")
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to load user {user_id}: {e}")
            return None
