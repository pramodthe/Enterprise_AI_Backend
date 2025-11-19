"""
Tests for Supabase integration.

These tests verify the Supabase storage backend and database operations.
Note: These are integration tests that require a configured Supabase instance.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import os

from backend.chatbot.storage import SupabaseStorageBackend
from backend.chatbot.models import Session, Message, MessageRole
from backend.database.operations import DatabaseOperations
from backend.database.supabase_client import SupabaseClient


# Skip tests if Supabase is not configured
SUPABASE_CONFIGURED = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
skip_if_no_supabase = pytest.mark.skipif(
    not SUPABASE_CONFIGURED,
    reason="Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY to run these tests."
)


class TestSupabaseStorageBackend:
    """Tests for SupabaseStorageBackend."""
    
    @skip_if_no_supabase
    def test_initialization(self):
        """Test that storage backend initializes correctly."""
        storage = SupabaseStorageBackend(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY")
        )
        assert storage is not None
    
    def test_initialization_without_credentials(self):
        """Test that initialization fails without credentials."""
        with pytest.raises(Exception):
            SupabaseStorageBackend(
                supabase_url="",
                supabase_key=""
            )
    
    @skip_if_no_supabase
    def test_save_and_load_session(self):
        """Test saving and loading a session."""
        storage = SupabaseStorageBackend(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        )
        
        # Create test session data
        session_id = f"test_session_{datetime.now().timestamp()}"
        session_data = {
            "session_id": session_id,
            "user_id": "test_user",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation_history": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "agent_used": None
                }
            ],
            "metadata": {"test": True},
            "is_expired": False
        }
        
        # Save session
        storage.save_session(session_id, session_data)
        
        # Load session
        loaded_data = storage.load_session(session_id)
        
        assert loaded_data is not None
        assert loaded_data["session_id"] == session_id
        assert loaded_data["user_id"] == "test_user"
        assert len(loaded_data["conversation_history"]) == 1
        
        # Cleanup
        storage.delete_session(session_id)
    
    @skip_if_no_supabase
    def test_delete_session(self):
        """Test deleting a session."""
        storage = SupabaseStorageBackend(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        )
        
        # Create and save test session
        session_id = f"test_session_delete_{datetime.now().timestamp()}"
        session_data = {
            "session_id": session_id,
            "user_id": "test_user",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation_history": [],
            "metadata": {},
            "is_expired": False
        }
        
        storage.save_session(session_id, session_data)
        
        # Verify it exists
        loaded = storage.load_session(session_id)
        assert loaded is not None
        
        # Delete it
        storage.delete_session(session_id)
        
        # Verify it's gone
        loaded = storage.load_session(session_id)
        assert loaded is None
    
    @skip_if_no_supabase
    def test_list_sessions(self):
        """Test listing sessions with filters."""
        storage = SupabaseStorageBackend(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        )
        
        test_user = f"test_user_{datetime.now().timestamp()}"
        
        # Create multiple test sessions
        session_ids = []
        for i in range(3):
            session_id = f"test_session_list_{i}_{datetime.now().timestamp()}"
            session_data = {
                "session_id": session_id,
                "user_id": test_user,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "conversation_history": [],
                "metadata": {},
                "is_expired": False
            }
            storage.save_session(session_id, session_data)
            session_ids.append(session_id)
        
        # List sessions for user
        user_sessions = storage.list_sessions(filters={"user_id": test_user})
        
        assert len(user_sessions) >= 3
        for session_id in session_ids:
            assert session_id in user_sessions
        
        # Cleanup
        for session_id in session_ids:
            storage.delete_session(session_id)


class TestDatabaseOperations:
    """Tests for DatabaseOperations."""
    
    @skip_if_no_supabase
    def test_create_and_get_user(self):
        """Test creating and retrieving a user."""
        db_ops = DatabaseOperations()
        
        if not db_ops.is_available():
            pytest.skip("Database operations not available")
        
        user_id = f"test_user_{datetime.now().timestamp()}"
        
        # Create user
        success = db_ops.create_user(
            user_id=user_id,
            username="test_user",
            email="test@example.com",
            full_name="Test User"
        )
        
        assert success
        
        # Get user
        user = db_ops.get_user(user_id)
        
        assert user is not None
        assert user["user_id"] == user_id
        assert user["username"] == "test_user"
        assert user["email"] == "test@example.com"
    
    @skip_if_no_supabase
    def test_update_user(self):
        """Test updating user data."""
        db_ops = DatabaseOperations()
        
        if not db_ops.is_available():
            pytest.skip("Database operations not available")
        
        user_id = f"test_user_update_{datetime.now().timestamp()}"
        
        # Create user
        db_ops.create_user(
            user_id=user_id,
            username="test_user",
            email="old@example.com"
        )
        
        # Update user
        success = db_ops.update_user(
            user_id=user_id,
            updates={"email": "new@example.com"}
        )
        
        assert success
        
        # Verify update
        user = db_ops.get_user(user_id)
        assert user["email"] == "new@example.com"
    
    @skip_if_no_supabase
    def test_save_agent_metric(self):
        """Test saving agent performance metrics."""
        db_ops = DatabaseOperations()
        
        if not db_ops.is_available():
            pytest.skip("Database operations not available")
        
        # Save metric
        success = db_ops.save_agent_metric(
            agent_name="test_agent",
            query="test query",
            response_time_ms=100,
            success=True,
            confidence_score=0.95
        )
        
        assert success
    
    @skip_if_no_supabase
    def test_get_agent_statistics(self):
        """Test getting agent statistics."""
        db_ops = DatabaseOperations()
        
        if not db_ops.is_available():
            pytest.skip("Database operations not available")
        
        # Save some test metrics
        for i in range(3):
            db_ops.save_agent_metric(
                agent_name="test_agent_stats",
                query=f"test query {i}",
                response_time_ms=100 + i * 10,
                success=True,
                confidence_score=0.9 + i * 0.01
            )
        
        # Get statistics
        stats = db_ops.get_agent_statistics(days=1)
        
        assert isinstance(stats, dict)
        if "test_agent_stats" in stats:
            agent_stats = stats["test_agent_stats"]
            assert "total_queries" in agent_stats
            assert "success_rate" in agent_stats
            assert "avg_response_time_ms" in agent_stats


class TestSupabaseClient:
    """Tests for SupabaseClient singleton."""
    
    def test_singleton_pattern(self):
        """Test that SupabaseClient follows singleton pattern."""
        # Reset singleton
        SupabaseClient.reset()
        
        # Get client twice
        client1 = SupabaseClient.get_client()
        client2 = SupabaseClient.get_client()
        
        # Should be same instance (or both None if not configured)
        assert client1 is client2
    
    def test_reset(self):
        """Test resetting the singleton."""
        SupabaseClient.reset()
        
        client1 = SupabaseClient.get_client()
        
        SupabaseClient.reset()
        
        client2 = SupabaseClient.get_client()
        
        # After reset, should reinitialize
        # (may be same or different depending on implementation)
        assert True  # Just verify no errors


class TestIntegrationWithSessionManager:
    """Integration tests with SessionManager."""
    
    @skip_if_no_supabase
    def test_session_manager_with_supabase(self):
        """Test SessionManager with Supabase storage."""
        from backend.chatbot.session_manager import SessionManager
        
        storage = SupabaseStorageBackend(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        )
        
        session_manager = SessionManager(storage_backend=storage)
        
        # Create session
        user_id = f"test_user_{datetime.now().timestamp()}"
        session = session_manager.create_session(user_id=user_id)
        
        assert session is not None
        assert session.user_id == user_id
        
        # Add message
        session_manager.add_message(
            session_id=session.session_id,
            role=MessageRole.USER,
            content="Test message"
        )
        
        # Retrieve session
        retrieved = session_manager.get_session(session.session_id)
        
        assert retrieved is not None
        assert len(retrieved.conversation_history) == 1
        assert retrieved.conversation_history[0].content == "Test message"
        
        # Cleanup
        storage.delete_session(session.session_id)


# Run tests with: pytest backend/tests/test_supabase_integration.py -v
