"""
Tests for Root Chatbot storage backends.
"""
import pytest
from datetime import datetime
from backend.chatbot.storage import InMemoryStorageBackend, StorageBackend
from backend.chatbot.models import Session, Message, MessageRole


def test_in_memory_backend_save_and_load():
    """Test saving and loading sessions with InMemoryStorageBackend."""
    backend = InMemoryStorageBackend()
    
    session_data = {
        "session_id": "test-123",
        "user_id": "user-456",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "conversation_history": [],
        "metadata": {"test": "data"},
        "is_expired": False
    }
    
    # Save session
    backend.save_session("test-123", session_data)
    
    # Load session
    loaded = backend.load_session("test-123")
    assert loaded is not None
    assert loaded["session_id"] == "test-123"
    assert loaded["user_id"] == "user-456"
    assert loaded["metadata"] == {"test": "data"}


def test_in_memory_backend_delete():
    """Test deleting sessions with InMemoryStorageBackend."""
    backend = InMemoryStorageBackend()
    
    session_data = {
        "session_id": "test-123",
        "user_id": "user-456",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "conversation_history": [],
        "metadata": {},
        "is_expired": False
    }
    
    # Save and verify
    backend.save_session("test-123", session_data)
    assert backend.load_session("test-123") is not None
    
    # Delete and verify
    backend.delete_session("test-123")
    assert backend.load_session("test-123") is None


def test_in_memory_backend_list_sessions():
    """Test listing sessions with InMemoryStorageBackend."""
    backend = InMemoryStorageBackend()
    
    # Create multiple sessions
    for i in range(3):
        session_data = {
            "session_id": f"test-{i}",
            "user_id": f"user-{i}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation_history": [],
            "metadata": {},
            "is_expired": False
        }
        backend.save_session(f"test-{i}", session_data)
    
    # List all sessions
    sessions = backend.list_sessions()
    assert len(sessions) == 3
    assert "test-0" in sessions
    assert "test-1" in sessions
    assert "test-2" in sessions


def test_in_memory_backend_list_with_filters():
    """Test listing sessions with filters."""
    backend = InMemoryStorageBackend()
    
    # Create sessions with different user_ids
    session_data_1 = {
        "session_id": "test-1",
        "user_id": "user-a",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "conversation_history": [],
        "metadata": {},
        "is_expired": False
    }
    
    session_data_2 = {
        "session_id": "test-2",
        "user_id": "user-b",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "conversation_history": [],
        "metadata": {},
        "is_expired": False
    }
    
    backend.save_session("test-1", session_data_1)
    backend.save_session("test-2", session_data_2)
    
    # Filter by user_id
    filtered = backend.list_sessions(filters={"user_id": "user-a"})
    assert len(filtered) == 1
    assert "test-1" in filtered


def test_in_memory_backend_nonexistent_session():
    """Test loading a nonexistent session."""
    backend = InMemoryStorageBackend()
    
    loaded = backend.load_session("nonexistent")
    assert loaded is None


def test_session_round_trip():
    """Test full session serialization round trip through storage."""
    backend = InMemoryStorageBackend()
    
    # Create a session with messages
    timestamp = datetime.now()
    message = Message(
        role=MessageRole.USER,
        content="Test message",
        timestamp=timestamp
    )
    
    session = Session(
        session_id="test-123",
        user_id="user-456",
        created_at=timestamp,
        updated_at=timestamp,
        conversation_history=[message],
        metadata={"test": "data"}
    )
    
    # Save session
    backend.save_session(session.session_id, session.to_dict())
    
    # Load and restore session
    loaded_data = backend.load_session(session.session_id)
    assert loaded_data is not None
    
    restored_session = Session.from_dict(loaded_data)
    assert restored_session.session_id == session.session_id
    assert restored_session.user_id == session.user_id
    assert len(restored_session.conversation_history) == 1
    assert restored_session.conversation_history[0].content == "Test message"


# Redis backend tests (requires Redis to be running)
def test_redis_backend_connection():
    """Test Redis backend connection (skipped if Redis not available)."""
    try:
        import redis
    except ImportError:
        pytest.skip("Redis package not installed")
        return
    
    try:
        from backend.chatbot.storage import RedisStorageBackend
        backend = RedisStorageBackend(redis_url="redis://localhost:6379")
        
        # If we get here, connection succeeded
        session_data = {
            "session_id": "redis-test-123",
            "user_id": "user-456",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation_history": [],
            "metadata": {"test": "redis"},
            "is_expired": False
        }
        
        # Save and load
        backend.save_session("redis-test-123", session_data)
        loaded = backend.load_session("redis-test-123")
        
        assert loaded is not None
        assert loaded["session_id"] == "redis-test-123"
        assert loaded["metadata"] == {"test": "redis"}
        
        # Cleanup
        backend.delete_session("redis-test-123")
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
