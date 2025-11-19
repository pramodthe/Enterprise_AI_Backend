"""
Tests for SessionManager component.
"""
import pytest
from datetime import datetime, timedelta
from backend.chatbot.session_manager import SessionManager
from backend.chatbot.storage import InMemoryStorageBackend
from backend.chatbot.models import Session, Message, MessageRole


@pytest.fixture
def storage_backend():
    """Provide an in-memory storage backend for testing."""
    return InMemoryStorageBackend()


@pytest.fixture
def session_manager(storage_backend):
    """Provide a SessionManager instance for testing."""
    return SessionManager(storage_backend)


def test_create_session(session_manager):
    """Test creating a new session with unique ID."""
    session = session_manager.create_session(user_id="test-user")
    
    assert session is not None
    assert session.session_id is not None
    assert len(session.session_id) > 0
    assert session.user_id == "test-user"
    assert session.conversation_history == []
    assert session.is_expired is False
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)


def test_create_session_unique_ids(session_manager):
    """Test that multiple sessions get unique IDs."""
    session1 = session_manager.create_session()
    session2 = session_manager.create_session()
    
    assert session1.session_id != session2.session_id


def test_get_session(session_manager):
    """Test retrieving an existing session."""
    # Create a session
    created_session = session_manager.create_session(user_id="test-user")
    
    # Retrieve it
    retrieved_session = session_manager.get_session(created_session.session_id)
    
    assert retrieved_session is not None
    assert retrieved_session.session_id == created_session.session_id
    assert retrieved_session.user_id == "test-user"


def test_get_nonexistent_session(session_manager):
    """Test retrieving a session that doesn't exist."""
    session = session_manager.get_session("nonexistent-id")
    assert session is None


def test_update_session(session_manager):
    """Test updating a session."""
    # Create a session
    session = session_manager.create_session(user_id="test-user")
    original_updated_at = session.updated_at
    
    # Modify the session
    session.metadata["test_key"] = "test_value"
    
    # Small delay to ensure timestamp changes
    import time
    time.sleep(0.01)
    
    # Update the session
    session_manager.update_session(session)
    
    # Retrieve and verify
    retrieved = session_manager.get_session(session.session_id)
    assert retrieved.metadata["test_key"] == "test_value"
    assert retrieved.updated_at > original_updated_at


def test_add_message(session_manager):
    """Test adding a message to a session."""
    # Create a session
    session = session_manager.create_session(user_id="test-user")
    
    # Add a message
    session_manager.add_message(
        session=session,
        role=MessageRole.USER,
        content="Hello, chatbot!"
    )
    
    # Verify message was added
    assert len(session.conversation_history) == 1
    assert session.conversation_history[0].role == MessageRole.USER
    assert session.conversation_history[0].content == "Hello, chatbot!"
    assert isinstance(session.conversation_history[0].timestamp, datetime)


def test_add_message_with_metadata(session_manager):
    """Test adding a message with metadata."""
    session = session_manager.create_session()
    
    metadata = {"agent_used": "hr_agent", "confidence": 0.95}
    session_manager.add_message(
        session=session,
        role=MessageRole.ASSISTANT,
        content="I can help with that.",
        metadata=metadata
    )
    
    message = session.conversation_history[0]
    assert message.metadata == metadata
    assert message.agent_used == "hr_agent"


def test_add_multiple_messages(session_manager):
    """Test adding multiple messages to a session."""
    session = session_manager.create_session()
    
    # Add multiple messages
    session_manager.add_message(session, MessageRole.USER, "First message")
    session_manager.add_message(session, MessageRole.ASSISTANT, "First response")
    session_manager.add_message(session, MessageRole.USER, "Second message")
    
    assert len(session.conversation_history) == 3
    assert session.conversation_history[0].content == "First message"
    assert session.conversation_history[1].content == "First response"
    assert session.conversation_history[2].content == "Second message"


def test_get_conversation_history(session_manager):
    """Test retrieving conversation history."""
    session = session_manager.create_session()
    
    # Add messages
    session_manager.add_message(session, MessageRole.USER, "Message 1")
    session_manager.add_message(session, MessageRole.ASSISTANT, "Response 1")
    session_manager.add_message(session, MessageRole.USER, "Message 2")
    
    # Get all history
    history = session_manager.get_conversation_history(session)
    
    assert len(history) == 3
    assert history[0].content == "Message 1"
    assert history[1].content == "Response 1"
    assert history[2].content == "Message 2"


def test_get_conversation_history_with_limit(session_manager):
    """Test retrieving conversation history with message limit."""
    session = session_manager.create_session()
    
    # Add 5 messages
    for i in range(5):
        session_manager.add_message(session, MessageRole.USER, f"Message {i}")
    
    # Get last 2 messages
    history = session_manager.get_conversation_history(session, max_messages=2)
    
    assert len(history) == 2
    assert history[0].content == "Message 3"
    assert history[1].content == "Message 4"


def test_get_conversation_history_limit_exceeds_total(session_manager):
    """Test getting history when limit exceeds total messages."""
    session = session_manager.create_session()
    
    # Add 2 messages
    session_manager.add_message(session, MessageRole.USER, "Message 1")
    session_manager.add_message(session, MessageRole.USER, "Message 2")
    
    # Request 10 messages (more than available)
    history = session_manager.get_conversation_history(session, max_messages=10)
    
    assert len(history) == 2


def test_expire_old_sessions(session_manager, storage_backend):
    """Test expiring old sessions."""
    # Create a recent session
    recent_session = session_manager.create_session(user_id="recent-user")
    
    # Create an old session by manually setting timestamps
    old_session = session_manager.create_session(user_id="old-user")
    old_time = datetime.now() - timedelta(hours=25)
    old_session.updated_at = old_time
    
    # Manually save to storage without updating timestamp
    storage_backend.save_session(old_session.session_id, old_session.to_dict())
    
    # Expire sessions older than 24 hours
    expired_count = session_manager.expire_old_sessions(max_age_hours=24)
    
    assert expired_count == 1
    
    # Verify the old session is marked as expired
    retrieved_old = session_manager.get_session(old_session.session_id)
    assert retrieved_old.is_expired is True
    
    # Verify the recent session is not expired
    retrieved_recent = session_manager.get_session(recent_session.session_id)
    assert retrieved_recent.is_expired is False


def test_expire_old_sessions_no_old_sessions(session_manager):
    """Test expiring when there are no old sessions."""
    # Create only recent sessions
    session_manager.create_session()
    session_manager.create_session()
    
    # Try to expire old sessions
    expired_count = session_manager.expire_old_sessions(max_age_hours=24)
    
    assert expired_count == 0


def test_session_persistence_across_retrieval(session_manager):
    """Test that session data persists across multiple retrievals."""
    # Create and populate a session
    session = session_manager.create_session(user_id="test-user")
    session_manager.add_message(session, MessageRole.USER, "Test message")
    
    # Retrieve the session multiple times
    retrieved1 = session_manager.get_session(session.session_id)
    retrieved2 = session_manager.get_session(session.session_id)
    
    # Verify data is consistent
    assert retrieved1.session_id == retrieved2.session_id
    assert len(retrieved1.conversation_history) == len(retrieved2.conversation_history)
    assert retrieved1.conversation_history[0].content == "Test message"
    assert retrieved2.conversation_history[0].content == "Test message"


def test_add_message_persists_to_storage(session_manager):
    """Test that adding a message persists to storage."""
    # Create a session
    session = session_manager.create_session()
    session_id = session.session_id
    
    # Add a message
    session_manager.add_message(session, MessageRole.USER, "Persisted message")
    
    # Create a new session manager with the same storage
    new_manager = SessionManager(session_manager.storage_backend)
    
    # Retrieve the session with the new manager
    retrieved = new_manager.get_session(session_id)
    
    assert retrieved is not None
    assert len(retrieved.conversation_history) == 1
    assert retrieved.conversation_history[0].content == "Persisted message"
