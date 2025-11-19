"""
Tests for Root Chatbot data models.
"""
import pytest
from datetime import datetime
from backend.chatbot.models import (
    Message, Session, ChatResponse, RoutingDecision, AgentResponse, MessageRole
)


def test_message_serialization():
    """Test Message to_dict and from_dict methods."""
    timestamp = datetime.now()
    message = Message(
        role=MessageRole.USER,
        content="Hello, world!",
        timestamp=timestamp,
        metadata={"key": "value"},
        agent_used="test_agent"
    )
    
    # Serialize
    data = message.to_dict()
    assert data["role"] == MessageRole.USER
    assert data["content"] == "Hello, world!"
    assert data["metadata"] == {"key": "value"}
    assert data["agent_used"] == "test_agent"
    
    # Deserialize
    restored = Message.from_dict(data)
    assert restored.role == message.role
    assert restored.content == message.content
    assert restored.metadata == message.metadata
    assert restored.agent_used == message.agent_used


def test_session_serialization():
    """Test Session to_dict and from_dict methods."""
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
        metadata={"test": "data"},
        is_expired=False
    )
    
    # Serialize
    data = session.to_dict()
    assert data["session_id"] == "test-123"
    assert data["user_id"] == "user-456"
    assert len(data["conversation_history"]) == 1
    assert data["metadata"] == {"test": "data"}
    assert data["is_expired"] is False
    
    # Deserialize
    restored = Session.from_dict(data)
    assert restored.session_id == session.session_id
    assert restored.user_id == session.user_id
    assert len(restored.conversation_history) == 1
    assert restored.conversation_history[0].content == "Test message"
    assert restored.metadata == session.metadata
    assert restored.is_expired == session.is_expired


def test_routing_decision_threshold():
    """Test RoutingDecision threshold checking."""
    decision = RoutingDecision(
        agent_name="hr_agent",
        confidence=0.75,
        reasoning="HR keywords detected",
        fallback_agents=["general"]
    )
    
    assert decision.should_use_agent(threshold=0.5) is True
    assert decision.should_use_agent(threshold=0.8) is False


def test_chat_response_serialization():
    """Test ChatResponse to_dict method."""
    timestamp = datetime.now()
    response = ChatResponse(
        message="Response text",
        session_id="session-123",
        agent_used="hr_agent",
        confidence=0.9,
        timestamp=timestamp,
        metadata={"tokens": 100}
    )
    
    data = response.to_dict()
    assert data["message"] == "Response text"
    assert data["session_id"] == "session-123"
    assert data["agent_used"] == "hr_agent"
    assert data["confidence"] == 0.9
    assert data["metadata"] == {"tokens": 100}


def test_agent_response():
    """Test AgentResponse data model."""
    response = AgentResponse(
        content="Agent response",
        agent_name="analytics_agent",
        metadata={"processing_time": 1.5},
        success=True,
        error=None
    )
    
    assert response.content == "Agent response"
    assert response.agent_name == "analytics_agent"
    assert response.success is True
    assert response.error is None
    
    # Test failed response
    failed_response = AgentResponse(
        content="",
        agent_name="analytics_agent",
        metadata={},
        success=False,
        error="Connection timeout"
    )
    
    assert failed_response.success is False
    assert failed_response.error == "Connection timeout"
