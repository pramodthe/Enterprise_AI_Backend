"""
Data models for the Root Chatbot.

This module defines the core data structures used throughout the Root Chatbot system,
including Session, Message, ChatResponse, RoutingDecision, and AgentResponse.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MessageRole(str, Enum):
    """Enumeration of message roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender (user, assistant, system)
        content: The text content of the message
        timestamp: When the message was created
        metadata: Optional additional data about the message
        agent_used: Optional name of the agent that generated this message
    """
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    agent_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "agent_used": self.agent_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Deserialize message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata"),
            agent_used=data.get("agent_used")
        )


@dataclass
class Session:
    """
    Represents a conversation session with persistent state.
    
    Attributes:
        session_id: Unique identifier for the session
        user_id: Optional identifier for the user
        created_at: When the session was created
        updated_at: When the session was last modified
        conversation_history: List of messages in the conversation
        metadata: Additional session data
        is_expired: Whether the session has expired
    """
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    conversation_history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_expired: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "metadata": self.metadata,
            "is_expired": self.is_expired
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Deserialize session from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            conversation_history=[
                Message.from_dict(msg) for msg in data.get("conversation_history", [])
            ],
            metadata=data.get("metadata", {}),
            is_expired=data.get("is_expired", False)
        )


@dataclass
class ChatResponse:
    """
    Represents a response from the Root Chatbot.
    
    Attributes:
        message: The response message content
        session_id: The session this response belongs to
        agent_used: Name of the agent that handled the query
        confidence: Confidence score for the routing decision
        timestamp: When the response was generated
        metadata: Additional response data
    """
    message: str
    session_id: str
    agent_used: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        return {
            "message": self.message,
            "session_id": self.session_id,
            "agent_used": self.agent_used,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class RoutingDecision:
    """
    Represents a decision about which agent should handle a query.
    
    Attributes:
        agent_name: Name of the selected agent
        confidence: Confidence score for this decision (0.0 to 1.0)
        reasoning: Explanation of why this agent was selected
        fallback_agents: List of alternative agents if primary fails
    """
    agent_name: str
    confidence: float
    reasoning: str
    fallback_agents: List[str] = field(default_factory=list)
    
    def should_use_agent(self, threshold: float = 0.5) -> bool:
        """
        Check if confidence exceeds threshold.
        
        Args:
            threshold: Minimum confidence required (default: 0.5)
            
        Returns:
            True if confidence is above threshold
        """
        return self.confidence >= threshold


@dataclass
class AgentResponse:
    """
    Represents a response from a specialized agent.
    
    Attributes:
        content: The response content from the agent
        agent_name: Name of the agent that generated the response
        metadata: Additional data about the response
        success: Whether the agent call was successful
        error: Optional error message if the call failed
    """
    content: str
    agent_name: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
