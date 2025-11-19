"""
SessionManager component for managing conversation sessions.

This module provides the SessionManager class that handles session lifecycle,
conversation history management, and persistence through storage backends.
"""
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backend.chatbot.models import Session, Message
from backend.chatbot.storage import StorageBackend

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions including creation, retrieval, and persistence.
    
    The SessionManager provides a high-level interface for session operations,
    abstracting away the underlying storage implementation through dependency injection.
    
    Attributes:
        storage_backend: The storage backend used for session persistence
    """
    
    def __init__(self, storage_backend: StorageBackend):
        """
        Initialize the SessionManager with a storage backend.
        
        Args:
            storage_backend: Storage backend implementation for session persistence
        """
        self.storage_backend = storage_backend
        logger.info(f"Initialized SessionManager with {type(storage_backend).__name__}")
    
    def create_session(self, user_id: Optional[str] = None) -> Session:
        """
        Create a new session with a unique ID.
        
        Generates a unique session identifier using UUID4 and initializes
        a new Session object with empty conversation history.
        
        Args:
            user_id: Optional identifier for the user
            
        Returns:
            A new Session object with unique session_id
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            conversation_history=[],
            metadata={},
            is_expired=False
        )
        
        # Persist the new session
        self.update_session(session)
        
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve an existing session by ID.
        
        Loads session data from the storage backend and deserializes it
        into a Session object.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Session object if found, None otherwise
        """
        session_data = self.storage_backend.load_session(session_id)
        
        if session_data is None:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        session = Session.from_dict(session_data)
        logger.debug(f"Retrieved session: {session_id}")
        return session
    
    def update_session(self, session: Session) -> None:
        """
        Persist session updates to storage.
        
        Updates the session's updated_at timestamp and saves the complete
        session data to the storage backend.
        
        Args:
            session: Session object to persist
        """
        # Update the timestamp
        session.updated_at = datetime.now()
        
        # Serialize and save
        session_data = session.to_dict()
        self.storage_backend.save_session(session.session_id, session_data)
        
        logger.debug(f"Updated session: {session.session_id}")
    
    def add_message(
        self, 
        session: Session, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a message to the session's conversation history.
        
        Creates a new Message object and appends it to the session's
        conversation history. The session is automatically persisted
        after adding the message.
        
        Args:
            session: Session to add the message to
            role: Role of the message sender (user, assistant, system)
            content: Text content of the message
            metadata: Optional additional data about the message
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            agent_used=metadata.get('agent_used') if metadata else None
        )
        
        session.conversation_history.append(message)
        self.update_session(session)
        
        logger.debug(
            f"Added {role} message to session {session.session_id}: "
            f"{content[:50]}..." if len(content) > 50 else content
        )
    
    def get_conversation_history(
        self, 
        session: Session,
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve conversation history with optional message limit.
        
        Returns the conversation history from the session, optionally
        limited to the most recent N messages.
        
        Args:
            session: Session to retrieve history from
            max_messages: Optional limit on number of messages to return.
                         If provided, returns the most recent messages.
            
        Returns:
            List of Message objects from the conversation history
        """
        history = session.conversation_history
        
        if max_messages is not None and max_messages > 0:
            # Return the most recent max_messages
            history = history[-max_messages:]
            logger.debug(
                f"Retrieved last {max_messages} messages from session {session.session_id}"
            )
        else:
            logger.debug(
                f"Retrieved all {len(history)} messages from session {session.session_id}"
            )
        
        return history
    
    def expire_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Mark inactive sessions as expired.
        
        Scans all sessions and marks those that haven't been updated
        within the specified time period as expired. Expired sessions
        are not deleted but flagged for potential cleanup.
        
        Args:
            max_age_hours: Maximum age in hours before a session is expired (default: 24)
            
        Returns:
            Number of sessions that were marked as expired
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_count = 0
        
        # Get all non-expired sessions
        all_session_ids = self.storage_backend.list_sessions(
            filters={'is_expired': False}
        )
        
        for session_id in all_session_ids:
            session = self.get_session(session_id)
            
            if session is None:
                continue
            
            # Check if session is older than cutoff
            if session.updated_at < cutoff_time:
                session.is_expired = True
                self.update_session(session)
                expired_count += 1
                logger.info(
                    f"Expired session {session_id} "
                    f"(last updated: {session.updated_at})"
                )
        
        logger.info(
            f"Expired {expired_count} sessions older than {max_age_hours} hours"
        )
        return expired_count
