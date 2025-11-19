"""
Common database operations for Supabase.

This module provides high-level functions for common database operations
like saving metrics, querying analytics, and managing users.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from backend.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """High-level database operations wrapper."""
    
    def __init__(self):
        """Initialize database operations."""
        self.client = get_supabase_client()
        if not self.client:
            logger.warning("Supabase client not available. Database operations will be disabled.")
    
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.client is not None
    
    # User Operations
    
    def create_user(self, user_id: str, username: str, email: Optional[str] = None,
                   full_name: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Create a new user.
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: User email (optional)
            full_name: User's full name (optional)
            metadata: Additional user data (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            user_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "metadata": metadata or {}
            }
            
            self.client.table("users").insert(user_data).execute()
            logger.info(f"Created user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User data dictionary or None
        """
        if not self.client:
            return None
        
        try:
            response = self.client.table("users").select("*").eq("user_id", user_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def update_user(self, user_id: str, updates: Dict) -> bool:
        """
        Update user data.
        
        Args:
            user_id: User identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.table("users").update(updates).eq("user_id", user_id).execute()
            logger.info(f"Updated user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
    
    # Session Operations
    
    def get_user_sessions(self, user_id: str, include_expired: bool = False) -> List[Dict]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            include_expired: Whether to include expired sessions
            
        Returns:
            List of session dictionaries
        """
        if not self.client:
            return []
        
        try:
            query = self.client.table("sessions").select("*").eq("user_id", user_id)
            
            if not include_expired:
                query = query.eq("is_expired", False)
            
            response = query.order("updated_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    def expire_old_sessions(self, days: int = 30) -> int:
        """
        Mark old sessions as expired.
        
        Args:
            days: Number of days of inactivity before expiring
            
        Returns:
            Number of sessions expired
        """
        if not self.client:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            response = self.client.table("sessions").update({
                "is_expired": True
            }).lt("updated_at", cutoff_date.isoformat()).eq("is_expired", False).execute()
            
            count = len(response.data) if response.data else 0
            logger.info(f"Expired {count} old sessions")
            return count
        except Exception as e:
            logger.error(f"Failed to expire old sessions: {e}")
            return 0
    
    # Agent Metrics Operations
    
    def save_agent_metric(self, agent_name: str, query: str, response_time_ms: int,
                         success: bool, session_id: Optional[str] = None,
                         error_message: Optional[str] = None,
                         confidence_score: Optional[float] = None,
                         metadata: Optional[Dict] = None) -> bool:
        """
        Save agent performance metric.
        
        Args:
            agent_name: Name of the agent
            query: User query
            response_time_ms: Response time in milliseconds
            success: Whether the query was successful
            session_id: Associated session ID (optional)
            error_message: Error message if failed (optional)
            confidence_score: Confidence score (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            metric_data = {
                "agent_name": agent_name,
                "query": query,
                "response_time_ms": response_time_ms,
                "success": success,
                "session_id": session_id,
                "error_message": error_message,
                "confidence_score": confidence_score,
                "metadata": metadata or {}
            }
            
            self.client.table("agent_metrics").insert(metric_data).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save agent metric: {e}")
            return False
    
    def get_agent_performance(self, agent_name: Optional[str] = None,
                            days: int = 7) -> List[Dict]:
        """
        Get agent performance metrics.
        
        Args:
            agent_name: Filter by agent name (optional)
            days: Number of days to look back
            
        Returns:
            List of performance metrics
        """
        if not self.client:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = self.client.table("agent_metrics").select("*").gte(
                "created_at", cutoff_date.isoformat()
            )
            
            if agent_name:
                query = query.eq("agent_name", agent_name)
            
            response = query.order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get agent performance: {e}")
            return []
    
    def get_agent_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get aggregated agent statistics.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with statistics per agent
        """
        if not self.client:
            return {}
        
        try:
            metrics = self.get_agent_performance(days=days)
            
            stats = {}
            for metric in metrics:
                agent = metric["agent_name"]
                if agent not in stats:
                    stats[agent] = {
                        "total_queries": 0,
                        "successful_queries": 0,
                        "failed_queries": 0,
                        "total_response_time": 0,
                        "confidence_scores": []
                    }
                
                stats[agent]["total_queries"] += 1
                if metric["success"]:
                    stats[agent]["successful_queries"] += 1
                else:
                    stats[agent]["failed_queries"] += 1
                
                stats[agent]["total_response_time"] += metric["response_time_ms"]
                
                if metric.get("confidence_score"):
                    stats[agent]["confidence_scores"].append(metric["confidence_score"])
            
            # Calculate averages
            for agent, data in stats.items():
                data["success_rate"] = (
                    data["successful_queries"] / data["total_queries"]
                    if data["total_queries"] > 0 else 0
                )
                data["avg_response_time_ms"] = (
                    data["total_response_time"] / data["total_queries"]
                    if data["total_queries"] > 0 else 0
                )
                data["avg_confidence"] = (
                    sum(data["confidence_scores"]) / len(data["confidence_scores"])
                    if data["confidence_scores"] else 0
                )
                del data["total_response_time"]
                del data["confidence_scores"]
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get agent statistics: {e}")
            return {}
    
    # Document Operations
    
    def save_document_metadata(self, document_id: str, filename: str, file_type: str,
                              file_size: int, uploaded_by: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> bool:
        """
        Save document metadata.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_type: File type/extension
            file_size: File size in bytes
            uploaded_by: User ID who uploaded (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            doc_data = {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "uploaded_by": uploaded_by,
                "metadata": metadata or {}
            }
            
            self.client.table("documents").insert(doc_data).execute()
            logger.info(f"Saved document metadata: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save document metadata: {e}")
            return False
    
    def mark_document_processed(self, document_id: str, chunk_count: int) -> bool:
        """
        Mark a document as processed.
        
        Args:
            document_id: Document identifier
            chunk_count: Number of chunks created
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.table("documents").update({
                "processed": True,
                "chunk_count": chunk_count
            }).eq("document_id", document_id).execute()
            
            logger.info(f"Marked document {document_id} as processed")
            return True
        except Exception as e:
            logger.error(f"Failed to mark document as processed: {e}")
            return False
    
    def get_user_documents(self, user_id: str) -> List[Dict]:
        """
        Get all documents uploaded by a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of document metadata
        """
        if not self.client:
            return []
        
        try:
            response = self.client.table("documents").select("*").eq(
                "uploaded_by", user_id
            ).order("upload_date", desc=True).execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []


# Global instance
db_ops = DatabaseOperations()
