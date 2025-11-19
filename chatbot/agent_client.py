"""
AgentClient for A2A (Agent-to-Agent) communication.

This module provides a client for communicating with specialized agents
using the A2A protocol from the Strands framework.
"""
import logging
import time
from typing import Dict, List, Optional, Any
import asyncio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.chatbot.models import AgentResponse

# Configure logging
logger = logging.getLogger(__name__)


class AgentClient:
    """
    Client for communicating with specialized agents via A2A protocol.
    
    This client handles connection management, query sending, health checking,
    and implements retry logic with exponential backoff for resilient communication.
    
    Attributes:
        agent_url: The base URL of the agent's A2A server
        agent_name: Human-readable name of the agent
        timeout: Request timeout in seconds (default: 30)
    """
    
    def __init__(
        self, 
        agent_url: str, 
        agent_name: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        """
        Initialize client for a specific agent.
        
        Args:
            agent_url: Base URL of the agent's A2A server (e.g., "http://localhost:8000")
            agent_name: Human-readable name of the agent
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Multiplier for exponential backoff (default: 1.0)
        """
        self.agent_url = agent_url.rstrip('/')
        self.agent_name = agent_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Create a session with retry logic
        self.session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(
            f"Initialized AgentClient for {agent_name} at {agent_url} "
            f"(timeout={timeout}s, max_retries={max_retries})"
        )
    
    async def query(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Send a query to the specialized agent.
        
        This method sends a message to the agent via A2A protocol and returns
        the agent's response. It implements retry logic with exponential backoff
        for handling transient failures.
        
        Args:
            message: The query to send to the agent
            context: Optional context information (e.g., conversation history)
            
        Returns:
            AgentResponse containing the agent's reply and metadata
            
        Raises:
            Exception: If all retry attempts fail
        """
        endpoint = f"{self.agent_url}/query"
        payload = {
            "message": message,
            "context": context or {}
        }
        
        logger.debug(f"Sending query to {self.agent_name}: {message[:100]}...")
        
        # Track retry attempts for manual exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Use asyncio to run the synchronous request in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.post(
                        endpoint,
                        json=payload,
                        timeout=self.timeout
                    )
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                logger.info(
                    f"Successfully received response from {self.agent_name} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                
                return AgentResponse(
                    content=response_data.get("response", response_data.get("content", "")),
                    agent_name=self.agent_name,
                    metadata={
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds(),
                        "attempt": attempt + 1,
                        **response_data.get("metadata", {})
                    },
                    success=True,
                    error=None
                )
                
            except requests.exceptions.Timeout as e:
                logger.warning(
                    f"Timeout querying {self.agent_name} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    # Calculate backoff delay
                    delay = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    error_msg = f"Agent {self.agent_name} timed out after {self.max_retries} attempts"
                    logger.error(error_msg)
                    return AgentResponse(
                        content="",
                        agent_name=self.agent_name,
                        metadata={"attempts": self.max_retries},
                        success=False,
                        error=error_msg
                    )
                    
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Error querying {self.agent_name} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    # Calculate backoff delay
                    delay = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    error_msg = f"Agent {self.agent_name} failed after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    return AgentResponse(
                        content="",
                        agent_name=self.agent_name,
                        metadata={"attempts": self.max_retries},
                        success=False,
                        error=error_msg
                    )
            
            except Exception as e:
                logger.error(f"Unexpected error querying {self.agent_name}: {str(e)}")
                return AgentResponse(
                    content="",
                    agent_name=self.agent_name,
                    metadata={"attempts": attempt + 1},
                    success=False,
                    error=f"Unexpected error: {str(e)}"
                )
        
        # Should not reach here, but return error response as fallback
        return AgentResponse(
            content="",
            agent_name=self.agent_name,
            metadata={},
            success=False,
            error="Maximum retries exceeded"
        )
    
    def is_available(self) -> bool:
        """
        Check if the agent is currently available.
        
        Performs a health check by attempting to connect to the agent's
        health endpoint or base URL.
        
        Returns:
            True if the agent is available and responding, False otherwise
        """
        # Try health endpoint first
        health_endpoint = f"{self.agent_url}/health"
        
        try:
            response = self.session.get(health_endpoint, timeout=5)
            if response.status_code == 200:
                logger.debug(f"Agent {self.agent_name} is available (health check passed)")
                return True
        except requests.exceptions.RequestException:
            # Health endpoint might not exist, try base URL
            pass
        
        # Fallback: try base URL
        try:
            response = self.session.get(self.agent_url, timeout=5)
            is_available = response.status_code < 500
            
            if is_available:
                logger.debug(f"Agent {self.agent_name} is available (base URL check passed)")
            else:
                logger.warning(f"Agent {self.agent_name} returned status {response.status_code}")
            
            return is_available
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Agent {self.agent_name} is not available: {str(e)}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.
        
        Queries the agent for its capabilities, which may include supported
        query types, available tools, or other features.
        
        Returns:
            List of capability strings. Returns empty list if capabilities
            cannot be retrieved.
        """
        capabilities_endpoint = f"{self.agent_url}/capabilities"
        
        try:
            response = self.session.get(capabilities_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            capabilities = data.get("capabilities", [])
            
            logger.info(f"Retrieved {len(capabilities)} capabilities from {self.agent_name}")
            return capabilities
            
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Could not retrieve capabilities from {self.agent_name}: {str(e)}. "
                "Returning empty list."
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving capabilities from {self.agent_name}: {str(e)}"
            )
            return []
    
    def close(self):
        """Close the HTTP session and clean up resources."""
        if self.session:
            self.session.close()
            logger.debug(f"Closed session for {self.agent_name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
