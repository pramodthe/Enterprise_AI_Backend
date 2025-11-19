"""
AWS Bedrock Integration for Root Chatbot.

This module provides a wrapper around the Strands BedrockModel with:
- Proper configuration management
- Request formatting for Bedrock API
- Response parsing and metadata extraction
- Retry logic with exponential backoff
- Error handling for rate limiting and API failures
"""
import logging
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from strands.models.bedrock import BedrockModel
from strands.models.anthropic import AnthropicModel

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BedrockResponse:
    """Structured response from Bedrock API."""
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class BedrockIntegration:
    """
    Wrapper for AWS Bedrock integration with retry logic and error handling.
    
    This class handles:
    - Model initialization with AWS credentials
    - Request formatting for Bedrock API
    - Response parsing and metadata extraction
    - Retry logic with exponential backoff (up to 3 attempts)
    - Error handling for rate limiting and API failures
    
    Attributes:
        model: BedrockModel or AnthropicModel instance
        max_retries: Maximum number of retry attempts (default: 3)
        initial_retry_delay: Initial delay in seconds for exponential backoff (default: 1)
        use_bedrock: Whether to use Bedrock or Anthropic API
    """
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        model_id: str = "anthropic.claude-sonnet-v1:0",
        max_tokens: int = 1028,
        temperature: float = 0.3,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        use_bedrock: Optional[bool] = None,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize Bedrock integration.
        
        Args:
            aws_access_key_id: AWS access key ID (defaults to env var)
            aws_secret_access_key: AWS secret access key (defaults to env var)
            region_name: AWS region (default: us-east-1)
            model_id: Bedrock model ID (default: anthropic.claude-sonnet-v1:0)
            max_tokens: Maximum tokens for response (default: 1028)
            temperature: Model temperature (default: 0.3)
            max_retries: Maximum retry attempts (default: 3)
            initial_retry_delay: Initial retry delay in seconds (default: 1.0)
            use_bedrock: Whether to use Bedrock (defaults to USE_BEDROCK env var)
            anthropic_api_key: Anthropic API key for direct API usage
        """
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Determine whether to use Bedrock or Anthropic API
        if use_bedrock is None:
            self.use_bedrock = os.getenv("USE_BEDROCK", "False").lower() == "true"
        else:
            self.use_bedrock = use_bedrock
        
        # Get credentials from environment if not provided
        if self.use_bedrock:
            aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0")
            
            # Validate AWS credentials
            if not aws_access_key_id or not aws_secret_access_key:
                raise ValueError(
                    "AWS credentials required for Bedrock. "
                    "Provide aws_access_key_id and aws_secret_access_key or set "
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
                )
            
            logger.info(
                f"Initializing BedrockModel with region={region_name}, "
                f"model_id={model_id}"
            )
            
            # Initialize BedrockModel
            self.model = BedrockModel(
                client_args={
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                    "region_name": region_name,
                },
                max_tokens=max_tokens,
                model_id=model_id,
                params={"temperature": temperature}
            )
        else:
            # Use Anthropic API directly
            anthropic_api_key = anthropic_api_key or os.getenv("api_key")
            model_id = model_id or os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219")
            
            if not anthropic_api_key:
                raise ValueError(
                    "Anthropic API key required. "
                    "Provide anthropic_api_key or set api_key environment variable."
                )
            
            logger.info(f"Initializing AnthropicModel with model_id={model_id}")
            
            self.model = AnthropicModel(
                client_args={"api_key": anthropic_api_key},
                max_tokens=max_tokens,
                model_id=model_id,
                params={"temperature": temperature}
            )
        
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(
            f"Initialized {'Bedrock' if self.use_bedrock else 'Anthropic'} integration "
            f"with max_retries={max_retries}"
        )

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format messages for Bedrock API.
        
        Ensures messages follow the required format with 'role' and 'content' fields.
        Validates that roles alternate between 'user' and 'assistant'.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt (handled separately by Strands)
            
        Returns:
            Formatted messages list
            
        Raises:
            ValueError: If message format is invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        formatted_messages = []
        
        for i, msg in enumerate(messages):
            # Validate message structure
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Message at index {i} must have 'role' and 'content' fields"
                )
            
            role = msg["role"]
            content = msg["content"]
            
            # Validate role
            if role not in ["user", "assistant", "system"]:
                raise ValueError(
                    f"Invalid role '{role}' at index {i}. "
                    "Must be 'user', 'assistant', or 'system'"
                )
            
            # Validate content
            if not isinstance(content, str):
                raise ValueError(f"Content at index {i} must be a string")
            
            if not content.strip():
                raise ValueError(f"Content at index {i} cannot be empty")
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        logger.debug(f"Formatted {len(formatted_messages)} messages for Bedrock API")
        
        return formatted_messages

    def parse_response(self, response: Any) -> BedrockResponse:
        """
        Parse response from Bedrock API and extract metadata.
        
        Args:
            response: Raw response from Bedrock model
            
        Returns:
            BedrockResponse with content and metadata
        """
        try:
            # Convert response to string
            content = str(response)
            
            # Extract metadata if available
            metadata = {
                "model_id": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "provider": "bedrock" if self.use_bedrock else "anthropic"
            }
            
            # Try to extract additional metadata from response object
            if hasattr(response, "__dict__"):
                response_dict = response.__dict__
                
                # Extract common metadata fields
                if "usage" in response_dict:
                    metadata["usage"] = response_dict["usage"]
                
                if "stop_reason" in response_dict:
                    metadata["stop_reason"] = response_dict["stop_reason"]
                
                if "model" in response_dict:
                    metadata["model"] = response_dict["model"]
            
            logger.debug(f"Parsed response with {len(content)} characters")
            
            return BedrockResponse(
                content=content,
                metadata=metadata,
                success=True
            )
        
        except Exception as e:
            logger.error(f"Error parsing Bedrock response: {str(e)}")
            return BedrockResponse(
                content="",
                metadata={},
                success=False,
                error=f"Failed to parse response: {str(e)}"
            )

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> BedrockResponse:
        """
        Generate response with retry logic and error handling.
        
        Implements exponential backoff retry strategy:
        - Attempt 1: immediate
        - Attempt 2: wait 1 second
        - Attempt 3: wait 2 seconds
        - Attempt 4: wait 4 seconds (if max_retries >= 3)
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            BedrockResponse with content and metadata
        """
        # Format messages
        try:
            formatted_messages = self.format_messages(messages, system_prompt)
        except ValueError as e:
            logger.error(f"Message formatting error: {str(e)}")
            return BedrockResponse(
                content="",
                metadata={},
                success=False,
                error=f"Invalid message format: {str(e)}"
            )
        
        # Retry loop with exponential backoff
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} "
                    f"to generate response"
                )
                
                # Use Strands Agent for response generation
                from strands import Agent
                
                # Create agent with the model
                agent = Agent(
                    model=self.model,
                    name="Root Chatbot",
                    description="General purpose conversational assistant",
                    system_prompt=system_prompt or ""
                )
                
                # Get the last user message
                user_message = None
                for msg in reversed(formatted_messages):
                    if msg["role"] == "user":
                        user_message = msg["content"]
                        break
                
                if not user_message:
                    raise ValueError("No user message found in conversation")
                
                # Generate response
                response = agent(user_message)
                
                # Parse and return response
                parsed_response = self.parse_response(response)
                
                if parsed_response.success:
                    logger.info(
                        f"Successfully generated response on attempt {attempt + 1}"
                    )
                    return parsed_response
                else:
                    last_error = parsed_response.error
                    logger.warning(
                        f"Response parsing failed on attempt {attempt + 1}: "
                        f"{last_error}"
                    )
            
            except Exception as e:
                last_error = str(e)
                error_type = type(e).__name__
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: "
                    f"{error_type}: {last_error}"
                )
                
                # Check if it's a rate limiting error
                is_rate_limit = (
                    "rate" in last_error.lower() or
                    "throttl" in last_error.lower() or
                    "429" in last_error
                )
                
                # If this is not the last attempt, wait before retrying
                if attempt < self.max_retries:
                    # Calculate exponential backoff delay
                    delay = self.initial_retry_delay * (2 ** attempt)
                    
                    # Add extra delay for rate limiting
                    if is_rate_limit:
                        delay *= 2
                        logger.info(
                            f"Rate limiting detected. "
                            f"Waiting {delay} seconds before retry..."
                        )
                    else:
                        logger.info(f"Waiting {delay} seconds before retry...")
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Last error: {last_error}"
                    )
        
        # All retries exhausted
        return BedrockResponse(
            content="",
            metadata={
                "attempts": self.max_retries + 1,
                "last_error": last_error
            },
            success=False,
            error=f"Failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def get_model(self):
        """
        Get the underlying model instance.
        
        Returns:
            BedrockModel or AnthropicModel instance
        """
        return self.model
    
    def is_using_bedrock(self) -> bool:
        """
        Check if using Bedrock or Anthropic API.
        
        Returns:
            True if using Bedrock, False if using Anthropic API
        """
        return self.use_bedrock
