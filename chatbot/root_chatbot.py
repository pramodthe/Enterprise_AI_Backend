"""
RootChatbot - Main conversational agent for the Enterprise AI Assistant Platform.

This module implements the Root Chatbot that serves as the primary interface for users,
managing conversation context, routing queries to specialized agents, and providing
a seamless multi-turn conversation experience.
"""
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from backend.chatbot.models import (
    ChatResponse, Message, MessageRole, Session, RoutingDecision
)
from backend.chatbot.session_manager import SessionManager
from backend.chatbot.agent_router import AgentRouter
from backend.chatbot.bedrock_integration import BedrockIntegration
from backend.core.guardrail import create_organization_guardrail, GuardrailResult

# Configure logging
logger = logging.getLogger(__name__)


class RootChatbot:
    """
    Main conversational agent that orchestrates user interactions.
    
    The RootChatbot maintains conversation context, intelligently routes queries
    to specialized agents, and provides responses for general queries. It uses
    AWS Bedrock (via Strands SDK) for language model capabilities.
    
    Attributes:
        model: BedrockModel instance for generating responses
        session_manager: SessionManager for handling conversation state
        agent_router: AgentRouter for determining query routing
        max_context_tokens: Maximum tokens for context window (default: 4000)
        system_prompt: System prompt for the chatbot
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant for an enterprise platform.
You can help with general questions and coordinate with specialized agents for:
- HR queries (employee information, organizational structure, skills)
- Analytics queries (calculations, data analysis, reporting)
- Document queries (policies, procedures, guidelines)

Be professional, concise, and helpful in your responses."""
    
    def __init__(
        self,
        bedrock_integration: BedrockIntegration,
        session_manager: SessionManager,
        agent_router: AgentRouter,
        max_context_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        enable_guardrails: bool = True
    ):
        """
        Initialize the Root Chatbot with dependencies.
        
        Args:
            bedrock_integration: BedrockIntegration instance for model interactions
            session_manager: SessionManager for conversation state management
            agent_router: AgentRouter for query routing decisions
            max_context_tokens: Maximum tokens for context window (default: 4000)
            system_prompt: Optional custom system prompt
            enable_guardrails: Enable guardrail checks (default: True)
        """
        self.bedrock_integration = bedrock_integration
        self.model = bedrock_integration.get_model()
        self.session_manager = session_manager
        self.agent_router = agent_router
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Initialize guardrail
        self.enable_guardrails = enable_guardrails
        if enable_guardrails:
            self.guardrail = create_organization_guardrail()
            logger.info("Guardrails enabled for RootChatbot")
        else:
            self.guardrail = None
            logger.info("Guardrails disabled for RootChatbot")
        
        logger.info(
            f"Initialized RootChatbot with max_context_tokens={max_context_tokens}, "
            f"using {'Bedrock' if bedrock_integration.is_using_bedrock() else 'Anthropic'}"
        )
    
    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a user message and return a response.
        
        This is the main entry point for the Root Chatbot. It handles session
        management, routing decisions, agent coordination, and response generation.
        
        Args:
            message: The user's message
            session_id: Optional session identifier for continuing conversations
            user_id: Optional user identifier
            
        Returns:
            ChatResponse containing the reply, session_id, and metadata
        """
        logger.info(f"Processing message: {message[:100]}...")
        
        # Step 0: Guardrail check (before any processing)
        if self.enable_guardrails and self.guardrail:
            guardrail_result = self.guardrail.check(message)
            
            if not guardrail_result.is_safe:
                logger.warning(
                    f"Guardrail violation detected: {guardrail_result.violation_type.value}"
                )
                
                # Return intervention message without processing
                return ChatResponse(
                    message=guardrail_result.intervention_message,
                    session_id=session_id or "guardrail_blocked",
                    agent_used="guardrail",
                    confidence=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        "guardrail_blocked": True,
                        "violation_type": guardrail_result.violation_type.value,
                        "reason": guardrail_result.reason
                    }
                )
        
        # Step 1: Session retrieval or creation
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session is None:
                logger.warning(
                    f"Session {session_id} not found. Creating new session."
                )
                session = self.session_manager.create_session(user_id)
        else:
            session = self.session_manager.create_session(user_id)
        
        # Step 2: Add user message to conversation history
        self.session_manager.add_message(
            session=session,
            role=MessageRole.USER,
            content=message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Step 3: Build context from conversation history
        context = self._build_context(session)
        
        # Step 4: Determine if we should route to a specialized agent
        should_route = self._should_route_to_agent(message, context)
        
        # Step 5: Get previous agent for sticky routing
        previous_agent = self._get_previous_agent(session)
        
        # Step 6: Make routing decision
        routing_decision = self.agent_router.route_query(
            query=message,
            context=context,
            previous_agent=previous_agent
        )
        
        logger.info(
            f"Routing decision: {routing_decision.agent_name} "
            f"(confidence: {routing_decision.confidence:.2f})"
        )
        
        # Step 7: Process based on routing decision
        if routing_decision.should_use_agent() and routing_decision.agent_name != "root":
            # Route to specialized agent
            response_content, agent_used = await self._handle_agent_query(
                message=message,
                routing_decision=routing_decision,
                context=context
            )
        else:
            # Handle directly with Root Chatbot
            response_content = await self._handle_general_query(
                message=message,
                context=context
            )
            agent_used = "root"
        
        # Step 8: Add assistant response to conversation history
        self.session_manager.add_message(
            session=session,
            role=MessageRole.ASSISTANT,
            content=response_content,
            metadata={
                "agent_used": agent_used,
                "confidence": routing_decision.confidence,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Step 9: Create and return ChatResponse
        response = ChatResponse(
            message=response_content,
            session_id=session.session_id,
            agent_used=agent_used,
            confidence=routing_decision.confidence,
            timestamp=datetime.now(),
            metadata={
                "routing_reasoning": routing_decision.reasoning,
                "fallback_agents": routing_decision.fallback_agents,
                "conversation_length": len(session.conversation_history)
            }
        )
        
        logger.info(
            f"Completed processing message for session {session.session_id}"
        )
        
        return response
    
    def _build_context(self, session: Session) -> str:
        """
        Build context string from conversation history.
        
        Formats the conversation history into a string suitable for providing
        context to the language model or for routing analysis. Implements
        sliding window strategy to stay within token limits.
        
        Args:
            session: Session containing conversation history
            
        Returns:
            Formatted context string
        """
        # Get conversation history with sliding window
        history = self._apply_sliding_window(session.conversation_history)
        
        # Format messages into context string
        context_parts = []
        for msg in history:
            role_label = msg.role.upper() if hasattr(msg.role, 'upper') else str(msg.role).upper()
            context_parts.append(f"{role_label}: {msg.content}")
        
        context = "\n".join(context_parts)
        
        logger.debug(
            f"Built context with {len(history)} messages "
            f"({len(context)} characters)"
        )
        
        return context
    
    def _apply_sliding_window(self, messages: List[Message]) -> List[Message]:
        """
        Apply sliding window strategy to conversation history.
        
        Keeps the most recent messages that fit within the context window limit.
        Uses a simple heuristic of ~4 characters per token for estimation.
        
        Args:
            messages: Full conversation history
            
        Returns:
            List of messages that fit within the context window
        """
        if not messages:
            return []
        
        # Simple heuristic: ~4 characters per token
        chars_per_token = 4
        max_chars = self.max_context_tokens * chars_per_token
        
        # Start from the most recent messages and work backwards
        selected_messages = []
        current_chars = 0
        
        for message in reversed(messages):
            message_chars = len(message.content) + 20  # +20 for role and formatting
            
            if current_chars + message_chars > max_chars:
                # Would exceed limit, stop here
                break
            
            selected_messages.insert(0, message)
            current_chars += message_chars
        
        # Ensure we have at least the most recent message
        if not selected_messages and messages:
            selected_messages = [messages[-1]]
        
        if len(selected_messages) < len(messages):
            logger.debug(
                f"Applied sliding window: kept {len(selected_messages)}/{len(messages)} messages"
            )
        
        return selected_messages
    
    def _should_route_to_agent(self, message: str, context: str) -> bool:
        """
        Determine if message should be routed to a specialized agent.
        
        This is a preliminary check before making the full routing decision.
        It helps optimize by avoiding unnecessary routing for clearly general queries.
        
        Args:
            message: The user's message
            context: Conversation context
            
        Returns:
            True if the message might benefit from specialized agent routing
        """
        # Simple heuristic: check for common general query patterns
        general_patterns = [
            "hello", "hi", "hey", "thanks", "thank you",
            "what can you do", "help", "capabilities",
            "how are you", "goodbye", "bye"
        ]
        
        message_lower = message.lower().strip()
        
        # If it's a very short greeting or thanks, probably general
        if any(pattern in message_lower for pattern in general_patterns):
            if len(message.split()) <= 5:  # Short message
                logger.debug("Message appears to be a general query (greeting/thanks)")
                return False
        
        # Otherwise, let the router decide
        return True
    
    def _get_previous_agent(self, session: Session) -> Optional[str]:
        """
        Get the agent that handled the previous message.
        
        Used for sticky routing to maintain conversation continuity.
        
        Args:
            session: Current session
            
        Returns:
            Name of the previous agent, or None
        """
        # Look for the most recent assistant message
        for message in reversed(session.conversation_history):
            if message.role == MessageRole.ASSISTANT and message.agent_used:
                return message.agent_used
        
        return None
    
    async def _handle_agent_query(
        self,
        message: str,
        routing_decision: RoutingDecision,
        context: str
    ) -> tuple[str, str]:
        """
        Handle a query by routing to a specialized agent.
        
        Communicates with the specialized agent via A2A protocol and handles
        failures with fallback strategies.
        
        Args:
            message: The user's message
            routing_decision: Routing decision with selected agent
            context: Conversation context
            
        Returns:
            Tuple of (response_content, agent_name)
        """
        agent_name = routing_decision.agent_name
        
        # Get the agent client
        if agent_name not in self.agent_router.agents:
            logger.error(f"Agent {agent_name} not found in router")
            return await self._handle_general_query(message, context), "root"
        
        agent_client = self.agent_router.agents[agent_name]
        
        # Prepare context for agent
        agent_context = {
            "conversation_history": context,
            "routing_confidence": routing_decision.confidence
        }
        
        # Query the agent
        try:
            agent_response = await agent_client.query(
                message=message,
                context=agent_context
            )
            
            if agent_response.success:
                logger.info(f"Successfully received response from {agent_name}")
                return agent_response.content, agent_name
            else:
                # Agent failed, try fallback
                logger.warning(
                    f"Agent {agent_name} failed: {agent_response.error}. "
                    "Attempting fallback."
                )
                return await self._handle_agent_failure(
                    message=message,
                    context=context,
                    failed_agent=agent_name,
                    routing_decision=routing_decision
                )
        
        except Exception as e:
            logger.error(f"Exception querying agent {agent_name}: {str(e)}")
            return await self._handle_agent_failure(
                message=message,
                context=context,
                failed_agent=agent_name,
                routing_decision=routing_decision
            )
    
    async def _handle_agent_failure(
        self,
        message: str,
        context: str,
        failed_agent: str,
        routing_decision: RoutingDecision
    ) -> tuple[str, str]:
        """
        Handle agent failure with fallback strategies.
        
        Tries fallback agents or falls back to general query handling.
        
        Args:
            message: The user's message
            context: Conversation context
            failed_agent: Name of the agent that failed
            routing_decision: Original routing decision with fallback options
            
        Returns:
            Tuple of (response_content, agent_name)
        """
        # Try fallback agents
        for fallback_agent in routing_decision.fallback_agents:
            if fallback_agent in self.agent_router.agents:
                logger.info(f"Trying fallback agent: {fallback_agent}")
                
                try:
                    agent_client = self.agent_router.agents[fallback_agent]
                    agent_response = await agent_client.query(
                        message=message,
                        context={"conversation_history": context}
                    )
                    
                    if agent_response.success:
                        logger.info(f"Fallback agent {fallback_agent} succeeded")
                        return agent_response.content, fallback_agent
                
                except Exception as e:
                    logger.warning(f"Fallback agent {fallback_agent} also failed: {str(e)}")
                    continue
        
        # All agents failed, handle with root chatbot
        logger.warning(
            f"All agents failed for query. Handling with root chatbot. "
            f"Failed agent: {failed_agent}"
        )
        
        response = await self._handle_general_query(message, context)
        
        # Add disclaimer about agent failure
        disclaimer = (
            f"\n\n(Note: The specialized {failed_agent} agent is currently unavailable. "
            "I've provided a general response, but it may not be as accurate as the "
            "specialized agent would provide.)"
        )
        
        return response + disclaimer, "root"
    
    async def _handle_general_query(
        self,
        message: str,
        context: str
    ) -> str:
        """
        Handle a general query directly with the Root Chatbot.
        
        Uses the language model to generate a response for queries that don't
        require specialized agent knowledge.
        
        Args:
            message: The user's message
            context: Conversation context
            
        Returns:
            Response content string
        """
        logger.debug("Handling query with root chatbot (general knowledge)")
        
        # Build conversation messages
        messages = []
        
        if context:
            # Parse context back into messages
            for line in context.split('\n'):
                if line.startswith('USER:'):
                    messages.append({"role": "user", "content": line[5:].strip()})
                elif line.startswith('ASSISTANT:'):
                    messages.append({"role": "assistant", "content": line[10:].strip()})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response using BedrockIntegration with retry logic
        bedrock_response = await self.bedrock_integration.generate_response(
            messages=messages,
            system_prompt=self.system_prompt
        )
        
        if bedrock_response.success:
            logger.debug(
                f"Generated response: {bedrock_response.content[:100]}... "
                f"(metadata: {bedrock_response.metadata})"
            )
            return bedrock_response.content
        else:
            logger.error(
                f"Error generating response with Bedrock: {bedrock_response.error}"
            )
            return (
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again in a moment."
            )
