"""
AgentRouter for intelligent query routing to specialized agents.

This module provides routing logic to determine which specialized agent
should handle a given query based on keyword analysis, context awareness,
and conversation history.
"""
import logging
import re
from typing import Dict, List, Optional, Set
from collections import defaultdict

from backend.chatbot.models import RoutingDecision
from backend.chatbot.agent_client import AgentClient

# Configure logging
logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes queries to appropriate specialized agents based on content analysis.
    
    The router uses keyword-based scoring, context analysis, and sticky routing
    to intelligently determine which agent should handle each query. It supports
    dynamic agent registration and fallback strategies for low-confidence decisions.
    
    Attributes:
        agents: Dictionary mapping agent names to AgentClient instances
        agent_keywords: Dictionary mapping agent names to their keyword sets
        confidence_threshold: Minimum confidence for routing to an agent (default: 0.5)
    """
    
    # Default keyword mappings for specialized agents
    DEFAULT_KEYWORDS = {
        "hr": {
            # Core HR terms
            "employee", "staff", "hire", "hiring", "skill", "skills", "team", "teams",
            "org chart", "organization", "organizational", "personnel", "workforce",
            # HR processes
            "onboarding", "offboarding", "recruitment", "recruiting", "interview",
            "performance", "review", "evaluation", "promotion", "termination",
            # Benefits and compensation
            "salary", "compensation", "benefits", "vacation", "leave", "pto",
            "insurance", "retirement", "401k", "bonus",
            # Training and development
            "training", "development", "learning", "course", "certification",
            "mentoring", "coaching", "career",
            # Employee relations
            "manager", "supervisor", "report", "reporting", "hierarchy",
            "department", "role", "position", "job", "title"
        },
        "analytics": {
            # Mathematical operations
            "calculate", "compute", "sum", "total", "average", "mean", "median",
            "percentage", "percent", "ratio", "rate", "count", "number",
            # Financial terms
            "payroll", "budget", "cost", "expense", "revenue", "profit", "loss",
            "financial", "accounting", "invoice", "payment",
            # Data analysis
            "analyze", "analysis", "metric", "metrics", "statistics", "stats",
            "data", "report", "reporting", "dashboard", "trend", "trends",
            "forecast", "projection", "comparison", "compare",
            # Aggregations
            "aggregate", "summarize", "summary", "breakdown", "distribution",
            "maximum", "minimum", "highest", "lowest", "top", "bottom"
        },
        "document": {
            # Document types
            "policy", "policies", "document", "documents", "handbook", "manual",
            "procedure", "procedures", "guideline", "guidelines", "protocol",
            "standard", "standards", "regulation", "regulations",
            # Document actions
            "search", "find", "lookup", "look up", "retrieve", "locate",
            "read", "view", "check", "reference", "consult",
            # Document content
            "rule", "rules", "requirement", "requirements", "compliance",
            "code of conduct", "ethics", "legal", "contract", "agreement",
            "form", "forms", "template", "templates",
            # Specific document areas
            "hr policy", "company policy", "employee handbook", "safety",
            "security", "privacy", "confidentiality", "intellectual property"
        }
    }
    
    def __init__(
        self,
        agents: Optional[Dict[str, AgentClient]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the AgentRouter with available agents.
        
        Args:
            agents: Dictionary mapping agent names to AgentClient instances
            confidence_threshold: Minimum confidence for routing (default: 0.5)
        """
        self.agents: Dict[str, AgentClient] = agents or {}
        self.agent_keywords: Dict[str, Set[str]] = {}
        self.confidence_threshold = confidence_threshold
        
        # Initialize default keywords for known agents
        for agent_name in self.agents.keys():
            agent_key = agent_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if agent_key in self.DEFAULT_KEYWORDS:
                self.agent_keywords[agent_name] = self.DEFAULT_KEYWORDS[agent_key]
            else:
                self.agent_keywords[agent_name] = set()
        
        logger.info(
            f"Initialized AgentRouter with {len(self.agents)} agents "
            f"(threshold={confidence_threshold})"
        )
    
    def route_query(
        self,
        query: str,
        context: Optional[str] = None,
        previous_agent: Optional[str] = None
    ) -> RoutingDecision:
        """
        Determine which agent should handle the query.
        
        This method analyzes the query content, considers conversation context,
        and applies sticky routing logic to select the most appropriate agent.
        
        Args:
            query: The user's query text
            context: Optional conversation context for context-aware routing
            previous_agent: Optional name of agent that handled the previous query
            
        Returns:
            RoutingDecision with selected agent, confidence, and reasoning
        """
        logger.debug(f"Routing query: {query[:100]}...")
        
        # Calculate keyword-based scores for all agents
        keyword_scores = self._calculate_keyword_scores(query)
        
        # Apply context analysis if context is provided
        if context:
            context_boost = self._analyze_context(query, context)
            logger.debug(f"Context analysis suggests: {context_boost}")
            
            # Boost the score of the context-suggested agent
            if context_boost and context_boost in keyword_scores:
                keyword_scores[context_boost] *= 1.3  # 30% boost for context match
        
        # Apply sticky routing if previous agent is provided
        if previous_agent and previous_agent in keyword_scores:
            # Check if the query might be a follow-up
            if self._is_followup_query(query):
                keyword_scores[previous_agent] *= 1.5  # 50% boost for sticky routing
                logger.debug(f"Applied sticky routing boost to {previous_agent}")
        
        # Find the agent with the highest score
        if not keyword_scores:
            # No agents available or no scores calculated
            return RoutingDecision(
                agent_name="root",
                confidence=0.0,
                reasoning="No specialized agents available",
                fallback_agents=[]
            )
        
        # Sort agents by score
        sorted_agents = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        best_agent, best_score = sorted_agents[0]
        
        # Normalize score to 0-1 range (assuming max possible score is around 10)
        # This is a heuristic normalization
        normalized_confidence = min(best_score / 10.0, 1.0)
        
        # Prepare fallback agents (next best options)
        fallback_agents = [agent for agent, score in sorted_agents[1:3] if score > 0]
        
        # Determine reasoning
        reasoning_parts = []
        if keyword_scores[best_agent] > 0:
            reasoning_parts.append(f"keyword match (score: {best_score:.2f})")
        if context and self._analyze_context(query, context) == best_agent:
            reasoning_parts.append("context alignment")
        if previous_agent == best_agent and self._is_followup_query(query):
            reasoning_parts.append("follow-up to previous query")
        
        reasoning = f"Selected {best_agent} based on: {', '.join(reasoning_parts)}"
        
        # Check if confidence meets threshold
        if normalized_confidence < self.confidence_threshold:
            logger.info(
                f"Low confidence ({normalized_confidence:.2f}) for routing. "
                f"Falling back to root agent."
            )
            return RoutingDecision(
                agent_name="root",
                confidence=normalized_confidence,
                reasoning=f"Confidence below threshold. {reasoning}",
                fallback_agents=[best_agent] + fallback_agents
            )
        
        logger.info(
            f"Routed to {best_agent} with confidence {normalized_confidence:.2f}"
        )
        
        return RoutingDecision(
            agent_name=best_agent,
            confidence=normalized_confidence,
            reasoning=reasoning,
            fallback_agents=fallback_agents
        )
    
    def _calculate_keyword_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate keyword-based routing scores for all agents.
        
        Analyzes the query text for keywords associated with each agent
        and returns a score for each agent based on keyword matches.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Dictionary mapping agent names to their keyword match scores
        """
        # Normalize query for matching
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scores: Dict[str, float] = defaultdict(float)
        
        for agent_name, keywords in self.agent_keywords.items():
            if not keywords:
                continue
            
            # Count exact keyword matches
            for keyword in keywords:
                # Handle multi-word keywords
                if ' ' in keyword:
                    if keyword in query_lower:
                        scores[agent_name] += 2.0  # Multi-word matches are stronger
                else:
                    if keyword in query_words:
                        scores[agent_name] += 1.0
            
            # Bonus for multiple keyword matches (indicates stronger relevance)
            if scores[agent_name] >= 3:
                scores[agent_name] *= 1.2
        
        logger.debug(f"Keyword scores: {dict(scores)}")
        return dict(scores)
    
    def _analyze_context(self, query: str, context: str) -> str:
        """
        Analyze context to improve routing decision.
        
        Examines the conversation context to identify patterns that might
        indicate which agent is most appropriate for the current query.
        
        Args:
            query: The current query
            context: The conversation context (previous messages)
            
        Returns:
            Suggested agent name based on context analysis, or empty string
        """
        if not context:
            return ""
        
        context_lower = context.lower()
        
        # Count agent-related keywords in context
        context_scores: Dict[str, int] = defaultdict(int)
        
        for agent_name, keywords in self.agent_keywords.items():
            for keyword in keywords:
                # Count occurrences in context
                if ' ' in keyword:
                    context_scores[agent_name] += context_lower.count(keyword) * 2
                else:
                    # Use word boundary matching for single words
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    context_scores[agent_name] += len(re.findall(pattern, context_lower))
        
        # Return agent with highest context score
        if context_scores:
            best_context_agent = max(context_scores.items(), key=lambda x: x[1])
            if best_context_agent[1] > 0:
                return best_context_agent[0]
        
        return ""
    
    def _is_followup_query(self, query: str) -> bool:
        """
        Determine if a query appears to be a follow-up question.
        
        Analyzes the query for linguistic patterns that suggest it's
        continuing a previous conversation thread.
        
        Args:
            query: The query text to analyze
            
        Returns:
            True if the query appears to be a follow-up
        """
        query_lower = query.lower().strip()
        
        # Follow-up indicators
        followup_patterns = [
            # Pronouns and references
            r'\b(it|this|that|these|those|they|them)\b',
            # Continuation words
            r'\b(also|additionally|furthermore|moreover|besides)\b',
            # Reference to previous
            r'\b(same|previous|earlier|before|above|mentioned)\b',
            # Questions about previous topic
            r'\b(what about|how about|tell me more|more info|explain|clarify)\b',
            # Short queries (often follow-ups)
            r'^.{1,20}$',  # Very short queries
            # Comparative language
            r'\b(another|other|different|similar|like that)\b',
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def register_agent(
        self,
        agent_name: str,
        agent_client: AgentClient,
        keywords: Optional[Set[str]] = None
    ) -> None:
        """
        Register a new specialized agent for routing.
        
        This method allows dynamic addition of agents without modifying
        the router's core logic, supporting system extensibility.
        
        Args:
            agent_name: Unique name for the agent
            agent_client: AgentClient instance for communicating with the agent
            keywords: Optional set of keywords for routing to this agent
        """
        self.agents[agent_name] = agent_client
        
        # Use provided keywords or try to get from default mappings
        if keywords:
            self.agent_keywords[agent_name] = keywords
        else:
            agent_key = agent_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if agent_key in self.DEFAULT_KEYWORDS:
                self.agent_keywords[agent_name] = self.DEFAULT_KEYWORDS[agent_key]
            else:
                self.agent_keywords[agent_name] = set()
                logger.warning(
                    f"No keywords provided for agent '{agent_name}'. "
                    "Agent registered but may not receive routed queries."
                )
        
        logger.info(
            f"Registered agent '{agent_name}' with {len(self.agent_keywords[agent_name])} keywords"
        )
    
    def unregister_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the router.
        
        Args:
            agent_name: Name of the agent to remove
            
        Returns:
            True if agent was removed, False if agent was not found
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            if agent_name in self.agent_keywords:
                del self.agent_keywords[agent_name]
            logger.info(f"Unregistered agent '{agent_name}'")
            return True
        
        logger.warning(f"Attempted to unregister unknown agent '{agent_name}'")
        return False
    
    def get_registered_agents(self) -> List[str]:
        """
        Get list of all registered agent names.
        
        Returns:
            List of agent names currently registered with the router
        """
        return list(self.agents.keys())
    
    def set_agent_keywords(self, agent_name: str, keywords: Set[str]) -> bool:
        """
        Update the keywords for a registered agent.
        
        Args:
            agent_name: Name of the agent
            keywords: New set of keywords for routing
            
        Returns:
            True if keywords were updated, False if agent not found
        """
        if agent_name not in self.agents:
            logger.warning(f"Cannot set keywords for unknown agent '{agent_name}'")
            return False
        
        self.agent_keywords[agent_name] = keywords
        logger.info(f"Updated keywords for agent '{agent_name}' ({len(keywords)} keywords)")
        return True
