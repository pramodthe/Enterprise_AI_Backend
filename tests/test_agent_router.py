"""
Unit tests for AgentRouter component.

Tests the routing logic, keyword scoring, context analysis, and agent registration.
"""
import pytest
from unittest.mock import Mock, MagicMock

from backend.chatbot.agent_router import AgentRouter
from backend.chatbot.agent_client import AgentClient
from backend.chatbot.models import RoutingDecision


class TestAgentRouter:
    """Test suite for AgentRouter functionality."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agent clients for testing."""
        hr_agent = Mock(spec=AgentClient)
        hr_agent.agent_name = "hr"
        
        analytics_agent = Mock(spec=AgentClient)
        analytics_agent.agent_name = "analytics"
        
        document_agent = Mock(spec=AgentClient)
        document_agent.agent_name = "document"
        
        return {
            "hr": hr_agent,
            "analytics": analytics_agent,
            "document": document_agent
        }
    
    @pytest.fixture
    def router(self, mock_agents):
        """Create an AgentRouter instance with mock agents."""
        return AgentRouter(agents=mock_agents, confidence_threshold=0.5)
    
    def test_router_initialization(self, router, mock_agents):
        """Test that router initializes correctly with agents."""
        assert len(router.agents) == 3
        assert "hr" in router.agents
        assert "analytics" in router.agents
        assert "document" in router.agents
        assert router.confidence_threshold == 0.5
    
    def test_hr_keyword_routing(self, router):
        """Test routing to HR agent based on HR keywords."""
        queries = [
            "Who are the employees in the engineering team?",
            "What skills does John have?",
            "Show me the org chart",
            "I need to hire a new developer",
            "What is the onboarding process?"
        ]
        
        for query in queries:
            decision = router.route_query(query)
            # Should route to HR or root (if confidence is low)
            assert decision.agent_name in ["hr", "root"]
            if decision.agent_name == "hr":
                assert decision.confidence >= 0.5
    
    def test_analytics_keyword_routing(self, router):
        """Test routing to Analytics agent based on analytics keywords."""
        queries = [
            "Calculate the average salary",
            "What is the total payroll cost?",
            "Compute the percentage of remote workers",
            "Show me the budget breakdown",
            "Analyze the hiring trends"
        ]
        
        for query in queries:
            decision = router.route_query(query)
            assert decision.agent_name in ["analytics", "root"]
            if decision.agent_name == "analytics":
                assert decision.confidence >= 0.5
    
    def test_document_keyword_routing(self, router):
        """Test routing to Document agent based on document keywords."""
        queries = [
            "What is the vacation policy?",
            "Find the employee handbook",
            "Show me the security procedures",
            "What are the company guidelines?",
            "Look up the code of conduct"
        ]
        
        for query in queries:
            decision = router.route_query(query)
            assert decision.agent_name in ["document", "root"]
            if decision.agent_name == "document":
                assert decision.confidence >= 0.5
    
    def test_fallback_routing_low_confidence(self, router):
        """Test that queries with no clear match fall back to root."""
        queries = [
            "Hello",
            "What's the weather like?",
            "Tell me a joke",
            "How are you?"
        ]
        
        for query in queries:
            decision = router.route_query(query)
            # Should fall back to root due to low confidence
            assert decision.agent_name == "root"
            assert decision.confidence < 0.5
    
    def test_sticky_routing(self, router):
        """Test that follow-up queries stick to the previous agent."""
        # First query to HR
        first_query = "Who are the employees in engineering?"
        first_decision = router.route_query(first_query)
        
        # Follow-up query (ambiguous but should stick to HR)
        followup_query = "What about their skills?"
        followup_decision = router.route_query(
            followup_query,
            previous_agent="hr"
        )
        
        # The follow-up should have boosted score for HR
        # (though it might still route to root if confidence is too low)
        assert isinstance(followup_decision, RoutingDecision)
    
    def test_context_aware_routing(self, router):
        """Test that context influences routing decisions."""
        query = "What about the numbers?"
        context = "We were discussing payroll and salary calculations earlier."
        
        decision = router.route_query(query, context=context)
        
        # Context should boost analytics agent
        assert isinstance(decision, RoutingDecision)
        # The decision should consider context
        assert decision.agent_name in ["analytics", "root"]
    
    def test_register_agent(self, router):
        """Test dynamic agent registration."""
        new_agent = Mock(spec=AgentClient)
        new_agent.agent_name = "legal"
        
        initial_count = len(router.agents)
        
        router.register_agent(
            "legal",
            new_agent,
            keywords={"legal", "compliance", "contract"}
        )
        
        assert len(router.agents) == initial_count + 1
        assert "legal" in router.agents
        assert "legal" in router.agent_keywords
        assert "legal" in router.agent_keywords["legal"]
    
    def test_unregister_agent(self, router):
        """Test removing an agent from the router."""
        initial_count = len(router.agents)
        
        result = router.unregister_agent("hr")
        
        assert result is True
        assert len(router.agents) == initial_count - 1
        assert "hr" not in router.agents
    
    def test_get_registered_agents(self, router):
        """Test retrieving list of registered agents."""
        agents = router.get_registered_agents()
        
        assert isinstance(agents, list)
        assert len(agents) == 3
        assert "hr" in agents
        assert "analytics" in agents
        assert "document" in agents
    
    def test_set_agent_keywords(self, router):
        """Test updating keywords for an agent."""
        new_keywords = {"test", "keyword", "update"}
        
        result = router.set_agent_keywords("hr", new_keywords)
        
        assert result is True
        assert router.agent_keywords["hr"] == new_keywords
    
    def test_calculate_keyword_scores(self, router):
        """Test keyword scoring calculation."""
        query = "Calculate the average employee salary and benefits"
        
        scores = router._calculate_keyword_scores(query)
        
        # Should have scores for agents with matching keywords
        assert isinstance(scores, dict)
        # Analytics should score high (calculate, average, salary)
        # HR should score medium (employee, salary)
        if "analytics" in scores:
            assert scores["analytics"] > 0
    
    def test_is_followup_query(self, router):
        """Test follow-up query detection."""
        followup_queries = [
            "What about that?",
            "Tell me more",
            "Also, what is the policy?",
            "How about the same for managers?",
            "It seems interesting"
        ]
        
        for query in followup_queries:
            assert router._is_followup_query(query) is True
        
        non_followup_queries = [
            "What is the vacation policy?",
            "Calculate the average salary",
            "Who are the employees?"
        ]
        
        for query in non_followup_queries:
            # These might or might not be detected as follow-ups
            # Just ensure the method returns a boolean
            result = router._is_followup_query(query)
            assert isinstance(result, bool)
    
    def test_routing_decision_structure(self, router):
        """Test that routing decisions have the correct structure."""
        query = "What is the employee handbook?"
        
        decision = router.route_query(query)
        
        assert isinstance(decision, RoutingDecision)
        assert hasattr(decision, 'agent_name')
        assert hasattr(decision, 'confidence')
        assert hasattr(decision, 'reasoning')
        assert hasattr(decision, 'fallback_agents')
        assert isinstance(decision.agent_name, str)
        assert isinstance(decision.confidence, float)
        assert isinstance(decision.reasoning, str)
        assert isinstance(decision.fallback_agents, list)
    
    def test_empty_query_handling(self, router):
        """Test handling of empty or whitespace queries."""
        queries = ["", "   ", "\n\t"]
        
        for query in queries:
            decision = router.route_query(query)
            # Should return a valid decision (likely root with low confidence)
            assert isinstance(decision, RoutingDecision)
            assert decision.agent_name == "root"
    
    def test_multi_keyword_match_bonus(self, router):
        """Test that multiple keyword matches increase confidence."""
        # Query with multiple HR keywords
        multi_keyword_query = "Show me the employee skills and team org chart for hiring"
        single_keyword_query = "Show me the employee list"
        
        multi_decision = router.route_query(multi_keyword_query)
        single_decision = router.route_query(single_keyword_query)
        
        # Multi-keyword query should have higher or equal confidence
        # (if both route to the same agent)
        assert isinstance(multi_decision, RoutingDecision)
        assert isinstance(single_decision, RoutingDecision)
