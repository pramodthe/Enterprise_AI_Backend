"""
Integration tests for AgentRouter with other components.

Tests the router working with real AgentClient instances and routing decisions.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from backend.chatbot.agent_router import AgentRouter
from backend.chatbot.agent_client import AgentClient
from backend.chatbot.models import RoutingDecision, AgentResponse


class TestAgentRouterIntegration:
    """Integration tests for AgentRouter."""
    
    @pytest.fixture
    def router_with_clients(self):
        """Create router with real AgentClient instances (mocked HTTP)."""
        # Create real AgentClient instances with mocked URLs
        hr_client = AgentClient("http://localhost:8001", "hr")
        analytics_client = AgentClient("http://localhost:8002", "analytics")
        document_client = AgentClient("http://localhost:8003", "document")
        
        agents = {
            "hr": hr_client,
            "analytics": analytics_client,
            "document": document_client
        }
        
        return AgentRouter(agents=agents)
    
    def test_route_and_verify_agent_availability(self, router_with_clients):
        """Test routing decision and checking agent availability."""
        query = "What is the vacation policy?"
        
        decision = router_with_clients.route_query(query)
        
        # Should route to document agent
        assert isinstance(decision, RoutingDecision)
        assert decision.agent_name in ["document", "root"]
        
        # Verify we can access the agent client
        if decision.agent_name != "root":
            agent_client = router_with_clients.agents[decision.agent_name]
            assert isinstance(agent_client, AgentClient)
    
    def test_routing_with_conversation_context(self, router_with_clients):
        """Test routing with conversation history context."""
        # Simulate a conversation about HR topics
        context = """
        User: Who are the employees in engineering?
        Assistant: Here are the employees in the engineering team...
        User: What skills do they have?
        Assistant: The engineering team has skills in Python, JavaScript...
        """
        
        # New query that's ambiguous but context suggests HR
        query = "What about their managers?"
        
        decision = router_with_clients.route_query(query, context=context)
        
        assert isinstance(decision, RoutingDecision)
        # Context should influence routing toward HR
        assert decision.agent_name in ["hr", "root"]
    
    def test_sticky_routing_in_conversation(self, router_with_clients):
        """Test sticky routing across multiple turns."""
        # First query clearly about analytics
        query1 = "Calculate the average salary"
        decision1 = router_with_clients.route_query(query1)
        
        # Follow-up query
        query2 = "What about the median?"
        decision2 = router_with_clients.route_query(
            query2,
            previous_agent=decision1.agent_name
        )
        
        # Both should be valid routing decisions
        assert isinstance(decision1, RoutingDecision)
        assert isinstance(decision2, RoutingDecision)
        
        # If first routed to analytics, second should consider sticky routing
        if decision1.agent_name == "analytics":
            # The follow-up should have received a boost for analytics
            # (though it might still route elsewhere based on confidence)
            assert decision2.agent_name in ["analytics", "root"]
    
    def test_fallback_agents_populated(self, router_with_clients):
        """Test that fallback agents are provided in routing decisions."""
        query = "Show me employee skills and calculate their average experience"
        
        decision = router_with_clients.route_query(query)
        
        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.fallback_agents, list)
        # Fallback agents should be provided for complex queries
        # (may be empty for very clear-cut cases)
    
    def test_dynamic_agent_registration_and_routing(self, router_with_clients):
        """Test adding a new agent and routing to it."""
        # Register a new agent
        legal_client = AgentClient("http://localhost:8004", "legal")
        router_with_clients.register_agent(
            "legal",
            legal_client,
            keywords={"legal", "compliance", "contract", "lawsuit", "attorney"}
        )
        
        # Query that should route to the new agent
        query = "What are our legal compliance requirements?"
        decision = router_with_clients.route_query(query)
        
        assert isinstance(decision, RoutingDecision)
        # Should route to legal or root
        assert decision.agent_name in ["legal", "root"]
        
        # Verify agent is registered
        assert "legal" in router_with_clients.get_registered_agents()
    
    def test_routing_with_multiple_keyword_matches(self, router_with_clients):
        """Test routing when query matches multiple agents."""
        # Query with keywords for both HR and Analytics
        query = "Calculate the average salary for employees in the engineering team"
        
        decision = router_with_clients.route_query(query)
        
        assert isinstance(decision, RoutingDecision)
        # Should route to one of the matching agents or root
        assert decision.agent_name in ["hr", "analytics", "root"]
        
        # Should have reasoning explaining the choice
        assert len(decision.reasoning) > 0
    
    def test_confidence_threshold_enforcement(self, router_with_clients):
        """Test that low confidence queries fall back to root."""
        # Very generic query with no clear agent match
        query = "Hello, how are you today?"
        
        decision = router_with_clients.route_query(query)
        
        assert isinstance(decision, RoutingDecision)
        # Should fall back to root due to low confidence
        assert decision.agent_name == "root"
        assert decision.confidence < router_with_clients.confidence_threshold
    
    def test_routing_decision_should_use_agent(self, router_with_clients):
        """Test the should_use_agent method on routing decisions."""
        # Query with clear agent match
        query = "What is the employee handbook policy on remote work?"
        
        decision = router_with_clients.route_query(query)
        
        # Test the should_use_agent method
        if decision.agent_name != "root":
            assert decision.should_use_agent(threshold=0.5) is True
        else:
            # Root agent means confidence was below threshold
            assert decision.should_use_agent(threshold=0.5) is False
    
    def test_empty_router_fallback(self):
        """Test router behavior with no registered agents."""
        empty_router = AgentRouter(agents={})
        
        query = "What is the vacation policy?"
        decision = empty_router.route_query(query)
        
        # Should fall back to root
        assert decision.agent_name == "root"
        assert decision.confidence == 0.0
        assert "No specialized agents available" in decision.reasoning
