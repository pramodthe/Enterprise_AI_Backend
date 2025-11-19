"""
Tests for the organization guardrail.
"""
import pytest
from backend.core.guardrail import (
    create_organization_guardrail,
    GuardrailViolationType,
    OrganizationGuardrail
)


class TestOrganizationGuardrail:
    """Test suite for OrganizationGuardrail"""
    
    def test_financial_advice_blocked(self):
        """Test that financial advice requests are blocked"""
        guardrail = create_organization_guardrail()
        
        test_cases = [
            "Should I invest in stocks?",
            "What's the best cryptocurrency to buy?",
            "How should I allocate my portfolio?",
            "Is it a good time to trade forex?",
            "Should I buy Bitcoin or Ethereum?",
        ]
        
        for message in test_cases:
            result = guardrail.check(message)
            assert not result.is_safe, f"Expected '{message}' to be blocked"
            assert result.violation_type == GuardrailViolationType.FINANCIAL_ADVICE
            assert "financial" in result.intervention_message.lower()
    
    def test_negative_employee_comments_blocked(self):
        """Test that negative employee comments are blocked"""
        guardrail = create_organization_guardrail()
        
        test_cases = [
            "John is a terrible employee",
            "Which employee is the most incompetent?",
            "Sarah is lazy and useless",
            "That staff member is an idiot",
        ]
        
        for message in test_cases:
            result = guardrail.check(message)
            assert not result.is_safe, f"Expected '{message}' to be blocked"
            assert result.violation_type == GuardrailViolationType.NEGATIVE_EMPLOYEE_COMMENT
    
    def test_out_of_scope_blocked(self):
        """Test that out-of-scope questions are blocked"""
        guardrail = create_organization_guardrail()
        
        test_cases = [
            "What's the weather like today?",
            "Who won the Super Bowl?",
            "Tell me a joke",
            "What's the capital of France?",
            "How do I make a cake?",
        ]
        
        for message in test_cases:
            result = guardrail.check(message)
            assert not result.is_safe, f"Expected '{message}' to be blocked"
            assert result.violation_type == GuardrailViolationType.OUT_OF_SCOPE
            assert "organization" in result.intervention_message.lower()
    
    def test_valid_organization_queries_allowed(self):
        """Test that valid organization queries are allowed"""
        guardrail = create_organization_guardrail()
        
        test_cases = [
            "Who is the manager of John Smith?",
            "What skills does Sarah have?",
            "Show me the organizational structure",
            "What are the company policies?",
            "Who reports to the CTO?",
            "Find employees with Python skills",
        ]
        
        for message in test_cases:
            result = guardrail.check(message)
            assert result.is_safe, f"Expected '{message}' to be allowed"
            assert result.violation_type == GuardrailViolationType.NONE
    
    def test_greetings_allowed(self):
        """Test that greetings and general queries are allowed"""
        guardrail = create_organization_guardrail()
        
        test_cases = [
            "Hello",
            "Hi, how are you?",
            "What can you help me with?",
            "Thanks!",
            "Goodbye",
        ]
        
        for message in test_cases:
            result = guardrail.check(message)
            assert result.is_safe, f"Expected '{message}' to be allowed"
            assert result.violation_type == GuardrailViolationType.NONE
    
    def test_disable_financial_check(self):
        """Test that financial check can be disabled"""
        guardrail = create_organization_guardrail(enable_financial_check=False)
        
        result = guardrail.check("Should I invest in stocks?")
        # Should not be blocked for financial advice when check is disabled
        # (might still be blocked for other reasons)
        assert result.violation_type != GuardrailViolationType.FINANCIAL_ADVICE
    
    def test_disable_negative_check(self):
        """Test that negative comment check can be disabled"""
        guardrail = create_organization_guardrail(enable_negative_comment_check=False)
        
        result = guardrail.check("John is a terrible employee")
        # Should not be blocked for negative comments when check is disabled
        assert result.violation_type != GuardrailViolationType.NEGATIVE_EMPLOYEE_COMMENT
    
    def test_disable_scope_check(self):
        """Test that scope check can be disabled"""
        guardrail = create_organization_guardrail(enable_scope_check=False)
        
        result = guardrail.check("What's the weather today?")
        # Should not be blocked for scope when check is disabled
        assert result.violation_type != GuardrailViolationType.OUT_OF_SCOPE
    
    def test_intervention_messages(self):
        """Test that appropriate intervention messages are returned"""
        guardrail = create_organization_guardrail()
        
        # Financial advice
        result = guardrail.check("Should I buy stocks?")
        assert "financial advisor" in result.intervention_message.lower()
        
        # Negative comment
        result = guardrail.check("John is a terrible employee")
        assert "cannot engage" in result.intervention_message.lower()
        
        # Out of scope
        result = guardrail.check("What's the weather?")
        assert "organization" in result.intervention_message.lower()
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        guardrail = create_organization_guardrail()
        
        # Empty message
        result = guardrail.check("")
        assert result.is_safe  # Empty messages are allowed (will be handled elsewhere)
        
        # Very long message with organization content
        long_message = "employee " * 100 + "What are their skills?"
        result = guardrail.check(long_message)
        assert result.is_safe
        
        # Mixed content (organization + financial)
        result = guardrail.check("Who is the CFO and should I invest in stocks?")
        assert not result.is_safe  # Should be blocked for financial content
        assert result.violation_type == GuardrailViolationType.FINANCIAL_ADVICE
