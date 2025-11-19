"""
Guardrail utilities for the Enterprise AI Assistant Platform.

This module provides guardrail functionality to prevent misuse of the root agent:
- No financial advice
- No negative comments about employees
- Only organization-relevant questions
"""
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class GuardrailViolationType(Enum):
    """Types of guardrail violations"""
    FINANCIAL_ADVICE = "financial_advice"
    NEGATIVE_EMPLOYEE_COMMENT = "negative_employee_comment"
    OUT_OF_SCOPE = "out_of_scope"
    NONE = "none"


class GuardrailResult:
    """Result of guardrail check"""
    
    def __init__(
        self,
        is_safe: bool,
        violation_type: GuardrailViolationType = GuardrailViolationType.NONE,
        reason: Optional[str] = None,
        intervention_message: Optional[str] = None
    ):
        self.is_safe = is_safe
        self.violation_type = violation_type
        self.reason = reason
        self.intervention_message = intervention_message or self._default_intervention_message()
    
    def _default_intervention_message(self) -> str:
        """Get default intervention message based on violation type"""
        messages = {
            GuardrailViolationType.FINANCIAL_ADVICE: (
                "I'm not able to provide financial advice. "
                "Please consult with a qualified financial advisor for investment or financial planning guidance."
            ),
            GuardrailViolationType.NEGATIVE_EMPLOYEE_COMMENT: (
                "I'm here to provide helpful information about our organization. "
                "I cannot engage in negative discussions about employees."
            ),
            GuardrailViolationType.OUT_OF_SCOPE: (
                "I'm sorry, I can only answer questions relevant to our organization. "
                "I can help with HR queries, analytics, and document-related questions."
            ),
            GuardrailViolationType.NONE: ""
        }
        return messages.get(self.violation_type, "I can't assist with that request.")


class OrganizationGuardrail:
    """
    Guardrail for the Enterprise AI Assistant to prevent misuse.
    
    Rules:
    1. No financial advice (investments, stocks, trading, etc.)
    2. No negative comments about employees
    3. Only organization-relevant questions
    """
    
    # Financial advice keywords
    FINANCIAL_KEYWORDS = [
        "invest", "investment", "stock", "stocks", "trading", "trade",
        "portfolio", "mutual fund", "etf", "bond", "bonds", "dividend",
        "crypto", "cryptocurrency", "bitcoin", "forex", "options",
        "financial advice", "should i buy", "should i sell", "market",
        "401k", "ira", "retirement fund", "hedge fund", "asset allocation"
    ]
    
    # Negative employee comment patterns
    NEGATIVE_PATTERNS = [
        "bad employee", "worst employee", "terrible", "incompetent",
        "useless", "lazy", "stupid", "idiot", "hate", "fire",
        "get rid of", "should be fired", "doesn't deserve", "awful",
        "pathetic", "worthless", "garbage", "trash"
    ]
    
    # Organization-relevant topics (allowed)
    ORGANIZATION_TOPICS = [
        "employee", "staff", "team", "department", "organization",
        "company", "hr", "human resources", "skill", "skills",
        "report", "analytics", "data", "document", "policy",
        "procedure", "guideline", "structure", "hierarchy", "manager",
        "who reports to", "org chart", "capabilities", "what can you do"
    ]
    
    def __init__(
        self,
        enable_financial_check: bool = True,
        enable_negative_comment_check: bool = True,
        enable_scope_check: bool = True
    ):
        """
        Initialize guardrail with configurable checks.
        
        Args:
            enable_financial_check: Check for financial advice requests
            enable_negative_comment_check: Check for negative employee comments
            enable_scope_check: Check if question is organization-relevant
        """
        self.enable_financial_check = enable_financial_check
        self.enable_negative_comment_check = enable_negative_comment_check
        self.enable_scope_check = enable_scope_check
        
        logger.info(
            f"Initialized OrganizationGuardrail: "
            f"financial={enable_financial_check}, "
            f"negative={enable_negative_comment_check}, "
            f"scope={enable_scope_check}"
        )
    
    def check(self, message: str, context: Optional[str] = None) -> GuardrailResult:
        """
        Check if a message violates guardrails.
        
        Args:
            message: User message to check
            context: Optional conversation context
            
        Returns:
            GuardrailResult indicating if message is safe
        """
        message_lower = message.lower()
        
        # Check 1: Financial advice
        if self.enable_financial_check:
            if self._contains_financial_request(message_lower):
                logger.warning(f"Guardrail violation: Financial advice request detected")
                return GuardrailResult(
                    is_safe=False,
                    violation_type=GuardrailViolationType.FINANCIAL_ADVICE,
                    reason="Message contains financial advice keywords"
                )
        
        # Check 2: Negative employee comments (pass original message for name detection)
        if self.enable_negative_comment_check:
            if self._contains_negative_employee_comment(message, message_lower):
                logger.warning(f"Guardrail violation: Negative employee comment detected")
                return GuardrailResult(
                    is_safe=False,
                    violation_type=GuardrailViolationType.NEGATIVE_EMPLOYEE_COMMENT,
                    reason="Message contains negative employee comments"
                )
        
        # Check 3: Out of scope (only if not a greeting/general query)
        if self.enable_scope_check:
            if not self._is_organization_relevant(message_lower):
                logger.warning(f"Guardrail violation: Out of scope question")
                return GuardrailResult(
                    is_safe=False,
                    violation_type=GuardrailViolationType.OUT_OF_SCOPE,
                    reason="Question is not organization-relevant"
                )
        
        # All checks passed
        return GuardrailResult(is_safe=True)
    
    def _contains_financial_request(self, message: str) -> bool:
        """Check if message contains financial advice request"""
        return any(keyword in message for keyword in self.FINANCIAL_KEYWORDS)
    
    def _contains_negative_employee_comment(self, message_original: str, message_lower: str) -> bool:
        """Check if message contains negative employee comments"""
        # Check if message mentions employees
        mentions_employee = any(
            word in message_lower 
            for word in ["employee", "staff", "worker", "person", "people", "team member"]
        )
        
        # Check if message has a name pattern (capitalized word) + negative pattern
        # This catches "John is terrible" or "Sarah is lazy"
        words = message_original.split()
        has_name_pattern = any(
            word[0].isupper() and len(word) > 1 
            for word in words 
            if word.isalpha()
        )
        
        # If mentions employee OR has name pattern, check for negative patterns
        if mentions_employee or has_name_pattern:
            return any(pattern in message_lower for pattern in self.NEGATIVE_PATTERNS)
        
        return False
    
    def _is_organization_relevant(self, message: str) -> bool:
        """
        Check if message is organization-relevant.
        
        Allows:
        - Questions about employees, HR, analytics, documents
        - General greetings and capability questions
        - Short conversational messages
        
        Blocks:
        - Unrelated topics (weather, sports, general knowledge, etc.)
        """
        # Allow short messages (greetings, thanks, etc.)
        if len(message.split()) <= 5:
            greeting_patterns = [
                "hello", "hi", "hey", "thanks", "thank you",
                "what can you do", "help", "capabilities",
                "how are you", "goodbye", "bye"
            ]
            if any(pattern in message for pattern in greeting_patterns):
                return True
        
        # Check if message contains organization-relevant topics
        is_relevant = any(
            topic in message 
            for topic in self.ORGANIZATION_TOPICS
        )
        
        if is_relevant:
            return True
        
        # Check for common out-of-scope patterns
        out_of_scope_patterns = [
            "weather", "sports", "recipe", "movie", "music", "game",
            "celebrity", "news", "politics", "religion", "joke",
            "story", "poem", "song", "translate", "definition of",
            "capital of", "population of", "history of", "who invented",
            "super bowl", "world cup", "olympics", "championship",
            "make a cake", "cook", "bake"
        ]
        
        if any(pattern in message for pattern in out_of_scope_patterns):
            return False
        
        # If we can't determine, be permissive (let the agent handle it)
        # This prevents false positives
        return True


def create_organization_guardrail(
    enable_financial_check: bool = True,
    enable_negative_comment_check: bool = True,
    enable_scope_check: bool = True
) -> OrganizationGuardrail:
    """
    Factory function to create an OrganizationGuardrail instance.
    
    Args:
        enable_financial_check: Check for financial advice requests
        enable_negative_comment_check: Check for negative employee comments
        enable_scope_check: Check if question is organization-relevant
        
    Returns:
        OrganizationGuardrail instance
    """
    return OrganizationGuardrail(
        enable_financial_check=enable_financial_check,
        enable_negative_comment_check=enable_negative_comment_check,
        enable_scope_check=enable_scope_check
    )
