"""
Example usage of the Organization Guardrail system.

This demonstrates how to use the guardrail both standalone and integrated
with the RootChatbot.
"""
import asyncio
from backend.core.guardrail import create_organization_guardrail


def example_standalone_usage():
    """Example: Using guardrail standalone"""
    print("=" * 70)
    print("Example 1: Standalone Guardrail Usage")
    print("=" * 70)
    print()
    
    # Create guardrail
    guardrail = create_organization_guardrail()
    
    # Test various messages
    test_messages = [
        "Should I invest in Bitcoin?",
        "Who is the CTO?",
        "John is a terrible employee",
        "What's the weather today?",
        "Show me the org chart"
    ]
    
    for message in test_messages:
        result = guardrail.check(message)
        
        print(f"Message: {message}")
        print(f"Status: {'✓ ALLOWED' if result.is_safe else '✗ BLOCKED'}")
        
        if not result.is_safe:
            print(f"Violation: {result.violation_type.value}")
            print(f"Response: {result.intervention_message}")
        
        print()


def example_custom_configuration():
    """Example: Custom guardrail configuration"""
    print("=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)
    print()
    
    # Create guardrail with only financial check enabled
    guardrail = create_organization_guardrail(
        enable_financial_check=True,
        enable_negative_comment_check=False,
        enable_scope_check=False
    )
    
    print("Configuration: Only financial advice check enabled")
    print()
    
    # This will be blocked
    result1 = guardrail.check("Should I buy stocks?")
    print(f"'Should I buy stocks?' -> {'BLOCKED' if not result1.is_safe else 'ALLOWED'}")
    
    # This will be allowed (negative check disabled)
    result2 = guardrail.check("John is terrible")
    print(f"'John is terrible' -> {'BLOCKED' if not result2.is_safe else 'ALLOWED'}")
    
    # This will be allowed (scope check disabled)
    result3 = guardrail.check("What's the weather?")
    print(f"'What's the weather?' -> {'BLOCKED' if not result3.is_safe else 'ALLOWED'}")
    print()


async def example_with_root_chatbot():
    """Example: Guardrail integrated with RootChatbot"""
    print("=" * 70)
    print("Example 3: Integration with RootChatbot")
    print("=" * 70)
    print()
    
    # Note: This is a conceptual example. In production, you would have
    # actual BedrockIntegration, SessionManager, and AgentRouter instances.
    
    print("When integrated with RootChatbot:")
    print("1. User sends message")
    print("2. Guardrail checks message BEFORE processing")
    print("3. If blocked: Return intervention message immediately")
    print("4. If allowed: Continue to routing and agent processing")
    print()
    
    print("Example blocked message flow:")
    print("  User: 'Should I invest in Bitcoin?'")
    print("  Guardrail: BLOCKED (financial_advice)")
    print("  Response: 'I'm not able to provide financial advice...'")
    print()
    
    print("Example allowed message flow:")
    print("  User: 'Who is the CTO?'")
    print("  Guardrail: ALLOWED")
    print("  → Routes to HR agent")
    print("  Response: 'The CTO is Jennifer Lee...'")
    print()


def main():
    """Run all examples"""
    example_standalone_usage()
    example_custom_configuration()
    asyncio.run(example_with_root_chatbot())
    
    print("=" * 70)
    print("For more information, see:")
    print("  - backend/core/GUARDRAIL_README.md")
    print("  - backend/tests/test_guardrail.py")
    print("  - backend/scripts/deploy_guardrails.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
