"""
Example usage of Supabase storage and database operations.

This script demonstrates how to use the Supabase integration
for session management, user operations, and metrics tracking.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chatbot.storage import SupabaseStorageBackend
from backend.chatbot.session_manager import SessionManager
from backend.chatbot.models import Session, Message, MessageRole
from backend.database.operations import db_ops
from backend.core.config import settings


def example_session_management():
    """Example: Using Supabase for session management."""
    print("\n" + "="*70)
    print("Example 1: Session Management with Supabase")
    print("="*70)
    
    # Check if Supabase is configured
    if not settings.supabase_url or not settings.supabase_key:
        print("⚠ Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in .env")
        return
    
    # Create Supabase storage backend
    storage = SupabaseStorageBackend(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_role_key or settings.supabase_key
    )
    
    # Create session manager with Supabase storage
    session_manager = SessionManager(storage_backend=storage)
    
    # Create a new session
    print("\n1. Creating a new session...")
    session = session_manager.create_session(user_id="demo_user_123")
    print(f"   ✓ Created session: {session.session_id}")
    
    # Add messages to the session
    print("\n2. Adding messages to session...")
    session_manager.add_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content="Hello! Can you help me find employee information?",
        metadata={"source": "web_ui"}
    )
    print("   ✓ Added user message")
    
    session_manager.add_message(
        session_id=session.session_id,
        role=MessageRole.ASSISTANT,
        content="Of course! I can help you find employee information. What would you like to know?",
        agent_used="hr_agent",
        metadata={"confidence": 0.95}
    )
    print("   ✓ Added assistant message")
    
    # Retrieve the session
    print("\n3. Retrieving session from Supabase...")
    retrieved_session = session_manager.get_session(session.session_id)
    if retrieved_session:
        print(f"   ✓ Retrieved session with {len(retrieved_session.conversation_history)} messages")
        for msg in retrieved_session.conversation_history:
            print(f"     - {msg.role}: {msg.content[:50]}...")
    
    # List all sessions for user
    print("\n4. Listing all sessions for user...")
    user_sessions = storage.list_sessions(filters={"user_id": "demo_user_123"})
    print(f"   ✓ Found {len(user_sessions)} session(s) for user")
    
    return session.session_id


def example_user_operations():
    """Example: User management operations."""
    print("\n" + "="*70)
    print("Example 2: User Management")
    print("="*70)
    
    if not db_ops.is_available():
        print("⚠ Database operations not available")
        return
    
    # Create a user
    print("\n1. Creating a user...")
    success = db_ops.create_user(
        user_id="demo_user_123",
        username="demo_user",
        email="demo@example.com",
        full_name="Demo User",
        metadata={"department": "Engineering", "role": "Developer"}
    )
    if success:
        print("   ✓ User created successfully")
    else:
        print("   ℹ User may already exist")
    
    # Get user data
    print("\n2. Retrieving user data...")
    user = db_ops.get_user("demo_user_123")
    if user:
        print(f"   ✓ User: {user['username']} ({user['email']})")
        print(f"     Metadata: {user.get('metadata', {})}")
    
    # Update user
    print("\n3. Updating user data...")
    success = db_ops.update_user(
        "demo_user_123",
        {"full_name": "Demo User Updated", "metadata": {"department": "Product"}}
    )
    if success:
        print("   ✓ User updated successfully")
    
    # Get user sessions
    print("\n4. Getting user sessions...")
    sessions = db_ops.get_user_sessions("demo_user_123")
    print(f"   ✓ Found {len(sessions)} session(s)")


def example_agent_metrics():
    """Example: Tracking agent performance metrics."""
    print("\n" + "="*70)
    print("Example 3: Agent Performance Metrics")
    print("="*70)
    
    if not db_ops.is_available():
        print("⚠ Database operations not available")
        return
    
    # Save some sample metrics
    print("\n1. Saving agent metrics...")
    
    agents = ["hr_agent", "analytics_agent", "document_agent"]
    queries = [
        "Who is the CEO?",
        "Calculate 15% of 200",
        "What is our vacation policy?"
    ]
    
    for i, (agent, query) in enumerate(zip(agents, queries)):
        success = db_ops.save_agent_metric(
            agent_name=agent,
            query=query,
            response_time_ms=150 + (i * 50),
            success=True,
            confidence_score=0.85 + (i * 0.05),
            metadata={"model": "claude-3-sonnet"}
        )
        if success:
            print(f"   ✓ Saved metric for {agent}")
    
    # Get agent statistics
    print("\n2. Getting agent statistics (last 7 days)...")
    stats = db_ops.get_agent_statistics(days=7)
    
    if stats:
        print("\n   Agent Performance Summary:")
        print("   " + "-"*66)
        for agent, data in stats.items():
            print(f"\n   {agent}:")
            print(f"     Total Queries: {data['total_queries']}")
            print(f"     Success Rate: {data['success_rate']:.1%}")
            print(f"     Avg Response Time: {data['avg_response_time_ms']:.0f}ms")
            print(f"     Avg Confidence: {data['avg_confidence']:.2f}")
    else:
        print("   ℹ No metrics found")


def example_document_tracking():
    """Example: Document metadata tracking."""
    print("\n" + "="*70)
    print("Example 4: Document Metadata Tracking")
    print("="*70)
    
    if not db_ops.is_available():
        print("⚠ Database operations not available")
        return
    
    # Save document metadata
    print("\n1. Saving document metadata...")
    success = db_ops.save_document_metadata(
        document_id="doc_12345",
        filename="employee_handbook.pdf",
        file_type="pdf",
        file_size=2048576,  # 2MB
        uploaded_by="demo_user_123",
        metadata={"category": "HR", "version": "2024.1"}
    )
    if success:
        print("   ✓ Document metadata saved")
    
    # Mark as processed
    print("\n2. Marking document as processed...")
    success = db_ops.mark_document_processed(
        document_id="doc_12345",
        chunk_count=42
    )
    if success:
        print("   ✓ Document marked as processed (42 chunks)")
    
    # Get user documents
    print("\n3. Getting user's documents...")
    documents = db_ops.get_user_documents("demo_user_123")
    if documents:
        print(f"   ✓ Found {len(documents)} document(s):")
        for doc in documents:
            print(f"     - {doc['filename']} ({doc['file_type']}, {doc['file_size']} bytes)")
            print(f"       Processed: {doc['processed']}, Chunks: {doc['chunk_count']}")


def example_complete_workflow():
    """Example: Complete workflow combining all features."""
    print("\n" + "="*70)
    print("Example 5: Complete Workflow")
    print("="*70)
    
    if not settings.supabase_url or not settings.supabase_key:
        print("⚠ Supabase not configured")
        return
    
    print("\nSimulating a complete user interaction workflow...")
    
    # 1. User logs in (create/get user)
    print("\n1. User authentication...")
    db_ops.create_user(
        user_id="workflow_user",
        username="workflow_demo",
        email="workflow@example.com"
    )
    print("   ✓ User authenticated")
    
    # 2. Create session
    print("\n2. Creating chat session...")
    storage = SupabaseStorageBackend(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_role_key or settings.supabase_key
    )
    session_manager = SessionManager(storage_backend=storage)
    session = session_manager.create_session(user_id="workflow_user")
    print(f"   ✓ Session created: {session.session_id}")
    
    # 3. User sends query
    print("\n3. Processing user query...")
    start_time = datetime.now()
    
    session_manager.add_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content="What documents do I have access to?"
    )
    
    # Simulate agent processing
    response_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    session_manager.add_message(
        session_id=session.session_id,
        role=MessageRole.ASSISTANT,
        content="You have access to 3 documents: Employee Handbook, Benefits Guide, and Code of Conduct.",
        agent_used="document_agent"
    )
    print("   ✓ Query processed by document_agent")
    
    # 4. Save metrics
    print("\n4. Saving performance metrics...")
    db_ops.save_agent_metric(
        agent_name="document_agent",
        query="What documents do I have access to?",
        response_time_ms=response_time,
        success=True,
        session_id=session.session_id,
        confidence_score=0.92
    )
    print(f"   ✓ Metrics saved (response time: {response_time}ms)")
    
    # 5. Get session history
    print("\n5. Retrieving session history...")
    retrieved = session_manager.get_session(session.session_id)
    if retrieved:
        print(f"   ✓ Session has {len(retrieved.conversation_history)} messages")
    
    print("\n✓ Workflow complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Supabase Integration Examples")
    print("="*70)
    print("\nThese examples demonstrate the Supabase integration features.")
    print("Make sure you have configured SUPABASE_URL and SUPABASE_KEY in .env")
    
    try:
        # Run examples
        example_session_management()
        example_user_operations()
        example_agent_metrics()
        example_document_tracking()
        example_complete_workflow()
        
        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70)
        print("\nNext steps:")
        print("1. Check your Supabase dashboard to see the data")
        print("2. Explore the Table Editor to view users, sessions, messages")
        print("3. Try querying the agent_performance view for analytics")
        print("4. Integrate these patterns into your application")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nMake sure:")
        print("1. Supabase is configured in .env")
        print("2. Database schema has been initialized")
        print("3. You have internet connectivity")


if __name__ == "__main__":
    main()
