"""Quick test of Supabase connection."""
import os
from dotenv import load_dotenv

# Load .env from backend directory
load_dotenv()

print("Testing Supabase Connection")
print("=" * 70)

# Check environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print(f"\nSUPABASE_URL: {supabase_url[:30]}..." if supabase_url else "SUPABASE_URL: Not set")
print(f"SUPABASE_KEY: {supabase_key[:20]}..." if supabase_key else "SUPABASE_KEY: Not set")
print(f"SUPABASE_SERVICE_ROLE_KEY: {supabase_service_key[:20]}..." if supabase_service_key else "SUPABASE_SERVICE_ROLE_KEY: Not set")

if not supabase_url or not supabase_key:
    print("\n❌ Supabase credentials not found!")
    exit(1)

print("\n✓ Credentials found!")

# Test connection
try:
    from supabase import create_client
    
    print("\nConnecting to Supabase...")
    client = create_client(supabase_url, supabase_service_key or supabase_key)
    
    print("✓ Connected successfully!")
    
    # Test query
    print("\nTesting database query...")
    result = client.table("users").select("count").execute()
    print(f"✓ Query successful! Users table is accessible.")
    
    # Test user creation first
    print("\nTesting user creation...")
    from datetime import datetime
    
    test_user_id = f"test_user_{datetime.now().timestamp()}"
    user_data = {
        "user_id": test_user_id,
        "username": "test_user",
        "email": "test@example.com",
        "metadata": {}
    }
    
    client.table("users").insert(user_data).execute()
    print(f"✓ Created test user: {test_user_id}")
    
    # Test session creation
    print("\nTesting session storage...")
    test_session_id = f"test_{datetime.now().timestamp()}"
    session_data = {
        "session_id": test_session_id,
        "user_id": test_user_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {"test": True},
        "is_expired": False
    }
    
    client.table("sessions").insert(session_data).execute()
    print(f"✓ Created test session: {test_session_id}")
    
    # Retrieve it
    result = client.table("sessions").select("*").eq("session_id", test_session_id).execute()
    if result.data:
        print(f"✓ Retrieved test session successfully!")
    
    # Clean up
    client.table("sessions").delete().eq("session_id", test_session_id).execute()
    print(f"✓ Cleaned up test session")
    
    client.table("users").delete().eq("user_id", test_user_id).execute()
    print(f"✓ Cleaned up test user")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Supabase is working correctly.")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    exit(1)
