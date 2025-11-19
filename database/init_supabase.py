"""
Supabase database initialization script.

This script initializes the Supabase database with the required schema.
Run this once when setting up a new Supabase project.
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def init_database():
    """Initialize Supabase database with schema."""
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        print("Please add them to your .env file")
        return False
    
    print(f"Connecting to Supabase at {supabase_url}...")
    
    try:
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Read schema file
        schema_path = Path(__file__).parent / "supabase_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        print("Schema loaded successfully")
        print("\nIMPORTANT: You need to run the SQL schema manually in Supabase SQL Editor")
        print("=" * 70)
        print("\nSteps to initialize your database:")
        print("1. Go to your Supabase project dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Create a new query")
        print("4. Copy the contents of backend/database/supabase_schema.sql")
        print("5. Paste and run the SQL")
        print("\nAlternatively, you can use the Supabase CLI:")
        print("  supabase db push")
        print("\n" + "=" * 70)
        
        # Test connection by checking if we can query
        print("\nTesting connection...")
        
        # Try to query users table (will fail if not created yet)
        try:
            result = supabase.table("users").select("count").execute()
            print("✓ Database tables are accessible")
            print(f"✓ Connection successful!")
            return True
        except Exception as e:
            if "relation" in str(e).lower() or "does not exist" in str(e).lower():
                print("⚠ Tables not yet created. Please run the schema SQL as described above.")
            else:
                print(f"⚠ Connection test failed: {e}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_tables():
    """Verify that all required tables exist."""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: Supabase credentials not found")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        
        required_tables = ["users", "sessions", "messages", "agent_metrics", "documents"]
        
        print("\nVerifying tables...")
        all_exist = True
        
        for table in required_tables:
            try:
                supabase.table(table).select("count").limit(1).execute()
                print(f"✓ {table}")
            except Exception as e:
                print(f"✗ {table} - {e}")
                all_exist = False
        
        if all_exist:
            print("\n✓ All required tables exist!")
            return True
        else:
            print("\n⚠ Some tables are missing. Please run the schema SQL.")
            return False
            
    except Exception as e:
        print(f"Error verifying tables: {e}")
        return False


if __name__ == "__main__":
    print("Supabase Database Initialization")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_tables()
    else:
        init_database()
