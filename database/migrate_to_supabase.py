"""
Migration script to move data from existing storage to Supabase.

This script helps migrate sessions and data from InMemoryStorageBackend
or RedisStorageBackend to SupabaseStorageBackend.
"""
import sys
from pathlib import Path
from typing import List
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chatbot.storage import (
    InMemoryStorageBackend,
    RedisStorageBackend,
    SupabaseStorageBackend
)
from backend.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_from_memory_to_supabase(
    source: InMemoryStorageBackend,
    target: SupabaseStorageBackend
) -> tuple[int, int]:
    """
    Migrate sessions from in-memory storage to Supabase.
    
    Args:
        source: Source in-memory storage backend
        target: Target Supabase storage backend
        
    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    logger.info("Starting migration from InMemoryStorageBackend to Supabase...")
    
    # Get all session IDs
    session_ids = source.list_sessions()
    logger.info(f"Found {len(session_ids)} sessions to migrate")
    
    successful = 0
    failed = 0
    
    for session_id in session_ids:
        try:
            # Load session data from source
            session_data = source.load_session(session_id)
            
            if not session_data:
                logger.warning(f"Could not load session {session_id}, skipping")
                failed += 1
                continue
            
            # Save to target
            target.save_session(session_id, session_data)
            logger.info(f"✓ Migrated session {session_id}")
            successful += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to migrate session {session_id}: {e}")
            failed += 1
    
    logger.info(f"\nMigration complete: {successful} successful, {failed} failed")
    return successful, failed


def migrate_from_redis_to_supabase(
    source: RedisStorageBackend,
    target: SupabaseStorageBackend
) -> tuple[int, int]:
    """
    Migrate sessions from Redis storage to Supabase.
    
    Args:
        source: Source Redis storage backend
        target: Target Supabase storage backend
        
    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    logger.info("Starting migration from RedisStorageBackend to Supabase...")
    
    # Get all session IDs
    session_ids = source.list_sessions()
    logger.info(f"Found {len(session_ids)} sessions to migrate")
    
    successful = 0
    failed = 0
    
    for session_id in session_ids:
        try:
            # Load session data from source
            session_data = source.load_session(session_id)
            
            if not session_data:
                logger.warning(f"Could not load session {session_id}, skipping")
                failed += 1
                continue
            
            # Save to target
            target.save_session(session_id, session_data)
            logger.info(f"✓ Migrated session {session_id}")
            successful += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to migrate session {session_id}: {e}")
            failed += 1
    
    logger.info(f"\nMigration complete: {successful} successful, {failed} failed")
    return successful, failed


def verify_migration(
    source_session_ids: List[str],
    target: SupabaseStorageBackend
) -> tuple[int, int]:
    """
    Verify that all sessions were migrated successfully.
    
    Args:
        source_session_ids: List of session IDs from source
        target: Target Supabase storage backend
        
    Returns:
        Tuple of (verified_count, missing_count)
    """
    logger.info("\nVerifying migration...")
    
    verified = 0
    missing = 0
    
    for session_id in source_session_ids:
        session_data = target.load_session(session_id)
        if session_data:
            verified += 1
        else:
            logger.warning(f"✗ Session {session_id} not found in target")
            missing += 1
    
    logger.info(f"Verification complete: {verified} verified, {missing} missing")
    return verified, missing


def main():
    """Main migration script."""
    print("="*70)
    print("Migration Script: Moving Data to Supabase")
    print("="*70)
    
    # Check Supabase configuration
    if not settings.supabase_url or not settings.supabase_key:
        print("\n✗ Error: Supabase not configured")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        return
    
    # Create target Supabase storage
    target = SupabaseStorageBackend(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_role_key or settings.supabase_key
    )
    
    print("\nSelect source storage type:")
    print("1. In-Memory Storage")
    print("2. Redis Storage")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n⚠ Warning: In-memory storage is empty unless you have a running instance")
        print("This option is mainly for testing the migration process")
        confirm = input("Continue? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("Migration cancelled")
            return
        
        source = InMemoryStorageBackend()
        successful, failed = migrate_from_memory_to_supabase(source, target)
        
        if successful > 0:
            session_ids = source.list_sessions()
            verify_migration(session_ids, target)
    
    elif choice == "2":
        redis_url = input("Enter Redis URL (default: redis://localhost:6379): ").strip()
        if not redis_url:
            redis_url = "redis://localhost:6379"
        
        try:
            source = RedisStorageBackend(redis_url=redis_url)
            
            # Preview sessions
            session_ids = source.list_sessions()
            print(f"\nFound {len(session_ids)} sessions in Redis")
            
            if len(session_ids) == 0:
                print("No sessions to migrate")
                return
            
            # Show sample
            if len(session_ids) > 0:
                print(f"\nSample session IDs:")
                for sid in session_ids[:5]:
                    print(f"  - {sid}")
                if len(session_ids) > 5:
                    print(f"  ... and {len(session_ids) - 5} more")
            
            confirm = input(f"\nMigrate {len(session_ids)} sessions to Supabase? (yes/no): ").strip().lower()
            
            if confirm != "yes":
                print("Migration cancelled")
                return
            
            successful, failed = migrate_from_redis_to_supabase(source, target)
            
            if successful > 0:
                verify_migration(session_ids, target)
            
        except Exception as e:
            print(f"\n✗ Error connecting to Redis: {e}")
            print("Make sure Redis is running and the URL is correct")
            return
    
    elif choice == "3":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*70)
    print("Migration Summary")
    print("="*70)
    print(f"✓ Successfully migrated: {successful} sessions")
    if failed > 0:
        print(f"✗ Failed migrations: {failed} sessions")
    print("\nNext steps:")
    print("1. Verify data in Supabase dashboard")
    print("2. Update your application to use SupabaseStorageBackend")
    print("3. Test thoroughly before decommissioning old storage")


if __name__ == "__main__":
    main()
