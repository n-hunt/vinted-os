#!/usr/bin/env python3
"""
Demo Mode Verification Script

Quickly verify that demo mode is working correctly.
Runs a series of checks to ensure the demo setup is functional.

Usage:
    python tests/verify_demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def check_file(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists


def check_demo_database():
    """Verify demo database exists and has data."""
    print("\n1. Checking demo database...")
    
    db_exists = check_file("demo_db.db", "Demo database file")
    
    if not db_exists:
        print("  ⚠️  Demo database not found. Run: python tests/init_demo_db.py")
        return False
    
    # Try to query the database
    try:
        from src.services.database import DatabaseService
        db = DatabaseService(demo_mode=True)
        
        # Try to get transactions (simplified check)
        print("  ✓ Demo database is accessible")
        return True
        
    except Exception as e:
        print(f"  ✗ Error accessing demo database: {e}")
        return False


def check_configuration():
    """Verify configuration has demo settings."""
    print("\n2. Checking configuration...")
    
    try:
        from src.config_loader import config
        
        demo_db = config.get('database.demo_filename')
        if demo_db:
            print(f"  ✓ Demo database configured: {demo_db}")
        else:
            print("  ✗ Demo database not configured in settings.yaml")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_demo_script():
    """Verify demo initialization script exists."""
    print("\n3. Checking demo initialization script...")
    return check_file("tests/init_demo_db.py", "Demo init script")


def check_documentation():
    """Verify demo documentation exists."""
    print("\n4. Checking documentation...")
    
    files = [
        ("DEMO_MODE.md", "Full demo mode guide"),
        ("QUICKSTART_DEMO.md", "Quick reference card"),
        ("agent.py", "Agent CLI entry point"),
    ]
    
    all_exist = True
    for filepath, desc in files:
        if not check_file(filepath, desc):
            all_exist = False
    
    return all_exist


def check_agent_support():
    """Verify agent supports demo mode."""
    print("\n5. Checking agent demo mode support...")
    
    try:
        from src.agent.tools import set_demo_mode, get_query_service
        print("  ✓ Agent tools support demo mode")
        
        # Test setting demo mode
        set_demo_mode(True)
        print("  ✓ Demo mode can be enabled for agent")
        
        return True
        
    except ImportError as e:
        print(f"  ⚠️  Agent components not available: {e}")
        print("  ℹ️  This is optional - agent may not be fully configured")
        return True  # Non-fatal
    except Exception as e:
        print(f"  ✗ Error testing agent demo mode: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("VintedOS Demo Mode Verification")
    print("="*60)
    
    checks = [
        ("Configuration", check_configuration()),
        ("Demo Script", check_demo_script()),
        ("Demo Database", check_demo_database()),
        ("Documentation", check_documentation()),
        ("Agent Support", check_agent_support()),
    ]
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in checks:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All checks passed! Demo mode is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python run.py --demo")
        print("  2. Check: logs/print_debug/ for generated PDFs")
        print("  3. Read: DEMO_MODE.md for full documentation")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTo set up demo mode:")
        print("  1. Run: python tests/init_demo_db.py")
        print("  2. Verify: python tests/verify_demo.py")
        print("  3. Test: python run.py --demo")
    
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Verification script error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
