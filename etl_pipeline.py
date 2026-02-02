#!/usr/bin/env python3
"""
VintedOS CLI Entry Point

Usage:
    python run.py                 # Production mode (actual printing)
    python run.py --dry-run       # Test mode (save to debug folder)
    python run.py --demo          # Demo mode (uses demo database)
    python run.py --help          # Show this help message

Modes:
    PRODUCTION: Connects to Gmail, processes emails, prints to physical printer
    DRY-RUN:    Uses production database but saves PDFs instead of printing
    DEMO:       Uses demo database with sample data, saves PDFs (safe testing)

First time setup:
    1. Initialize demo database: python tests/init_demo_db.py
    2. Test the pipeline:        python run.py --demo
    3. Check DEMO_MODE.md for detailed instructions
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.main import main

if __name__ == "__main__":
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    demo_mode = "--demo" in sys.argv
    
    print("\n" + "="*60)
    print("VintedOS - P2P Commerce ETL Pipeline")
    print("="*60)
    
    if demo_mode:
        print("Mode: DEMO (using demo database)")
        dry_run = True  # Demo mode implies dry-run
    elif dry_run:
        print("Mode: DRY-RUN (debugging - no actual printing)")
    else:
        print("Mode: PRODUCTION (will print to physical printer)")
    
    print("="*60 + "\n")
    
    try:
        summary = main(dry_run=dry_run, demo_mode=demo_mode)
        
        # Print summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        sys.exit(0 if summary.get("success") else 1)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        sys.exit(1)
