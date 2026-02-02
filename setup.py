#!/usr/bin/env python3
"""
VintedOS Setup Script

Quick setup for API credentials.
Usage:
    python setup.py                           # Interactive mode
    python setup.py --api-key YOUR_KEY        # Direct mode
"""

import argparse
import sys
from pathlib import Path


def setup_env_file(api_key: str, force: bool = False) -> bool:
    """
    Create or update .env file with API key.
    
    Args:
        api_key: The Gemini API key
        force: Overwrite existing .env file
    
    Returns:
        True if successful
    """
    root_dir = Path(__file__).parent
    env_file = root_dir / ".env"
    env_example = root_dir / ".env.example"
    
    # Check if .env already exists
    if env_file.exists() and not force:
        print(f"⚠️  .env file already exists at {env_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Read template
    if not env_example.exists():
        print(f"❌ Error: .env.example template not found at {env_example}")
        return False
    
    template = env_example.read_text()
    
    # Replace placeholder with actual key
    env_content = template.replace(
        "GEMINI_API_KEY=your_gemini_api_key_here",
        f"GEMINI_API_KEY={api_key}"
    )
    
    # Write .env file
    env_file.write_text(env_content)
    print(f"✅ Created {env_file}")
    print(f"✅ API key configured successfully!")
    
    return True


def interactive_setup():
    """Interactive setup mode."""
    print("=" * 60)
    print("VintedOS Setup - API Configuration")
    print("=" * 60)
    print()
    print("Get your free Gemini API key from:")
    print("👉 https://aistudio.google.com/app/apikey")
    print()
    
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ Error: API key cannot be empty")
        return False
    
    if api_key == "your_gemini_api_key_here":
        print("❌ Error: Please provide a real API key, not the placeholder")
        return False
    
    print()
    return setup_env_file(api_key)


def main():
    parser = argparse.ArgumentParser(
        description="VintedOS Setup - Configure API credentials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                                    # Interactive mode
  python setup.py --api-key YOUR_KEY                 # Direct mode
  python setup.py --api-key YOUR_KEY --force         # Overwrite existing
        """
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .env file without asking"
    )
    
    args = parser.parse_args()
    
    # Direct mode with --api-key
    if args.api_key:
        success = setup_env_file(args.api_key, force=args.force)
        sys.exit(0 if success else 1)
    
    # Interactive mode
    success = interactive_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
