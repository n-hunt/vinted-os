#!/usr/bin/env python3
"""
VintedOS Configuration Script

Quick setup for API credentials.
Usage:
    python configure.py                           # Interactive mode
    python configure.py --api-key YOUR_KEY        # Direct mode
    python configure.py --model MODEL_NAME        # Change LLM model
"""

import argparse
import sys
from pathlib import Path
import re


# Official Gemini model names (as of February 2026)
VALID_GEMINI_MODELS = {
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
}


def is_valid_gemini_model(model_name: str) -> bool:
    """
    Check if model name is a valid Gemini model.
    
    Args:
        model_name: Model name to validate
    
    Returns:
        True if valid Gemini model
    """
    return model_name in VALID_GEMINI_MODELS


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


def update_model_config(model_name: str) -> bool:
    """
    Update the LLM model in config/settings.yaml.
    
    Args:
        model_name: The Gemini model name to use
    
    Returns:
        True if successful
    """
    # Validate model name
    if not is_valid_gemini_model(model_name):
        print(f"❌ Error: '{model_name}' is not a recognized Gemini model name.")
        print()
        print("Valid Gemini models:")
        for model in sorted(VALID_GEMINI_MODELS):
            print(f"  • {model}")
        print()
        print("Please use an official Gemini model name.")
        return False
    
    root_dir = Path(__file__).parent
    config_file = root_dir / "config" / "settings.yaml"
    
    if not config_file.exists():
        print(f"❌ Error: Configuration file not found at {config_file}")
        return False
    
    # Read config file
    content = config_file.read_text()
    
    # Update the model line using regex
    # Match the line: model: "gemini-..." with potential comment
    pattern = r'(    model: )"[^"]*"(.*?)$'
    replacement = rf'\1"{model_name}"\2'
    
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    
    if count == 0:
        print(f"❌ Error: Could not find model configuration in {config_file}")
        return False
    
    # Write updated config
    config_file.write_text(new_content)
    print(f"✅ Updated model to: {model_name}")
    print(f"✅ Configuration saved to {config_file}")
    
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
  python configure.py                                    # Interactive mode
  python configure.py --api-key YOUR_KEY                 # Direct mode
  python configure.py --api-key YOUR_KEY --force         # Overwrite existing
  python configure.py --model gemini-2.5-flash           # Change LLM model
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
    
    parser.add_argument(
        "--model",
        type=str,
        help="Gemini model name to use (e.g., gemini-2.5-flash, gemini-3-flash-preview)"
    )
    
    args = parser.parse_args()
    
    # Handle model change
    if args.model:
        success = update_model_config(args.model)
        sys.exit(0 if success else 1)
    
    # Direct mode with --api-key
    if args.api_key:
        success = setup_env_file(args.api_key, force=args.force)
        sys.exit(0 if success else 1)
    
    # Interactive mode
    success = interactive_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
