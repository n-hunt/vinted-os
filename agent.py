#!/usr/bin/env python3
"""
VintedOS RAG Agent - Interactive CLI

Interactive agent for querying the VintedOS database and knowledge base.
Perfect for exploring the system, testing queries, and reviewing capabilities.

Usage:
    python agent.py              # Production mode (uses production database)
    python agent.py --demo       # Demo mode (uses demo database - recommended for review)
    python agent.py --help       # Show this help message

Quick Start for Reviewers:
    1. Run: python agent.py --demo
    2. Try queries like:
       - "Show me recent transactions"
       - "What's the total revenue?"
       - "Are there any failed transactions?"
       - "Give me a dashboard summary"
    3. Type 'quit' to exit

The agent can:
- Answer questions about transactions and orders in natural language
- Provide business analytics and statistics
- Search through system documentation and logs
- Handle multi-turn conversations with context awareness
- Execute complex queries without SQL knowledge

For a guided demonstration, see: demo_rag_showcase.ipynb
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Main entry point for the agent CLI."""
    # Parse arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    demo_mode = "--demo" in sys.argv
    
    # Set demo mode BEFORE importing tools
    if demo_mode:
        from src.agent.tools import set_demo_mode
        set_demo_mode(True)
        print("\n🎯 Running in DEMO mode (using demo database)")
    else:
        print("\n⚙️  Running in PRODUCTION mode (using production database)")
    
    # Import agent components (after setting demo mode)
    try:
        from src.agent.core.llm_client import get_llm_client
        from src.agent.core.memory import create_memory
        from src.agent.core.react_loop import create_react_agent
        from src.agent.tools import ALL_QUERY_TOOLS
        from src.agent.core.router import create_router
    except ImportError as e:
        print(f"\n❌ Error importing agent components: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  Warning: GEMINI_API_KEY not set in environment")
        print("The agent requires an API key to function.")
        print("\nSet it with:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("VintedOS RAG Agent")
    print("="*60)
    print("\nInitializing agent components...")
    
    try:
        # Initialize components
        llm_client = get_llm_client()
        memory = create_memory(session_id="cli_session")
        router = create_router(llm_client=llm_client, tools=ALL_QUERY_TOOLS)
        agent = create_react_agent(
            llm_client=llm_client,
            tools=ALL_QUERY_TOOLS,
            memory=memory
        )
        
        print("✓ Agent initialized successfully")
        print("\n" + "="*60)
        print("Type your questions below. Type 'quit' or 'exit' to stop.")
        print("="*60 + "\n")
        
        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Route and respond
                print("\n🤖 Agent: ", end="", flush=True)
                
                # Simple routing
                decision = router.route(user_input)
                
                # Execute with ReAct loop
                result = agent.run(user_input)
                
                if result.success:
                    print(result.answer)
                else:
                    print(f"Error: {result.error}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
        
        # Save conversation
        if memory:
            filepath = memory.save()
            print(f"\n💾 Conversation saved to: {filepath}")
        
    except Exception as e:
        print(f"\n❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
