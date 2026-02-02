"""
Conversation Memory for Agent

Manages conversation history with:
- Message storage and retrieval
- Context window management (token limits)
- Sliding window strategy
- Tool call tracking
- Persistence to disk
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict
)

from ...config_loader import config

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history for multi-turn interactions.
    
    Features:
    - Message history storage
    - Automatic context window trimming
    - Tool call tracking
    - Persistence to JSON
    - Token counting (approximate)
    """
    
    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep (None = unlimited)
            max_tokens: Maximum tokens in context (None = unlimited)
            session_id: Unique session identifier
            persist_path: Directory to save/load conversations
        """
        self.max_messages = max_messages or config.get('agent.memory.max_messages', 50)
        self.max_tokens = max_tokens or config.get('agent.memory.max_tokens', 32000)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.persist_path = persist_path or Path(config.get('agent.memory.persist_path', 'data/conversations'))
        
        # Message storage
        self.messages: List[BaseMessage] = []
        self.system_prompt: Optional[str] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "tool_calls": 0
        }
        
        logger.info(
            f"ConversationMemory initialized: session={self.session_id}, "
            f"max_messages={self.max_messages}, max_tokens={self.max_tokens}"
        )
    
    def set_system_prompt(self, prompt: str):
        """
        Set system prompt (persistent across conversation).
        
        Args:
            prompt: System prompt text
        """
        self.system_prompt = prompt
        logger.debug(f"System prompt set: {prompt[:50]}...")
    
    def add_message(self, message: BaseMessage):
        """
        Add message to conversation history.
        
        Args:
            message: LangChain message to add
        """
        self.messages.append(message)
        self.metadata["message_count"] += 1
        
        # Track tool calls
        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
            self.metadata["tool_calls"] += len(message.tool_calls)
        
        # Trim if needed
        self._trim_if_needed()
        
        logger.debug(
            f"Message added ({message.__class__.__name__}): "
            f"{len(self.messages)} total messages"
        )
    
    def add_user_message(self, content: str):
        """
        Add user message (convenience method).
        
        Args:
            content: User message text
        """
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str, tool_calls: Optional[List] = None):
        """
        Add AI message (convenience method).
        
        Args:
            content: AI message text
            tool_calls: Optional list of tool calls
        """
        message = AIMessage(content=content)
        if tool_calls:
            message.tool_calls = tool_calls
        self.add_message(message)
    
    def add_tool_message(self, content: str, tool_call_id: str):
        """
        Add tool result message (convenience method).
        
        Args:
            content: Tool result
            tool_call_id: ID of the tool call this responds to
        """
        self.add_message(ToolMessage(content=content, tool_call_id=tool_call_id))
    
    def get_messages(
        self,
        include_system: bool = True,
        last_n: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        Get conversation messages.
        
        Args:
            include_system: Whether to include system prompt
            last_n: Get only last N messages (None = all)
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add system prompt if requested
        if include_system and self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Add conversation messages
        if last_n is not None:
            messages.extend(self.messages[-last_n:])
        else:
            messages.extend(self.messages)
        
        return messages
    
    def get_recent_context(self, n_messages: int = 10) -> List[BaseMessage]:
        """
        Get recent conversation context.
        
        Args:
            n_messages: Number of recent messages to retrieve
            
        Returns:
            List of recent messages with system prompt
        """
        return self.get_messages(include_system=True, last_n=n_messages)
    
    def clear(self, keep_system_prompt: bool = True):
        """
        Clear conversation history.
        
        Args:
            keep_system_prompt: Whether to keep system prompt
        """
        self.messages.clear()
        self.metadata["message_count"] = 0
        self.metadata["tool_calls"] = 0
        
        if not keep_system_prompt:
            self.system_prompt = None
        
        logger.info("Conversation history cleared")
    
    def _trim_if_needed(self):
        """
        Trim old messages if limits exceeded.
        
        Uses sliding window strategy:
        - Keeps most recent messages
        - Removes oldest messages first
        - Never removes system prompt
        """
        # Check message count limit
        if self.max_messages and len(self.messages) > self.max_messages:
            excess = len(self.messages) - self.max_messages
            removed = self.messages[:excess]
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} old messages (message limit)")
        
        # Check token limit (approximate)
        if self.max_tokens:
            total_tokens = self._estimate_tokens()
            if total_tokens > self.max_tokens:
                # Remove oldest messages until under limit
                while total_tokens > self.max_tokens and len(self.messages) > 1:
                    removed = self.messages.pop(0)
                    total_tokens = self._estimate_tokens()
                logger.debug(f"Trimmed messages to stay under token limit: {total_tokens} tokens")
    
    def _estimate_tokens(self) -> int:
        """
        Estimate total tokens in conversation.
        
        Uses rough approximation: 1 token ≈ 4 characters
        
        Returns:
            Estimated token count
        """
        total_chars = 0
        
        # Count system prompt
        if self.system_prompt:
            total_chars += len(self.system_prompt)
        
        # Count messages
        for msg in self.messages:
            if hasattr(msg, 'content') and msg.content:
                total_chars += len(str(msg.content))
            
            # Count tool calls
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(json.dumps(tc))
        
        # Rough estimate: 4 chars per token
        return total_chars // 4
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dict with message counts, tokens, etc.
        """
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if isinstance(m, HumanMessage)),
            "ai_messages": sum(1 for m in self.messages if isinstance(m, AIMessage)),
            "tool_messages": sum(1 for m in self.messages if isinstance(m, ToolMessage)),
            "estimated_tokens": self._estimate_tokens(),
            "tool_calls": self.metadata["tool_calls"],
            "created_at": self.metadata["created_at"],
            "has_system_prompt": self.system_prompt is not None
        }
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save conversation to JSON file.
        
        Args:
            filename: Optional filename (default: session_id.json)
            
        Returns:
            Path to saved file
        """
        filename = filename or f"{self.session_id}.json"
        filepath = self.persist_path / filename
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize messages
        serialized_messages = [message_to_dict(m) for m in self.messages]
        
        # Create save data
        data = {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": serialized_messages,
            "metadata": self.metadata,
            "saved_at": datetime.now().isoformat()
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Conversation saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'ConversationMemory':
        """
        Load conversation from JSON file.
        
        Args:
            filepath: Path to conversation JSON file
            
        Returns:
            ConversationMemory instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create memory instance
        memory = cls(
            session_id=data["session_id"],
            persist_path=filepath.parent
        )
        
        # Restore system prompt
        memory.system_prompt = data.get("system_prompt")
        
        # Restore messages
        memory.messages = messages_from_dict(data["messages"])
        
        # Restore metadata
        memory.metadata = data.get("metadata", {})
        
        logger.info(f"Conversation loaded from {filepath}: {len(memory.messages)} messages")
        return memory
    
    def get_last_user_message(self) -> Optional[str]:
        """
        Get content of last user message.
        
        Returns:
            User message content or None
        """
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None
    
    def get_last_ai_message(self) -> Optional[str]:
        """
        Get content of last AI message.
        
        Returns:
            AI message content or None
        """
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return None
    
    def __len__(self) -> int:
        """Get number of messages."""
        return len(self.messages)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConversationMemory(session={self.session_id}, "
            f"messages={len(self.messages)}, "
            f"tokens≈{self._estimate_tokens()})"
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_memory(
    session_id: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> ConversationMemory:
    """
    Create new conversation memory.
    
    Args:
        session_id: Optional session ID
        system_prompt: Optional system prompt
        
    Returns:
        ConversationMemory instance
    """
    memory = ConversationMemory(session_id=session_id)
    if system_prompt:
        memory.set_system_prompt(system_prompt)
    return memory


def load_or_create_memory(
    session_id: str,
    persist_path: Optional[Path] = None
) -> ConversationMemory:
    """
    Load existing conversation or create new one.
    
    Args:
        session_id: Session ID
        persist_path: Directory containing saved conversations
        
    Returns:
        ConversationMemory instance
    """
    persist_path = persist_path or Path(config.get('agent.memory.persist_path', 'data/conversations'))
    filepath = persist_path / f"{session_id}.json"
    
    if filepath.exists():
        logger.info(f"Loading existing conversation: {session_id}")
        return ConversationMemory.load(filepath)
    else:
        logger.info(f"Creating new conversation: {session_id}")
        return ConversationMemory(session_id=session_id, persist_path=persist_path)


# ============================================================
# EXAMPLE USAGE - FOR DEV
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Basic usage - FOR DEV
    print("\n=== Example 1: Basic Usage ===")
    memory = create_memory(
        session_id="demo_session",
        system_prompt="You are a helpful assistant for VintedOS."
    )
    
    memory.add_user_message("Hello! How are you?")
    memory.add_ai_message("I'm doing well! How can I help you today?")
    memory.add_user_message("Can you explain how the system works?")
    memory.add_ai_message("VintedOS is an automation system for processing Vinted orders...")
    
    print(f"Memory: {memory}")
    print(f"Stats: {memory.get_statistics()}")
    
    # Example 2: Get recent context
    print("\n=== Example 2: Recent Context ===")
    recent = memory.get_recent_context(n_messages=2)
    for msg in recent:
        print(f"{msg.__class__.__name__}: {msg.content[:50]}...")
    
    # Example 3: Save and load
    print("\n=== Example 3: Persistence ===")
    saved_path = memory.save()
    print(f"Saved to: {saved_path}")
    
    loaded_memory = ConversationMemory.load(saved_path)
    print(f"Loaded: {loaded_memory}")
    print(f"Last user message: {loaded_memory.get_last_user_message()}")
    
    # Example 4: Context window trimming
    print("\n=== Example 4: Context Trimming ===")
    small_memory = ConversationMemory(max_messages=3)
    for i in range(5):
        small_memory.add_user_message(f"Message {i}")
    print(f"Added 5 messages, kept {len(small_memory)} (max=3)")
    
    # Example 5: Tool call tracking
    print("\n=== Example 5: Tool Calls ===")
    memory.add_ai_message(
        "Let me check the database...",
        tool_calls=[{"name": "search_transactions", "args": {"status": "completed"}}]
    )
    memory.add_tool_message(
        content='{"success": true, "count": 10}',
        tool_call_id="call_123"
    )
    print(f"Tool calls tracked: {memory.get_statistics()['tool_calls']}")
    
    # Cleanup
    saved_path.unlink()
    print(f"\nCleaned up: {saved_path}")
