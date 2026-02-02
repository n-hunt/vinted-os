"""
LLM Client for Agent Orchestrator

Provides unified interface for LLM interactions with:
- Gemini 2.0 Flash (dev/fast)
- Gemini 3.0 Pro (production/quality)
- Tool/function calling support
- Streaming and non-streaming responses
- Token tracking and cost estimation
"""

import os
import logging
from typing import Optional, List, Dict, Any, Iterator, Union
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from ...config_loader import config

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GOOGLE = "google"
    OPENAI = "openai"


class LLMModel(str, Enum):
    """Supported LLM models."""
    # Gemini models
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_3_PRO = "gemini-3.0-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    
    # OpenAI models (future support)
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class LLMClient:
    """
    Unified LLM client for agent interactions.
    
    Features:
    - Multi-provider support (Gemini, OpenAI)
    - Tool/function calling
    - Streaming responses
    - Token tracking
    - Automatic retries
    - Configuration from settings.yaml
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ('google', 'openai'). Defaults to config.
            model: Model name. Defaults to config.
            temperature: Sampling temperature (0-1). Defaults to config.
            max_tokens: Max output tokens. Defaults to config.
            tools: List of LangChain tools for function calling
        """
        # Load from config if not provided
        self.provider = provider or config.get('agent.llm.provider', 'google')
        self.model = model or config.get('agent.llm.model', 'gemini-2.0-flash')
        self.temperature = temperature if temperature is not None else config.get('agent.llm.temperature', 0.7)
        self.max_tokens = max_tokens or config.get('agent.llm.max_tokens', 4096)
        
        # Get API key from environment
        api_key_env = config.get('agent.llm.api_key_env', 'GEMINI_API_KEY')
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            logger.warning(f"API key not found in environment variable: {api_key_env}")
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Bind tools if provided
        self.tools = tools or []
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        logger.info(
            f"LLM Client initialized: {self.provider}/{self.model} "
            f"(temp={self.temperature}, max_tokens={self.max_tokens}, tools={len(self.tools)})"
        )
    
    def _create_llm(self) -> BaseChatModel:
        """
        Create LLM instance based on provider.
        
        Returns:
            LangChain chat model instance
        """
        if self.provider == LLMProvider.GOOGLE:
            return self._create_google_llm()
        elif self.provider == LLMProvider.OPENAI:
            return self._create_openai_llm()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_google_llm(self) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            convert_system_message_to_human=True,  # Gemini compatibility
        )
    
    def _create_openai_llm(self):
        """Create OpenAI LLM instance (future support)."""
        # Import here to avoid dependency if not using OpenAI
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def invoke(
        self,
        messages: Union[str, List[BaseMessage]],
        use_tools: bool = True
    ) -> AIMessage:
        """
        Invoke LLM with messages (non-streaming).
        
        Args:
            messages: String prompt or list of LangChain messages
            use_tools: Whether to enable tool calling
            
        Returns:
            AIMessage response
        """
        # Convert string to message list
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        # Select LLM (with or without tools)
        llm = self.llm_with_tools if (use_tools and self.tools) else self.llm
        
        try:
            # Invoke LLM
            response = llm.invoke(messages)
            
            # Track tokens (if available)
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                self.total_input_tokens += usage.get('input_tokens', 0)
                self.total_output_tokens += usage.get('output_tokens', 0)
            
            logger.debug(f"LLM response: {response.content[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise
    
    def stream(
        self,
        messages: Union[str, List[BaseMessage]],
        use_tools: bool = True
    ) -> Iterator[AIMessage]:
        """
        Stream LLM response in chunks.
        
        Args:
            messages: String prompt or list of LangChain messages
            use_tools: Whether to enable tool calling
            
        Yields:
            AIMessage chunks
        """
        # Convert string to message list
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        # Select LLM (with or without tools)
        llm = self.llm_with_tools if (use_tools and self.tools) else self.llm
        
        try:
            # Stream LLM response
            for chunk in llm.stream(messages):
                yield chunk
                
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}", exc_info=True)
            raise
    
    def bind_tools(self, tools: List[BaseTool]) -> 'LLMClient':
        """
        Create new client instance with tools bound.
        
        Args:
            tools: List of LangChain tools
            
        Returns:
            New LLMClient instance with tools
        """
        return LLMClient(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools
        )
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get cumulative token usage.
        
        Returns:
            Dict with input_tokens, output_tokens, total_tokens
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def estimate_cost(self) -> float:
        """
        Estimate API cost based on token usage.
        
        Returns:
            Estimated cost in USD
        """
        # Pricing (as of Feb 2026, approximate)
        pricing = {
            "gemini-2.5-flash-lite": {"input": 0.0 / 1_000_000, "output": 0.0 / 1_000_000},  # Free tier
            "gemini-2.0-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
            "gemini-3.0-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
            "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
            "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        }
        
        if self.model not in pricing:
            logger.warning(f"No pricing data for model: {self.model}")
            return 0.0
        
        model_pricing = pricing[self.model]
        input_cost = self.total_input_tokens * model_pricing["input"]
        output_cost = self.total_output_tokens * model_pricing["output"]
        
        return input_cost + output_cost
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.debug("Token usage counters reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMClient(provider={self.provider}, model={self.model}, "
            f"temp={self.temperature}, tools={len(self.tools)})"
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_llm_client(
    environment: str = "dev",
    tools: Optional[List[BaseTool]] = None
) -> LLMClient:
    """
    Get LLM client configured for environment.
    
    Args:
        environment: 'dev' (Gemini 2.5 Flash Lite) or 'prod' (Gemini 3.0 Pro)
        tools: Optional list of tools to bind
        
    Returns:
        Configured LLMClient instance
    """
    if environment == "prod":
        model = "gemini-3.0-pro"
        temperature = 0.3  # Lower temp for production (more deterministic)
    else:
        model = "gemini-2.5-flash-lite"  # Free tier with good quota
        temperature = 0.7  # Higher temp for dev (more creative)
    
    return LLMClient(
        model=model,
        temperature=temperature,
        tools=tools
    )


def create_message_list(
    system_prompt: Optional[str] = None,
    user_message: Optional[str] = None,
    history: Optional[List[BaseMessage]] = None
) -> List[BaseMessage]:
    """
    Create message list for LLM.
    
    Args:
        system_prompt: Optional system prompt
        user_message: Optional user message
        history: Optional conversation history
        
    Returns:
        List of LangChain messages
    """
    messages = []
    
    # Add system prompt
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    # Add history
    if history:
        messages.extend(history)
    
    # Add user message
    if user_message:
        messages.append(HumanMessage(content=user_message))
    
    return messages


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Simple query (dev)
    print("\n=== Example 1: Simple Query (Dev) ===")
    client_dev = get_llm_client(environment="dev")
    response = client_dev.invoke("What is the capital of France?")
    print(f"Response: {response.content}")
    print(f"Usage: {client_dev.get_token_usage()}")
    print(f"Cost: ${client_dev.estimate_cost():.6f}")
    
    # Example 2: Production client
    print("\n=== Example 2: Production Client ===")
    client_prod = get_llm_client(environment="prod")
    print(f"Client: {client_prod}")
    
    # Example 3: With conversation history
    print("\n=== Example 3: Multi-turn Conversation ===")
    messages = create_message_list(
        system_prompt="You are a helpful assistant for a Vinted OS automation system.",
        user_message="Hello! Can you help me understand how the system works?"
    )
    response = client_dev.invoke(messages)
    print(f"Response: {response.content[:100]}...")
    
    # Example 4: Streaming
    print("\n=== Example 4: Streaming Response ===")
    print("Streaming: ", end="", flush=True)
    for chunk in client_dev.stream("Tell me a short joke about databases."):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    
    # Final stats
    print("\n=== Final Stats ===")
    print(f"Total tokens: {client_dev.get_token_usage()}")
    print(f"Estimated cost: ${client_dev.estimate_cost():.6f}")
