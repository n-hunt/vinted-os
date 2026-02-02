"""
ReAct Loop Implementation

Implements the Reasoning and Acting (ReAct) pattern:
- Thought: LLM reasons about what to do
- Action: LLM calls tools or provides answer
- Observation: Tool results are fed back to LLM
- Repeat until answer is found or max iterations reached
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langchain_core.tools import BaseTool

from .llm_client import LLMClient
from .memory import ConversationMemory
from ...config_loader import config

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """Single step in ReAct loop."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class ReActResult:
    """Result from ReAct loop execution."""
    answer: str
    steps: List[ReActStep]
    total_steps: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReActLoop:
    """
    ReAct (Reasoning and Acting) loop implementation.
    
    Orchestrates interaction between:
    - LLM for reasoning and decision making
    - Tools for actions
    - Memory for conversation context
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tools: List[BaseTool],
        memory: Optional[ConversationMemory] = None,
        max_iterations: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize ReAct loop.
        
        Args:
            llm_client: LLM client for reasoning
            tools: Available tools for actions
            memory: Conversation memory (optional)
            max_iterations: Maximum reasoning steps
            system_prompt: System prompt for agent
        """
        self.llm_client = llm_client
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations or config.get('agent.max_iterations', 10)
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm_client.bind_tools(tools)
        
        # Track execution
        self.steps: List[ReActStep] = []
        
        logger.info(
            f"ReActLoop initialized: {len(tools)} tools, "
            f"max_iterations={self.max_iterations}"
        )
    
    def _default_system_prompt(self) -> str:
        """
        Default system prompt for ReAct agent.
        
        Returns:
            System prompt text
        """
        return """You are VintedOS Assistant, an AI agent that helps users manage and analyze their Vinted automation system.

You have access to tools that can:
- Query transaction data
- Analyze print failures
- Generate revenue reports
- Monitor system health
- Search the knowledge base

**How to respond:**
1. Think about what information is needed
2. Use tools to gather that information
3. Analyze the results
4. Provide clear, helpful answers

**Guidelines:**
- Use tools when you need specific data
- Call multiple tools if needed for complete answers
- Explain your findings clearly
- Format data in readable tables when appropriate
- If you don't know something, say so

**Tool Usage:**
- Always verify parameters before calling tools
- Handle errors gracefully
- Summarize tool results for the user

Be helpful, accurate, and concise."""
    
    def run(
        self,
        user_query: str,
        use_memory: bool = True
    ) -> ReActResult:
        """
        Run ReAct loop to answer user query.
        
        Args:
            user_query: User's question or request
            use_memory: Whether to use conversation memory
            
        Returns:
            ReActResult with answer and execution details
        """
        logger.info(f"Starting ReAct loop: '{user_query[:50]}...'")
        
        # Reset state
        self.steps = []
        current_iteration = 0
        
        try:
            # Add user query to memory at the start
            if use_memory and self.memory is not None:
                self.memory.add_message(HumanMessage(content=user_query))
                logger.debug("User query added to memory")
            
            # Build initial messages
            messages = self._build_messages(user_query, use_memory)
            
            # ReAct loop
            while current_iteration < self.max_iterations:
                current_iteration += 1
                logger.debug(f"ReAct iteration {current_iteration}/{self.max_iterations}")
                
                # Get LLM response
                response = self.llm_with_tools.invoke(messages)
                
                # Check if LLM provided final answer
                if not response.tool_calls:
                    # No tool calls = final answer
                    final_answer = self._extract_content(response.content)
                    
                    # Record final step
                    step = ReActStep(
                        step_number=current_iteration,
                        thought=final_answer,
                        is_final=True
                    )
                    self.steps.append(step)
                    
                    # Add AI response to memory
                    if use_memory and self.memory is not None:
                        self.memory.add_message(response)
                        logger.debug("AI response added to memory")
                    
                    logger.info(f"ReAct completed in {current_iteration} steps")
                    
                    return ReActResult(
                        answer=final_answer,
                        steps=self.steps,
                        total_steps=current_iteration,
                        success=True,
                        metadata=self._get_metadata()
                    )
                
                # Execute tool calls
                messages.append(response)  # Add AI response with tool calls
                
                for tool_call in response.tool_calls:
                    # Record step
                    step = ReActStep(
                        step_number=current_iteration,
                        thought=self._extract_content(response.content) or f"Calling {tool_call['name']}",
                        action=tool_call['name'],
                        action_input=tool_call['args']
                    )
                    
                    # Execute tool
                    observation = self._execute_tool(tool_call)
                    step.observation = observation
                    self.steps.append(step)
                    
                    # Add tool result to messages
                    tool_message = ToolMessage(
                        content=observation,
                        tool_call_id=tool_call['id']
                    )
                    messages.append(tool_message)
                    
                    logger.debug(
                        f"Tool executed: {tool_call['name']} -> "
                        f"{observation[:100]}..."
                    )
            
            # Max iterations reached
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            
            # Try to get final answer anyway
            final_response = self.llm_client.invoke(messages + [
                HumanMessage(content="Please provide your final answer based on the information gathered.")
            ])
            
            # Add AI response to memory even when max iterations reached
            if use_memory and self.memory is not None:
                self.memory.add_message(final_response)
                logger.debug("AI response added to memory (max iterations)")
            
            return ReActResult(
                answer=self._extract_content(final_response.content),
                steps=self.steps,
                total_steps=current_iteration,
                success=False,
                error=f"Max iterations ({self.max_iterations}) reached",
                metadata=self._get_metadata()
            )
        
        except Exception as e:
            logger.error(f"ReAct loop failed: {e}", exc_info=True)
            
            return ReActResult(
                answer=f"I encountered an error: {str(e)}",
                steps=self.steps,
                total_steps=current_iteration,
                success=False,
                error=str(e),
                metadata=self._get_metadata()
            )
    
    def _build_messages(
        self,
        user_query: str,
        use_memory: bool
    ) -> List[BaseMessage]:
        """
        Build message list for LLM.
        
        Args:
            user_query: Current user query
            use_memory: Whether to include conversation history
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add system prompt
        messages.append(SystemMessage(content=self.system_prompt))
        
        # Add conversation history if using memory
        if use_memory and self.memory is not None:
            history = self.memory.get_messages(include_system=False)
            messages.extend(history)
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        return messages
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """
        Execute a tool call.
        
        Args:
            tool_call: Tool call dict with name, args, id
            
        Returns:
            Tool execution result as string
        """
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        # Find tool
        tool = self._find_tool(tool_name)
        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        try:
            # Execute tool
            logger.debug(f"Executing {tool_name} with args: {tool_args}")
            result = tool.invoke(tool_args)
            
            # Convert result to string if needed
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            elif isinstance(result, (list, tuple)):
                return json.dumps(result, indent=2)
            else:
                return str(result)
        
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"{error_msg} (tool={tool_name}, args={tool_args})", exc_info=True)
            return json.dumps({"error": error_msg})
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Find tool by name.
        
        Args:
            tool_name: Name of tool to find
            
        Returns:
            Tool or None if not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _extract_content(self, content: Any) -> str:
        """
        Extract text content from LLM response.
        
        Handles both string responses and structured content blocks
        (e.g., Gemini's [{'type': 'text', 'text': '...'}] format).
        
        Args:
            content: Content from LLM response (string or list)
            
        Returns:
            Extracted text as string
        """
        # If already a string, return as-is
        if isinstance(content, str):
            return content
        
        # If list of content blocks, extract text
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            return '\n'.join(text_parts) if text_parts else str(content)
        
        # Fallback: convert to string
        return str(content)
    
    def _get_metadata(self) -> Dict[str, Any]:
        """
        Get execution metadata.
        
        Returns:
            Metadata dict
        """
        return {
            "total_steps": len(self.steps),
            "tools_used": list(set(
                step.action for step in self.steps if step.action
            )),
            "tool_calls": sum(1 for step in self.steps if step.action),
            "token_usage": self.llm_client.get_token_usage(),
            "estimated_cost": self.llm_client.estimate_cost()
        }
    
    def get_trace(self) -> str:
        """
        Get human-readable trace of execution.
        
        Returns:
            Formatted trace string
        """
        lines = ["=== ReAct Execution Trace ===\n"]
        
        for step in self.steps:
            lines.append(f"Step {step.step_number}:")
            lines.append(f"  Thought: {step.thought}")
            
            if step.action:
                lines.append(f"  Action: {step.action}")
                lines.append(f"  Input: {json.dumps(step.action_input, indent=4)}")
                lines.append(f"  Observation: {step.observation[:200]}...")
            
            if step.is_final:
                lines.append("  [FINAL ANSWER]")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def clear_history(self):
        """Clear execution history."""
        self.steps = []
        logger.debug("ReAct history cleared")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_react_agent(
    llm_client: LLMClient,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    memory: Optional[ConversationMemory] = None
) -> ReActLoop:
    """
    Create ReAct agent.
    
    Args:
        llm_client: LLM client
        tools: Available tools
        system_prompt: Optional system prompt
        memory: Optional conversation memory
        
    Returns:
        ReActLoop instance
    """
    return ReActLoop(
        llm_client=llm_client,
        tools=tools,
        memory=memory,
        system_prompt=system_prompt
    )


def run_agent_query(
    agent: ReActLoop,
    query: str,
    verbose: bool = False
) -> str:
    """
    Run agent query and return answer.
    
    Args:
        agent: ReAct agent
        query: User query
        verbose: Whether to print trace
        
    Returns:
        Agent's answer
    """
    result = agent.run(query)
    
    if verbose:
        print(agent.get_trace())
        print(f"\nMetadata: {json.dumps(result.metadata, indent=2)}")
    
    return result.answer


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    from ..tools.query_tools import ALL_QUERY_TOOLS
    from .llm_client import get_llm_client
    from .memory import create_memory
    
    # Example 1: Simple query without memory
    print("\n=== Example 1: Simple Query ===")
    llm = get_llm_client(environment="dev")
    agent = create_react_agent(
        llm_client=llm,
        tools=ALL_QUERY_TOOLS
    )
    
    result = agent.run("What are the recent transactions?")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.total_steps}")
    print(f"Success: {result.success}")
    
    # Example 2: Multi-turn conversation with memory
    print("\n=== Example 2: Conversation with Memory ===")
    memory = create_memory(
        session_id="demo_session",
        system_prompt=agent.system_prompt
    )
    
    agent_with_memory = create_react_agent(
        llm_client=llm,
        tools=ALL_QUERY_TOOLS,
        memory=memory
    )
    
    # First query
    result1 = agent_with_memory.run("Show me failed transactions from the last 7 days")
    print(f"\nQ1: Show me failed transactions from the last 7 days")
    print(f"A1: {result1.answer[:200]}...")
    
    # Follow-up query (uses memory)
    result2 = agent_with_memory.run("What about print failures?")
    print(f"\nQ2: What about print failures?")
    print(f"A2: {result2.answer[:200]}...")
    
    # Example 3: Detailed trace
    print("\n=== Example 3: Execution Trace ===")
    result = agent.run("What's the total revenue for the last month?")
    print(agent.get_trace())
    
    # Example 4: Error handling
    print("\n=== Example 4: Error Handling ===")
    result = agent.run("Get transaction with ID 999999")
    print(f"Answer: {result.answer}")
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")
    
    # Example 5: Token usage and cost
    print("\n=== Example 5: Cost Tracking ===")
    result = agent.run("Give me a dashboard summary")
    print(f"Answer: {result.answer[:100]}...")
    print(f"\nMetadata:")
    print(f"  Steps: {result.metadata['total_steps']}")
    print(f"  Tools used: {result.metadata['tools_used']}")
    print(f"  Tokens: {result.metadata['token_usage']}")
    print(f"  Estimated cost: ${result.metadata['estimated_cost']:.4f}")
