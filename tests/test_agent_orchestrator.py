"""
Comprehensive end-to-end test for Agent Orchestrator.

Tests:
- Routing (RAG vs Function Calling vs Hybrid)
- Multi-turn conversation with memory
- ReAct loop reasoning and tool execution
- LLM interaction (using mock if needed or live if keys exist)
- State persistence and statistics tracking
"""

import unittest
import os
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Import components
from src.agent.core.llm_client import LLMClient, get_llm_client
from src.agent.core.memory import ConversationMemory, create_memory
from src.agent.core.react_loop import ReActLoop, create_react_agent
from src.agent.core.router import QueryRouter, create_router, QueryType, QueryIntent
from src.agent.tools.query_tools import ALL_QUERY_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgentOrchestrator(unittest.TestCase):
    """Test suite for the Agent Orchestrator components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create mock database for tool testing
        subprocess.run(["python", "create_mock_db.py", "vinted_os.db"], check=True)
        
        # Ensure directories exist
        Path("data/conversations").mkdir(parents=True, exist_ok=True)

    def setUp(self):
        """Set up individual tests."""
        self.session_id = "test_session_123"
        self.memory = create_memory(session_id=self.session_id)
        
        # Mock LLM Client to avoid actual API calls during logic testing
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.get_token_usage.return_value = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        self.mock_llm.estimate_cost.return_value = 0.0001
        
        # Router with mock LLM
        self.router = create_router(llm_client=self.mock_llm, tools=ALL_QUERY_TOOLS, use_llm=False)

    def test_routing_logic_rule_based(self):
        """Test rule-based routing for different query types."""
        # Function Calling
        decision = self.router.route("Show me the recent 5 transactions")
        self.assertEqual(decision.query_type, QueryType.FUNCTION_CALLING)
        self.assertEqual(decision.intent, QueryIntent.TRANSACTION_LOOKUP)
        self.assertIn("get_recent_transactions", decision.suggested_tools)
        
        # RAG
        decision = self.router.route("Explain the system architecture and design")
        self.assertEqual(decision.query_type, QueryType.RAG)
        self.assertEqual(decision.intent, QueryIntent.ARCHITECTURE)
        
        # Hybrid
        decision = self.router.route("hello")
        self.assertEqual(decision.query_type, QueryType.HYBRID)

    def test_memory_management(self):
        """Test conversation memory storage and trimming."""
        memory = ConversationMemory(max_messages=2)
        
        memory.add_user_message("First message")
        memory.add_ai_message("First response")
        self.assertEqual(len(memory.messages), 2)
        
        memory.add_user_message("Second message")
        # Should have trimmed the first user message
        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(memory.messages[0].content, "First response")
        self.assertEqual(memory.messages[1].content, "Second message")

    def test_memory_persistence(self):
        """Test saving and loading memory from disk."""
        self.memory.add_user_message("Persistent query")
        self.memory.add_ai_message("Persistent response")
        
        filepath = self.memory.save()
        self.assertTrue(filepath.exists())
        
        loaded_memory = ConversationMemory.load(filepath)
        self.assertEqual(len(loaded_memory.messages), 2)
        self.assertEqual(loaded_memory.session_id, self.session_id)
        
        # Cleanup
        filepath.unlink()

    @patch('src.agent.core.llm_client.LLMClient')
    def test_react_loop_iteration(self, mock_client_class):
        """Test the ReAct loop logic (thought -> tool call -> observation -> final)."""
        mock_client = mock_client_class.return_value
        
        # 1. First response: AI wants to call a tool
        tool_call = {
            "name": "get_recent_transactions",
            "args": {"limit": 1},
            "id": "call_1"
        }
        res1 = AIMessage(content="I should check recent transactions.", tool_calls=[tool_call])
        
        # 2. Second response: AI provides final answer
        res2 = AIMessage(content="I see one transaction for $25.00.")
        
        # Setup mock returns
        mock_client.invoke.side_effect = [res1, res2]
        mock_client.bind_tools.return_value = mock_client
        mock_client.get_token_usage.return_value = {"total_tokens": 200, "input_tokens": 100, "output_tokens": 100}
        mock_client.estimate_cost.return_value = 0.002
        
        agent = create_react_agent(llm_client=mock_client, tools=ALL_QUERY_TOOLS)
        
        result = agent.run("Check my transactions")
        
        self.assertTrue(result.success)
        # Fix: Check for substring without strict case matching if necessary, 
        # but here we just match exactly what we mocked
        self.assertIn("one transaction for $25.00", result.answer)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, "get_recent_transactions")
        self.assertTrue(result.steps[1].is_final)

    def test_tool_execution_safety(self):
        """Test that the ReAct loop handles non-existent tools safely."""
        agent = create_react_agent(llm_client=self.mock_llm, tools=[])
        
        # Manually invoke tool execution for a tool not in the list
        tool_call = {"name": "dangerous_delete_all", "args": {}, "id": "1"}
        observation = agent._execute_tool(tool_call)
        
        result_json = json.loads(observation)
        self.assertIn("error", result_json)
        self.assertIn("Tool not found", result_json["error"])

    def test_llm_client_token_tracking(self):
        """Test token tracking accumulation."""
        client = LLMClient(provider="google", model="gemini-2.0-flash")
        
        # Replace the real LLM with a mock to avoid Pydantic attribute errors
        mock_llm = MagicMock()
        client.llm = mock_llm
        
        # Simulate usage metadata injection
        mock_response = AIMessage(content="Hi")
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        
        mock_llm.invoke.return_value = mock_response
        
        client.invoke("test")
        client.invoke("test again")
            
        stats = client.get_token_usage()
        self.assertEqual(stats["input_tokens"], 20)
        self.assertEqual(stats["output_tokens"], 10)
        self.assertEqual(stats["total_tokens"], 30)

    def test_router_pydantic_validation(self):
        """Test that router output follows the Pydantic schema."""
        decision = self.router.route("Get failed prints")
        
        # Validate properties exist (Schema enforced by Pydantic)
        self.assertIsInstance(decision.query_type, QueryType)
        self.assertEqual(decision.query_type, QueryType.FUNCTION_CALLING)
        self.assertIsInstance(decision.confidence, float)
        self.assertGreater(len(decision.reasoning), 0)

if __name__ == "__main__":
    unittest.main()
