"""
Query Router for RAG vs Function Calling

Routes user queries to the appropriate retrieval method:
- RAG: Knowledge base search (architecture, specs, troubleshooting)
- Function Calling: Database queries and analytics
- Hybrid: Combination of both

Uses Pydantic for type safety and validation.
"""

import logging
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from .llm_client import LLMClient, get_llm_client
from ..retrieval.hybrid_retriever import HybridRetriever
from ...config_loader import config

logger = logging.getLogger(__name__)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class QueryType(str, Enum):
    """Type of query routing."""
    RAG = "rag"  # Knowledge base search
    FUNCTION_CALLING = "function_calling"  # Database/analytics tools
    HYBRID = "hybrid"  # Both RAG and function calling
    UNKNOWN = "unknown"  # Unable to classify


class QueryIntent(str, Enum):
    """Specific intent categories."""
    # RAG intents
    ARCHITECTURE = "architecture"
    SPECS = "specs"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO = "how_to"
    EXPLANATION = "explanation"
    
    # Function calling intents
    TRANSACTION_LOOKUP = "transaction_lookup"
    ANALYTICS = "analytics"
    ERROR_DIAGNOSIS = "error_diagnosis"
    SYSTEM_MONITORING = "system_monitoring"
    
    # Hybrid intents
    CONTEXT_SPECIFIC_QUERY = "context_specific_query"
    
    # Other
    GENERAL = "general"


class RoutingDecision(BaseModel):
    """Decision about how to route a query."""
    
    query_type: QueryType = Field(
        description="Primary routing type"
    )
    
    intent: QueryIntent = Field(
        description="Specific intent category"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for routing decision"
    )
    
    reasoning: str = Field(
        description="Explanation of routing decision"
    )
    
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="Suggested tools for function calling queries"
    )
    
    search_strategy: Optional[str] = Field(
        default=None,
        description="Search strategy for RAG queries (semantic, hybrid, etc.)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @field_validator('confidence')
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v


class RouterConfig(BaseModel):
    """Configuration for query router."""
    
    use_llm_classification: bool = Field(
        default=True,
        description="Use LLM for intelligent query classification"
    )
    
    fallback_to_hybrid: bool = Field(
        default=True,
        description="Use hybrid approach when confidence is low"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to use non-hybrid routing"
    )
    
    max_rag_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum RAG results to return"
    )
    
    enable_caching: bool = Field(
        default=True,
        description="Cache routing decisions for similar queries"
    )
    
    @classmethod
    def from_config(cls) -> 'RouterConfig':
        """Load from application config."""
        return cls(
            use_llm_classification=config.get('agent.router.use_llm_classification', True),
            fallback_to_hybrid=config.get('agent.router.fallback_to_hybrid', True),
            confidence_threshold=config.get('agent.router.confidence_threshold', 0.7),
            max_rag_results=config.get('agent.router.max_rag_results', 5),
            enable_caching=config.get('agent.router.enable_caching', True)
        )


class RetrievalResult(BaseModel):
    """Result from retrieval (RAG or function calling)."""
    
    source: Literal["rag", "function_calling", "hybrid"] = Field(
        description="Source of retrieval"
    )
    
    content: str = Field(
        description="Retrieved content or tool results"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (sources, scores, etc.)"
    )
    
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in retrieval results"
    )


# ============================================================
# QUERY ROUTER
# ============================================================

class QueryRouter:
    """
    Routes queries to appropriate retrieval method.
    
    Uses rule-based and LLM-based classification to determine
    whether to use RAG, function calling, or both.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[RouterConfig] = None
    ):
        """
        Initialize query router.
        
        Args:
            llm_client: LLM for intelligent classification
            hybrid_retriever: RAG retriever
            tools: Function calling tools
            config: Router configuration
        """
        self.llm_client = llm_client or get_llm_client(environment="dev")
        self.hybrid_retriever = hybrid_retriever
        self.tools = tools or []
        self.config = config or RouterConfig.from_config()
        
        # Decision cache
        self._cache: Dict[str, RoutingDecision] = {}
        
        # Keywords for rule-based routing
        self._rag_keywords = {
            'architecture', 'design', 'how does', 'explain', 'what is',
            'documentation', 'guide', 'tutorial', 'workflow', 'process',
            'troubleshoot', 'debug', 'fix', 'error message', 'specs',
            'constraints', 'requirements'
        }
        
        self._function_keywords = {
            'transaction', 'order', 'revenue', 'profit', 'sales',
            'failed', 'print', 'database', 'query', 'stats', 'analytics',
            'count', 'total', 'recent', 'show me', 'get', 'list',
            'dashboard', 'monitoring', 'pipeline'
        }
        
        logger.info(
            f"QueryRouter initialized: llm={self.llm_client is not None}, "
            f"rag={self.hybrid_retriever is not None}, "
            f"tools={len(self.tools)}"
        )
    
    def route(self, query: str) -> RoutingDecision:
        """
        Route query to appropriate method.
        
        Args:
            query: User query
            
        Returns:
            RoutingDecision with routing information
        """
        logger.info(f"Routing query: '{query[:50]}...'")
        
        # Check cache
        if self.config.enable_caching and query in self._cache:
            logger.debug("Using cached routing decision")
            return self._cache[query]
        
        # Get routing decision
        if self.config.use_llm_classification:
            decision = self._llm_classify(query)
        else:
            decision = self._rule_based_classify(query)
        
        # Fallback to hybrid if confidence is low
        if (self.config.fallback_to_hybrid and 
            decision.confidence < self.config.confidence_threshold and
            decision.query_type != QueryType.HYBRID):
            
            logger.debug(
                f"Low confidence ({decision.confidence:.2f}), "
                f"falling back to hybrid"
            )
            decision.query_type = QueryType.HYBRID
            decision.reasoning += " [Fallback to hybrid due to low confidence]"
        
        # Cache decision
        if self.config.enable_caching:
            self._cache[query] = decision
        
        logger.info(
            f"Routed to {decision.query_type.value} "
            f"(confidence: {decision.confidence:.2f})"
        )
        
        return decision
    
    def _llm_classify(self, query: str) -> RoutingDecision:
        """
        Use LLM to classify query.
        
        Args:
            query: User query
            
        Returns:
            RoutingDecision
        """
        classification_prompt = f"""Classify this user query for routing:

Query: "{query}"

Determine:
1. Is this asking about system architecture, documentation, or troubleshooting? → RAG (knowledge base)
2. Is this asking for specific data, transactions, analytics, or metrics? → FUNCTION_CALLING (database tools)
3. Does it need both context and data? → HYBRID

Available function calling tools:
- get_transaction_full, get_transaction_by_order_id
- search_transactions, get_recent_transactions
- get_failed_transactions, get_print_failures
- get_pipeline_stats, get_revenue_by_period
- get_top_selling_items, get_dashboard_summary

Available RAG knowledge:
- Architecture documentation
- System specifications and constraints
- Troubleshooting guides

Respond in JSON format:
{{
  "query_type": "rag" | "function_calling" | "hybrid",
  "intent": "<specific intent>",
  "confidence": 0.0-1.0,
  "reasoning": "<brief explanation>",
  "suggested_tools": ["tool1", "tool2"] (if function_calling),
  "search_strategy": "hybrid" (if rag)
}}"""
        
        try:
            response = self.llm_client.invoke([HumanMessage(content=classification_prompt)])
            
            # Parse JSON response
            import json
            content = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            classification = json.loads(content)
            
            # Validate and convert enum values
            try:
                query_type = QueryType(classification['query_type'])
            except (KeyError, ValueError):
                query_type = QueryType.UNKNOWN
            
            try:
                intent = QueryIntent(classification.get('intent', 'general'))
            except ValueError:
                intent = QueryIntent.GENERAL
            
            return RoutingDecision(
                query_type=query_type,
                intent=intent,
                confidence=classification.get('confidence', 0.5),
                reasoning=classification.get('reasoning', 'LLM classification'),
                suggested_tools=classification.get('suggested_tools', []),
                search_strategy=classification.get('search_strategy'),
                metadata={'method': 'llm_classification'}
            )
        
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to rules")
            return self._rule_based_classify(query)
    
    def _rule_based_classify(self, query: str) -> RoutingDecision:
        """
        Use rule-based classification.
        
        Args:
            query: User query
            
        Returns:
            RoutingDecision
        """
        query_lower = query.lower()
        
        # Count keyword matches
        rag_score = sum(1 for kw in self._rag_keywords if kw in query_lower)
        func_score = sum(1 for kw in self._function_keywords if kw in query_lower)
        
        # Specific patterns
        has_id = any(pattern in query_lower for pattern in ['id', 'transaction #', 'order #'])
        has_time = any(pattern in query_lower for pattern in ['last', 'recent', 'today', 'yesterday', 'week', 'month', 'days'])
        has_count = any(pattern in query_lower for pattern in ['how many', 'count', 'total', 'number of'])
        has_question = any(pattern in query_lower for pattern in ['how', 'what', 'why', 'when', 'where'])
        
        # Determine routing
        if func_score > rag_score:
            query_type = QueryType.FUNCTION_CALLING
            confidence = min(0.9, 0.6 + (func_score * 0.1))
            intent = self._determine_function_intent(query_lower)
            suggested_tools = self._suggest_tools(query_lower)
            reasoning = f"Query contains {func_score} function-calling keywords"
        
        elif rag_score > func_score:
            query_type = QueryType.RAG
            confidence = min(0.9, 0.6 + (rag_score * 0.1))
            intent = self._determine_rag_intent(query_lower)
            suggested_tools = []
            reasoning = f"Query contains {rag_score} RAG keywords"
        
        elif has_id or has_count or has_time:
            query_type = QueryType.FUNCTION_CALLING
            confidence = 0.75
            intent = QueryIntent.TRANSACTION_LOOKUP if has_id else QueryIntent.ANALYTICS
            suggested_tools = self._suggest_tools(query_lower)
            reasoning = "Query has specific lookup/metric patterns"
        
        elif has_question and not has_time:
            query_type = QueryType.RAG
            confidence = 0.65
            intent = QueryIntent.EXPLANATION
            suggested_tools = []
            reasoning = "Conceptual question detected"
        
        else:
            query_type = QueryType.HYBRID
            confidence = 0.5
            intent = QueryIntent.GENERAL
            suggested_tools = []
            reasoning = "Ambiguous query, using hybrid approach"
        
        return RoutingDecision(
            query_type=query_type,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            suggested_tools=suggested_tools,
            search_strategy="hybrid" if query_type == QueryType.RAG else None,
            metadata={'method': 'rule_based'}
        )
    
    def _determine_function_intent(self, query_lower: str) -> QueryIntent:
        """Determine specific function calling intent."""
        if any(kw in query_lower for kw in ['failed', 'error', 'failure', 'problem']):
            return QueryIntent.ERROR_DIAGNOSIS
        elif any(kw in query_lower for kw in ['revenue', 'profit', 'sales', 'selling']):
            return QueryIntent.ANALYTICS
        elif any(kw in query_lower for kw in ['dashboard', 'monitoring', 'health', 'status']):
            return QueryIntent.SYSTEM_MONITORING
        elif any(kw in query_lower for kw in ['transaction', 'order', 'get', 'show']):
            return QueryIntent.TRANSACTION_LOOKUP
        else:
            return QueryIntent.GENERAL
    
    def _determine_rag_intent(self, query_lower: str) -> QueryIntent:
        """Determine specific RAG intent."""
        if any(kw in query_lower for kw in ['architecture', 'design', 'structure']):
            return QueryIntent.ARCHITECTURE
        elif any(kw in query_lower for kw in ['spec', 'constraint', 'requirement']):
            return QueryIntent.SPECS
        elif any(kw in query_lower for kw in ['troubleshoot', 'debug', 'fix', 'solve']):
            return QueryIntent.TROUBLESHOOTING
        elif any(kw in query_lower for kw in ['how to', 'how do', 'guide']):
            return QueryIntent.HOW_TO
        else:
            return QueryIntent.EXPLANATION
    
    def _suggest_tools(self, query_lower: str) -> List[str]:
        """Suggest specific tools based on query."""
        tools = []
        
        if 'failed' in query_lower or 'failure' in query_lower:
            tools.extend(['get_failed_transactions', 'get_print_failures'])
        
        if 'revenue' in query_lower or 'profit' in query_lower:
            tools.append('get_revenue_by_period')
        
        if 'top' in query_lower or 'best' in query_lower or 'selling' in query_lower:
            tools.append('get_top_selling_items')
        
        if 'dashboard' in query_lower or 'summary' in query_lower:
            tools.append('get_dashboard_summary')
        
        if 'stats' in query_lower or 'analytics' in query_lower:
            tools.append('get_pipeline_stats')
        
        if 'recent' in query_lower or 'latest' in query_lower:
            tools.append('get_recent_transactions')
        
        if any(kw in query_lower for kw in ['id', 'transaction #', 'order #']):
            tools.extend(['get_transaction_full', 'get_transaction_by_order_id'])
        
        if 'search' in query_lower:
            tools.append('search_transactions')
        
        return tools
    
    def clear_cache(self):
        """Clear routing decision cache."""
        self._cache.clear()
        logger.debug("Router cache cleared")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_router(
    llm_client: Optional[LLMClient] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    tools: Optional[List[BaseTool]] = None,
    use_llm: bool = True
) -> QueryRouter:
    """
    Create query router.
    
    Args:
        llm_client: LLM for classification
        hybrid_retriever: RAG retriever
        tools: Function calling tools
        use_llm: Use LLM-based classification
        
    Returns:
        QueryRouter instance
    """
    # Create config with override
    router_config = RouterConfig(
        use_llm_classification=use_llm,
        fallback_to_hybrid=config.get('agent.router.fallback_to_hybrid', True),
        confidence_threshold=config.get('agent.router.confidence_threshold', 0.7),
        max_rag_results=config.get('agent.router.max_rag_results', 5),
        enable_caching=config.get('agent.router.enable_caching', True)
    )
    
    return QueryRouter(
        llm_client=llm_client,
        hybrid_retriever=hybrid_retriever,
        tools=tools,
        config=router_config
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    from ..tools.query_tools import ALL_QUERY_TOOLS
    
    # Example 1: Rule-based routing - FOR DEV
    print("\n=== Example 1: Rule-Based Routing ===")
    router = create_router(tools=ALL_QUERY_TOOLS, use_llm=False)
    
    test_queries = [
        "Show me recent transactions",
        "How does the pipeline architecture work?",
        "What are the failed transactions from last week?",
        "Explain the troubleshooting process",
        "Get me the dashboard summary"
    ]
    
    for query in test_queries:
        decision = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {decision.query_type.value}")
        print(f"  Intent: {decision.intent.value}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        if decision.suggested_tools:
            print(f"  Suggested tools: {', '.join(decision.suggested_tools)}")
    
    # Example 2: LLM-based routing
    print("\n\n=== Example 2: LLM-Based Routing ===")
    llm = get_llm_client(environment="dev")
    router_llm = create_router(
        llm_client=llm,
        tools=ALL_QUERY_TOOLS,
        use_llm=True
    )
    
    complex_query = "I need to understand why prints are failing and see the actual failure data"
    decision = router_llm.route(complex_query)
    print(f"\nQuery: {complex_query}")
    print(f"  Type: {decision.query_type.value}")
    print(f"  Intent: {decision.intent.value}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Reasoning: {decision.reasoning}")
    print(f"  Method: {decision.metadata.get('method')}")
    
    # Example 3: Pydantic validation
    print("\n\n=== Example 3: Pydantic Validation ===")
    try:
        invalid_decision = RoutingDecision(
            query_type=QueryType.RAG,
            intent=QueryIntent.ARCHITECTURE,
            confidence=1.5,  # Invalid: > 1.0
            reasoning="Test"
        )
    except Exception as e:
        print(f"Validation error caught: {e}")
    
    # Example 4: Configuration
    print("\n\n=== Example 4: Router Configuration ===")
    custom_config = RouterConfig(
        use_llm_classification=False,
        confidence_threshold=0.8,
        max_rag_results=10,
        fallback_to_hybrid=True
    )
    print(f"Config: {custom_config.model_dump_json(indent=2)}")
    
    # Example 5: Caching
    print("\n\n=== Example 5: Decision Caching ===")
    router.clear_cache()
    query = "Show me the dashboard"
    
    # First call (not cached)
    decision1 = router.route(query)
    print(f"First call: {decision1.query_type.value}")
    
    # Second call (cached)
    decision2 = router.route(query)
    print(f"Second call (cached): {decision2.query_type.value}")
    print(f"Cache size: {len(router._cache)}")
