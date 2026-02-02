"""
Query Tools for LLM Function Calling

Wraps QueryService methods as LangChain-compatible tools.
Each tool:
- Accepts validated Pydantic input
- Executes database query via QueryService
- Returns standardized ToolResult
- Handles errors gracefully
- Tracks execution metadata
"""

import logging
import time
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

from .schemas import (
    # Input schemas
    GetTransactionFullInput,
    GetTransactionByOrderIdInput,
    GetRecentTransactionsInput,
    SearchTransactionsInput,
    GetFailedTransactionsInput,
    GetPrintFailuresInput,
    GetProcessingErrorsByStepInput,
    GetPipelineStatsInput,
    GetRevenueByPeriodInput,
    GetTopSellingItemsInput,
    GetPipelineRunsInput,
    GetDashboardSummaryInput,
    # Output wrapper
    ToolResult,
)
from ...services.query_service import QueryService
from ...models import TransactionStatus

logger = logging.getLogger(__name__)

# Global QueryService instance (reused across tool calls)
_query_service: Optional[QueryService] = None
_demo_mode: bool = False  # Track current demo mode state


def set_demo_mode(demo_mode: bool) -> None:
    """
    Set demo mode for agent tools.
    
    This resets the QueryService singleton to use the demo database.
    Call this BEFORE using any agent tools.
    
    Args:
        demo_mode: If True, tools will use demo database
    """
    global _query_service, _demo_mode
    if _demo_mode != demo_mode:
        _demo_mode = demo_mode
        _query_service = None  # Force recreation with new mode
        logger.info(f"Agent demo mode {'enabled' if demo_mode else 'disabled'}")


def get_query_service() -> QueryService:
    """Get or create QueryService singleton."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService(demo_mode=_demo_mode)
        mode_info = " (demo mode)" if _demo_mode else ""
        logger.info(f"Initialized QueryService for tools{mode_info}")
    return _query_service


def format_tool_result(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    execution_time: Optional[float] = None,
    **metadata
) -> Dict[str, Any]:
    """
    Format standardized tool result.
    
    Args:
        success: Whether tool executed successfully
        data: Tool output data
        error: Error message if failed
        execution_time: Execution time in seconds
        **metadata: Additional metadata
        
    Returns:
        ToolResult dict
    """
    result = ToolResult(
        success=success,
        data=data,
        error=error,
        metadata={
            "execution_time_seconds": execution_time,
            **metadata
        }
    )
    return result.model_dump()


# ============================================================
# TRANSACTION QUERY TOOLS
# ============================================================

@tool
def get_transaction_full(db_id: int) -> Dict[str, Any]:
    """
    Get complete transaction details by database ID.
    
    Returns full transaction with items, messages, print jobs, and logs.
    Use this when you need all details about a specific transaction.
    
    Args:
        db_id: Database primary key ID of the transaction
        
    Returns:
        ToolResult with complete transaction data
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetTransactionFullInput(db_id=db_id)
        
        # Execute query
        qs = get_query_service()
        result = qs.get_transaction_full(input_data.db_id)
        
        execution_time = time.time() - start_time
        
        if result is None:
            return format_tool_result(
                success=False,
                error=f"Transaction {db_id} not found",
                execution_time=execution_time
            )
        
        return format_tool_result(
            success=True,
            data=result,
            execution_time=execution_time,
            transaction_id=result["id"],
            items_count=len(result["items"])
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_transaction_full failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_transaction_by_order_id(vinted_order_id: str) -> Dict[str, Any]:
    """
    Get transaction by Vinted order ID.
    
    Returns full transaction details for a given Vinted order ID.
    Use this when you have the customer's order number.
    
    Args:
        vinted_order_id: Vinted transaction/order ID (e.g., "12345")
        
    Returns:
        ToolResult with complete transaction data
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetTransactionByOrderIdInput(vinted_order_id=vinted_order_id)
        
        # Execute query
        qs = get_query_service()
        result = qs.get_transaction_by_order_id(input_data.vinted_order_id)
        
        execution_time = time.time() - start_time
        
        if result is None:
            return format_tool_result(
                success=False,
                error=f"Transaction with order ID '{vinted_order_id}' not found",
                execution_time=execution_time
            )
        
        return format_tool_result(
            success=True,
            data=result,
            execution_time=execution_time,
            transaction_id=result["id"]
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_transaction_by_order_id failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_recent_transactions(limit: int = 10) -> Dict[str, Any]:
    """
    Get most recent transactions.
    
    Returns summary of recent transactions ordered by creation date.
    Use this for overview of recent activity.
    
    Args:
        limit: Maximum number of transactions to return (1-100)
        
    Returns:
        ToolResult with list of transaction summaries
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetRecentTransactionsInput(limit=limit)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_recent_transactions(input_data.limit)
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results)
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_recent_transactions failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def search_transactions(
    query: Optional[str] = None,
    status: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Search transactions with filters.
    
    Search by customer name, order ID, or item title.
    Filter by status and value range.
    
    Args:
        query: Search text (matches customer name, order ID, item titles)
        status: Filter by status (pending, processing, completed, failed)
        min_value: Minimum total transaction value
        max_value: Maximum total transaction value
        limit: Maximum number of results (1-1000)
        
    Returns:
        ToolResult with list of matching transactions
    """
    start_time = time.time()
    
    try:
        # Convert status string to enum if provided
        status_enum = None
        if status:
            # Convert to uppercase to match enum values
            status_enum = TransactionStatus(status.upper())
        
        # Validate input
        input_data = SearchTransactionsInput(
            query=query,
            status=status_enum,
            min_value=min_value,
            max_value=max_value,
            limit=limit
        )
        
        # Execute query
        qs = get_query_service()
        model_status = input_data.status
        
        results = qs.search_transactions(
            query=input_data.query,
            status=model_status,
            min_value=input_data.min_value,
            max_value=input_data.max_value,
            limit=input_data.limit
        )
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results),
            filters_applied={
                "query": query,
                "status": status,
                "min_value": min_value,
                "max_value": max_value
            }
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"search_transactions failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


# ============================================================
# ERROR & FAILURE QUERY TOOLS
# ============================================================

@tool
def get_failed_transactions(days: int = 7) -> Dict[str, Any]:
    """
    Get transactions that failed processing.
    
    Returns transactions with 'failed' status in the specified time window.
    Use this for troubleshooting recent failures.
    
    Args:
        days: Look back this many days from now (1-365)
        
    Returns:
        ToolResult with list of failed transactions
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetFailedTransactionsInput(days=days)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_failed_transactions(input_data.days)
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results),
            days_back=days
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_failed_transactions failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_print_failures(days: int = 7) -> Dict[str, Any]:
    """
    Get print job failures.
    
    Returns all failed print jobs in the specified time window.
    Use this to diagnose printer issues.
    
    Args:
        days: Look back this many days from now (1-365)
        
    Returns:
        ToolResult with list of print failures
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetPrintFailuresInput(days=days)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_print_failures(input_data.days)
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results),
            days_back=days
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_print_failures failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_processing_errors_by_step(days: int = 7) -> Dict[str, Any]:
    """
    Get processing errors grouped by pipeline step.
    
    Returns errors organized by step (email_fetch, pdf_parse, print, etc.).
    Use this to identify which pipeline steps are problematic.
    
    Args:
        days: Look back this many days from now (1-365)
        
    Returns:
        ToolResult with errors grouped by step
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetProcessingErrorsByStepInput(days=days)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_processing_errors_by_step(input_data.days)
        
        execution_time = time.time() - start_time
        
        # Count total errors
        total_errors = sum(len(errors) for errors in results.values())
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            total_errors=total_errors,
            steps_with_errors=len(results),
            days_back=days
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_processing_errors_by_step failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


# ============================================================
# ANALYTICS & REPORTING TOOLS
# ============================================================

@tool
def get_pipeline_stats(days: int = 30) -> Dict[str, Any]:
    """
    Get pipeline performance statistics.
    
    Returns metrics like success rate, avg duration, total processed.
    Use this for performance monitoring and reporting.
    
    Args:
        days: Calculate stats for this many days from now (1-365)
        
    Returns:
        ToolResult with pipeline statistics
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetPipelineStatsInput(days=days)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_pipeline_stats(input_data.days)
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            days_back=days
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_pipeline_stats failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_revenue_by_period(days: int = 30, period: str = "day") -> Dict[str, Any]:
    """
    Get revenue aggregated by time period.
    
    Returns revenue grouped by day, week, or month.
    Use this for revenue analysis and trending.
    
    Args:
        days: Look back this many days from now (1-365)
        period: Group by 'day', 'week', or 'month'
        
    Returns:
        ToolResult with revenue data by period
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetRevenueByPeriodInput(days=days, period=period)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_revenue_by_period(
            days=input_data.days,
            group_by=input_data.period.value  # Changed from 'period' to 'group_by'
        )
        
        execution_time = time.time() - start_time
        
        # Calculate totals (QueryService only returns period and revenue)
        total_revenue = sum(r["revenue"] for r in results) if results else 0
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            total_revenue=total_revenue,
            period_count=len(results),
            days_back=days,
            grouping=period
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_revenue_by_period failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_top_selling_items(days: int = 30, limit: int = 10) -> Dict[str, Any]:
    """
    Get best-selling items by quantity sold.
    
    Returns top items ranked by number of times sold.
    Use this for inventory and sales insights.
    
    Args:
        days: Look back this many days from now (1-365)
        limit: Maximum number of items to return (1-100)
        
    Returns:
        ToolResult with top-selling items
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetTopSellingItemsInput(days=days, limit=limit)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_top_selling_items(
            days=input_data.days,
            limit=input_data.limit
        )
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results),
            days_back=days
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_top_selling_items failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


# ============================================================
# SYSTEM MONITORING TOOLS
# ============================================================

@tool
def get_pipeline_runs(limit: int = 10) -> Dict[str, Any]:
    """
    Get recent pipeline execution history.
    
    Returns history of pipeline runs with status and metrics.
    Use this to monitor pipeline health and execution history.
    
    Args:
        limit: Maximum number of pipeline runs to return (1-100)
        
    Returns:
        ToolResult with pipeline run history
    """
    start_time = time.time()
    
    try:
        # Validate input
        input_data = GetPipelineRunsInput(limit=limit)
        
        # Execute query
        qs = get_query_service()
        results = qs.get_pipeline_runs(input_data.limit)
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time,
            count=len(results)
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_pipeline_runs failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


@tool
def get_dashboard_summary() -> Dict[str, Any]:
    """
    Get overall system health dashboard.
    
    Returns high-level metrics for system overview.
    Use this for a quick system health check.
    
    Returns:
        ToolResult with dashboard summary
    """
    start_time = time.time()
    
    try:
        # Execute query (no input validation needed)
        qs = get_query_service()
        results = qs.get_dashboard_summary()
        
        execution_time = time.time() - start_time
        
        return format_tool_result(
            success=True,
            data=results,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_dashboard_summary failed: {e}", exc_info=True)
        return format_tool_result(
            success=False,
            error=str(e),
            execution_time=execution_time
        )


# ============================================================
# TOOL REGISTRY
# ============================================================

ALL_QUERY_TOOLS = [
    get_transaction_full,
    get_transaction_by_order_id,
    get_recent_transactions,
    search_transactions,
    get_failed_transactions,
    get_print_failures,
    get_processing_errors_by_step,
    get_pipeline_stats,
    get_revenue_by_period,
    get_top_selling_items,
    get_pipeline_runs,
    get_dashboard_summary,
]


def get_tools_by_category() -> Dict[str, List]:
    """
    Get tools organized by category.
    
    Returns:
        Dict mapping category name to list of tools
    """
    return {
        "transaction_queries": [
            get_transaction_full,
            get_transaction_by_order_id,
            get_recent_transactions,
            search_transactions,
        ],
        "error_diagnostics": [
            get_failed_transactions,
            get_print_failures,
            get_processing_errors_by_step,
        ],
        "analytics": [
            get_pipeline_stats,
            get_revenue_by_period,
            get_top_selling_items,
        ],
        "system_monitoring": [
            get_pipeline_runs,
            get_dashboard_summary,
        ],
    }
