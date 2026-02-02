"""
Pydantic Schemas for Function Calling Tools

Defines type-safe input/output schemas for LLM tool calling.
Each schema includes:
- Field validation and constraints
- Documentation for LLM understanding
- Type hints for runtime safety
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from ...models import TransactionStatus


# ============================================================
# ENUMS
# ============================================================

class PeriodEnum(str, Enum):
    """Time period grouping for aggregations."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


# ============================================================
# INPUT SCHEMAS
# ============================================================

class GetTransactionFullInput(BaseModel):
    """Get complete transaction details by database ID."""
    
    db_id: int = Field(
        ...,
        description="Database primary key ID of the transaction",
        gt=0,
        examples=[1, 42, 100]
    )


class GetTransactionByOrderIdInput(BaseModel):
    """Get transaction by Vinted order ID."""
    
    vinted_order_id: str = Field(
        ...,
        description="Vinted transaction/order ID (e.g., '12345')",
        min_length=1,
        examples=["12345", "TX-2024-001"]
    )


class GetRecentTransactionsInput(BaseModel):
    """Get most recent transactions."""
    
    limit: int = Field(
        default=10,
        description="Maximum number of transactions to return",
        ge=1,
        le=100,
        examples=[10, 20, 50]
    )


class SearchTransactionsInput(BaseModel):
    """Search transactions with filters."""
    
    query: Optional[str] = Field(
        default=None,
        description="Search text (matches customer name, order ID, item titles)",
        examples=["John Doe", "shirt", "TX-2024"]
    )
    
    status: Optional[TransactionStatus] = Field(
        default=None,
        description="Filter by transaction status"
    )
    
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum total transaction value",
        ge=0,
        examples=[10.0, 50.0, 100.0]
    )
    
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum total transaction value",
        ge=0,
        examples=[100.0, 500.0, 1000.0]
    )
    
    limit: int = Field(
        default=100,
        description="Maximum number of results to return",
        ge=1,
        le=1000
    )
    
    @field_validator('max_value')
    @classmethod
    def validate_value_range(cls, v, info):
        """Ensure max_value >= min_value if both provided."""
        if v is not None and info.data.get('min_value') is not None:
            if v < info.data['min_value']:
                raise ValueError("max_value must be >= min_value")
        return v


class GetFailedTransactionsInput(BaseModel):
    """Get transactions that failed processing."""
    
    days: int = Field(
        default=7,
        description="Look back this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )


class GetPrintFailuresInput(BaseModel):
    """Get print job failures."""
    
    days: int = Field(
        default=7,
        description="Look back this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )


class GetProcessingErrorsByStepInput(BaseModel):
    """Get processing errors grouped by pipeline step."""
    
    days: int = Field(
        default=7,
        description="Look back this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )


class GetPipelineStatsInput(BaseModel):
    """Get pipeline performance statistics."""
    
    days: int = Field(
        default=30,
        description="Calculate stats for this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )


class GetRevenueByPeriodInput(BaseModel):
    """Get revenue aggregated by time period."""
    
    days: int = Field(
        default=30,
        description="Look back this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )
    
    period: PeriodEnum = Field(
        default=PeriodEnum.DAY,
        description="Group revenue by day, week, or month"
    )


class GetTopSellingItemsInput(BaseModel):
    """Get best-selling items by quantity sold."""
    
    days: int = Field(
        default=30,
        description="Look back this many days from now",
        ge=1,
        le=365,
        examples=[7, 30, 90]
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of items to return",
        ge=1,
        le=100,
        examples=[5, 10, 20]
    )


class GetPipelineRunsInput(BaseModel):
    """Get recent pipeline execution history."""
    
    limit: int = Field(
        default=10,
        description="Maximum number of pipeline runs to return",
        ge=1,
        le=100,
        examples=[10, 20, 50]
    )


class GetDashboardSummaryInput(BaseModel):
    """Get overall system health dashboard (no parameters)."""
    pass


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class ItemSummary(BaseModel):
    """Summary of a sold item."""
    id: int
    title: str
    price: Optional[float]
    currency: str


class GmailMessageSummary(BaseModel):
    """Summary of a Gmail message."""
    id: int
    gmail_id: str
    subject: str
    received_at: str  # ISO format datetime


class PrintJobSummary(BaseModel):
    """Summary of a print job."""
    id: int
    printer_name: str
    status: str
    error_message: Optional[str]
    attempted_at: str  # ISO format datetime


class ProcessingLogSummary(BaseModel):
    """Summary of a processing log entry."""
    id: int
    step: str
    status: str
    details: Optional[str]
    timestamp: str  # ISO format datetime


class TransactionFull(BaseModel):
    """Complete transaction details with all relationships."""
    id: int
    vinted_order_id: str
    customer_name: str
    status: str
    label_path: Optional[str]
    created_at: str
    updated_at: str
    items: List[ItemSummary]
    gmail_messages: List[GmailMessageSummary]
    print_jobs: List[PrintJobSummary]
    logs: List[ProcessingLogSummary]
    total_value: float


class TransactionSummary(BaseModel):
    """Summary of a transaction (less detail than TransactionFull)."""
    id: int
    vinted_order_id: str
    customer_name: str
    status: str
    items_count: int
    total_value: float
    created_at: str
    has_print_errors: bool


class PrintFailure(BaseModel):
    """Print job failure details."""
    id: int
    transaction_id: int
    vinted_order_id: str
    customer_name: str
    printer_name: str
    error_message: Optional[str]
    attempted_at: str
    retry_count: int


class ProcessingErrorGroup(BaseModel):
    """Group of processing errors for a pipeline step."""
    step: str
    errors: List[Dict[str, Any]]


class PipelineStats(BaseModel):
    """Pipeline performance statistics."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    avg_duration_seconds: Optional[float]
    total_transactions_processed: int
    total_items_processed: int
    total_labels_printed: int
    failed_prints: int
    print_success_rate: float


class RevenuePeriod(BaseModel):
    """Revenue for a time period."""
    period: str
    revenue: float
    transaction_count: int
    item_count: int


class TopSellingItem(BaseModel):
    """Top-selling item statistics."""
    title: str
    times_sold: int
    total_revenue: float
    avg_price: float


class PipelineRunSummary(BaseModel):
    """Summary of a pipeline execution."""
    id: int
    start_time: str  # ISO format
    end_time: Optional[str]  # ISO format
    status: str
    items_processed: int
    items_failed: int
    duration_seconds: Optional[float]


class DashboardSummary(BaseModel):
    """Overall system health dashboard."""
    total_transactions: int
    pending_transactions: int
    failed_transactions: int
    completed_transactions: int
    total_items: int
    total_revenue: float
    recent_print_failures: int
    last_pipeline_run: Optional[str]
    system_health: str


# ============================================================
# TOOL OUTPUT WRAPPERS
# ============================================================

class ToolResult(BaseModel):
    """Standardized tool execution result."""
    
    success: bool = Field(
        ...,
        description="Whether the tool executed successfully"
    )
    
    data: Optional[Any] = Field(
        default=None,
        description="Tool output data (structure depends on tool)"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if tool failed"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (execution time, count, etc.)"
    )


# ============================================================
# SQL VALIDATION SCHEMAS
# ============================================================

class SQLValidationResult(BaseModel):
    """Result of SQL query safety validation."""
    
    is_safe: bool = Field(
        ...,
        description="Whether the SQL query is safe to execute"
    )
    
    issues: List[str] = Field(
        default_factory=list,
        description="List of safety issues found (if any)"
    )
    
    query_type: Optional[str] = Field(
        default=None,
        description="Detected query type (SELECT, INSERT, etc.)"
    )
    
    affected_tables: List[str] = Field(
        default_factory=list,
        description="Tables referenced in the query"
    )
