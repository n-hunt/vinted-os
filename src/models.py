"""
VintedOS Data Models

SQLModel definitions for database persistence.
Uses SQLite with WAL mode for concurrent access.
"""

from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


# ============================================================
# ENUMS
# ============================================================

class TransactionStatus(str, Enum):
    """Transaction processing status."""
    PENDING = "PENDING"
    PARSED = "PARSED"
    MATCHED = "MATCHED"
    PRINTED = "PRINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PrintStatus(str, Enum):
    """Print job status."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ============================================================
# TOP-LEVEL MODEL
# ============================================================

class PipelineRun(SQLModel, table=True):
    """
    Tracks a specific execution of the pipeline.
    
    Each run represents one invocation of the bot (e.g., "Run at 9:00 AM").
    """
    __tablename__ = "pipeline_runs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    items_processed: int = Field(default=0)
    items_failed: int = Field(default=0)
    status: str = Field(default="running")


# ============================================================
# CORE MODEL
# ============================================================

class Transaction(SQLModel, table=True):
    """
    Central hub representing one Vinted sale.
    
    Links together items, messages, print jobs, and processing logs.
    """
    __tablename__ = "transactions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_run_id: Optional[int] = Field(default=None, foreign_key="pipeline_runs.id")
    vinted_order_id: str = Field(index=True)  # e.g., "Order #123" or transaction ID
    customer_name: Optional[str] = None
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    label_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships (cascade delete enabled)
    items: List["Item"] = Relationship(
        back_populates="transaction",
        sa_relationship_kwargs={"cascade": "all, delete"}
    )
    gmail_messages: List["GmailMessage"] = Relationship(
        back_populates="transaction",
        sa_relationship_kwargs={"cascade": "all, delete"}
    )
    print_jobs: List["PrintJob"] = Relationship(
        back_populates="transaction",
        sa_relationship_kwargs={"cascade": "all, delete"}
    )
    logs: List["ProcessingLog"] = Relationship(
        back_populates="transaction",
        sa_relationship_kwargs={"cascade": "all, delete"}
    )


# ============================================================
# CHILD MODELS (belong to Transaction)
# ============================================================

class Item(SQLModel, table=True):
    """
    Specific product sold in a transaction.
    """
    __tablename__ = "items"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(foreign_key="transactions.id")
    title: str
    price: Optional[float] = None
    currency: Optional[str] = "GBP"
    
    # Relationship
    transaction: Optional[Transaction] = Relationship(back_populates="items")


class GmailMessage(SQLModel, table=True):
    """
    Links the Gmail message that triggered this transaction.
    """
    __tablename__ = "gmail_messages"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(foreign_key="transactions.id")
    gmail_id: str = Field(index=True)  # Unique ID from Gmail API
    subject: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    transaction: Optional[Transaction] = Relationship(back_populates="gmail_messages")
    attachments: List["Attachment"] = Relationship(
        back_populates="message",
        sa_relationship_kwargs={"cascade": "all, delete"}
    )


class PrintJob(SQLModel, table=True):
    """
    History of print attempts for this transaction.
    """
    __tablename__ = "print_jobs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(foreign_key="transactions.id")
    printer_name: str
    status: PrintStatus
    error_message: Optional[str] = None
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationship
    transaction: Optional[Transaction] = Relationship(back_populates="print_jobs")


class ProcessingLog(SQLModel, table=True):
    """
    Audit trail for each transaction step.
    
    Replaces text file logs with structured database records.
    """
    __tablename__ = "processing_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(foreign_key="transactions.id")
    step: str  # e.g., "parsing", "downloading", "printing"
    status: str  # "success", "error"
    details: Optional[str] = None  # Error message or context
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationship
    transaction: Optional[Transaction] = Relationship(back_populates="logs")


# ============================================================
# GRANDCHILD MODEL (belongs to GmailMessage)
# ============================================================

class Attachment(SQLModel, table=True):
    """
    Tracks files found in Gmail messages.
    """
    __tablename__ = "attachments"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="gmail_messages.id")
    filename: str
    content_type: Optional[str] = None
    local_path: Optional[str] = None
    
    # Relationship
    message: Optional[GmailMessage] = Relationship(back_populates="attachments")
