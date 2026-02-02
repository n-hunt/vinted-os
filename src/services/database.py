"""
Database Service

Handles all SQLite database operations with WAL mode and proper session management.
Uses context managers to prevent dangling locks.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from sqlmodel import SQLModel, Session, select, create_engine, text

from ..config_loader import config
from ..models import (
    PipelineRun,
    Transaction,
    TransactionStatus,
    Item,
    GmailMessage,
    Attachment,
    PrintJob,
    PrintStatus,
    ProcessingLog
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    SQLite database service with WAL mode for concurrent access.
    
    All write operations use context managers to prevent lock issues.
    """
    
    def __init__(self, demo_mode: bool = False):
        """
        Initialize database service with WAL mode enabled.
        
        Args:
            demo_mode: If True, use demo database instead of production database
        """
        # Get database filename from config
        if demo_mode:
            db_filename = config.get('database.demo_filename', 'demo_db.db')
            logger.info("Running in DEMO mode - using demo database")
        else:
            db_filename = config.get('database.filename', 'vinted_os.db')
        
        echo_sql = config.get('database.echo_sql', False)
        
        # Create engine with anti-freeze settings
        db_url = f"sqlite:///{db_filename}"
        self.engine = create_engine(
            db_url,
            echo=echo_sql,
            connect_args={
                "check_same_thread": False,  # Allow threading
                "timeout": 30                # Wait 30s for lock release (default is 5s)
            }
        )
        
        logger.info(f"Database engine initialized: {db_filename}")
        
        # Create tables first, then enable WAL mode
        self._create_tables()
        self._enable_wal_mode()
    
    def _enable_wal_mode(self) -> None:
        """
        Enable Write-Ahead Logging mode for concurrent access.
        
        WAL mode allows multiple readers and one writer simultaneously.
        Creates .db-wal and .db-shm temporary files.
        """
        try:
            with self.engine.connect() as connection:
                # Enable WAL mode (allows concurrent reads during writes)
                connection.execute(text("PRAGMA journal_mode=WAL;"))
                
                # Optimize disk syncing
                connection.execute(text("PRAGMA synchronous=NORMAL;"))
                
                connection.commit()
                
            logger.info("WAL mode enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable WAL mode: {e}", exc_info=True)
            # Non-fatal - continue with default mode
    
    def _create_tables(self) -> None:
        """
        Create all database tables if they don't exist.
        """
        try:
            SQLModel.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}", exc_info=True)
            raise
    
    # ============================================================
    # TRANSACTION CRUD OPERATIONS
    # ============================================================
    
    def create_transaction(
        self,
        transaction_id: str,
        customer: Optional[str] = None,
        items_data: Optional[List[Dict[str, Any]]] = None,
        pipeline_run_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Create a new transaction with associated items.
        
        Args:
            transaction_id: Vinted order ID (e.g., "12345")
            customer: Customer name (optional)
            items_data: List of dicts with 'item' and 'price' keys
            pipeline_run_id: ID of current pipeline run (optional)
            
        Returns:
            Database ID of created transaction, or None if failed
        """
        try:
            with Session(self.engine) as session:
                # Create transaction
                transaction = Transaction(
                    vinted_order_id=transaction_id,
                    customer_name=customer,
                    status=TransactionStatus.PENDING,
                    pipeline_run_id=pipeline_run_id
                )
                
                session.add(transaction)
                session.commit()
                session.refresh(transaction)
                
                # Create items if provided
                if items_data:
                    for item_dict in items_data:
                        item = Item(
                            transaction_id=transaction.id,
                            title=item_dict.get('item', 'Unknown Item'),
                            price=item_dict.get('price'),
                            currency=item_dict.get('currency', 'GBP')
                        )
                        session.add(item)
                    
                    session.commit()
                
                logger.info(
                    f"Created transaction {transaction_id} (DB ID: {transaction.id}) "
                    f"with {len(items_data) if items_data else 0} items"
                )
                
                return transaction.id
                
        except Exception as e:
            logger.warning(f"Failed to create transaction {transaction_id} (non-fatal): {e}")
            return None
    
    def update_transaction_status(
        self,
        transaction_id: int,
        status: TransactionStatus,
        label_path: Optional[str] = None
    ) -> bool:
        """
        Update transaction status and optionally label path.
        
        Args:
            transaction_id: Database ID of transaction
            status: New status
            label_path: Path to generated label (optional)
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Guard clause: Null safety
        if transaction_id is None:
            logger.debug("Skipping DB update: No active transaction ID")
            return False
        
        try:
            with Session(self.engine) as session:
                # Fetch transaction
                statement = select(Transaction).where(Transaction.id == transaction_id)
                transaction = session.exec(statement).first()
                
                if not transaction:
                    logger.warning(f"Transaction ID {transaction_id} not found")
                    return False
                
                # Update fields
                transaction.status = status
                transaction.updated_at = datetime.now(timezone.utc)
                
                if label_path:
                    transaction.label_path = label_path
                
                session.add(transaction)
                session.commit()
                
                logger.debug(f"Updated transaction {transaction_id} status to {status}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to update transaction {transaction_id} (non-fatal): {e}")
            return False
    
    def link_gmail_message(
        self,
        transaction_id: int,
        message_data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Link a Gmail message to a transaction.
        
        Args:
            transaction_id: Database ID of transaction
            message_data: Dict with 'gmail_id', 'subject', 'received_at' keys
            
        Returns:
            Database ID of created message, or None if failed
        """
        # Guard clause: Null safety
        if transaction_id is None:
            logger.debug("Skipping DB link: No active transaction ID")
            return None
        
        try:
            with Session(self.engine) as session:
                # Create Gmail message record
                gmail_message = GmailMessage(
                    transaction_id=transaction_id,
                    gmail_id=message_data.get('gmail_id', ''),
                    subject=message_data.get('subject', ''),
                    received_at=message_data.get('received_at', datetime.now(timezone.utc))
                )
                
                session.add(gmail_message)
                session.commit()
                session.refresh(gmail_message)
                
                logger.debug(
                    f"Linked Gmail message {gmail_message.gmail_id} "
                    f"to transaction {transaction_id}"
                )
                
                return gmail_message.id
                
        except Exception as e:
            logger.warning(f"Failed to link Gmail message (non-fatal): {e}")
            return None
    
    # ============================================================
    # LOGGING & TELEMETRY OPERATIONS
    # ============================================================
    
    def log_step(
        self,
        transaction_id: int,
        step: str,
        status: str,
        details: Optional[str] = None
    ) -> bool:
        """
        Log a processing step for audit trail.
        
        Replaces text file logging with structured database records.
        
        Args:
            transaction_id: Database ID of transaction
            step: Step name (e.g., "parsing", "downloading", "printing")
            status: Status ("success", "error")
            details: Optional error message or context
            
        Returns:
            True if logged successfully, False otherwise
        """
        # Guard clause: Null safety
        if transaction_id is None:
            logger.debug("Skipping DB log: No active transaction ID")
            return False
        
        try:
            with Session(self.engine) as session:
                log_entry = ProcessingLog(
                    transaction_id=transaction_id,
                    step=step,
                    status=status,
                    details=details,
                    timestamp=datetime.now(timezone.utc)
                )
                
                session.add(log_entry)
                session.commit()
                
                logger.debug(f"Logged step '{step}' for transaction {transaction_id}: {status}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to log step '{step}' (non-fatal): {e}")
            return False
    
    def add_print_job(
        self,
        transaction_id: int,
        printer_name: str,
        status: PrintStatus,
        error_message: Optional[str] = None
    ) -> Optional[int]:
        """
        Record a print job attempt.
        
        Args:
            transaction_id: Database ID of transaction
            printer_name: Name of printer used
            status: Print status (SUCCESS, FAILED, SKIPPED)
            error_message: Optional error message if failed
            
        Returns:
            Database ID of created print job, or None if failed
        """
        # Guard clause: Null safety
        if transaction_id is None:
            logger.debug("Skipping DB print job: No active transaction ID")
            return None
        
        try:
            with Session(self.engine) as session:
                print_job = PrintJob(
                    transaction_id=transaction_id,
                    printer_name=printer_name,
                    status=status,
                    error_message=error_message,
                    attempted_at=datetime.now(timezone.utc)
                )
                
                session.add(print_job)
                session.commit()
                session.refresh(print_job)
                
                logger.debug(
                    f"Recorded print job for transaction {transaction_id}: {status.value}"
                )
                
                return print_job.id
                
        except Exception as e:
            logger.warning(f"Failed to record print job (non-fatal): {e}")
            return None
    
    # ============================================================
    # PIPELINE RUN TRACKING
    # ============================================================
    
    def start_run(self) -> Optional[int]:
        """
        Start a new pipeline run.
        
        Creates a PipelineRun record to track this execution.
        
        Returns:
            Database ID of created run, or None if failed
        """
        try:
            with Session(self.engine) as session:
                pipeline_run = PipelineRun(
                    start_time=datetime.now(timezone.utc),
                    status="running",
                    items_processed=0,
                    items_failed=0
                )
                
                session.add(pipeline_run)
                session.commit()
                session.refresh(pipeline_run)
                
                logger.info(f"Started pipeline run {pipeline_run.id}")
                return pipeline_run.id
                
        except Exception as e:
            logger.warning(f"Failed to start pipeline run (non-fatal): {e}")
            return None
    
    def end_run(
        self,
        run_id: int,
        items_processed: int,
        items_failed: int,
        status: str = "completed"
    ) -> bool:
        """
        End a pipeline run with final statistics.
        
        Args:
            run_id: Database ID of pipeline run
            items_processed: Number of items successfully processed
            items_failed: Number of items that failed
            status: Final status (default: "completed")
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with Session(self.engine) as session:
                # Fetch pipeline run
                statement = select(PipelineRun).where(PipelineRun.id == run_id)
                pipeline_run = session.exec(statement).first()
                
                if not pipeline_run:
                    logger.warning(f"Pipeline run ID {run_id} not found")
                    return False
                
                # Update fields
                pipeline_run.end_time = datetime.now(timezone.utc)
                pipeline_run.status = status
                pipeline_run.items_processed = items_processed
                pipeline_run.items_failed = items_failed
                
                session.add(pipeline_run)
                session.commit()
                
                logger.info(
                    f"Ended pipeline run {run_id}: "
                    f"{items_processed} processed, {items_failed} failed"
                )
                return True
                
        except Exception as e:
            logger.warning(f"Failed to end pipeline run {run_id} (non-fatal): {e}")
            return False
    
    # ============================================================
    # QUERY OPERATIONS
    # ============================================================
    
    def get_transaction_by_vinted_id(self, vinted_order_id: str) -> Optional[int]:
        """
        Get database transaction ID by Vinted order ID.
        
        Args:
            vinted_order_id: Vinted transaction ID (e.g., "12345")
            
        Returns:
            Database ID of transaction, or None if not found
        """
        try:
            with Session(self.engine) as session:
                statement = select(Transaction).where(
                    Transaction.vinted_order_id == vinted_order_id
                )
                transaction = session.exec(statement).first()
                
                if transaction:
                    return transaction.id
                
                return None
                
        except Exception as e:
            logger.warning(f"Failed to query transaction {vinted_order_id} (non-fatal): {e}")
            return None
