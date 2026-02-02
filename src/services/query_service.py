"""
Query Service

Read-only database query layer for analytics, reporting, and LLM agents.
Provides high-level query methods with structured data output.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from sqlmodel import Session, select, func, text, col

from .database import DatabaseService
from ..models import (
    Transaction,
    TransactionStatus,
    Item,
    GmailMessage,
    Attachment,
    PrintJob,
    PrintStatus,
    ProcessingLog,
    PipelineRun
)

logger = logging.getLogger(__name__)


class QueryService:
    """
    Read-only query service for complex database queries.
    
    Provides structured data for:
    - LLM agents and assistants
    - Analytics dashboards
    - Troubleshooting and debugging
    - Business intelligence
    """
    
    def __init__(self, db_service: Optional[DatabaseService] = None, demo_mode: bool = False):
        """
        Initialize query service.
        
        Args:
            db_service: DatabaseService instance. If None, creates new instance.
            demo_mode: If True, use demo database instead of production database.
        """
        self.db = db_service or DatabaseService(demo_mode=demo_mode)
        self.engine = self.db.engine
        mode_info = " (demo mode)" if demo_mode else ""
        logger.info(f"QueryService initialized{mode_info}")
    
    # ============================================================
    # TRANSACTION QUERIES
    # ============================================================
    
    def get_transaction_full(self, db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get complete transaction details with all relationships.
        
        Args:
            db_id: Database primary key ID of transaction
            
        Returns:
            Dict with transaction, items, messages, print_jobs, logs or None
        """
        try:
            with Session(self.engine) as session:
                stmt = select(Transaction).where(Transaction.id == db_id)
                tx = session.exec(stmt).first()
                
                if not tx:
                    logger.warning(f"Transaction {db_id} not found")
                    return None
                
                return {
                    "id": tx.id,
                    "vinted_order_id": tx.vinted_order_id,
                    "customer_name": tx.customer_name,
                    "status": tx.status.value,
                    "label_path": tx.label_path,
                    "created_at": tx.created_at.isoformat(),
                    "updated_at": tx.updated_at.isoformat(),
                    "items": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "price": item.price,
                            "currency": item.currency
                        }
                        for item in tx.items
                    ],
                    "gmail_messages": [
                        {
                            "id": msg.id,
                            "gmail_id": msg.gmail_id,
                            "subject": msg.subject,
                            "received_at": msg.received_at.isoformat()
                        }
                        for msg in tx.gmail_messages
                    ],
                    "print_jobs": [
                        {
                            "id": pj.id,
                            "printer_name": pj.printer_name,
                            "status": pj.status.value,
                            "error_message": pj.error_message,
                            "attempted_at": pj.attempted_at.isoformat()
                        }
                        for pj in tx.print_jobs
                    ],
                    "logs": [
                        {
                            "id": log.id,
                            "step": log.step,
                            "status": log.status,
                            "details": log.details,
                            "timestamp": log.timestamp.isoformat()
                        }
                        for log in tx.logs
                    ],
                    "total_value": sum(item.price or 0 for item in tx.items)
                }
                
        except Exception as e:
            logger.error(f"Failed to get transaction {db_id}: {e}", exc_info=True)
            return None
    
    def get_transaction_by_order_id(self, vinted_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transaction by Vinted order ID with full details.
        
        Args:
            vinted_order_id: Vinted transaction ID (e.g., "12345")
            
        Returns:
            Complete transaction dict or None
        """
        try:
            with Session(self.engine) as session:
                stmt = select(Transaction).where(
                    Transaction.vinted_order_id == vinted_order_id
                )
                tx = session.exec(stmt).first()
                
                if not tx:
                    return None
                
                return self.get_transaction_full(tx.id)
                
        except Exception as e:
            logger.error(f"Failed to get transaction by order ID {vinted_order_id}: {e}")
            return None
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent transactions with summary info.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction summary dicts
        """
        try:
            with Session(self.engine) as session:
                stmt = select(Transaction).order_by(
                    Transaction.created_at.desc()
                ).limit(limit)
                transactions = session.exec(stmt).all()
                
                return [
                    {
                        "id": tx.id,
                        "vinted_order_id": tx.vinted_order_id,
                        "customer_name": tx.customer_name,
                        "status": tx.status.value,
                        "items_count": len(tx.items),
                        "total_value": sum(item.price or 0 for item in tx.items),
                        "created_at": tx.created_at.isoformat(),
                        "has_print_errors": any(
                            pj.status == PrintStatus.FAILED for pj in tx.print_jobs
                        )
                    }
                    for tx in transactions
                ]
                
        except Exception as e:
            logger.error(f"Failed to get recent transactions: {e}", exc_info=True)
            return []
    
    def search_transactions(
        self,
        query: Optional[str] = None,
        status: Optional[TransactionStatus] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search transactions with multiple filters.
        
        Args:
            query: Search in order ID or customer name
            status: Filter by transaction status
            min_value: Minimum total transaction value
            max_value: Maximum total transaction value
            limit: Maximum results to return
            
        Returns:
            List of matching transactions
        """
        try:
            with Session(self.engine) as session:
                stmt = select(Transaction)
                
                # Apply filters
                if query:
                    stmt = stmt.where(
                        (Transaction.vinted_order_id.contains(query)) |
                        (Transaction.customer_name.contains(query))
                    )
                
                if status:
                    stmt = stmt.where(Transaction.status == status)
                
                stmt = stmt.order_by(Transaction.created_at.desc()).limit(limit)
                transactions = session.exec(stmt).all()
                
                # Post-filter by value if needed
                results = []
                for tx in transactions:
                    total_value = sum(item.price or 0 for item in tx.items)
                    
                    if min_value is not None and total_value < min_value:
                        continue
                    if max_value is not None and total_value > max_value:
                        continue
                    
                    results.append({
                        "id": tx.id,
                        "vinted_order_id": tx.vinted_order_id,
                        "customer_name": tx.customer_name,
                        "status": tx.status.value,
                        "items_count": len(tx.items),
                        "total_value": total_value,
                        "created_at": tx.created_at.isoformat()
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search transactions: {e}", exc_info=True)
            return []
    
    # ============================================================
    # FAILURE & TROUBLESHOOTING QUERIES
    # ============================================================
    
    def get_failed_transactions(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get all failed transactions from last N days with error details.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of failed transactions with error logs
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                stmt = select(Transaction).where(
                    Transaction.status == TransactionStatus.FAILED,
                    Transaction.created_at >= cutoff
                ).order_by(Transaction.created_at.desc())
                
                transactions = session.exec(stmt).all()
                
                return [
                    {
                        "id": tx.id,
                        "vinted_order_id": tx.vinted_order_id,
                        "customer_name": tx.customer_name,
                        "created_at": tx.created_at.isoformat(),
                        "items_count": len(tx.items),
                        "error_logs": [
                            {
                                "step": log.step,
                                "details": log.details,
                                "timestamp": log.timestamp.isoformat()
                            }
                            for log in tx.logs if log.status == "error"
                        ],
                        "print_errors": [
                            {
                                "printer": pj.printer_name,
                                "error": pj.error_message,
                                "attempted_at": pj.attempted_at.isoformat()
                            }
                            for pj in tx.print_jobs if pj.status == PrintStatus.FAILED
                        ]
                    }
                    for tx in transactions
                ]
                
        except Exception as e:
            logger.error(f"Failed to get failed transactions: {e}", exc_info=True)
            return []
    
    def get_print_failures(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get all print job failures with details.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of failed print jobs
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                stmt = select(PrintJob).where(
                    PrintJob.status == PrintStatus.FAILED,
                    PrintJob.attempted_at >= cutoff
                ).order_by(PrintJob.attempted_at.desc())
                
                print_jobs = session.exec(stmt).all()
                
                return [
                    {
                        "id": pj.id,
                        "db_id": pj.transaction_id,
                        "vinted_order_id": pj.transaction.vinted_order_id,
                        "printer_name": pj.printer_name,
                        "error_message": pj.error_message,
                        "attempted_at": pj.attempted_at.isoformat()
                    }
                    for pj in print_jobs
                ]
                
        except Exception as e:
            logger.error(f"Failed to get print failures: {e}", exc_info=True)
            return []
    
    def get_processing_errors_by_step(self, days: int = 7) -> Dict[str, List[Dict]]:
        """
        Get all processing errors grouped by step.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict mapping step name to list of errors
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                stmt = select(ProcessingLog).where(
                    ProcessingLog.status == "error",
                    ProcessingLog.timestamp >= cutoff
                ).order_by(ProcessingLog.timestamp.desc())
                
                logs = session.exec(stmt).all()
                
                # Group by step
                errors_by_step = defaultdict(list)
                for log in logs:
                    errors_by_step[log.step].append({
                        "db_id": log.transaction_id,
                        "vinted_order_id": log.transaction.vinted_order_id,
                        "details": log.details,
                        "timestamp": log.timestamp.isoformat()
                    })
                
                return dict(errors_by_step)
                
        except Exception as e:
            logger.error(f"Failed to get processing errors: {e}", exc_info=True)
            return {}
    
    # ============================================================
    # ANALYTICS & STATISTICS QUERIES
    # ============================================================
    
    def get_pipeline_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with various statistics
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                
                # Total transactions
                total_stmt = select(Transaction).where(Transaction.created_at >= cutoff)
                total_txs = session.exec(total_stmt).all()
                total = len(total_txs)
                
                # By status
                status_counts = {}
                for status in TransactionStatus:
                    count = len([tx for tx in total_txs if tx.status == status])
                    status_counts[status.value] = count
                
                # Print statistics
                print_stmt = select(PrintJob).where(PrintJob.attempted_at >= cutoff)
                print_jobs = session.exec(print_stmt).all()
                
                print_success = sum(1 for pj in print_jobs if pj.status == PrintStatus.SUCCESS)
                print_failed = sum(1 for pj in print_jobs if pj.status == PrintStatus.FAILED)
                print_total = len(print_jobs)
                
                # Revenue statistics
                total_revenue = sum(
                    sum(item.price or 0 for item in tx.items)
                    for tx in total_txs
                )
                
                # Average items per transaction
                total_items = sum(len(tx.items) for tx in total_txs)
                avg_items = total_items / total if total > 0 else 0
                
                # Pipeline runs
                run_stmt = select(PipelineRun).where(PipelineRun.start_time >= cutoff)
                runs = session.exec(run_stmt).all()
                
                return {
                    "period_days": days,
                    "total_transactions": total,
                    "status_breakdown": status_counts,
                    "completed_transactions": status_counts.get("COMPLETED", 0),
                    "failed_transactions": status_counts.get("FAILED", 0),
                    "success_rate": (status_counts.get("COMPLETED", 0) / total * 100) if total > 0 else 0,
                    "print_jobs": {
                        "total": print_total,
                        "successful": print_success,
                        "failed": print_failed,
                        "success_rate": (print_success / print_total * 100) if print_total > 0 else 0
                    },
                    "revenue": {
                        "total": round(total_revenue, 2),
                        "average_per_transaction": round(total_revenue / total, 2) if total > 0 else 0
                    },
                    "items": {
                        "total": total_items,
                        "average_per_transaction": round(avg_items, 2)
                    },
                    "pipeline_runs": len(runs)
                }
                
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}", exc_info=True)
            return {}
    
    def get_revenue_by_period(
        self,
        days: int = 30,
        group_by: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get revenue grouped by time period.
        
        Args:
            days: Number of days to analyze
            group_by: Grouping period ("day", "week", "month")
            
        Returns:
            List of dicts with period and revenue
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                stmt = select(Transaction).where(Transaction.created_at >= cutoff)
                transactions = session.exec(stmt).all()
                
                # Group by period
                period_revenue = defaultdict(float)
                
                for tx in transactions:
                    total = sum(item.price or 0 for item in tx.items)
                    
                    if group_by == "day":
                        period_key = tx.created_at.strftime("%Y-%m-%d")
                    elif group_by == "week":
                        period_key = tx.created_at.strftime("%Y-W%W")
                    elif group_by == "month":
                        period_key = tx.created_at.strftime("%Y-%m")
                    else:
                        period_key = tx.created_at.strftime("%Y-%m-%d")
                    
                    period_revenue[period_key] += total
                
                # Convert to sorted list
                results = [
                    {"period": period, "revenue": round(revenue, 2)}
                    for period, revenue in sorted(period_revenue.items())
                ]
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get revenue by period: {e}", exc_info=True)
            return []
    
    def get_top_selling_items(self, days: int = 30, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently sold items.
        
        Args:
            days: Number of days to analyze
            limit: Maximum number of items to return
            
        Returns:
            List of items with frequency and revenue
        """
        try:
            with Session(self.engine) as session:
                cutoff = datetime.now() - timedelta(days=days)
                
                # Get items from recent transactions
                stmt = select(Item).join(Transaction).where(
                    Transaction.created_at >= cutoff
                )
                items = session.exec(stmt).all()
                
                # Count occurrences and sum revenue
                item_stats = defaultdict(lambda: {"count": 0, "revenue": 0.0})
                
                for item in items:
                    # Normalize title for grouping
                    title = item.title.lower().strip()
                    item_stats[title]["count"] += 1
                    item_stats[title]["revenue"] += item.price or 0
                
                # Convert to sorted list
                results = [
                    {
                        "item_title": title,
                        "times_sold": stats["count"],
                        "total_revenue": round(stats["revenue"], 2),
                        "average_price": round(stats["revenue"] / stats["count"], 2)
                    }
                    for title, stats in item_stats.items()
                ]
                
                # Sort by times sold
                results.sort(key=lambda x: x["times_sold"], reverse=True)
                
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get top selling items: {e}", exc_info=True)
            return []
    
    # ============================================================
    # PIPELINE RUN QUERIES
    # ============================================================
    
    def get_pipeline_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent pipeline run history.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of pipeline run summaries
        """
        try:
            with Session(self.engine) as session:
                stmt = select(PipelineRun).order_by(
                    PipelineRun.start_time.desc()
                ).limit(limit)
                runs = session.exec(stmt).all()
                
                return [
                    {
                        "id": run.id,
                        "start_time": run.start_time.isoformat(),
                        "end_time": run.end_time.isoformat() if run.end_time else None,
                        "status": run.status,
                        "items_processed": run.items_processed,
                        "items_failed": run.items_failed,
                        "duration_seconds": (
                            (run.end_time - run.start_time).total_seconds()
                            if run.end_time else None
                        )
                    }
                    for run in runs
                ]
                
        except Exception as e:
            logger.error(f"Failed to get pipeline runs: {e}", exc_info=True)
            return []
    
    # ============================================================
    # SAFE SQL EXECUTION (for LLM-generated queries)
    # ============================================================
    
    def execute_safe_query(self, sql_query: str) -> Tuple[bool, Any]:
        """
        Execute read-only SQL query safely.
        
        Validates query before execution to prevent destructive operations.
        Use this for LLM-generated SQL queries.
        
        Args:
            sql_query: SQL SELECT statement
            
        Returns:
            Tuple of (success: bool, result: List[Dict] or error message)
        """
        # Validate it's a SELECT query
        cleaned = sql_query.strip().upper()
        
        if not cleaned.startswith("SELECT"):
            return False, "Only SELECT queries are allowed"
        
        # Check for destructive operations
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
        if any(keyword in cleaned for keyword in forbidden):
            return False, f"Forbidden operation detected: {sql_query}"
        
        try:
            with Session(self.engine) as session:
                result = session.exec(text(sql_query))
                
                # Convert to list of dicts
                if result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    return True, rows
                else:
                    return True, []
                    
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return False, f"Query execution error: {str(e)}"
    
    # ============================================================
    # SUMMARY QUERIES
    # ============================================================
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get high-level dashboard summary for quick overview.
        
        Returns:
            Dict with key metrics for dashboard display
        """
        try:
            with Session(self.engine) as session:
                # Today's transactions
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                today_stmt = select(Transaction).where(Transaction.created_at >= today_start)
                today_txs = session.exec(today_stmt).all()
                
                # Pending transactions (any status except COMPLETED/FAILED)
                pending_stmt = select(Transaction).where(
                    Transaction.status.in_([
                        TransactionStatus.PENDING,
                        TransactionStatus.PARSED,
                        TransactionStatus.MATCHED,
                        TransactionStatus.PRINTED
                    ])
                )
                pending = len(session.exec(pending_stmt).all())
                
                # Failed transactions needing attention
                failed_stmt = select(Transaction).where(Transaction.status == TransactionStatus.FAILED)
                failed = len(session.exec(failed_stmt).all())
                
                # Recent print success rate
                recent_prints = select(PrintJob).order_by(PrintJob.attempted_at.desc()).limit(100)
                prints = session.exec(recent_prints).all()
                print_success_rate = (
                    sum(1 for p in prints if p.status == PrintStatus.SUCCESS) / len(prints) * 100
                    if prints else 0
                )
                
                return {
                    "today": {
                        "transactions": len(today_txs),
                        "revenue": round(sum(
                            sum(item.price or 0 for item in tx.items)
                            for tx in today_txs
                        ), 2)
                    },
                    "pending_transactions": pending,
                    "failed_transactions": failed,
                    "recent_print_success_rate": round(print_success_rate, 1),
                    "needs_attention": failed > 0 or print_success_rate < 90
                }
                
        except Exception as e:
            logger.error(f"Failed to get dashboard summary: {e}", exc_info=True)
            return {}
