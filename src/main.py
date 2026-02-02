"""
VintedOS Main Orchestrator

Coordinates the ETL pipeline:
1. Fetch emails from Gmail
2. Parse PDFs and extract data
3. Process and enhance labels
4. Print labels
5. Clean up processed messages

"""

import os
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

# Initialize logging first
from .logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

# Import services and utilities
from .config_loader import config
from .services.gmail import GmailConnector, AttachmentData
from .services.printer import PrintService, PrinterStatus
from .services.database import DatabaseService
from .docuflow.parser import TransactionParser, extract_transaction_id_from_filename
from .docuflow.vision import PDFProcessor
from .docuflow.generator import LabelGenerator


class TransactionData:
    """
    Data structure for tracking a complete transaction.
    
    Attributes:
        transaction_id: Unique transaction identifier
        message_ids: Gmail message IDs associated with this transaction
        items: List of items with prices
        label_filename: Original label filename
        label_path: Path to saved label file
    """
    
    def __init__(self, transaction_id: str):
        self.transaction_id = transaction_id
        self.message_ids: Set[str] = set()
        self.items: List[Dict[str, any]] = []
        self.label_filename: str = ""
        self.label_path: str = ""
    
    def is_complete(self) -> bool:
        """Check if transaction has all required components."""
        return bool(
            self.transaction_id and
            self.items and
            self.label_path and
            os.path.exists(self.label_path)
        )
    
    def __repr__(self) -> str:
        return (
            f"TransactionData(id={self.transaction_id}, "
            f"items={len(self.items)}, "
            f"messages={len(self.message_ids)}, "
            f"complete={self.is_complete()})"
        )


class VintedOSPipeline:
    """
    Main ETL pipeline orchestrator.
    
    Coordinates all services and processing steps.
    """
    
    def __init__(self, dry_run: bool = False, demo_mode: bool = False):
        """
        Initialize pipeline with services.
        
        Args:
            dry_run: If True, save PDFs to debug folder instead of printing
            demo_mode: If True, use demo database instead of production database
        """
        logger.info("Initializing VintedOS Pipeline")
        
        self.dry_run = dry_run
        self.demo_mode = demo_mode
        
        # Initialize services
        self.db = DatabaseService(demo_mode=demo_mode)
        self.gmail = GmailConnector()
        self.printer = PrintService(dry_run=dry_run)
        self.parser = TransactionParser()
        self.pdf_processor = PDFProcessor()
        self.label_generator = LabelGenerator()
        
        # Start pipeline run tracking
        self.run_id = None
        try:
            self.run_id = self.db.start_run()
        except Exception as e:
            logger.warning(f"Database telemetry failed (start_run): {e}")
            # PASS - Pipeline continues without run tracking
        
        # Track current transaction across methods
        self.current_tx_id = None
        
        # Setup output directory
        self.labels_dir = Path(config.get('paths.labels_output', './static/services/labels'))
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        mode_info = f"dry_run={dry_run}"
        if demo_mode:
            mode_info += f", demo_mode={demo_mode}"
        logger.info(f"Pipeline initialized ({mode_info}, run_id={self.run_id})")
    
    def _safe_db_call(self, func, *args, context: str = "", **kwargs):
        """
        Fire-and-forget wrapper for database telemetry calls.
        
        Prevents database errors from affecting the core pipeline.
        
        Args:
            func: Database method to call
            *args: Positional arguments for the method
            context: Optional context string for error messages
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result of func if successful, None otherwise
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            func_name = getattr(func, '__name__', str(func))
            context_msg = f" ({context})" if context else ""
            
            # Special handling for "database is locked" errors
            error_str = str(e).lower()
            if "database is locked" in error_str or "database locked" in error_str:
                logger.error(
                    f"DB telemetry failed [{func_name}]{context_msg}: {e}\n"
                    "    → HINT: Database is locked. This usually means a tool like "
                    "'DB Browser for SQLite' has the file open in 'Write' mode. "
                    "Close the tool or switch to 'Read Only' mode."
                )
            else:
                logger.warning(f"DB telemetry failed [{func_name}]{context_msg}: {e}")
            
            return None
    
    def run(self) -> Dict[str, any]:
        """
        Execute the complete ETL pipeline.
        
        Returns:
            Dict with pipeline execution summary
        """
        logger.info("="*60)
        logger.info("Starting VintedOS ETL Pipeline")
        logger.info("="*60)
        
        summary = {}
        
        try:
            # Step 1: Fetch and parse return forms
            logger.info("\n=== Step 1: Processing Return Forms ===")
            return_forms_data = self._fetch_and_parse_return_forms()
            logger.info(f"Parsed {len(return_forms_data)} return form(s)")
            
            # Step 2: Fetch and process shipping labels
            logger.info("\n=== Step 2: Processing Shipping Labels ===")
            labels_data = self._fetch_and_process_labels()
            logger.info(f"Processed {len(labels_data)} shipping label(s)")
            
            # Step 3: Match and generate enhanced labels
            logger.info("\n=== Step 3: Matching and Generating Labels ===")
            transactions = self._match_and_generate(return_forms_data, labels_data)
            logger.info(f"Successfully matched {len(transactions)} transaction(s)")
            
            # Step 4: Print labels
            logger.info("\n=== Step 4: Printing Labels ===")
            print_results = self._print_labels(transactions)
            
            # Step 5: Cleanup
            logger.info("\n=== Step 5: Cleanup ===")
            cleanup_results = self._cleanup_messages(transactions)
            
            # Summary
            summary = self._generate_summary(
                return_forms_data,
                labels_data,
                transactions,
                print_results,
                cleanup_results
            )
            
            logger.info("\n" + "="*60)
            logger.info("Pipeline Execution Complete")
            logger.info("="*60)
            logger.info(f"Summary: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            summary = {
                "success": False,
                "error": str(e)
            }
            return summary
            
        finally:
            # Finalize pipeline run in database using fire-and-forget pattern
            if self.run_id:
                # Calculate statistics
                items_processed = summary.get("transactions_matched", 0)
                items_failed = summary.get("return_forms_found", 0) - items_processed
                
                # Determine final status
                if summary.get("success", False):
                    final_status = "completed"
                else:
                    final_status = "failed"
                
                # Fire-and-forget end run
                self._safe_db_call(
                    self.db.end_run,
                    self.run_id,
                    items_processed=items_processed,
                    items_failed=items_failed,
                    status=final_status,
                    context="pipeline finalization"
                )
    
    def _fetch_and_parse_return_forms(self) -> Dict[str, TransactionData]:
        """
        Fetch return form PDFs and extract transaction data.
        
        Returns:
            Dict mapping transaction_id -> TransactionData
        """
        # Get query and pattern from config
        query = config.get('gmail.queries.return_forms', 'has:attachment filename:pdf')
        pattern = config.get('patterns.return_form', '^Order-return-form')
        
        # Fetch attachments
        attachments = self.gmail.fetch_attachments_with_pattern(
            query=query,
            filename_pattern=pattern,
            extract_body_text=False
        )
        
        logger.info(f"Found {len(attachments)} return form attachment(s)")
        
        # Process each attachment
        transactions = {}
        
        for att in attachments:
            try:
                # Download PDF
                att_data = self.gmail.download_attachment(att)
                if not att_data:
                    logger.warning(f"Failed to download attachment: {att.filename}")
                    continue
                
                # Parse PDF
                parsed = self.parser.parse_pdf(att_data.raw_bytes)
                tid = parsed["transaction_id"]
                items = parsed["items"]
                
                if tid == "0" or not items:
                    logger.warning(
                        f"Invalid return form data: TID={tid}, items={len(items)}"
                    )
                    continue
                
                # Create or update transaction record
                if tid not in transactions:
                    transactions[tid] = TransactionData(tid)
                
                transactions[tid].message_ids.add(att.message_id)
                transactions[tid].items = items
                
                logger.info(
                    f"Parsed return form for TID {tid}: {len(items)} item(s)"
                )
                
                # Save to database (after parsing completes)
                # Reset transaction ID to None for safety
                db_tx_id = None
                try:
                    db_tx_id = self.db.create_transaction(
                        transaction_id=tid,
                        customer=None,  # Customer name not available in return form
                        items_data=items,
                        pipeline_run_id=self.run_id
                    )
                    
                    if db_tx_id:
                        self.db.log_step(db_tx_id, "parsing", "success", f"Parsed {len(items)} items")
                        
                        # Link Gmail message to transaction
                        self.db.link_gmail_message(
                            db_tx_id,
                            {
                                'gmail_id': att.message_id,
                                'subject': f"Return form for {tid}",
                                'received_at': datetime.now(timezone.utc)
                            }
                        )
                except Exception as e:
                    logger.error(f"[Non-Fatal] Failed to create DB transaction for TID {tid}: {e}")
                    # Pipeline proceeds, but db_tx_id remains None
                
            except Exception as e:
                logger.error(
                    f"Error processing return form {att.filename}: {e}",
                    exc_info=True
                )
                # Continue processing other attachments (fault tolerance)
                continue
        
        return transactions
    
    def _fetch_and_process_labels(self) -> Dict[str, AttachmentData]:
        """
        Fetch shipping labels and process images.
        
        Returns:
            Dict mapping transaction_id -> AttachmentData (with processed PDF bytes)
        """
        # Get query and pattern from config
        query = config.get('gmail.queries.shipping_labels', 
                          'has:attachment filename:pdf subject:\'shipping label\'')
        pattern = config.get('patterns.shipping_label', '^Vinted-(?:Label|Digital-Label)')
        
        # Fetch attachments
        attachments = self.gmail.fetch_attachments_with_pattern(
            query=query,
            filename_pattern=pattern,
            extract_body_text=True  # May contain useful info
        )
        
        logger.info(f"Found {len(attachments)} shipping label(s)")
        
        # Download and process each label
        labels_data = {}
        
        for att in attachments:
            try:
                # Download PDF
                att_data = self.gmail.download_attachment(att)
                if not att_data:
                    logger.warning(f"Failed to download label: {att.filename}")
                    continue
                
                # Extract transaction ID from filename
                tid = extract_transaction_id_from_filename(att.filename)
                if not tid:
                    logger.warning(
                        f"Could not extract transaction ID from: {att.filename}"
                    )
                    continue
                
                # Process PDF (crop, resize, binarize) - CPU intensive, NO DB SESSION
                processed_bytes = self.pdf_processor.crop_and_resize_to_4x6(
                    att_data.raw_bytes
                )
                
                # Update attachment data with processed bytes
                att_data.raw_bytes = processed_bytes
                
                labels_data[tid] = att_data
                
                logger.info(f"Processed label for TID {tid}")
                
                # Save to database (after processing completes) - Quick write
                # Note: We need to find the DB transaction ID by vinted_order_id
                # For now, we'll create a helper method or update in the matching phase
                # This is logged in Step 3.4 during matching instead
                
            except Exception as e:
                logger.error(
                    f"Error processing label {att.filename}: {e}",
                    exc_info=True
                )
                # Continue processing other labels (fault tolerance)
                continue
        
        return labels_data
    
    def _match_and_generate(
        self,
        return_forms: Dict[str, TransactionData],
        labels: Dict[str, AttachmentData]
    ) -> Dict[str, TransactionData]:
        """
        Match return forms with labels and generate enhanced PDFs.
        
        This replaces the monolithic coupler() function.
        
        Args:
            return_forms: Dict of TransactionData from return forms
            labels: Dict of AttachmentData from shipping labels
            
        Returns:
            Dict of complete TransactionData objects
        """
        matched_transactions = {}
        
        for tid, label_data in labels.items():
            # Check if we have a matching return form
            if tid not in return_forms:
                logger.warning(
                    f"No return form found for label TID {tid}, skipping"
                )
                continue
            
            transaction = return_forms[tid]
            
            try:
                # Add item overlay to label - CPU intensive, NO DB SESSION
                enhanced_pdf = self.label_generator.add_items_to_pdf(
                    pdf_bytes=label_data.raw_bytes,
                    items=transaction.items,
                    transaction_id=tid
                )
                
                # Save enhanced label - File I/O, NO DB SESSION
                label_filename = self._generate_label_filename(tid, label_data.filename)
                label_path = self.labels_dir / label_filename
                
                with open(label_path, 'wb') as f:
                    f.write(enhanced_pdf)
                
                logger.info(f"Saved enhanced label: {label_path}")
                
                # Update transaction record
                transaction.label_filename = label_data.filename
                transaction.label_path = str(label_path)
                transaction.message_ids.add(label_data.message_id)
                
                # Check if multi-item order
                multi_item_threshold = config.get('processing.multi_item_threshold', 1)
                if len(transaction.items) > multi_item_threshold:
                    self._generate_multi_item_lists(transaction)
                
                matched_transactions[tid] = transaction
                
                logger.info(
                    f"Matched transaction {tid}: "
                    f"{len(transaction.items)} item(s), "
                    f"{len(transaction.message_ids)} message(s)"
                )
                
                # Save to database (after file operations complete) - Quick write
                try:
                    db_tx_id = self.db.get_transaction_by_vinted_id(tid)
                    if db_tx_id:
                        # Update status to MATCHED and save label path
                        from .models import TransactionStatus
                        self.db.update_transaction_status(
                            db_tx_id,
                            TransactionStatus.MATCHED,
                            label_path=str(label_path)
                        )
                        
                        # Link shipping label Gmail message
                        self.db.link_gmail_message(
                            db_tx_id,
                            {
                                'gmail_id': label_data.message_id,
                                'subject': f"Shipping label for {tid}",
                                'received_at': datetime.now(timezone.utc)
                            }
                        )
                        
                        # Log successful matching
                        self.db.log_step(
                            db_tx_id,
                            "matching",
                            "success",
                            f"Matched and generated label: {label_filename}"
                        )
                except Exception as e:
                    logger.warning(f"Database telemetry failed for TID {tid}: {e}")
                    # PASS - Continue processing, user still needs their label
                
            except Exception as e:
                logger.error(
                    f"Error generating label for TID {tid}: {e}",
                    exc_info=True
                )
                # Continue processing other transactions (fault tolerance)
                continue
        
        return matched_transactions
    
    def _generate_multi_item_lists(self, transaction: TransactionData) -> None:
        """
        Generate multi-item list PDFs for orders with multiple items.
        
        Args:
            transaction: TransactionData to generate lists for
        """
        try:
            list_pages = self.label_generator.create_items_list_pdf(
                items=transaction.items,
                transaction_id=transaction.transaction_id
            )
            
            # Save each page
            for page_num, pdf_bytes in enumerate(list_pages, 1):
                filename_template = config.get(
                    'paths.templates.multi_item_list',
                    '{transaction_id}_YOINKED_PAGE_{page_num}.pdf'
                )
                filename = filename_template.format(
                    transaction_id=transaction.transaction_id,
                    page_num=page_num
                )
                
                file_path = self.labels_dir / filename
                
                with open(file_path, 'wb') as f:
                    f.write(pdf_bytes)
                
                logger.info(f"Generated multi-item list: {file_path}")
                
        except Exception as e:
            logger.error(
                f"Error generating multi-item lists for {transaction.transaction_id}: {e}",
                exc_info=True
            )
    
    def _generate_label_filename(self, transaction_id: str, original_filename: str) -> str:
        """
        Generate safe filename for saved label.
        
        Args:
            transaction_id: Transaction ID
            original_filename: Original attachment filename
            
        Returns:
            Safe filename string
        """
        # Ensure .pdf extension
        if not original_filename.lower().endswith('.pdf'):
            original_filename = os.path.splitext(original_filename)[0] + '.pdf'
        
        template = config.get(
            'paths.templates.coupled_label',
            '{transaction_id}_{original_filename}'
        )
        
        return template.format(
            transaction_id=transaction_id,
            original_filename=original_filename
        )
    
    def _print_labels(self, transactions: Dict[str, TransactionData]) -> List:
        """
        Print all matched labels.
        
        Args:
            transactions: Dict of complete TransactionData objects
            
        Returns:
            List of PrintJob results
        """
        if not transactions:
            logger.info("No labels to print")
            return []
        
        # Prepare batch print
        file_paths = []
        transaction_ids = []
        
        for tid, transaction in transactions.items():
            if transaction.is_complete():
                file_paths.append(transaction.label_path)
                transaction_ids.append(tid)
        
        logger.info(f"Printing {len(file_paths)} label(s)")
        
        # Execute batch print - Hardware operation (slow)
        results = self.printer.print_batch(file_paths, transaction_ids)
        
        # Record print jobs in database (quick write after hardware operation)
        from .models import PrintStatus as DBPrintStatus
        
        for result in results:
            try:
                # Get database transaction ID by vinted order ID
                db_tx_id = self.db.get_transaction_by_vinted_id(result.transaction_id)
                
                if db_tx_id:
                    # Map PrinterStatus to database PrintStatus
                    if result.status == PrinterStatus.SUCCESS:
                        db_status = DBPrintStatus.SUCCESS
                        tx_status_str = "PRINTED"
                    elif result.status == PrinterStatus.DRY_RUN:
                        db_status = DBPrintStatus.SUCCESS  # Treat dry-run as success
                        tx_status_str = "PRINTED"
                    elif result.status == PrinterStatus.FAILED:
                        db_status = DBPrintStatus.FAILED
                        tx_status_str = "FAILED"
                    else:
                        db_status = DBPrintStatus.SKIPPED
                        tx_status_str = "FAILED"
                    
                    # Record print job
                    self.db.add_print_job(
                        transaction_id=db_tx_id,
                        printer_name=self.printer.printer_name,
                        status=db_status,
                        error_message=result.message if db_status == DBPrintStatus.FAILED else None
                    )
                    
                    # Update transaction status
                    from .models import TransactionStatus
                    if db_status == DBPrintStatus.SUCCESS:
                        self.db.update_transaction_status(
                            db_tx_id,
                            TransactionStatus.PRINTED
                        )
                        self.db.log_step(db_tx_id, "printing", "success", f"Printed to {self.printer.printer_name}")
                    else:
                        self.db.update_transaction_status(
                            db_tx_id,
                            TransactionStatus.FAILED
                        )
                        self.db.log_step(db_tx_id, "printing", "error", result.message)
            except Exception as e:
                logger.warning(f"Database telemetry failed for print job {result.transaction_id}: {e}")
                # PASS - Continue processing, printing already completed
        
        return results
    
    def _cleanup_messages(self, transactions: Dict[str, TransactionData]) -> Dict:
        """
        Trash processed Gmail messages and clean up local files.
        
        Args:
            transactions: Dict of processed TransactionData objects
            
        Returns:
            Dict with cleanup results
        """
        if not transactions:
            logger.info("No messages to clean up")
            return {"trashed": 0, "files_deleted": 0}
        
        # Collect all message IDs
        all_message_ids = set()
        for transaction in transactions.values():
            all_message_ids.update(transaction.message_ids)
        
        logger.info(f"Cleaning up {len(all_message_ids)} message(s)")
        
        # Trash messages
        trash_results = self.gmail.trash_messages(all_message_ids)
        trashed_count = sum(1 for success in trash_results.values() if success)
        
        # Delete local label files (skip during dry run)
        deleted_count = 0
        for transaction in transactions.values():
            if transaction.label_path and os.path.exists(transaction.label_path):
                if self.dry_run:
                    logger.debug(f"Dry run: Skipping deletion of {transaction.label_path}")
                    # Count as "deleted" for summary purposes, but don't actually delete
                    deleted_count += 1
                else:
                    try:
                        os.remove(transaction.label_path)
                        deleted_count += 1
                        logger.debug(f"Deleted local file: {transaction.label_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {transaction.label_path}: {e}")
                        continue
                
                # Log cleanup step to database using fire-and-forget pattern
                try:
                    db_tx_id = self._safe_db_call(
                        self.db.get_transaction_by_vinted_id,
                        transaction.transaction_id,
                        context=f"cleanup lookup for {transaction.transaction_id}"
                    )
                    
                    if db_tx_id:
                        from .models import TransactionStatus
                        
                        # Fire-and-forget status update
                        self._safe_db_call(
                            self.db.update_transaction_status,
                            db_tx_id,
                            TransactionStatus.COMPLETED,
                            context=f"cleanup status {transaction.transaction_id}"
                        )
                        
                        # Fire-and-forget log entry
                        cleanup_message = "Trashed emails" + ("" if self.dry_run else " and deleted local files")
                        self._safe_db_call(
                            self.db.log_step,
                            db_tx_id,
                            "cleanup",
                            "success",
                            cleanup_message,
                            context=f"cleanup log {transaction.transaction_id}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to log cleanup step for {transaction.transaction_id}: {e}")
        
        logger.info(
            f"Cleanup complete: {trashed_count} messages trashed, "
            f"{deleted_count} files deleted"
        )
        
        return {
            "trashed": trashed_count,
            "files_deleted": deleted_count
        }
    
    def _generate_summary(
        self,
        return_forms: Dict,
        labels: Dict,
        transactions: Dict,
        print_results: List,
        cleanup_results: Dict
    ) -> Dict[str, any]:
        """
        Generate execution summary.
        
        Returns:
            Dict with pipeline metrics
        """
        successful_prints = sum(
            1 for r in print_results 
            if r.status.value in ['success', 'dry_run']
        )
        
        return {
            "success": True,
            "return_forms_found": len(return_forms),
            "labels_found": len(labels),
            "transactions_matched": len(transactions),
            "labels_printed": successful_prints,
            "messages_trashed": cleanup_results.get("trashed", 0),
            "files_cleaned": cleanup_results.get("files_deleted", 0),
            "dry_run": self.dry_run
        }


def main(dry_run: bool = False, demo_mode: bool = False) -> Dict[str, any]:
    """
    Main entry point for VintedOS pipeline.
    
    Args:
        dry_run: If True, save to debug folder instead of printing
        demo_mode: If True, use demo database instead of production database
        
    Returns:
        Execution summary dict
    """
    pipeline = VintedOSPipeline(dry_run=dry_run, demo_mode=demo_mode)
    return pipeline.run()


if __name__ == "__main__":
    import sys
    
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        logger.info("Running in DRY-RUN mode (no actual printing)")
    
    summary = main(dry_run=dry_run)
    
    # Exit with appropriate code
    sys.exit(0 if summary.get("success") else 1)
