"""
Printer Service Adapter

Wraps CUPS printing system with dry-run support for testing.
"""

import os
import subprocess
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..config_loader import config

logger = logging.getLogger(__name__)


class PrinterStatus(Enum):
    """Printing operation status (service layer)."""
    SUCCESS = "success"
    FAILED = "failed"
    DRY_RUN = "dry_run"
    SKIPPED = "skipped"


@dataclass
class PrintJob:
    """Data object representing a print job result."""
    file_path: str
    transaction_id: str
    status: PrinterStatus
    message: str
    timestamp: float


class PrintService:
    """
    CUPS printer service with dry-run capability.
    
    In production mode, sends PDFs to physical printer via `lp` command.
    In dry-run mode, saves PDFs to debug folder for inspection.
    """
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize printer service.
        
        Args:
            dry_run: If True, save to debug folder instead of printing
        """
        self.dry_run = dry_run
        
        # Load printer configuration
        printer_config = config.get_section('printer')
        self.printer_name = printer_config.get('name', 'Zebra_GK420d')
        self.darkness = printer_config.get('options', {}).get('darkness', 30)
        self.fit_to_page = printer_config.get('options', {}).get('fit_to_page', True)
        self.job_delay = printer_config.get('delay_between_jobs', 0.5)
        
        # Debug output directory
        self.debug_dir = Path("./logs/print_debug")
        if self.dry_run:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dry-run mode enabled. PDFs will be saved to: {self.debug_dir}")
        else:
            logger.info(f"Print mode enabled. Printer: {self.printer_name}")
    
    def print_label(
        self,
        file_path: str,
        transaction_id: Optional[str] = None
    ) -> PrintJob:
        """
        Print a single label PDF.
        
        Args:
            file_path: Path to PDF file
            transaction_id: Optional transaction ID for logging
            
        Returns:
            PrintJob result object
        """
        transaction_id = transaction_id or "unknown"
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.FAILED,
                message="File not found",
                timestamp=time.time()
            )
        
        if self.dry_run:
            return self._dry_run_save(file_path, transaction_id)
        else:
            return self._execute_print(file_path, transaction_id)
    
    def print_batch(
        self,
        file_paths: List[str],
        transaction_ids: Optional[List[str]] = None
    ) -> List[PrintJob]:
        """
        Print multiple labels with delays between jobs.
        
        Args:
            file_paths: List of PDF file paths
            transaction_ids: Optional list of transaction IDs (must match file_paths length)
            
        Returns:
            List of PrintJob results
        """
        if transaction_ids and len(transaction_ids) != len(file_paths):
            logger.warning(
                f"Transaction ID count mismatch: {len(transaction_ids)} IDs "
                f"for {len(file_paths)} files"
            )
            transaction_ids = None
        
        results = []
        total = len(file_paths)
        
        logger.info(f"Starting batch print job: {total} label(s)")
        
        for idx, file_path in enumerate(file_paths):
            tid = transaction_ids[idx] if transaction_ids else f"batch_{idx}"
            
            result = self.print_label(file_path, tid)
            results.append(result)
            
            # Delay between jobs (except after last job)
            if idx < total - 1:
                time.sleep(self.job_delay)
        
        # Summary
        success_count = sum(1 for r in results if r.status == PrinterStatus.SUCCESS)
        dry_run_count = sum(1 for r in results if r.status == PrinterStatus.DRY_RUN)
        
        if self.dry_run:
            logger.info(f"Batch dry-run complete: {dry_run_count}/{total} saved")
        else:
            logger.info(f"Batch print complete: {success_count}/{total} successful")
        
        return results
    
    def _execute_print(self, file_path: str, transaction_id: str) -> PrintJob:
        """
        Execute actual print command via CUPS.
        
        Args:
            file_path: Path to PDF file
            transaction_id: Transaction ID for logging
            
        Returns:
            PrintJob result
        """
        # Build lp command
        # sudo lp -d Zebra_GK420d -o fit-to-page -o Darkness=30 <file>
        cmd = ["sudo", "lp", "-d", self.printer_name]
        
        if self.fit_to_page:
            cmd.extend(["-o", "fit-to-page"])
        
        cmd.extend(["-o", f"Darkness={self.darkness}"])
        cmd.append(file_path)
        
        logger.debug(f"Executing print command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            logger.info(f"SUCCESS: Spooled label: {transaction_id}")
            
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.SUCCESS,
                message=result.stdout.strip() if result.stdout else "Spooled successfully",
                timestamp=time.time()
            )
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            logger.error(f"ERROR: Print error for {transaction_id}: {error_msg}")
            
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.FAILED,
                message=error_msg,
                timestamp=time.time()
            )
        
        except subprocess.TimeoutExpired:
            logger.error(f"ERROR: Print timeout for {transaction_id}")
            
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.FAILED,
                message="Print command timeout",
                timestamp=time.time()
            )
        
        except Exception as e:
            logger.error(f"ERROR: Unexpected error printing {transaction_id}: {e}")
            
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.FAILED,
                message=str(e),
                timestamp=time.time()
            )
    
    def _dry_run_save(self, file_path: str, transaction_id: str) -> PrintJob:
        """
        Save PDF to debug folder instead of printing.
        
        Args:
            file_path: Source PDF path
            transaction_id: Transaction ID for naming
            
        Returns:
            PrintJob result
        """
        try:
            # Create timestamped filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{transaction_id}_{Path(file_path).name}"
            debug_path = self.debug_dir / filename
            
            # Copy file
            import shutil
            shutil.copy2(file_path, debug_path)
            
            logger.info(f"[DRY-RUN] Saved: {transaction_id} -> {debug_path}")
            
            return PrintJob(
                file_path=str(debug_path),
                transaction_id=transaction_id,
                status=PrinterStatus.DRY_RUN,
                message=f"Saved to {debug_path}",
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"[DRY-RUN] Failed to save {transaction_id}: {e}")
            
            return PrintJob(
                file_path=file_path,
                transaction_id=transaction_id,
                status=PrinterStatus.FAILED,
                message=f"Dry-run save failed: {e}",
                timestamp=time.time()
            )
    
    def get_printer_status(self) -> Dict[str, any]:
        """
        Get printer status from CUPS.
        
        Returns:
            Dict with printer information or error
        """
        if self.dry_run:
            return {
                "mode": "dry_run",
                "available": True,
                "debug_dir": str(self.debug_dir)
            }
        
        try:
            # Run lpstat to get printer status
            result = subprocess.run(
                ["lpstat", "-p", self.printer_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                "mode": "print",
                "printer": self.printer_name,
                "available": result.returncode == 0,
                "status": result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get printer status: {e}")
            return {
                "mode": "print",
                "printer": self.printer_name,
                "available": False,
                "error": str(e)
            }
