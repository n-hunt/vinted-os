"""
DocuFlow Parser Module

Handles PDF text extraction and structured data parsing:
- Transaction ID extraction
- Item name and price parsing
- Multi-page PDF processing

All functions work with raw bytes/strings - no Gmail dependencies.
"""

import io
import re
import logging
from typing import Dict, List, Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber not installed. Install with: pip install pdfplumber")

from ..config_loader import config

logger = logging.getLogger(__name__)


class TransactionParser:
    """
    Parses Vinted return forms and extracts structured transaction data.
    """
    
    def __init__(self):
        """Initialize parser with regex patterns from config."""
        patterns = config.get_section('patterns')
        
        self.transaction_id_pattern = re.compile(patterns.get('transaction_id_text', r'Transaction ID:\s*(\d+)'))
        self.transaction_id_filename_pattern = re.compile(patterns.get('transaction_id_filename', r'(\d{10,})'))
        self.price_pattern = re.compile(patterns.get('price', r'£(\d+\.\d{2})$'))
        
        self.order_section_marker = patterns.get('order_section_start', 'Order Return code Price')
        self.order_end_markers = patterns.get('order_section_end', ['Total:', 'Return codes'])
        self.excluded_keywords = patterns.get('excluded_keywords', [
            'shipping', 'buyer protection', 'total', 'return codes'
        ])
    
    def extract_transaction_id_from_text(self, text: str) -> Optional[str]:
        """
        Extract transaction ID from PDF text content.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Transaction ID as string, or None if not found
        """
        match = self.transaction_id_pattern.search(text)
        if match:
            transaction_id = match.group(1)
            logger.debug(f"Extracted transaction ID from text: {transaction_id}")
            return transaction_id
        
        logger.warning("No transaction ID found in text")
        return None
    
    def extract_transaction_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract transaction ID from filename (e.g., Vinted-Label-16826065613.pdf).
        
        Args:
            filename: PDF filename
            
        Returns:
            Transaction ID as string, or None if not found
        """
        match = self.transaction_id_filename_pattern.search(filename)
        if match:
            transaction_id = match.group(1)
            logger.debug(f"Extracted transaction ID from filename '{filename}': {transaction_id}")
            return transaction_id
        
        logger.warning(f"No transaction ID found in filename: {filename}")
        return None
    
    def extract_items_and_prices_structured(self, text: str) -> Dict[str, any]:
        """
        Parse text to extract transaction ID and item list with prices.
        
        Looks for sections like:
            Order Return code Price
            Item Name 1           £12.50
            Item Name 2           £8.00
            Total:               £20.50
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Dict with keys:
                - transaction_id: str or 0 if not found
                - items: List[Dict] with 'item' and 'price' keys
        """
        items = []
        lines = text.split('\n')
        in_order_section = False
        transaction_id = None
        
        for line in lines:
            line = line.strip()
            
            # Check for order section start
            if self.order_section_marker in line:
                in_order_section = True
                logger.debug("Found order section marker")
                continue
            
            # Extract transaction ID if present
            tid = self.extract_transaction_id_from_text(line)
            if tid:
                transaction_id = tid
            
            # Parse items in order section
            if in_order_section and line:
                price_match = self.price_pattern.search(line)
                
                if price_match:
                    price_value = float(price_match.group(1))
                    item_name = line[:price_match.start()].strip()
                    
                    # Filter out excluded items
                    if not any(keyword in item_name.lower() for keyword in self.excluded_keywords):
                        if item_name:
                            items.append({
                                'item': item_name,
                                'price': price_value
                            })
                            logger.debug(f"Parsed item: {item_name} - £{price_value:.2f}")
            
            # Check for section end
            if any(marker in line for marker in self.order_end_markers):
                logger.debug("Reached order section end")
                break
        
        result = {
            "transaction_id": transaction_id or "0",
            "items": items
        }
        
        logger.info(f"Parsed {len(items)} items from text (TID: {transaction_id})")
        return result
    
    def parse_pdf(self, pdf_bytes: bytes) -> Dict[str, any]:
        """
        Extract all text from PDF and parse into structured data.
        
        Args:
            pdf_bytes: Raw PDF file as bytes
            
        Returns:
            Dict with keys:
                - transaction_id: str
                - items: List[Dict] with 'item' and 'price' keys
        """
        transaction_id = "0"
        all_items = []
        
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                logger.debug(f"Processing PDF with {len(pdf.pages)} page(s)")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text:
                        logger.debug(f"Extracted text from page {page_num}")
                        extracted = self.extract_items_and_prices_structured(text)
                        
                        # Use first found transaction ID
                        if extracted["transaction_id"] != "0":
                            transaction_id = extracted["transaction_id"]
                        
                        all_items.extend(extracted["items"])
                    else:
                        logger.warning(f"No text found on page {page_num}")
            
            logger.info(
                f"PDF parsing complete: TID={transaction_id}, Items={len(all_items)}"
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}", exc_info=True)
        
        return {
            "transaction_id": transaction_id,
            "items": all_items
        }


# Convenience functions for backward compatibility
def extract_items_and_prices_structured(text: str) -> Dict[str, any]:
    """
    Parse text to extract transaction ID and items.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Dict with transaction_id and items list
    """
    parser = TransactionParser()
    return parser.extract_items_and_prices_structured(text)


def pdfInfoScrape(raw_pdf_bytes: bytes) -> Dict[str, any]:
    """
    Extract structured data from PDF bytes.
    
    Args:
        raw_pdf_bytes: PDF file as bytes
        
    Returns:
        Dict with transaction_id and items list
    """
    parser = TransactionParser()
    return parser.parse_pdf(raw_pdf_bytes)


def extract_transaction_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract transaction ID from filename pattern.
    
    Args:
        filename: PDF filename (e.g., Vinted-Label-16826065613.pdf)
        
    Returns:
        Transaction ID string or None
    """
    parser = TransactionParser()
    return parser.extract_transaction_id_from_filename(filename)
