"""
DocuFlow Generator Module

Handles PDF creation and overlay generation:
- Adding item text overlays to shipping labels
- Creating multi-item list PDFs
- Scaling and positioning label content

All functions accept clean dict inputs and return PDF bytes.
"""

import io
import logging
from typing import List, Dict, Optional

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import inch
except ImportError:
    raise ImportError("reportlab not installed. Install with: pip install reportlab")

from ..config_loader import config

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generates enhanced PDF labels with item overlays and multi-item lists.
    """
    
    def __init__(self):
        """Initialize generator with configuration values."""
        self.page_width = config.get('image_processing.page_size.width', 4.0)
        self.page_height = config.get('image_processing.page_size.height', 6.0)
        
        self.scaled_width = config.get('image_processing.scaled_label_size.width', 3.8)
        self.scaled_height = config.get('image_processing.scaled_label_size.height', 5.5)
        self.scale_factor = config.get('image_processing.scaled_label_size.scale_factor', 0.95)
        
        self.y_offset = config.get('image_processing.positioning.y_offset', 0.15)
        
        # Text overlay config
        self.tid_config = config.get_section('image_processing').get('text_overlay', {}).get('transaction_id', {})
        self.item_config = config.get_section('image_processing').get('text_overlay', {}).get('item_text', {})
        
        # Multi-item list config
        self.multi_config = config.get_section('image_processing').get('multi_item_list', {})
    
    def add_items_to_pdf(
        self, 
        pdf_bytes: bytes, 
        items: List[Dict[str, any]], 
        transaction_id: str
    ) -> bytes:
        """
        Shrink original label and add item text at bottom of 4x6 page.
        
        Creates a 4x6 page with:
        - Scaled label (3.8x5.5) at top
        - Transaction ID text
        - Item name(s) or multi-item indicator
        
        Args:
            pdf_bytes: Original shipping label PDF bytes
            items: List of dicts with 'item' and 'price' keys
            transaction_id: Transaction ID string
            
        Returns:
            Enhanced PDF as bytes
        """
        try:
            # Read original PDF
            original_pdf = PdfReader(io.BytesIO(pdf_bytes))
            original_page = original_pdf.pages[0]
            
            # Create text overlay
            overlay_bytes = self._create_text_overlay(items, transaction_id)
            
            # Merge scaled label with text
            text_pdf = PdfReader(io.BytesIO(overlay_bytes))
            writer = PdfWriter()
            
            # Get the text overlay page
            final_page = text_pdf.pages[0]
            
            # Get original page dimensions in points (72 points = 1 inch)
            orig_width_pts = float(original_page.mediabox.width)
            orig_height_pts = float(original_page.mediabox.height)
            
            # Target dimensions in points
            target_width_pts = self.scaled_width * 72
            target_height_pts = self.scaled_height * 72
            
            # Calculate scale to fit within target while preserving aspect ratio
            scale_x = target_width_pts / orig_width_pts
            scale_y = target_height_pts / orig_height_pts
            scale = min(scale_x, scale_y)
            
            # Calculate actual dimensions after scaling
            scaled_width_pts = orig_width_pts * scale
            scaled_height_pts = orig_height_pts * scale
            
            # Position: center horizontally, position from top with margin
            x_offset_pts = (self.page_width * 72 - scaled_width_pts) / 2  # Center horizontally
            y_offset_pts = self.page_height * 72 - scaled_height_pts - self.y_offset * 72  # From top with margin
            
            logger.debug(
                f"Label scaling: orig={orig_width_pts/72:.2f}x{orig_height_pts/72:.2f}in, "
                f"scale={scale:.3f}, final={scaled_width_pts/72:.2f}x{scaled_height_pts/72:.2f}in, "
                f"position=({x_offset_pts/72:.2f}, {y_offset_pts/72:.2f})in"
            )
            
            # Apply transformation using a transformation matrix
            # Matrix format: [a, b, c, d, e, f] where a,d = scale, e,f = translation
            original_page.add_transformation([scale, 0, 0, scale, x_offset_pts, y_offset_pts])
            
            # Merge scaled label onto page with text
            final_page.merge_page(original_page)
            writer.add_page(final_page)
            
            # Write to bytes
            output = io.BytesIO()
            writer.write(output)
            
            logger.info(
                f"Generated label with items overlay for TID {transaction_id} "
                f"({len(items)} item(s))"
            )
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error adding items to PDF: {e}", exc_info=True)
            return pdf_bytes
    
    def _create_text_overlay(
        self, 
        items: List[Dict[str, any]], 
        transaction_id: str
    ) -> bytes:
        """
        Create PDF page with transaction ID and item text at bottom.
        
        Args:
            items: List of item dicts
            transaction_id: Transaction ID string
            
        Returns:
            PDF bytes containing text overlay
        """
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(self.page_width * inch, self.page_height * inch))
        
        # Add transaction ID
        can.setFont(
            self.tid_config.get('font', 'Helvetica-Bold'),
            self.tid_config.get('size', 7)
        )
        can.drawString(
            self.tid_config.get('x_position', 0.1) * inch,
            self.tid_config.get('y_position', 0.20) * inch,
            f"ID: {transaction_id}"
        )
        
        # Add item text
        can.setFont(
            self.item_config.get('font', 'Helvetica'),
            self.item_config.get('size', 7)
        )
        
        # Determine item text
        if len(items) > 1:
            item_text = f"MULTI-ITEM ORDER ({len(items)} items) - CHECK LIST"
        else:
            item_text = ", ".join([item['item'] for item in items])
            max_length = self.item_config.get('max_length', 55)
            
            # Truncate if too long
            if len(item_text) > max_length:
                truncate_at = max_length - len(self.item_config.get('truncate_suffix', '...'))
                item_text = item_text[:truncate_at] + self.item_config.get('truncate_suffix', '...')
        
        can.drawString(
            self.item_config.get('x_position', 0.1) * inch,
            self.item_config.get('y_position', 0.05) * inch,
            item_text
        )
        
        can.save()
        packet.seek(0)
        return packet.getvalue()
    
    def create_items_list_pdf(
        self, 
        items: List[Dict[str, any]], 
        transaction_id: str
    ) -> List[bytes]:
        """
        Create paginated 4x6 PDFs listing all items for multi-item orders.
        
        Args:
            items: List of dicts with 'item' and 'price' keys
            transaction_id: Transaction ID string
            
        Returns:
            List of PDF bytes (one per page needed)
        """
        try:
            items_per_page = self.multi_config.get('items_per_page', 20)
            pdf_pages = []
            
            # Calculate pages needed
            total_pages = (len(items) + items_per_page - 1) // items_per_page
            
            logger.debug(
                f"Creating {total_pages} list page(s) for {len(items)} items"
            )
            
            for page_num in range(total_pages):
                page_bytes = self._create_single_list_page(
                    items, transaction_id, page_num, total_pages, items_per_page
                )
                pdf_pages.append(page_bytes)
            
            logger.info(
                f"Generated {len(pdf_pages)} multi-item list PDF(s) for TID {transaction_id}"
            )
            
            return pdf_pages
            
        except Exception as e:
            logger.error(f"Error creating items list PDF: {e}", exc_info=True)
            return []
    
    def _create_single_list_page(
        self,
        items: List[Dict[str, any]],
        transaction_id: str,
        page_num: int,
        total_pages: int,
        items_per_page: int
    ) -> bytes:
        """
        Create a single page of the items list.
        
        Args:
            items: Full list of items
            transaction_id: Transaction ID
            page_num: Current page number (0-indexed)
            total_pages: Total number of pages
            items_per_page: Items to display per page
            
        Returns:
            PDF bytes for this page
        """
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(self.page_width * inch, self.page_height * inch))
        
        # Title
        title_cfg = self.multi_config.get('title', {})
        can.setFont(
            title_cfg.get('font', 'Helvetica-Bold'),
            title_cfg.get('size', 12)
        )
        
        page_title = f"Order Items - ID: {transaction_id}"
        if total_pages > 1:
            page_title += f" (Page {page_num + 1}/{total_pages})"
        
        can.drawString(
            title_cfg.get('x_position', 0.2) * inch,
            title_cfg.get('y_position', 5.7) * inch,
            page_title
        )
        
        # Title underline
        line_cfg = self.multi_config.get('line', {})
        can.line(
            line_cfg.get('x1', 0.2) * inch,
            line_cfg.get('y1', 5.55) * inch,
            line_cfg.get('x2', 3.8) * inch,
            line_cfg.get('y2', 5.55) * inch
        )
        
        # Items
        item_cfg = self.multi_config.get('item', {})
        can.setFont(
            item_cfg.get('font', 'Helvetica'),
            item_cfg.get('size', 9)
        )
        
        y_position = item_cfg.get('y_start', 5.3) * inch
        y_spacing = item_cfg.get('y_spacing', 0.25) * inch
        
        start_idx = page_num * items_per_page
        end_idx = min(start_idx + items_per_page, len(items))
        
        for idx in range(start_idx, end_idx):
            item = items[idx]
            item_name = item['item']
            item_price = f"£{item['price']:.2f}"
            
            # Truncate long names
            max_length = item_cfg.get('name_max_length', 35)
            if len(item_name) > max_length:
                truncate_at = item_cfg.get('name_truncate_length', 32)
                item_name = item_name[:truncate_at] + "..."
            
            # Draw item number and name
            text = f"{idx + 1}. {item_name}"
            can.drawString(
                item_cfg.get('x_position', 0.3) * inch,
                y_position,
                text
            )
            
            # Draw price (right-aligned)
            can.drawRightString(
                item_cfg.get('price_x_position', 3.7) * inch,
                y_position,
                item_price
            )
            
            y_position -= y_spacing
        
        can.save()
        packet.seek(0)
        return packet.getvalue()


# Convenience functions for backward compatibility
def add_items_to_pdf(
    pdf_bytes: bytes, 
    items: List[Dict[str, any]], 
    transaction_id: str
) -> bytes:
    """
    Add item overlay to shipping label PDF.
    
    Args:
        pdf_bytes: Original label PDF
        items: List of item dicts
        transaction_id: Transaction ID
        
    Returns:
        Enhanced PDF bytes
    """
    generator = LabelGenerator()
    return generator.add_items_to_pdf(pdf_bytes, items, transaction_id)


def create_items_list_pdf(
    items: List[Dict[str, any]], 
    transaction_id: str
) -> List[bytes]:
    """
    Create multi-item list PDF pages.
    
    Args:
        items: List of item dicts
        transaction_id: Transaction ID
        
    Returns:
        List of PDF bytes (one per page)
    """
    generator = LabelGenerator()
    return generator.create_items_list_pdf(items, transaction_id)
