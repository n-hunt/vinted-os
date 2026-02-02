"""
DocuFlow Vision Module

Handles image processing and PDF manipulation:
- Binarization (high contrast conversion)
- Auto-cropping whitespace
- Landscape -> Portrait rotation
- PDF dimension adjustments

All functions are pure (no side effects) and configurable via parameters.
"""

import io
import logging
from typing import Optional
from PIL import Image, ImageChops

try:
    from pdf2image import convert_from_bytes
except ImportError:
    raise ImportError(
        "pdf2image requires poppler to be installed.\n"
        "Install with: brew install poppler (macOS) or apt-get install poppler-utils (Linux)"
    )

from ..config_loader import config

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF-to-image conversion, processing, and back to PDF.
    Designed for shipping label optimization (4x6 format).
    """
    
    def __init__(self):
        """Initialize processor with configuration values."""
        self.dpi = config.get('image_processing.dpi', 203)
        self.threshold = config.get('image_processing.binarization_threshold', 200)
        self.crop_margin = config.get('image_processing.crop_margin', 10)
    
    def crop_and_resize_to_4x6(
        self, 
        pdf_bytes: bytes,
        threshold: Optional[int] = None,
        dpi: Optional[int] = None
    ) -> bytes:
        """
        Process PDF for optimal 4x6 label printing.
        
        Steps:
        1. Convert PDF to grayscale image
        2. Apply binarization (high contrast black/white)
        3. Auto-crop whitespace with margin
        4. Rotate landscape to portrait if needed
        5. Convert back to PDF
        
        Args:
            pdf_bytes: Raw PDF file as bytes
            threshold: Binarization threshold (0-255). Pixels > threshold become white.
                      If None, uses config value.
            dpi: Resolution for PDF->Image conversion. If None, uses config value.
            
        Returns:
            Processed PDF as bytes
            
        Raises:
            ValueError: If PDF cannot be converted to image
        """
        threshold = threshold if threshold is not None else self.threshold
        dpi = dpi if dpi is not None else self.dpi
        
        try:
            # Step 1: Convert PDF bytes to PIL Image
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            
            if not images:
                logger.warning("PDF conversion produced no images, returning original")
                return pdf_bytes
            
            # Work with first page only
            img = images[0].convert("L")  # Convert to Grayscale
            logger.debug(f"Converted PDF to grayscale image: {img.size}")
            
            # Step 2: High Contrast Binarization
            # Prevents gray edges that cause printer dithering/fading
            img = self._apply_binarization(img, threshold)
            
            # Step 3: Auto-crop whitespace
            cropped_img = self._auto_crop_whitespace(img)
            
            if cropped_img is None:
                logger.warning("No content detected for cropping, using full image")
                cropped_img = img
            
            # Step 4: Rotate landscape to portrait
            if cropped_img.width > cropped_img.height:
                cropped_img = cropped_img.rotate(90, expand=True)
                logger.debug("Rotated landscape image to portrait")
            
            # Step 5: Save back to PDF
            output = io.BytesIO()
            cropped_img.save(output, format="PDF", resolution=float(dpi))
            
            logger.info(
                f"Successfully processed PDF: {len(pdf_bytes)} -> {output.tell()} bytes"
            )
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}", exc_info=True)
            return pdf_bytes
    
    def _apply_binarization(self, img: Image.Image, threshold: int) -> Image.Image:
        """
        Apply binarization to create pure black/white image.
        
        Args:
            img: Grayscale PIL Image
            threshold: Pixel value threshold (0-255)
            
        Returns:
            Binarized 1-bit image
        """
        # Any pixel darker than threshold becomes pure black (0)
        # Any pixel lighter becomes pure white (255)
        img = img.point(lambda p: 255 if p > threshold else 0)
        img = img.convert("1")  # Convert to 1-bit (Pure Black/White)
        
        logger.debug(f"Applied binarization with threshold={threshold}")
        return img
    
    def _auto_crop_whitespace(self, img: Image.Image) -> Optional[Image.Image]:
        """
        Detect the rectangular black border that surrounds all shipping labels.
        
        Strategy:
        - Scan from edges inward to find the first black pixel
        - This represents the label's rectangular border
        - Crop to include the border and everything inside it
        
        Args:
            img: Binarized PIL Image (pure black/white)
            
        Returns:
            Cropped image or None if no border detected
        """
        import numpy as np
        
        # Convert to numpy array (for 1-bit images: 0 = black, 255 or True = white)
        img_array = np.array(img)
        height, width = img_array.shape
        
        # Find any black pixels (value of 0 or False)
        # For mode '1', black pixels are False/0, white are True/255
        black_pixels = (img_array == 0) if img_array.dtype == bool else (img_array < 128)
        
        # Scan from top - find first row with any black pixel
        top = 0
        for i in range(height):
            if np.any(black_pixels[i, :]):
                top = i
                break
        
        # Scan from bottom - find first row with any black pixel
        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if np.any(black_pixels[i, :]):
                bottom = i
                break
        
        # Scan from left - find first column with any black pixel
        left = 0
        for i in range(width):
            if np.any(black_pixels[:, i]):
                left = i
                break
        
        # Scan from right - find first column with any black pixel
        right = width - 1
        for i in range(width - 1, -1, -1):
            if np.any(black_pixels[:, i]):
                right = i
                break
        
        # Validate we found a valid border
        if right <= left or bottom <= top:
            logger.warning("Could not detect valid rectangular border")
            return None
        
        # Crop to the border (no additional margin needed since we want the border itself)
        bbox = (left, top, right + 1, bottom + 1)
        
        cropped = img.crop(bbox)
        logger.debug(
            f"Detected rectangular border at ({left}, {top}, {right}, {bottom}), "
            f"cropped from {img.size} to {cropped.size}"
        )
        
        return cropped


# Convenience function for backward compatibility
def crop_and_resize_to_4x6(pdf_bytes: bytes, threshold: Optional[int] = None) -> bytes:
    """
    Convenience wrapper around PDFProcessor.crop_and_resize_to_4x6
    
    Args:
        pdf_bytes: Raw PDF file as bytes
        threshold: Optional binarization threshold override
        
    Returns:
        Processed PDF as bytes
    """
    processor = PDFProcessor()
    return processor.crop_and_resize_to_4x6(pdf_bytes, threshold=threshold)
