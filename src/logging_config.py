"""
Logging Configuration for VintedOS

Sets up structured logging with file and console handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .config_loader import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application-wide logging.
    
    Sets up:
    - Console handler (stdout)
    - Rotating file handler (if enabled)
    - Structured formatting
    
    Args:
        log_level: Override config log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Override config log filename
    """
    # Get configuration
    logging_config = config.get_section('logging')
    
    level = log_level or logging_config.get('level', 'INFO')
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_config = logging_config.get('handlers', {}).get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = console_config.get('level', 'INFO')
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    file_config = logging_config.get('handlers', {}).get('file', {})
    if file_config.get('enabled', True):
        # Ensure logs directory exists
        logs_dir = Path(config.get('paths.logs', './logs'))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        log_filename = log_file or file_config.get('filename', 'vintedos.log')
        log_path = logs_dir / log_filename
        
        max_bytes = file_config.get('max_bytes', 10485760)  # 10MB
        backup_count = file_config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        file_level = file_config.get('level', 'DEBUG')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("VintedOS Logging Initialized")
    logger.info(f"Log Level: {level}")
    logger.info(f"Console Handler: {console_config.get('enabled', True)}")
    logger.info(f"File Handler: {file_config.get('enabled', True)}")
    logger.info("="*60)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
