"""
Configuration Loader for VintedOS

Provides centralized access to settings.yaml configuration.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    """
    Singleton configuration loader.
    Loads settings.yaml once and provides access throughout the application.
    """
    
    _instance = None
    _config_data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config_data is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from settings.yaml"""
        # Determine config file path
        config_dir = Path(__file__).parent.parent / "config"
        config_file = config_dir / "settings.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self._config_data = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'printer.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config = Config()
            >>> config.get('printer.name')
            'Zebra_GK420d'
            >>> config.get('image_processing.dpi')
            203
        """
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Top-level section name (e.g., 'printer', 'gmail')
            
        Returns:
            Dictionary containing the section's configuration
        """
        return self._config_data.get(section, {})
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config_data


# Create a global instance for easy importing
config = Config()
