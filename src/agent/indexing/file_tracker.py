"""
File Hash Tracker

Detects file changes for incremental indexing.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class FileHashTracker:
    """
    Tracks file content hashes to detect changes.
    
    Used for incremental updates - only re-index files that changed.
    """
    
    def __init__(self, registry_path: str = "data/.file_hashes.pkl"):
        """
        Initialize tracker.
        
        Args:
            registry_path: Path to store hash registry
        """
        self.registry_path = Path(registry_path)
        self.hashes: Dict[str, str] = {}
    
    def compute_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def load_registry(self) -> Dict[str, str]:
        """
        Load existing hash registry from disk.
        
        Returns:
            Dictionary of file_path -> hash
        """
        if self.registry_path.exists():
            with open(self.registry_path, 'rb') as f:
                self.hashes = pickle.load(f)
        else:
            self.hashes = {}
        
        return self.hashes
    
    def save_registry(self) -> None:
        """Save hash registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'wb') as f:
            pickle.dump(self.hashes, f)
    
    def detect_changes(self, file_paths: List[Path]) -> Dict[str, List[str]]:
        """
        Detect which files changed, were added, or deleted.
        
        Args:
            file_paths: Current list of files
            
        Returns:
            Dict with keys: 'modified', 'added', 'deleted'
        """
        # Load existing hashes
        old_hashes = self.load_registry()
        
        # Convert paths to strings for comparison
        current_files = {str(p): p for p in file_paths}
        old_files = set(old_hashes.keys())
        
        # Detect changes
        modified = []
        added = []
        deleted = list(old_files - set(current_files.keys()))
        
        for file_str, file_path in current_files.items():
            current_hash = self.compute_hash(file_path)
            
            if file_str in old_hashes:
                # File exists - check if modified
                if old_hashes[file_str] != current_hash:
                    modified.append(file_str)
                # Update hash
                self.hashes[file_str] = current_hash
            else:
                # New file
                added.append(file_str)
                self.hashes[file_str] = current_hash
        
        # Remove deleted files from registry
        for deleted_file in deleted:
            self.hashes.pop(deleted_file, None)
        
        return {
            'modified': modified,
            'added': added,
            'deleted': deleted
        }
    
    def update_hashes(self, file_paths: List[Path]) -> None:
        """
        Update hash registry with current files.
        
        Args:
            file_paths: Files to hash and store
        """
        for file_path in file_paths:
            file_hash = self.compute_hash(file_path)
            self.hashes[str(file_path)] = file_hash
        
        self.save_registry()
