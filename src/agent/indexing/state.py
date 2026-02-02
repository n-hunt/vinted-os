"""
Indexing Pipeline State

Defines the state structure for the LangGraph indexing workflow.
"""

from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document


class IndexingState(TypedDict):
    """
    State for the indexing pipeline workflow.
    
    Tracks data flow through: load → preprocess → embed → store
    """
    
    # Input configuration
    source_paths: List[str]  # Paths to KB files or log directories
    mode: str  # "full" | "incremental" | "md_only" | "logs_only"
    
    # Incremental update tracking
    file_hashes: Optional[Dict[str, str]]  # file_path -> hash (for change detection)
    modified_files: Optional[List[str]]  # Files that changed since last index
    
    # Data at each stage
    raw_documents: Optional[List[Document]]  # Stage 1: Loaded files
    md_chunks: Optional[List[Document]]  # Stage 2: Preprocessed markdown
    log_chunks: Optional[List[Document]]  # Stage 2: Preprocessed logs
    md_embeddings: Optional[List[List[float]]]  # Stage 3: MD vectors
    log_embeddings: Optional[List[List[float]]]  # Stage 3: Log vectors
    
    # Results
    vector_store_status: Optional[str]  # "success" | "failed"
    bm25_status: Optional[str]  # "success" | "failed"
    
    # Progress tracking
    current_step: str
    steps_completed: List[str]
    
    # Error handling
    errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # Metadata
    total_chunks: int
    total_embeddings: int
    start_time: Optional[float]
    end_time: Optional[float]
