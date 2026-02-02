"""
Embedder Node

Generates embeddings for preprocessed document chunks.
Handles both markdown and log documents with batch processing.
"""

import logging
from typing import List, Tuple
from pathlib import Path
from langchain_core.documents import Document

# Handle imports for both package and script execution
try:
    from ...retrieval.embeddings import EmbeddingService
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Generates embeddings for preprocessed document chunks.
    
    Features:
    - Batch processing for efficiency
    - Separate handling for MD and log documents
    - Progress tracking
    - Validation of embedding consistency
    """
    
    def __init__(self):
        """Initialize embedder with embedding service."""
        logger.info("Initializing DocumentEmbedder")
        self.embedding_service = EmbeddingService()
        logger.info(f"DocumentEmbedder ready (dimension: {self.embedding_service.dimension})")
    
    def embed_markdown_chunks(
        self,
        chunks: List[Document],
        show_progress: bool = True
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for markdown document chunks.
        
        Uses the original markdown page_content (preserves headers and formatting)
        for semantic embeddings.
        
        Args:
            chunks: List of preprocessed markdown Document chunks
            show_progress: Show progress logging
            
        Returns:
            Tuple of (chunks, embeddings) where embeddings align with chunk order
        """
        if not chunks:
            logger.warning("No markdown chunks provided")
            return [], []
        
        logger.info(f"Embedding {len(chunks)} markdown chunks")
        
        # Extract text content for embedding
        # Use page_content (original markdown) for dense semantic search
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(texts)
        
        # Validate counts match
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks"
            )
        
        if show_progress:
            logger.info(f"SUCCESS: Generated {len(embeddings)} markdown embeddings")
            self._log_sample_info(chunks[0], embeddings[0])
        
        return chunks, embeddings
    
    def embed_log_chunks(
        self,
        chunks: List[Document],
        show_progress: bool = True
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for log document chunks.
        
        Uses the structured log page_content (preserves log format)
        for semantic embeddings.
        
        Args:
            chunks: List of preprocessed log Document chunks
            show_progress: Show progress logging
            
        Returns:
            Tuple of (chunks, embeddings) where embeddings align with chunk order
        """
        if not chunks:
            logger.warning("No log chunks provided")
            return [], []
        
        logger.info(f"Embedding {len(chunks)} log chunks")
        
        # Extract text content for embedding
        # Use page_content (structured log format) for dense semantic search
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(texts)
        
        # Validate counts match
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks"
            )
        
        if show_progress:
            logger.info(f"SUCCESS: Generated {len(embeddings)} log embeddings")
            self._log_sample_info(chunks[0], embeddings[0])
        
        return chunks, embeddings
    
    def embed_documents(
        self,
        chunks: List[Document],
        document_type: str = "auto",
        show_progress: bool = True
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for any document type (auto-detects from chunk_id).
        
        Args:
            chunks: List of preprocessed Document chunks
            document_type: "markdown", "log", or "auto" (detect from chunk_id prefix)
            show_progress: Show progress logging
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        if not chunks:
            return [], []
        
        # Auto-detect document type from chunk_id prefix
        if document_type == "auto":
            first_chunk_id = chunks[0].metadata.get('chunk_id', '')
            if first_chunk_id.startswith('md_chunk'):
                document_type = "markdown"
            elif first_chunk_id.startswith('log_chunk'):
                document_type = "log"
            else:
                logger.warning(
                    f"Cannot auto-detect document type from chunk_id: {first_chunk_id}. "
                    f"Defaulting to markdown."
                )
                document_type = "markdown"
        
        # Route to appropriate method
        if document_type == "markdown":
            return self.embed_markdown_chunks(chunks, show_progress)
        elif document_type == "log":
            return self.embed_log_chunks(chunks, show_progress)
        else:
            raise ValueError(f"Unknown document type: {document_type}")
    
    def _log_sample_info(self, chunk: Document, embedding: List[float]) -> None:
        """Log sample information about chunk and embedding."""
        chunk_id = chunk.metadata.get('chunk_id', 'unknown')
        content_preview = chunk.page_content[:60].replace('\n', ' ')
        
        logger.debug(
            f"Sample: {chunk_id} | "
            f"Content: '{content_preview}...' | "
            f"Embedding dim: {len(embedding)} | "
            f"First 3 values: {embedding[:3]}"
        )
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_service.get_dimension()
