"""
Hybrid Retriever

Combines dense vector search (Qdrant) with sparse keyword search (BM25)
using Reciprocal Rank Fusion (RRF) for optimal retrieval.

Features:
- Parallel execution of vector and BM25 search
- Reciprocal Rank Fusion for result merging
- Metadata filtering support
- Configurable search weights
- Deduplication by chunk_id
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document

# Handle imports for both package and script execution
try:
    from .vector_store import VectorStore
    from .bm25_index import BM25Index
    from .embeddings import EmbeddingService
    from ...config_loader import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.retrieval.vector_store import VectorStore
    from agent.retrieval.bm25_index import BM25Index
    from agent.retrieval.embeddings import EmbeddingService
    from config_loader import config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining dense (vector) and sparse (BM25) search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both methods:
    - Vector search: Semantic similarity (understands meaning)
    - BM25 search: Keyword matching (exact terms)
    
    RRF Formula:
    score(doc) = Σ 1 / (k + rank_i)
    
    Where:
    - rank_i is the rank of doc in result set i (1-indexed)
    - k is a constant (typically 60) to prevent division by small numbers
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        embedding_service: Optional[EmbeddingService] = None,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore instance (created if None)
            bm25_index: BM25Index instance (created if None)
            embedding_service: EmbeddingService instance (created if None)
            rrf_k: RRF constant (higher = less penalty for lower ranks)
        """
        logger.info("Initializing HybridRetriever")
        
        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_service = embedding_service or EmbeddingService()
        
        # RRF configuration
        self.rrf_k = config.get('agent.hybrid_search.rrf_k', rrf_k)
        
        # Search weights (for weighted fusion)
        self.vector_weight = config.get('agent.hybrid_search.vector_weight', 0.5)
        self.bm25_weight = config.get('agent.hybrid_search.bm25_weight', 0.5)
        
        # Load BM25 indexes if they exist
        bm25_path = Path(config.get('agent.bm25.save_path', 'data/bm25_indexes'))
        if bm25_path.exists():
            self.bm25_index.load(bm25_path)
            logger.info("Loaded BM25 indexes from disk")
        
        logger.info(f"HybridRetriever ready (RRF k={self.rrf_k})")
    
    def search(
        self,
        query: str,
        collection: str = "md_chunks",
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        use_rrf: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query text
            collection: "md_chunks" or "log_chunks"
            top_k: Number of final results to return
            metadata_filter: Optional metadata filters (e.g., {'doc_type': 'architecture'})
            vector_weight: Override default vector weight (0-1)
            bm25_weight: Override default BM25 weight (0-1)
            use_rrf: If True, use RRF fusion; if False, use weighted score fusion
            
        Returns:
            List of (Document, score) tuples, ranked by hybrid score
        """
        logger.debug(f"Hybrid search: '{query}' in {collection} (top_k={top_k})")
        
        # Use default weights if not provided
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        b_weight = bm25_weight if bm25_weight is not None else self.bm25_weight
        
        # Embed query for vector search
        query_embedding = self.embedding_service.embed_query(query)
        
        # Execute searches in parallel (conceptually - could use threading)
        # Fetch more results than needed for better fusion
        fetch_k = top_k * 3
        
        # Vector search
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            collection_name=collection,
            top_k=fetch_k,
            metadata_filter=metadata_filter
        )
        
        # BM25 search
        bm25_results = self.bm25_index.search(
            query=query,
            collection=collection,
            top_k=fetch_k
        )
        
        # Apply metadata filter to BM25 results (BM25 doesn't support native filtering)
        if metadata_filter:
            bm25_results = self._filter_results(bm25_results, metadata_filter)
        
        # Merge results
        if use_rrf:
            merged_results = self._reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                v_weight,
                b_weight
            )
        else:
            merged_results = self._weighted_fusion(
                vector_results,
                bm25_results,
                v_weight,
                b_weight
            )
        
        # Return top-k
        final_results = merged_results[:top_k]
        
        logger.debug(
            f"Hybrid search complete: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(final_results)} merged"
        )
        
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0
    ) -> List[Tuple[Document, float]]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        RRF is rank-based (doesn't depend on raw scores), making it robust
        to score scale differences between vector and BM25.
        
        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            vector_weight: Weight for vector rankings
            bm25_weight: Weight for BM25 rankings
            
        Returns:
            Merged and deduplicated results
        """
        # Track RRF scores by chunk_id
        rrf_scores = defaultdict(float)
        doc_map = {}  # chunk_id -> Document
        
        # Process vector results (rank 1 = first result)
        for rank, (doc, score) in enumerate(vector_results, start=1):
            chunk_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            rrf_score = vector_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id] += rrf_score
            
            # Store document (prefer first occurrence)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Process BM25 results
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            chunk_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            rrf_score = bm25_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id] += rrf_score
            
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Create result list
        merged = [
            (doc_map[chunk_id], score)
            for chunk_id, score in rrf_scores.items()
        ]
        
        # Sort by RRF score descending
        merged.sort(key=lambda x: x[1], reverse=True)
        
        return merged
    
    def _weighted_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Merge results using weighted score fusion.
        
        Combines raw scores from both methods. Requires score normalization
        since vector and BM25 scores have different scales.
        
        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            
        Returns:
            Merged and deduplicated results
        """
        # Normalize scores to [0, 1] range
        vector_normalized = self._normalize_scores(vector_results)
        bm25_normalized = self._normalize_scores(bm25_results)
        
        # Track combined scores by chunk_id
        combined_scores = defaultdict(float)
        doc_map = {}
        
        # Add vector scores
        for doc, norm_score in vector_normalized:
            chunk_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            combined_scores[chunk_id] += vector_weight * norm_score
            
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Add BM25 scores
        for doc, norm_score in bm25_normalized:
            chunk_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            combined_scores[chunk_id] += bm25_weight * norm_score
            
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Create result list
        merged = [
            (doc_map[chunk_id], score)
            for chunk_id, score in combined_scores.items()
        ]
        
        # Sort by combined score descending
        merged.sort(key=lambda x: x[1], reverse=True)
        
        return merged
    
    def _normalize_scores(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Min-max normalize scores to [0, 1] range.
        
        Args:
            results: List of (Document, score) tuples
            
        Returns:
            List with normalized scores
        """
        if not results:
            return []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case where all scores are identical
        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]
        
        # Min-max normalization
        normalized = [
            (doc, (score - min_score) / (max_score - min_score))
            for doc, score in results
        ]
        
        return normalized
    
    def _filter_results(
        self,
        results: List[Tuple[Document, float]],
        metadata_filter: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """
        Apply metadata filters to results.
        
        Args:
            results: List of (Document, score) tuples
            metadata_filter: Filter conditions
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for doc, score in results:
            # Check each filter condition
            passes = True
            for key, value in metadata_filter.items():
                doc_value = doc.metadata.get(key)
                
                # List filter: check if doc_value is in the list
                if isinstance(value, list):
                    if doc_value not in value:
                        passes = False
                        break
                # Exact match
                else:
                    if doc_value != value:
                        passes = False
                        break
            
            if passes:
                filtered.append((doc, score))
        
        return filtered
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all retrieval components.
        
        Returns:
            Health status dict
        """
        vector_health = self.vector_store.health_check()
        bm25_health = self.bm25_index.health_check()
        
        return {
            'vector_store': vector_health,
            'bm25_index': bm25_health,
            'embedding_service': {
                'model': self.embedding_service.model_name,
                'dimension': self.embedding_service.dimension
            },
            'rrf_k': self.rrf_k,
            'weights': {
                'vector': self.vector_weight,
                'bm25': self.bm25_weight
            }
        }
