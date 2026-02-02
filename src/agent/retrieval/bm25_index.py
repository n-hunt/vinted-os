"""
BM25 Index

Sparse keyword-based retrieval using BM25 algorithm.
Complements dense vector search for hybrid retrieval.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
import re

from langchain_core.documents import Document

# Handle imports for both package and script execution
try:
    from ...config_loader import Config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config_loader import Config

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse retrieval index for keyword-based search.
    
    Features:
    - Separate indexes for MD and log documents
    - Uses normalized_text from metadata for tokenization
    - Configurable BM25 parameters (k1, b)
    - Persistent storage with pickle
    - Fast in-memory search
    
    BM25 Formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
    
    Where:
    - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    - f(qi,D) = frequency of term qi in document D
    - |D| = length of document D
    - avgdl = average document length in collection
    - k1 = term frequency saturation parameter (default 1.5)
    - b = length normalization parameter (default 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation (higher = more weight to term freq)
            b: Length normalization (0 = no norm, 1 = full norm)
        """
        logger.info(f"Initializing BM25Index (k1={k1}, b={b})")
        
        # Load config
        config = Config()
        self.k1 = config.get('agent.bm25.k1', k1)
        self.b = config.get('agent.bm25.b', b)
        
        # Separate indexes for MD and logs
        self.md_index: Optional[Dict] = None
        self.log_index: Optional[Dict] = None
        
        logger.info(f"BM25Index ready (k1={self.k1}, b={self.b})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Uses simple whitespace + punctuation splitting.
        Lowercases and removes very short tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def build_index(self, chunks: List[Document]) -> Dict:
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of preprocessed Document chunks
            
        Returns:
            Index dict with structure:
            {
                'doc_ids': List[str],  # chunk_ids
                'doc_lengths': List[int],  # token counts
                'avgdl': float,  # average doc length
                'doc_freqs': Dict[str, int],  # term -> doc frequency
                'term_docs': Dict[str, List[int]],  # term -> doc indices
                'term_freqs': List[Dict[str, int]],  # per-doc term frequencies
                'chunks': List[Document]  # original chunks
            }
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return self._empty_index()
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks")
        
        doc_ids = []
        doc_lengths = []
        term_freqs = []  # Per-document term frequencies
        doc_freqs = Counter()  # Global document frequencies
        term_docs = {}  # term -> list of doc indices containing it
        
        for idx, chunk in enumerate(chunks):
            # Get normalized_text from metadata (preprocessed for BM25)
            normalized_text = chunk.metadata.get('normalized_text', chunk.page_content)
            
            # Tokenize
            tokens = self.tokenize(normalized_text)
            
            # Store doc info
            chunk_id = chunk.metadata.get('chunk_id', f'chunk_{idx}')
            doc_ids.append(chunk_id)
            doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            tf = Counter(tokens)
            term_freqs.append(dict(tf))
            
            # Update global document frequencies and term->docs mapping
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] += 1
                if term not in term_docs:
                    term_docs[term] = []
                term_docs[term].append(idx)
        
        # Calculate average document length
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        index = {
            'doc_ids': doc_ids,
            'doc_lengths': doc_lengths,
            'avgdl': avgdl,
            'doc_freqs': dict(doc_freqs),
            'term_docs': term_docs,
            'term_freqs': term_freqs,
            'chunks': chunks,
            'num_docs': len(chunks)
        }
        
        logger.info(
            f"SUCCESS: Built BM25 index: {len(chunks)} docs, "
            f"{len(doc_freqs)} unique terms, avgdl={avgdl:.1f}"
        )
        
        return index
    
    def _empty_index(self) -> Dict:
        """Return empty index structure."""
        return {
            'doc_ids': [],
            'doc_lengths': [],
            'avgdl': 0,
            'doc_freqs': {},
            'term_docs': {},
            'term_freqs': [],
            'chunks': [],
            'num_docs': 0
        }
    
    def index_markdown(self, chunks: List[Document]) -> None:
        """Build and store markdown BM25 index."""
        logger.info("Building markdown BM25 index")
        self.md_index = self.build_index(chunks)
    
    def index_logs(self, chunks: List[Document]) -> None:
        """Build and store log BM25 index."""
        logger.info("Building log BM25 index")
        self.log_index = self.build_index(chunks)
    
    def _calculate_idf(self, term: str, index: Dict) -> float:
        """
        Calculate IDF score for a term.
        
        IDF = log((N - df + 0.5) / (df + 0.5))
        
        Args:
            term: Query term
            index: BM25 index dict
            
        Returns:
            IDF score
        """
        N = index['num_docs']
        df = index['doc_freqs'].get(term, 0)
        
        # Standard BM25 IDF formula
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        return idf
    
    def _calculate_bm25_score(
        self,
        query_terms: List[str],
        doc_idx: int,
        index: Dict
    ) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: Tokenized query
            doc_idx: Document index in the collection
            index: BM25 index dict
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = index['doc_lengths'][doc_idx]
        avgdl = index['avgdl']
        term_freqs_doc = index['term_freqs'][doc_idx]
        
        for term in query_terms:
            if term not in term_freqs_doc:
                continue
            
            # Term frequency in this document
            tf = term_freqs_doc[term]
            
            # IDF score
            idf = self._calculate_idf(term, index)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(
        self,
        query: str,
        collection: str = "md_chunks",
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Search BM25 index with query.
        
        Args:
            query: Search query
            collection: "md_chunks" or "log_chunks"
            top_k: Number of results to return
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        # Select index
        if collection == "md_chunks":
            index = self.md_index
        elif collection == "log_chunks":
            index = self.log_index
        else:
            raise ValueError(f"Unknown collection: {collection}")
        
        if index is None or index['num_docs'] == 0:
            logger.warning(f"Index for {collection} is empty or not built")
            return []
        
        # Tokenize query
        query_terms = self.tokenize(query)
        
        if not query_terms:
            logger.warning("No valid query terms after tokenization")
            return []
        
        # Find candidate documents (union of docs containing any query term)
        candidate_indices = set()
        for term in query_terms:
            if term in index['term_docs']:
                candidate_indices.update(index['term_docs'][term])
        
        if not candidate_indices:
            logger.debug(f"No documents found containing query terms: {query_terms}")
            return []
        
        # Score all candidate documents
        results = []
        for doc_idx in candidate_indices:
            score = self._calculate_bm25_score(query_terms, doc_idx, index)
            if score > 0:
                chunk = index['chunks'][doc_idx]
                results.append((chunk, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    def save(self, path: Path) -> None:
        """
        Save BM25 indexes to disk.
        
        Args:
            path: Directory path to save indexes
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.md_index is not None:
            md_path = path / "md_bm25_index.pkl"
            with open(md_path, 'wb') as f:
                pickle.dump(self.md_index, f)
            logger.info(f"SUCCESS: Saved markdown BM25 index to {md_path}")
        
        if self.log_index is not None:
            log_path = path / "log_bm25_index.pkl"
            with open(log_path, 'wb') as f:
                pickle.dump(self.log_index, f)
            logger.info(f"SUCCESS: Saved log BM25 index to {log_path}")
    
    def load(self, path: Path) -> None:
        """
        Load BM25 indexes from disk.
        
        Args:
            path: Directory path to load indexes from
        """
        path = Path(path)
        
        md_path = path / "md_bm25_index.pkl"
        if md_path.exists():
            with open(md_path, 'rb') as f:
                self.md_index = pickle.load(f)
            logger.info(f"SUCCESS: Loaded markdown BM25 index from {md_path}")
        else:
            logger.warning(f"Markdown BM25 index not found at {md_path}")
        
        log_path = path / "log_bm25_index.pkl"
        if log_path.exists():
            with open(log_path, 'rb') as f:
                self.log_index = pickle.load(f)
            logger.info(f"SUCCESS: Loaded log BM25 index from {log_path}")
        else:
            logger.warning(f"Log BM25 index not found at {log_path}")
    
    def health_check(self) -> Dict[str, any]:
        """
        Check BM25 index health.
        
        Returns:
            Health status dict
        """
        md_status = None
        log_status = None
        
        if self.md_index is not None:
            md_status = {
                'num_docs': self.md_index['num_docs'],
                'unique_terms': len(self.md_index['doc_freqs']),
                'avgdl': self.md_index['avgdl']
            }
        
        if self.log_index is not None:
            log_status = {
                'num_docs': self.log_index['num_docs'],
                'unique_terms': len(self.log_index['doc_freqs']),
                'avgdl': self.log_index['avgdl']
            }
        
        return {
            'k1': self.k1,
            'b': self.b,
            'md_index': md_status,
            'log_index': log_status
        }
