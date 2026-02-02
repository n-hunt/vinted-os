"""
Vector Store Service

Manages Qdrant vector database for semantic search:
- Collection creation with metadata schemas
- Document insertion with embeddings
- Vector search with metadata filtering
- Hybrid search support (dense + sparse vectors)
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchRequest,
    PayloadSchemaType,
)
from langchain_core.documents import Document

# Handle imports for both package and script execution
try:
    from ...config_loader import config
except ImportError:
    # Fallback for script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config_loader import config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant vector database client for VintedOS knowledge base.
    
    Handles two collections:
    - md_chunks: Markdown documentation chunks
    - log_chunks: Operational log entries
    
    Both support dense vector search (semantic) and metadata filtering.
    """
    
    def __init__(self):
        """Initialize Qdrant client and collection configurations."""
        # Load config using dot notation
        self.qdrant_path = Path(config.get('agent.vector_db.path', './data/qdrant_data'))
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        
        self.distance_metric = self._get_distance_metric(
            config.get('agent.vector_db.distance_metric', 'cosine')
        )
        
        # Collection names
        self.md_collection = config.get('agent.vector_db.collections.markdown.name', 'md_chunks')
        self.log_collection = config.get('agent.vector_db.collections.logs.name', 'log_chunks')
        
        # Embedding dimension
        self.embedding_dim = config.get('agent.embeddings.dimension', 768)
        
        # Initialize client
        logger.info(f"Connecting to Qdrant at: {self.qdrant_path}")
        self.client = QdrantClient(path=str(self.qdrant_path))
        
        logger.info("VectorStore initialized successfully")
    
    def _get_distance_metric(self, metric: str) -> Distance:
        """Convert config string to Qdrant Distance enum."""
        metric_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
            'dot': Distance.DOT
        }
        return metric_map.get(metric.lower(), Distance.COSINE)
    
    def initialize_collections(self, recreate: bool = False) -> None:
        """
        Create vector collections with proper schemas.
        
        Args:
            recreate: If True, delete and recreate existing collections
        """
        logger.info("Initializing vector collections")
        
        # Define collections
        collections = [
            (self.md_collection, "Markdown documentation chunks"),
            (self.log_collection, "Operational log entries")
        ]
        
        for collection_name, description in collections:
            # Check if exists
            exists = self.client.collection_exists(collection_name)
            
            if exists and recreate:
                logger.warning(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
                exists = False
            
            if not exists:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance_metric
                    )
                )
                
                # Create payload indexes for fast metadata filtering
                self._create_payload_indexes(collection_name)
                
                logger.info(f"SUCCESS: Collection created: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
    
    def _create_payload_indexes(self, collection_name: str) -> None:
        """
        Create indexes on metadata fields for fast filtering.
        
        Indexed fields enable O(log n) filtering vs O(n) scanning.
        """
        # Common indexes for both collections
        common_indexes = [
            ('chunk_id', PayloadSchemaType.KEYWORD),
            ('source', PayloadSchemaType.KEYWORD),
            ('doc_type', PayloadSchemaType.KEYWORD),
        ]
        
        # MD-specific indexes
        md_indexes = [
            ('category', PayloadSchemaType.KEYWORD),
            ('section', PayloadSchemaType.KEYWORD),
            ('priority', PayloadSchemaType.INTEGER),
            ('has_code', PayloadSchemaType.BOOL),
            ('mentioned_services', PayloadSchemaType.KEYWORD),
        ]
        
        # Log-specific indexes
        log_indexes = [
            ('transaction_ids', PayloadSchemaType.KEYWORD),
            ('modules', PayloadSchemaType.KEYWORD),
            ('log_levels', PayloadSchemaType.KEYWORD),
            ('has_errors', PayloadSchemaType.BOOL),
            ('error_types', PayloadSchemaType.KEYWORD),
        ]
        
        # Select appropriate indexes
        if collection_name == self.md_collection:
            indexes = common_indexes + md_indexes
        else:
            indexes = common_indexes + log_indexes
        
        # Create each index
        for field_name, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
                logger.debug(f"  Created index: {field_name} ({schema_type})")
            except Exception as e:
                # Index may already exist
                logger.debug(f"  Index exists or failed: {field_name} - {e}")
    
    def insert_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Insert documents with embeddings into collection.
        
        Args:
            documents: List of LangChain Documents with metadata
            embeddings: Corresponding embedding vectors
            collection_name: Target collection (auto-detected if None)
            batch_size: Number of points per batch upload
            
        Returns:
            Number of documents inserted
        """
        if not documents:
            logger.warning("No documents to insert")
            return 0
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) != embedding count ({len(embeddings)})"
            )
        
        # Auto-detect collection from chunk_id prefix
        if collection_name is None:
            first_chunk_id = documents[0].metadata.get('chunk_id', '')
            if first_chunk_id.startswith('md_chunk'):
                collection_name = self.md_collection
            elif first_chunk_id.startswith('log_chunk'):
                collection_name = self.log_collection
            else:
                raise ValueError(f"Cannot auto-detect collection from chunk_id: {first_chunk_id}")
        
        logger.info(f"Inserting {len(documents)} documents into {collection_name}")
        
        # Prepare points
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Validate embedding dimension
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at index {idx}: "
                    f"expected {self.embedding_dim}, got {len(embedding)}"
                )
            
            # Serialize metadata (convert datetime to ISO strings)
            payload = self._serialize_metadata(doc.metadata)
            payload['page_content'] = doc.page_content
            
            # Generate stable deterministic ID from chunk_id using MD5 hash
            # This ensures same chunk_id always produces same ID across Python runs
            chunk_id = doc.metadata.get('chunk_id', f'chunk_{idx}')
            hash_bytes = hashlib.md5(chunk_id.encode('utf-8')).digest()
            point_id = int.from_bytes(hash_bytes[:8], byteorder='big')  # Use first 8 bytes as uint64
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upload in batches
        inserted_count = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                inserted_count += len(batch)
                logger.debug(f"  Uploaded batch {i // batch_size + 1}: {len(batch)} points")
            except Exception as e:
                logger.error(f"Failed to upload batch at index {i}: {e}")
                raise
        
        logger.info(f"SUCCESS: Inserted {inserted_count} documents into {collection_name}")
        return inserted_count
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata to Qdrant-compatible types.
        
        Qdrant payload supports: int, float, str, bool, list, dict.
        Convert datetime objects to ISO strings.
        """
        serialized = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert datetime to ISO string
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            
            # Keep primitives and lists as-is
            elif isinstance(value, (int, float, str, bool, list, dict)):
                serialized[key] = value
            
            # Convert other types to string
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def search(
        self,
        query_vector: List[float],
        collection_name: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search using query embedding.
        
        Args:
            query_vector: Query embedding vector
            collection_name: Collection to search
            top_k: Number of results to return
            metadata_filter: Filter by metadata fields (e.g., {'doc_type': 'architecture'})
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        # Validate collection exists
        if not self.client.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Call initialize_collections() first."
            )
        
        # Validate embedding dimension
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(query_vector)}"
            )
        
        # Build filter
        query_filter = None
        if metadata_filter:
            query_filter = self._build_filter(metadata_filter)
        
        # Execute search
        try:
            # Local Qdrant uses query_points with direct vector list
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=min_score
            ).points
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        
        # Convert to Documents
        documents = []
        for result in results:
            # Copy payload to avoid mutation
            payload_copy = dict(result.payload)
            
            # Extract page_content from copy
            page_content = payload_copy.pop('page_content', '')
            
            # Reconstruct metadata
            metadata = payload_copy
            
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            
            documents.append((doc, result.score))
        
        logger.debug(f"Search returned {len(documents)} results")
        return documents
    
    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Convert metadata dict to Qdrant Filter.
        
        Examples:
            {'doc_type': 'architecture'} → MatchValue filter
            {'priority': [1, 2]} → MatchAny filter
            {'has_errors': True} → MatchValue filter
        """
        conditions = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                # Match any value in list
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            elif isinstance(value, bool):
                # Boolean match
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dict with count, config, and index info
        """
        try:
            info = self.client.get_collection(collection_name)
            
            # Local Qdrant has different attributes than server version
            return {
                'name': collection_name,
                'points_count': info.points_count if hasattr(info, 'points_count') else 0,
                'status': info.status if hasattr(info, 'status') else 'unknown',
                'config': str(info.config) if hasattr(info, 'config') else None,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Verify Qdrant connection and collections.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check collections exist
            md_exists = self.client.collection_exists(self.md_collection)
            log_exists = self.client.collection_exists(self.log_collection)
            
            if not md_exists or not log_exists:
                logger.warning(
                    f"Collections missing - MD: {md_exists}, Logs: {log_exists}"
                )
                return False
            
            # Get collection stats
            md_info = self.get_collection_info(self.md_collection)
            log_info = self.get_collection_info(self.log_collection)
            
            logger.info(
                f"Health check passed - "
                f"MD: {md_info.get('points_count', 0)} points, "
                f"Logs: {log_info.get('points_count', 0)} points"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False