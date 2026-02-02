"""
Embedding Service

Generates vector embeddings using Google's text-embedding-004 model.
Handles batch processing and dimension validation.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

try:
    from google import genai
except ImportError:
    raise ImportError(
        "Google GenAI library not installed. "
        "Install with: pip install google-genai"
    )

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).resolve().parents[3] / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, will use environment variables directly
    pass

# Handle imports for both package and script execution
try:
    from ...config_loader import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config_loader import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Google text-embedding-004 embedding service.
    
    Features:
    - Batch processing for efficiency
    - Dimension validation (768-dim)
    - API key management
    - Error handling and retries
    """
    
    def __init__(self):
        """Initialize Google embedding model."""
        # Load config
        embed_config = config.get('agent.embeddings', {})
        
        self.provider = embed_config.get('provider', 'google')
        self.model_name = embed_config.get('model', 'models/text-embedding-004')
        self.dimension = embed_config.get('dimension', 768)
        self.batch_size = embed_config.get('batch_size', 100)
        
        # Get API key from environment
        api_key_env = embed_config.get('api_key_env', 'GEMINI_API_KEY')
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(
                f"API key not found in environment variable '{api_key_env}'. "
                f"Set it with: export {api_key_env}='your-api-key'"
            )
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"EmbeddingService initialized: {self.model_name} ({self.dimension}d)")
    
    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding model:
                - "RETRIEVAL_DOCUMENT": For indexing documents
                - "RETRIEVAL_QUERY": For search queries
                - "SEMANTIC_SIMILARITY": For similarity comparison
                
        Returns:
            768-dimensional embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config={'task_type': task_type}
            )
            
            embedding = result.embeddings[0].values
            
            # Validate dimension
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding)}"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for embedding model
            show_progress: Print progress updates
            
        Returns:
            List of 768-dimensional embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={self.batch_size})")
        
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            current_batch_num = (batch_idx // self.batch_size) + 1
            
            if show_progress:
                logger.info(f"  Processing batch {current_batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                # Google's API supports batch embedding
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts,
                    config={'task_type': task_type}
                )
                
                batch_embeddings = [emb.values for emb in result.embeddings]
                
                # Validate dimensions
                for idx, emb in enumerate(batch_embeddings):
                    if len(emb) != self.dimension:
                        raise ValueError(
                            f"Embedding dimension mismatch at index {batch_idx + idx}: "
                            f"expected {self.dimension}, got {len(emb)}"
                        )
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch {current_batch_num} failed: {e}")
                raise
        
        logger.info(f"SUCCESS: Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convenience method for embedding documents (uses RETRIEVAL_DOCUMENT task type).
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors
        """
        return self.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convenience method for embedding queries (uses RETRIEVAL_QUERY task type).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def health_check(self) -> bool:
        """
        Verify embedding service is working.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Generate test embedding
            test_embedding = self.embed_text("test")
            
            # Validate dimension
            if len(test_embedding) != self.dimension:
                logger.error(f"Health check failed: dimension mismatch")
                return False
            
            logger.info("Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
