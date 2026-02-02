"""
Embedding Service Tests

Test Google text-embedding-004 model initialization and embedding generation.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from agent.retrieval.embeddings import EmbeddingService


def test_initialization():
    """Test EmbeddingService initialization."""
    print("\n" + "="*80)
    print("TEST 1: EmbeddingService Initialization")
    print("="*80)
    
    try:
        embedding_service = EmbeddingService()
        print(f"SUCCESS: EmbeddingService initialized")
        print(f"  Provider: {embedding_service.provider}")
        print(f"  Model: {embedding_service.model_name}")
        print(f"  Dimension: {embedding_service.dimension}")
        print(f"  Batch size: {embedding_service.batch_size}")
        print(f"  API key set: {'Yes' if embedding_service.api_key else 'No'}")
        return embedding_service
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        raise


def test_single_embedding(service: EmbeddingService):
    """Test single text embedding."""
    print("\n" + "="*80)
    print("TEST 2: Single Text Embedding")
    print("="*80)
    
    test_text = "VintedOS is a local-first ETL pipeline for P2P commerce."
    
    try:
        embedding = service.embed_text(test_text)
        
        print(f"SUCCESS: Generated embedding for text: '{test_text[:50]}...'")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Last 5 values: {embedding[-5:]}")
        
        assert len(embedding) == service.dimension, f"Dimension mismatch: {len(embedding)} != {service.dimension}"
        assert all(isinstance(v, float) for v in embedding), "Embedding contains non-float values"
        
    except Exception as e:
        print(f"ERROR: Single embedding failed: {e}")
        raise


def test_batch_embedding(service: EmbeddingService):
    """Test batch text embedding."""
    print("\n" + "="*80)
    print("TEST 3: Batch Text Embedding")
    print("="*80)
    
    test_texts = [
        "Gmail API polls inbox for new messages",
        "PDF parser extracts transaction IDs and items",
        "Computer vision processes shipping labels",
        "Thermal printer outputs physical labels",
        "SQLite database stores all transactions"
    ]
    
    try:
        embeddings = service.embed_batch(test_texts, show_progress=True)
        
        print(f"\nSUCCESS: Generated {len(embeddings)} embeddings")
        print(f"  Input texts: {len(test_texts)}")
        print(f"  Output embeddings: {len(embeddings)}")
        
        # Validate all embeddings
        for idx, emb in enumerate(embeddings):
            assert len(emb) == service.dimension, f"Embedding {idx} dimension mismatch"
            assert all(isinstance(v, float) for v in emb), f"Embedding {idx} contains non-float values"
        
        print(f"  All embeddings validated SUCCESS")
        
    except Exception as e:
        print(f"ERROR: Batch embedding failed: {e}")
        raise


def test_query_vs_document(service: EmbeddingService):
    """Test query vs document embedding (different task types)."""
    print("\n" + "="*80)
    print("TEST 4: Query vs Document Embeddings")
    print("="*80)
    
    text = "database locked error troubleshooting"
    
    try:
        # Document embedding
        doc_embedding = service.embed_documents([text])[0]
        
        # Query embedding
        query_embedding = service.embed_query(text)
        
        print(f"SUCCESS: Generated both embedding types for: '{text}'")
        print(f"  Document embedding dimension: {len(doc_embedding)}")
        print(f"  Query embedding dimension: {len(query_embedding)}")
        
        # They should have same dimension but potentially different values
        assert len(doc_embedding) == len(query_embedding) == service.dimension
        
        # Calculate cosine similarity (should be high but not necessarily 1.0)
        import math
        dot_product = sum(a * b for a, b in zip(doc_embedding, query_embedding))
        norm_a = math.sqrt(sum(a * a for a in doc_embedding))
        norm_b = math.sqrt(sum(b * b for b in query_embedding))
        similarity = dot_product / (norm_a * norm_b)
        
        print(f"  Cosine similarity: {similarity:.4f}")
        
    except Exception as e:
        print(f"ERROR: Query vs document test failed: {e}")
        raise


def test_health_check(service: EmbeddingService):
    """Test health check."""
    print("\n" + "="*80)
    print("TEST 5: Health Check")
    print("="*80)
    
    try:
        healthy = service.health_check()
        
        if healthy:
            print("SUCCESS: Health check passed")
        else:
            print("ERROR: Health check failed")
            raise AssertionError("Health check returned False")
            
    except Exception as e:
        print(f"ERROR: Health check error: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EMBEDDING SERVICE TEST SUITE")
    print("="*80)
    print("\nNote: Requires GEMINI_API_KEY environment variable to be set")
    
    try:
        # Test 1: Initialization
        service = test_initialization()
        
        # Test 2: Single embedding
        test_single_embedding(service)
        
        # Test 3: Batch embedding
        test_batch_embedding(service)
        
        # Test 4: Query vs document
        test_query_vs_document(service)
        
        # Test 5: Health check
        test_health_check(service)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED SUCCESS")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST SUITE FAILED ERROR")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
