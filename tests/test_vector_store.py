"""
Vector Store Tests

Test Qdrant collection creation, insertion, and search.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from agent.retrieval.vector_store import VectorStore
from langchain_core.documents import Document


def test_initialization():
    """Test VectorStore initialization."""
    print("\n" + "="*80)
    print("TEST 1: VectorStore Initialization")
    print("="*80)
    
    try:
        vector_store = VectorStore()
        print(f"SUCCESS: VectorStore initialized")
        print(f"  Qdrant path: {vector_store.qdrant_path}")
        print(f"  MD collection: {vector_store.md_collection}")
        print(f"  Log collection: {vector_store.log_collection}")
        print(f"  Embedding dimension: {vector_store.embedding_dim}")
        print(f"  Distance metric: {vector_store.distance_metric}")
        return vector_store
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        raise


def test_collection_creation(vector_store: VectorStore):
    """Test collection creation with schemas."""
    print("\n" + "="*80)
    print("TEST 2: Collection Creation")
    print("="*80)
    
    try:
        # Create collections (recreate to start fresh)
        vector_store.initialize_collections(recreate=True)
        
        # Verify collections exist
        md_exists = vector_store.client.collection_exists(vector_store.md_collection)
        log_exists = vector_store.client.collection_exists(vector_store.log_collection)
        
        print(f"SUCCESS: Collections created")
        print(f"  {vector_store.md_collection}: {md_exists}")
        print(f"  {vector_store.log_collection}: {log_exists}")
        
        # Get collection info
        md_info = vector_store.get_collection_info(vector_store.md_collection)
        log_info = vector_store.get_collection_info(vector_store.log_collection)
        
        print(f"\nCollection Stats:")
        print(f"  {vector_store.md_collection}:")
        print(f"    Points: {md_info.get('points_count', 0)}")
        print(f"    Status: {md_info.get('status', 'unknown')}")
        print(f"  {vector_store.log_collection}:")
        print(f"    Points: {log_info.get('points_count', 0)}")
        print(f"    Status: {log_info.get('status', 'unknown')}")
        
        assert md_exists and log_exists, "Collections not created"
        
    except Exception as e:
        print(f"ERROR: Collection creation failed: {e}")
        raise


def test_document_insertion(vector_store: VectorStore):
    """Test document insertion with mock embeddings."""
    print("\n" + "="*80)
    print("TEST 3: Document Insertion")
    print("="*80)
    
    # Create mock markdown documents
    md_docs = [
        Document(
            page_content="## Architecture\n\nVintedOS uses service-oriented architecture.",
            metadata={
                'chunk_id': 'md_chunk_0',
                'source': 'data/demo_knowledge_base/architecture_map.md',
                'category': 'Architecture',
                'section': 'Overview',
                'doc_type': 'architecture',
                'priority': 1,
                'has_code': False,
                'mentioned_services': ['gmail', 'database'],
                'normalized_text': 'VintedOS uses service-oriented architecture'
            }
        ),
        Document(
            page_content="### Troubleshooting\n\nDatabase locked error occurs when...",
            metadata={
                'chunk_id': 'md_chunk_1',
                'source': 'data/demo_knowledge_base/troubleshooting_guide.md',
                'category': 'Troubleshooting',
                'section': 'Database Issues',
                'doc_type': 'troubleshooting',
                'priority': 2,
                'has_code': True,
                'mentioned_services': ['database'],
                'normalized_text': 'Database locked error occurs when'
            }
        )
    ]
    
    # Create mock log documents
    log_docs = [
        Document(
            page_content="2024-02-01 10:15:23 | ERROR | database | TID:12345 | DatabaseLockError",
            metadata={
                'chunk_id': 'log_chunk_0',
                'source_file': 'logs/server_errors.log',
                'transaction_ids': ['12345'],
                'modules': ['database'],
                'log_levels': ['ERROR'],
                'has_errors': True,
                'error_types': ['DatabaseLockError'],
                'normalized_text': 'DatabaseLockError'
            }
        )
    ]
    
    # Create mock embeddings (768-dim zero vectors)
    md_embeddings = [[0.0] * 768 for _ in md_docs]
    log_embeddings = [[0.0] * 768 for _ in log_docs]
    
    try:
        # Insert MD documents
        md_count = vector_store.insert_documents(
            documents=md_docs,
            embeddings=md_embeddings,
            collection_name=vector_store.md_collection
        )
        print(f"SUCCESS: Inserted {md_count} markdown documents")
        
        # Insert log documents
        log_count = vector_store.insert_documents(
            documents=log_docs,
            embeddings=log_embeddings,
            collection_name=vector_store.log_collection
        )
        print(f"SUCCESS: Inserted {log_count} log documents")
        
        # Verify counts
        md_info = vector_store.get_collection_info(vector_store.md_collection)
        log_info = vector_store.get_collection_info(vector_store.log_collection)
        
        print(f"\nUpdated Collection Stats:")
        print(f"  {vector_store.md_collection}: {md_info.get('points_count', 0)} points")
        print(f"  {vector_store.log_collection}: {log_info.get('points_count', 0)} points")
        
        # Exact match since we know how many we inserted
        assert md_info.get('points_count', 0) == md_count, f"Expected {md_count} MD points, got {md_info.get('points_count', 0)}"
        assert log_info.get('points_count', 0) == log_count, f"Expected {log_count} log points, got {log_info.get('points_count', 0)}"
        
    except Exception as e:
        print(f"ERROR: Document insertion failed: {e}")
        raise


def test_search(vector_store: VectorStore):
    """Test vector search with metadata filtering."""
    print("\n" + "="*80)
    print("TEST 4: Vector Search")
    print("="*80)
    
    # Create mock query vector (768-dim zero vector)
    query_vector = [0.0] * 768
    
    try:
        # Search markdown collection
        print("\nSearching markdown collection...")
        md_results = vector_store.search(
            query_vector=query_vector,
            collection_name=vector_store.md_collection,
            top_k=5
        )
        print(f"SUCCESS: Found {len(md_results)} markdown results")
        for doc, score in md_results:
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            print(f"  {chunk_id} ({doc_type}) - Score: {score:.4f}")
        
        # Search with metadata filter
        print("\nSearching with metadata filter (doc_type='architecture')...")
        filtered_results = vector_store.search(
            query_vector=query_vector,
            collection_name=vector_store.md_collection,
            top_k=5,
            metadata_filter={'doc_type': 'architecture'}
        )
        print(f"SUCCESS: Found {len(filtered_results)} filtered results")
        for doc, score in filtered_results:
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            print(f"  {chunk_id} ({doc_type}) - Score: {score:.4f}")
        
        # Search log collection
        print("\nSearching log collection...")
        log_results = vector_store.search(
            query_vector=query_vector,
            collection_name=vector_store.log_collection,
            top_k=5
        )
        print(f"SUCCESS: Found {len(log_results)} log results")
        for doc, score in log_results:
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            tids = doc.metadata.get('transaction_ids', [])
            print(f"  {chunk_id} (TIDs: {tids}) - Score: {score:.4f}")
        
    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        raise


def test_health_check(vector_store: VectorStore):
    """Test health check."""
    print("\n" + "="*80)
    print("TEST 5: Health Check")
    print("="*80)
    
    try:
        healthy = vector_store.health_check()
        
        if healthy:
            print("SUCCESS: Health check passed")
        else:
            print("ERROR: Health check failed")
            
    except Exception as e:
        print(f"ERROR: Health check error: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VECTOR STORE TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Initialization
        vector_store = test_initialization()
        
        # Test 2: Collection creation
        test_collection_creation(vector_store)
        
        # Test 3: Document insertion
        test_document_insertion(vector_store)
        
        # Test 4: Search
        test_search(vector_store)
        
        # Test 5: Health check
        test_health_check(vector_store)
        
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
