"""
Document Embedder Tests

Test batch embedding generation for markdown and log chunks.
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

from agent.indexing.nodes.embedder import DocumentEmbedder
from agent.indexing.nodes.kb_preprocessing import MDPreprocessor
from langchain_core.documents import Document


def test_initialization():
    """Test DocumentEmbedder initialization."""
    print("\n" + "="*80)
    print("TEST 1: DocumentEmbedder Initialization")
    print("="*80)
    
    try:
        embedder = DocumentEmbedder()
        print(f"SUCCESS: DocumentEmbedder initialized")
        print(f"  Embedding dimension: {embedder.get_dimension()}")
        return embedder
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        raise


def test_markdown_embedding(embedder: DocumentEmbedder):
    """Test embedding markdown chunks."""
    print("\n" + "="*80)
    print("TEST 2: Markdown Chunk Embedding")
    print("="*80)
    
    # Create mock markdown chunks
    md_chunks = [
        Document(
            page_content="## Architecture\n\nVintedOS uses service-oriented architecture with Gmail, Database, and Printer services.",
            metadata={
                'chunk_id': 'md_chunk_0',
                'source': 'data/demo_knowledge_base/architecture_map.md',
                'category': 'Architecture',
                'section': 'Overview',
                'doc_type': 'architecture',
                'priority': 1,
                'has_code': False,
                'mentioned_services': ['gmail', 'database', 'printer'],
                'normalized_text': 'VintedOS uses service-oriented architecture with Gmail, Database, and Printer services.'
            }
        ),
        Document(
            page_content="### Database Service\n\nHandles SQLite persistence with WAL mode for concurrent access.",
            metadata={
                'chunk_id': 'md_chunk_1',
                'source': 'data/demo_knowledge_base/architecture_map.md',
                'category': 'Architecture',
                'section': 'Database Service',
                'doc_type': 'architecture',
                'priority': 1,
                'has_code': True,
                'mentioned_services': ['database'],
                'normalized_text': 'Handles SQLite persistence with WAL mode for concurrent access.'
            }
        ),
        Document(
            page_content="## Troubleshooting\n\n### Database Locked Error\n\nOccurs when DB Browser is open in write mode.",
            metadata={
                'chunk_id': 'md_chunk_2',
                'source': 'data/demo_knowledge_base/troubleshooting_guide.md',
                'category': 'Troubleshooting',
                'section': 'Database Issues',
                'doc_type': 'troubleshooting',
                'priority': 2,
                'has_code': False,
                'mentioned_services': ['database'],
                'normalized_text': 'Database Locked Error Occurs when DB Browser is open in write mode.'
            }
        )
    ]
    
    try:
        chunks, embeddings = embedder.embed_markdown_chunks(md_chunks, show_progress=True)
        
        print(f"\nSUCCESS: Embedded {len(chunks)} markdown chunks")
        print(f"  Chunks returned: {len(chunks)}")
        print(f"  Embeddings generated: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Validate
        assert len(chunks) == len(md_chunks), "Chunk count mismatch"
        assert len(embeddings) == len(md_chunks), "Embedding count mismatch"
        assert all(len(emb) == 768 for emb in embeddings), "Embedding dimension mismatch"
        
        print(f"\n  Sample chunk 0:")
        print(f"    ID: {chunks[0].metadata['chunk_id']}")
        print(f"    Section: {chunks[0].metadata['section']}")
        print(f"    Content: {chunks[0].page_content[:80]}...")
        print(f"    Embedding preview: {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"ERROR: Markdown embedding failed: {e}")
        raise


def test_log_embedding(embedder: DocumentEmbedder):
    """Test embedding log chunks."""
    print("\n" + "="*80)
    print("TEST 3: Log Chunk Embedding")
    print("="*80)
    
    # Create mock log chunks
    log_chunks = [
        Document(
            page_content="2024-02-01 10:15:23 | ERROR | database | TID:12345 | DatabaseLockError: database is locked",
            metadata={
                'chunk_id': 'log_chunk_0',
                'source_file': 'logs/server_errors.log',
                'transaction_ids': ['12345'],
                'modules': ['database'],
                'log_levels': ['ERROR'],
                'has_errors': True,
                'error_types': ['DatabaseLockError'],
                'normalized_text': 'DatabaseLockError database is locked'
            }
        ),
        Document(
            page_content="2024-02-01 10:15:30 | INFO | printer | TID:12346 | Label printed successfully",
            metadata={
                'chunk_id': 'log_chunk_1',
                'source_file': 'logs/server_errors.log',
                'transaction_ids': ['12346'],
                'modules': ['printer'],
                'log_levels': ['INFO'],
                'has_errors': False,
                'error_types': [],
                'normalized_text': 'Label printed successfully'
            }
        )
    ]
    
    try:
        chunks, embeddings = embedder.embed_log_chunks(log_chunks, show_progress=True)
        
        print(f"\nSUCCESS: Embedded {len(chunks)} log chunks")
        print(f"  Chunks returned: {len(chunks)}")
        print(f"  Embeddings generated: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Validate
        assert len(chunks) == len(log_chunks), "Chunk count mismatch"
        assert len(embeddings) == len(log_chunks), "Embedding count mismatch"
        assert all(len(emb) == 768 for emb in embeddings), "Embedding dimension mismatch"
        
        print(f"\n  Sample log chunk 0:")
        print(f"    ID: {chunks[0].metadata['chunk_id']}")
        print(f"    TID: {chunks[0].metadata['transaction_ids']}")
        print(f"    Error types: {chunks[0].metadata['error_types']}")
        print(f"    Embedding preview: {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"ERROR: Log embedding failed: {e}")
        raise


def test_auto_detection(embedder: DocumentEmbedder):
    """Test auto-detection of document type."""
    print("\n" + "="*80)
    print("TEST 4: Auto Document Type Detection")
    print("="*80)
    
    # Create mixed chunks
    chunks = [
        Document(
            page_content="## Test MD",
            metadata={'chunk_id': 'md_chunk_99', 'doc_type': 'test'}
        ),
        Document(
            page_content="2024-01-01 | INFO | test",
            metadata={'chunk_id': 'log_chunk_99', 'doc_type': 'test'}
        )
    ]
    
    try:
        # Test markdown auto-detect
        md_result = embedder.embed_documents([chunks[0]], document_type="auto")
        print(f"SUCCESS: Auto-detected markdown chunk: {md_result[0][0].metadata['chunk_id']}")
        
        # Test log auto-detect
        log_result = embedder.embed_documents([chunks[1]], document_type="auto")
        print(f"SUCCESS: Auto-detected log chunk: {log_result[0][0].metadata['chunk_id']}")
        
    except Exception as e:
        print(f"ERROR: Auto-detection failed: {e}")
        raise


def test_real_kb_preprocessing(embedder: DocumentEmbedder):
    """Test with real preprocessed markdown chunks."""
    print("\n" + "="*80)
    print("TEST 5: Real Knowledge Base Preprocessing + Embedding")
    print("="*80)
    
    try:
        # Run MDPreprocessor on demo KB
        preprocessor = MDPreprocessor()
        print("Running MDPreprocessor on demo knowledge base...")
        
        md_chunks, graph, tfidf = preprocessor.run(use_tfidf=False, use_graph=False)
        
        print(f"SUCCESS: Preprocessed {len(md_chunks)} markdown chunks")
        
        # Take first 3 chunks for embedding test
        sample_chunks = md_chunks[:3]
        
        print(f"Embedding sample of {len(sample_chunks)} chunks...")
        chunks, embeddings = embedder.embed_markdown_chunks(sample_chunks, show_progress=True)
        
        print(f"\nSUCCESS: Successfully embedded real KB chunks")
        print(f"  Sample chunk 0:")
        print(f"    Source: {chunks[0].metadata.get('source', 'unknown')}")
        print(f"    Category: {chunks[0].metadata.get('category', 'unknown')}")
        print(f"    Has code: {chunks[0].metadata.get('has_code', False)}")
        print(f"    Services: {chunks[0].metadata.get('mentioned_services', [])}")
        
    except Exception as e:
        print(f"ERROR: Real KB test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DOCUMENT EMBEDDER TEST SUITE")
    print("="*80)
    print("\nNote: Requires GEMINI_API_KEY environment variable")
    
    try:
        # Test 1: Initialization
        embedder = test_initialization()
        
        # Test 2: Markdown embedding
        test_markdown_embedding(embedder)
        
        # Test 3: Log embedding
        test_log_embedding(embedder)
        
        # Test 4: Auto-detection
        test_auto_detection(embedder)
        
        # Test 5: Real KB preprocessing
        test_real_kb_preprocessing(embedder)
        
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
