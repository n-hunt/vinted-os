"""
BM25 Index Tests

Test keyword-based sparse retrieval.
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

from agent.retrieval.bm25_index import BM25Index
from agent.indexing.nodes.kb_preprocessing import MDPreprocessor
from langchain_core.documents import Document


def test_initialization():
    """Test BM25Index initialization."""
    print("\n" + "="*80)
    print("TEST 1: BM25Index Initialization")
    print("="*80)
    
    try:
        bm25 = BM25Index()
        print(f"SUCCESS: BM25Index initialized")
        print(f"  k1: {bm25.k1}")
        print(f"  b: {bm25.b}")
        return bm25
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        raise


def test_tokenization(bm25: BM25Index):
    """Test tokenization."""
    print("\n" + "="*80)
    print("TEST 2: Tokenization")
    print("="*80)
    
    test_cases = [
        ("VintedOS is a local-first ETL pipeline", 
         ['vintedos', 'is', 'local', 'first', 'etl', 'pipeline']),
        ("Database locked error (SQLite WAL mode)", 
         ['database', 'locked', 'error', 'sqlite', 'wal', 'mode']),
        ("PDF parsing with pdfplumber & Pillow", 
         ['pdf', 'parsing', 'with', 'pdfplumber', 'pillow'])
    ]
    
    try:
        for text, expected_subset in test_cases:
            tokens = bm25.tokenize(text)
            print(f"\nText: '{text}'")
            print(f"Tokens: {tokens}")
            
            # Check expected tokens are present
            for expected in expected_subset:
                assert expected in tokens, f"Missing token: {expected}"
        
        print(f"\nSUCCESS: Tokenization working correctly")
        
    except Exception as e:
        print(f"ERROR: Tokenization failed: {e}")
        raise


def test_index_building(bm25: BM25Index):
    """Test building BM25 index."""
    print("\n" + "="*80)
    print("TEST 3: Index Building")
    print("="*80)
    
    # Create mock markdown chunks
    md_chunks = [
        Document(
            page_content="## Database Service\n\nHandles SQLite persistence with WAL mode.",
            metadata={
                'chunk_id': 'md_chunk_0',
                'normalized_text': 'Database Service Handles SQLite persistence with WAL mode',
                'doc_type': 'architecture'
            }
        ),
        Document(
            page_content="### Printer Service\n\nManages CUPS thermal printing.",
            metadata={
                'chunk_id': 'md_chunk_1',
                'normalized_text': 'Printer Service Manages CUPS thermal printing',
                'doc_type': 'architecture'
            }
        ),
        Document(
            page_content="## Database Locked Error\n\nOccurs when DB Browser is open.",
            metadata={
                'chunk_id': 'md_chunk_2',
                'normalized_text': 'Database Locked Error Occurs when DB Browser is open',
                'doc_type': 'troubleshooting'
            }
        )
    ]
    
    try:
        bm25.index_markdown(md_chunks)
        
        print(f"\nSUCCESS: Built markdown index")
        print(f"  Documents: {bm25.md_index['num_docs']}")
        print(f"  Unique terms: {len(bm25.md_index['doc_freqs'])}")
        print(f"  Average doc length: {bm25.md_index['avgdl']:.1f}")
        
        # Check some expected terms
        assert 'database' in bm25.md_index['doc_freqs'], "Missing term: database"
        assert 'sqlite' in bm25.md_index['doc_freqs'], "Missing term: sqlite"
        
        print(f"\n  Sample term frequencies:")
        for term in ['database', 'sqlite', 'printer']:
            df = bm25.md_index['doc_freqs'].get(term, 0)
            print(f"    '{term}': appears in {df} documents")
        
    except Exception as e:
        print(f"ERROR: Index building failed: {e}")
        raise


def test_search(bm25: BM25Index):
    """Test BM25 search."""
    print("\n" + "="*80)
    print("TEST 4: BM25 Search")
    print("="*80)
    
    queries = [
        ("database locked error", "Should find troubleshooting doc"),
        ("sqlite wal mode", "Should find database service doc"),
        ("thermal printer cups", "Should find printer service doc"),
        ("gmail api authentication", "Should find nothing (no matching docs)")
    ]
    
    try:
        for query, description in queries:
            print(f"\nQuery: '{query}' ({description})")
            results = bm25.search(query, collection="md_chunks", top_k=3)
            
            print(f"  Results: {len(results)}")
            for i, (doc, score) in enumerate(results, 1):
                chunk_id = doc.metadata.get('chunk_id')
                doc_type = doc.metadata.get('doc_type')
                content_preview = doc.page_content[:60].replace('\n', ' ')
                print(f"    {i}. [{chunk_id}] score={score:.3f} type={doc_type}")
                print(f"       '{content_preview}...'")
        
        print(f"\nSUCCESS: Search working correctly")
        
    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        raise


def test_persistence(bm25: BM25Index):
    """Test saving and loading indexes."""
    print("\n" + "="*80)
    print("TEST 5: Index Persistence")
    print("="*80)
    
    try:
        # Save
        save_path = Path(__file__).parent / "test_bm25_data"
        bm25.save(save_path)
        print(f"SUCCESS: Saved indexes to {save_path}")
        
        # Create new instance and load
        bm25_new = BM25Index()
        bm25_new.load(save_path)
        print(f"SUCCESS: Loaded indexes from {save_path}")
        
        # Verify
        assert bm25_new.md_index is not None, "MD index not loaded"
        assert bm25_new.md_index['num_docs'] == bm25.md_index['num_docs'], "Doc count mismatch"
        
        print(f"\n  Loaded markdown index:")
        print(f"    Documents: {bm25_new.md_index['num_docs']}")
        print(f"    Unique terms: {len(bm25_new.md_index['doc_freqs'])}")
        
        # Test search on loaded index
        results = bm25_new.search("database locked", collection="md_chunks", top_k=1)
        assert len(results) > 0, "Search failed on loaded index"
        print(f"\nSUCCESS: Search works on loaded index ({len(results)} results)")
        
    except Exception as e:
        print(f"ERROR: Persistence test failed: {e}")
        raise


def test_real_kb_indexing(bm25: BM25Index):
    """Test with real preprocessed knowledge base."""
    print("\n" + "="*80)
    print("TEST 6: Real Knowledge Base Indexing")
    print("="*80)
    
    try:
        # Preprocess KB
        preprocessor = MDPreprocessor()
        print("Running MDPreprocessor on demo knowledge base...")
        
        md_chunks, _, _ = preprocessor.run(use_tfidf=False, use_graph=False)
        print(f"SUCCESS: Preprocessed {len(md_chunks)} markdown chunks")
        
        # Build BM25 index
        print("Building BM25 index...")
        bm25_real = BM25Index()
        bm25_real.index_markdown(md_chunks)
        
        print(f"\nSUCCESS: Built real KB index:")
        print(f"  Documents: {bm25_real.md_index['num_docs']}")
        print(f"  Unique terms: {len(bm25_real.md_index['doc_freqs'])}")
        print(f"  Average doc length: {bm25_real.md_index['avgdl']:.1f}")
        
        # Test real queries
        real_queries = [
            "database locked error",
            "gmail authentication failure",
            "printer thermal label"
        ]
        
        print("\nTesting real queries:")
        for query in real_queries:
            results = bm25_real.search(query, collection="md_chunks", top_k=2)
            print(f"\n  Query: '{query}' → {len(results)} results")
            if results:
                top_doc, top_score = results[0]
                content_preview = top_doc.page_content[:80].replace('\n', ' ')
                print(f"    Top result (score={top_score:.3f}): '{content_preview}...'")
        
    except Exception as e:
        print(f"ERROR: Real KB indexing failed: {e}")
        raise


def test_health_check(bm25: BM25Index):
    """Test health check."""
    print("\n" + "="*80)
    print("TEST 7: Health Check")
    print("="*80)
    
    try:
        health = bm25.health_check()
        
        print(f"\nBM25 Health Status:")
        print(f"  k1: {health['k1']}")
        print(f"  b: {health['b']}")
        
        if health['md_index']:
            print(f"\n  Markdown Index:")
            print(f"    Documents: {health['md_index']['num_docs']}")
            print(f"    Unique terms: {health['md_index']['unique_terms']}")
            print(f"    Avg doc length: {health['md_index']['avgdl']:.1f}")
        
        if health['log_index']:
            print(f"\n  Log Index:")
            print(f"    Documents: {health['log_index']['num_docs']}")
            print(f"    Unique terms: {health['log_index']['unique_terms']}")
            print(f"    Avg doc length: {health['log_index']['avgdl']:.1f}")
        
        print(f"\nSUCCESS: Health check complete")
        
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BM25 INDEX TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Initialization
        bm25 = test_initialization()
        
        # Test 2: Tokenization
        test_tokenization(bm25)
        
        # Test 3: Index building
        test_index_building(bm25)
        
        # Test 4: Search
        test_search(bm25)
        
        # Test 5: Persistence
        test_persistence(bm25)
        
        # Test 6: Real KB
        test_real_kb_indexing(bm25)
        
        # Test 7: Health check
        test_health_check(bm25)
        
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
