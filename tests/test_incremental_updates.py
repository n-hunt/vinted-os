"""
Test incremental update functionality in the indexing pipeline.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.indexing.graph import IndexingPipeline
import logging

logging.basicConfig(level=logging.INFO)

def test_incremental_updates():
    """Test that incremental mode only processes changed files."""
    
    kb_path = Path(__file__).parent.parent / "data" / "demo_knowledge_base"
    
    print("\n" + "="*80)
    print("STEP 1: Full mode - establish baseline")
    print("="*80)
    
    pipeline = IndexingPipeline()
    result1 = pipeline.run(source_paths=[str(kb_path)], mode="full")
    
    full_chunks = len(result1.get("md_chunks", []))
    full_embeddings = len(result1.get("md_embeddings", []))
    
    print(f"\nFull mode results:")
    print(f"  Chunks: {full_chunks}")
    print(f"  Embeddings: {full_embeddings}")
    print(f"  Errors: {len(result1.get('errors', []))}")
    
    # Modify one file
    test_file = kb_path / "specs_constraints.md"
    original_content = test_file.read_text()
    
    print("\n" + "="*80)
    print("STEP 2: Modify one file")
    print("="*80)
    print(f"Modifying: {test_file.name}")
    
    # Add a comment to trigger change detection
    modified_content = original_content + "\n\n<!-- Test modification for incremental update -->\n"
    test_file.write_text(modified_content)
    
    print("\n" + "="*80)
    print("STEP 3: Incremental mode - should only process modified file")
    print("="*80)
    
    # Reuse same pipeline instance (Qdrant uses file locking)
    result2 = pipeline.run(source_paths=[str(kb_path)], mode="incremental")
    
    inc_modified = result2.get("modified_files", [])
    inc_chunks = len(result2.get("md_chunks", []))
    inc_embeddings = len(result2.get("md_embeddings", []))
    
    print(f"\nIncremental mode results:")
    print(f"  Modified files: {len(inc_modified)}")
    for f in inc_modified:
        print(f"    - {Path(f).name}")
    print(f"  Chunks processed: {inc_chunks}")
    print(f"  Embeddings generated: {inc_embeddings}")
    print(f"  Errors: {len(result2.get('errors', []))}")
    
    # Restore original content
    print("\n" + "="*80)
    print("CLEANUP: Restore original file")
    print("="*80)
    test_file.write_text(original_content)
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    success = True
    
    if len(inc_modified) != 1:
        print(f"ERROR: Expected 1 modified file, got {len(inc_modified)}")
        success = False
    else:
        print(f"SUCCESS: Correctly detected 1 modified file")
    
    # Validate incremental efficiency
    if inc_chunks < full_chunks:
        efficiency = (1 - inc_chunks / full_chunks) * 100
        print(f"SUCCESS: Incremental processed {inc_chunks}/{full_chunks} chunks ({efficiency:.1f}% reduction)")
    else:
        print(f"ERROR: Incremental should process fewer chunks ({inc_chunks} vs {full_chunks})")
        success = False
    
    if inc_embeddings < full_embeddings:
        api_savings = (1 - inc_embeddings / full_embeddings) * 100
        print(f"SUCCESS: Incremental generated {inc_embeddings}/{full_embeddings} embeddings ({api_savings:.1f}% API call savings)")
    else:
        print(f"ERROR: Incremental should generate fewer embeddings ({inc_embeddings} vs {full_embeddings})")
        success = False
    
    if result2.get("vector_store_status") == "success":
        print(f"SUCCESS: Vector store updated successfully")
    else:
        print(f"ERROR: Vector store status: {result2.get('vector_store_status')}")
        success = False
    
    if result2.get("bm25_status") == "success":
        print(f"SUCCESS: BM25 index updated successfully")
    else:
        print(f"ERROR: BM25 status: {result2.get('bm25_status')}")
        success = False
    
    # Validate file change detection
    if len(inc_modified) == 1 and "specs_constraints.md" in inc_modified[0]:
        print(f"SUCCESS: File change detection working correctly")
    else:
        print(f"ERROR: File change detection failed")
        success = False
    
    print("\n" + "="*80)
    if success:
        print("SUCCESS: INCREMENTAL UPDATE TEST PASSED")
    else:
        print("ERROR: INCREMENTAL UPDATE TEST FAILED")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    test_incremental_updates()
