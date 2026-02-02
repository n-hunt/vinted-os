"""
Indexing Pipeline Tests

Test the full LangGraph indexing workflow.
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

from agent.indexing.graph import IndexingPipeline


def test_basic_pipeline():
    """Test basic pipeline initialization and execution."""
    print("\n" + "="*80)
    print("TEST: Basic Indexing Pipeline")
    print("="*80)
    
    try:
        # Initialize pipeline
        print("\n1. Initializing pipeline...")
        pipeline = IndexingPipeline()
        print("   SUCCESS: Pipeline initialized")
        
        # Run on demo KB
        kb_path = Path(__file__).parent.parent / "data" / "demo_knowledge_base"
        
        if not kb_path.exists():
            print(f"\nWARNING: Demo KB not found at {kb_path}")
            print("   Skipping test")
            return
        
        print(f"\n2. Running pipeline on: {kb_path}")
        final_state = pipeline.run(
            source_paths=[str(kb_path)],
            mode="full"
        )
        
        # Validate results
        print("\n3. Validating results...")
        print(f"   Steps completed: {len(final_state['steps_completed'])}/4")
        print(f"   Total chunks: {final_state['total_chunks']}")
        print(f"   Total embeddings: {final_state['total_embeddings']}")
        print(f"   Errors: {len(final_state['errors'])}")
        print(f"   Vector store: {final_state['vector_store_status']}")
        print(f"   BM25 index: {final_state['bm25_status']}")
        
        # Assertions
        assert len(final_state['steps_completed']) == 4, "Should complete all 4 steps"
        assert final_state['total_chunks'] > 0, "Should process chunks"
        assert final_state['total_embeddings'] > 0, "Should generate embeddings"
        assert final_state['vector_store_status'] == "success", "Vector store should succeed"
        assert final_state['bm25_status'] == "success", "BM25 should succeed"
        assert len(final_state['errors']) == 0, "Should have no errors"
        
        print("\n" + "="*80)
        print("SUCCESS: TEST PASSED")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_basic_pipeline()
