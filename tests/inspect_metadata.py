"""
Inspect what metadata actually exists in the KB chunks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.indexing.graph import IndexingPipeline
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress info logs

def inspect_metadata():
    """Check what metadata values exist in chunks."""
    
    print("\n" + "="*80)
    print("Inspecting KB Metadata")
    print("="*80)
    
    # Index KB
    kb_path = Path(__file__).parent.parent / "data" / "demo_knowledge_base"
    pipeline = IndexingPipeline()
    result = pipeline.run(source_paths=[str(kb_path)], mode="full")
    
    # Get chunks from BM25 index (has all chunks)
    chunks = pipeline.bm25_index.md_index['chunks']
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Collect unique values for each metadata field
    categories = set()
    sections = set()
    doc_types = set()
    
    print("\n" + "="*80)
    print("Sample of first 5 chunks:")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i+1}:")
        print(f"  category: {chunk.metadata.get('category', 'N/A')}")
        print(f"  section: {chunk.metadata.get('section', 'N/A')}")
        print(f"  subsection: {chunk.metadata.get('subsection', 'N/A')}")
        print(f"  doc_type: {chunk.metadata.get('doc_type', 'N/A')}")
        print(f"  source: {Path(chunk.metadata.get('source', 'N/A')).name}")
        print(f"  preview: {chunk.page_content[:60]}...")
    
    # Collect all unique values
    for chunk in chunks:
        cat = chunk.metadata.get('category')
        sec = chunk.metadata.get('section')
        dtype = chunk.metadata.get('doc_type')
        
        if cat:
            categories.add(cat)
        if sec:
            sections.add(sec)
        if dtype:
            doc_types.add(dtype)
    
    print("\n" + "="*80)
    print("Unique Metadata Values Across All Chunks:")
    print("="*80)
    
    print(f"\nCategories ({len(categories)}):")
    for cat in sorted(categories):
        count = sum(1 for c in chunks if c.metadata.get('category') == cat)
        print(f"  - '{cat}' ({count} chunks)")
    
    print(f"\nSections ({len(sections)}):")
    for sec in sorted(list(sections)[:10]):  # Show first 10
        count = sum(1 for c in chunks if c.metadata.get('section') == sec)
        print(f"  - '{sec}' ({count} chunks)")
    if len(sections) > 10:
        print(f"  ... and {len(sections) - 10} more")
    
    print(f"\nDoc Types ({len(doc_types)}):")
    for dtype in sorted(doc_types):
        count = sum(1 for c in chunks if c.metadata.get('doc_type') == dtype)
        print(f"  - '{dtype}' ({count} chunks)")
    
    print("\n" + "="*80)
    print("Why 'Service Architecture' filter returned 0 results:")
    print("="*80)
    print("\nThe filter looked for: category='Service Architecture'")
    print("But actual category values are:", sorted(categories)[:5])
    print("\nTo get results, use an actual category value like:")
    print("  metadata_filter={'category': 'CUPS Printer Connection Timeout'}")
    print("  or")
    print("  metadata_filter={'doc_type': 'architecture'}")  # if doc_type exists
    
if __name__ == "__main__":
    inspect_metadata()
