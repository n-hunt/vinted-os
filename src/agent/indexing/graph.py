"""
Indexing Pipeline Graph

LangGraph workflow orchestrating: load → preprocess → embed → store
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# Handle imports
try:
    from .state import IndexingState
    from .nodes.kb_preprocessing import MDPreprocessor, LogPreprocessor
    from .nodes.embedder import DocumentEmbedder
    from .file_tracker import FileHashTracker
    from ..retrieval.vector_store import VectorStore
    from ..retrieval.bm25_index import BM25Index
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.indexing.state import IndexingState
    from agent.indexing.nodes.kb_preprocessing import MDPreprocessor, LogPreprocessor
    from agent.indexing.nodes.embedder import DocumentEmbedder
    from agent.indexing.file_tracker import FileHashTracker
    from agent.retrieval.vector_store import VectorStore
    from agent.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    LangGraph-based indexing pipeline.
    
    Current implementation: load_sources node only
    """
    
    def __init__(self):
        """Initialize pipeline (components will be added incrementally)."""
        logger.info("Initializing IndexingPipeline")
        
        # Initialize preprocessors
        self.md_preprocessor = MDPreprocessor()
        self.log_preprocessor = LogPreprocessor()
        
        # Initialize embedder
        self.embedder = DocumentEmbedder()
        
        # Initialize storage components
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        
        # Initialize file tracker for incremental updates
        self.file_tracker = FileHashTracker()
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("IndexingPipeline initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(IndexingState)
        
        # Add nodes
        workflow.add_node("load_sources", self.load_sources)
        workflow.add_node("preprocess_documents", self.preprocess_documents)
        workflow.add_node("generate_embeddings", self.generate_embeddings)
        workflow.add_node("store_indexes", self.store_indexes)
        
        # Define edges
        workflow.set_entry_point("load_sources")
        workflow.add_edge("load_sources", "preprocess_documents")
        workflow.add_edge("preprocess_documents", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "store_indexes")
        workflow.add_edge("store_indexes", END)
        
        return workflow.compile()
    
    def load_sources(self, state: IndexingState) -> IndexingState:
        """
        Node 1: Load source files from disk.
        
        Loads markdown files from KB directory.
        Supports incremental mode - only loads changed files.
        """
        logger.info("="*80)
        logger.info("STEP 1: Load Sources")
        logger.info("="*80)
        
        state["current_step"] = "load_sources"
        state["start_time"] = time.time()
        
        try:
            mode = state.get("mode", "full")
            raw_documents = []
            all_files = []
            
            # Collect all markdown files
            for source_path in state.get("source_paths", []):
                path = Path(source_path)
                
                if not path.exists():
                    state["warnings"].append(f"Path not found: {source_path}")
                    continue
                
                # Single markdown file
                if path.is_file() and path.suffix == ".md":
                    all_files.append(path)
                
                # Directory of markdown files
                elif path.is_dir():
                    all_files.extend(path.rglob("*.md"))
            
            # Detect changes for incremental mode
            if mode == "incremental":
                logger.info("Detecting file changes...")
                changes = self.file_tracker.detect_changes(all_files)
                
                modified = changes['modified']
                added = changes['added']
                deleted = changes['deleted']
                
                logger.info(f"  Modified: {len(modified)} files")
                logger.info(f"  Added: {len(added)} files")
                logger.info(f"  Deleted: {len(deleted)} files")
                
                # Only load changed files
                files_to_load = [Path(f) for f in modified + added]
                state["modified_files"] = modified + added
                
                if deleted:
                    state["warnings"].extend([f"Deleted file: {f}" for f in deleted])
            else:
                # Full mode - load all files and compute hashes
                files_to_load = all_files
                state["modified_files"] = None
                
                # Compute hashes for all files (needed for next incremental run)
                for file_path in all_files:
                    file_hash = self.file_tracker.compute_hash(file_path)
                    self.file_tracker.hashes[str(file_path)] = file_hash
            
            # Load files
            for file_path in files_to_load:
                logger.info(f"Loading: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={'source': str(file_path), 'type': 'markdown'}
                )
                raw_documents.append(doc)
            
            # Store file hashes for next incremental run
            state["file_hashes"] = self.file_tracker.hashes.copy()
            
            state["raw_documents"] = raw_documents
            state["steps_completed"].append("load_sources")
            
            logger.info(f"SUCCESS: Loaded {len(raw_documents)} source documents")
            
        except Exception as e:
            logger.error(f"Error loading sources: {e}")
            state["errors"].append({
                "step": "load_sources",
                "error": str(e),
                "type": type(e).__name__
            })
        
        return state
    
    def preprocess_documents(self, state: IndexingState) -> IndexingState:
        """
        Node 2: Preprocess documents into chunks.
        
        Runs MDPreprocessor and/or LogPreprocessor based on mode.
        """
        logger.info("="*80)
        logger.info("STEP 2: Preprocess Documents")
        logger.info("="*80)
        
        state["current_step"] = "preprocess_documents"
        
        try:
            mode = state.get("mode", "full")
            raw_documents = state.get("raw_documents", [])
            
            md_chunks = []
            log_chunks = []
            
            # Process markdown documents
            if mode in ["full", "md_only", "incremental"]:
                logger.info("Running MDPreprocessor...")
                md_docs = [d for d in raw_documents if d.metadata.get('type') == 'markdown']
                
                if md_docs:
                    # Pass documents to preprocessor for incremental mode support
                    if mode == "incremental":
                        # Incremental: only process filtered documents
                        md_chunks, _, _ = self.md_preprocessor.run(documents=md_docs, use_tfidf=False, use_graph=False)
                    else:
                        # Full/md_only: load all from disk (documents=None)
                        md_chunks, _, _ = self.md_preprocessor.run(use_tfidf=False, use_graph=False)
                    logger.info(f"SUCCESS: Preprocessed {len(md_chunks)} markdown chunks")
            
            # Process log documents (placeholder)
            if mode in ["full", "logs_only"]:
                logger.info("Log preprocessing not yet implemented")
                # Future: log_chunks = self.log_preprocessor.run()
            
            state["md_chunks"] = md_chunks
            state["log_chunks"] = log_chunks
            state["total_chunks"] = len(md_chunks) + len(log_chunks)
            state["steps_completed"].append("preprocess_documents")
            
            logger.info(f"SUCCESS: Total chunks: {state['total_chunks']}")
            
        except Exception as e:
            logger.error(f"Error preprocessing documents: {e}")
            state["errors"].append({
                "step": "preprocess_documents",
                "error": str(e),
                "type": type(e).__name__
            })
        
        return state
    
    def generate_embeddings(self, state: IndexingState) -> IndexingState:
        """
        Node 3: Generate embeddings for all chunks.
        
        Uses DocumentEmbedder for batch processing.
        """
        logger.info("="*80)
        logger.info("STEP 3: Generate Embeddings")
        logger.info("="*80)
        
        state["current_step"] = "generate_embeddings"
        
        try:
            md_chunks = state.get("md_chunks", [])
            log_chunks = state.get("log_chunks", [])
            
            md_embeddings = []
            log_embeddings = []
            
            # Embed markdown chunks
            if md_chunks:
                logger.info(f"Embedding {len(md_chunks)} markdown chunks...")
                _, md_embeddings = self.embedder.embed_markdown_chunks(
                    md_chunks,
                    show_progress=True
                )
            
            # Embed log chunks
            if log_chunks:
                logger.info(f"Embedding {len(log_chunks)} log chunks...")
                _, log_embeddings = self.embedder.embed_log_chunks(
                    log_chunks,
                    show_progress=True
                )
            
            state["md_embeddings"] = md_embeddings
            state["log_embeddings"] = log_embeddings
            state["total_embeddings"] = len(md_embeddings) + len(log_embeddings)
            state["steps_completed"].append("generate_embeddings")
            
            logger.info(f"SUCCESS: Generated {state['total_embeddings']} embeddings")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            state["errors"].append({
                "step": "generate_embeddings",
                "error": str(e),
                "type": type(e).__name__
            })
        
        return state
    
    def store_indexes(self, state: IndexingState) -> IndexingState:
        """
        Node 4: Store embeddings in vector store and BM25 index.
        
        Persists to Qdrant and pickle files.
        """
        logger.info("="*80)
        logger.info("STEP 4: Store Indexes")
        logger.info("="*80)
        
        state["current_step"] = "store_indexes"
        
        try:
            md_chunks = state.get("md_chunks", [])
            log_chunks = state.get("log_chunks", [])
            md_embeddings = state.get("md_embeddings", [])
            log_embeddings = state.get("log_embeddings", [])
            
            # Initialize collections
            logger.info("Initializing vector collections...")
            self.vector_store.initialize_collections()
            
            # Store markdown vectors
            if md_chunks and md_embeddings:
                logger.info(f"Storing {len(md_chunks)} markdown vectors...")
                self.vector_store.insert_documents(
                    documents=md_chunks,
                    embeddings=md_embeddings,
                    collection_name="md_chunks"
                )
                
                # Build BM25 index for markdown
                logger.info("Building markdown BM25 index...")
                self.bm25_index.index_markdown(md_chunks)
            
            # Store log vectors
            if log_chunks and log_embeddings:
                logger.info(f"Storing {len(log_chunks)} log vectors...")
                self.vector_store.insert_documents(
                    documents=log_chunks,
                    embeddings=log_embeddings,
                    collection_name="log_chunks"
                )
                
                # Build BM25 index for logs
                logger.info("Building log BM25 index...")
                self.bm25_index.index_logs(log_chunks)
            
            # Persist BM25 indexes
            logger.info("Saving BM25 indexes to disk...")
            self.bm25_index.save(Path("data/bm25_indexes"))
            
            # Save file hash registry (for incremental updates)
            if state.get("file_hashes"):
                logger.info("Saving file hash registry...")
                self.file_tracker.hashes = state["file_hashes"]
                self.file_tracker.save_registry()
            
            state["vector_store_status"] = "success"
            state["bm25_status"] = "success"
            state["steps_completed"].append("store_indexes")
            state["end_time"] = time.time()
            
            # Calculate duration
            duration = state["end_time"] - state.get("start_time", state["end_time"])
            logger.info(f"SUCCESS: Indexing complete in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error storing indexes: {e}")
            state["errors"].append({
                "step": "store_indexes",
                "error": str(e),
                "type": type(e).__name__
            })
            state["vector_store_status"] = "failed"
            state["bm25_status"] = "failed"
        
        return state
    
    def run(self, source_paths: List[str], mode: str = "full") -> Dict[str, Any]:
        """
        Run the full indexing pipeline.
        
        Args:
            source_paths: Paths to KB files or directories
            mode: "full" | "incremental" | "md_only" | "logs_only"
            
        Returns:
            Final state dict with results and errors
        """
        logger.info("\n" + "="*80)
        logger.info("INDEXING PIPELINE START")
        logger.info("="*80)
        logger.info(f"Mode: {mode}")
        logger.info(f"Sources: {source_paths}")
        
        # Initialize state
        initial_state: IndexingState = {
            "source_paths": source_paths,
            "mode": mode,
            "raw_documents": None,
            "md_chunks": None,
            "log_chunks": None,
            "md_embeddings": None,
            "log_embeddings": None,
            "vector_store_status": None,
            "bm25_status": None,
            "current_step": "initializing",
            "steps_completed": [],
            "errors": [],
            "warnings": [],
            "total_chunks": 0,
            "total_embeddings": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("INDEXING PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Steps completed: {final_state['steps_completed']}")
        logger.info(f"Total chunks: {final_state['total_chunks']}")
        logger.info(f"Total embeddings: {final_state['total_embeddings']}")
        logger.info(f"Errors: {len(final_state['errors'])}")
        logger.info(f"Warnings: {len(final_state['warnings'])}")
        
        if final_state['errors']:
            logger.error("\nErrors encountered:")
            for error in final_state['errors']:
                logger.error(f"  [{error['step']}] {error['type']}: {error['error']}")
        
        if final_state['warnings']:
            logger.warning("\nWarnings:")
            for warning in final_state['warnings']:
                logger.warning(f"  {warning}")
        
        return final_state
