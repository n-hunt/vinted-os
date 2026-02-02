from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path
from datetime import datetime
import ftfy
import markdown
import re
import networkx as nx
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import anytree
from collections import defaultdict
import hashlib


ROOT = Path(__file__).resolve().parents[4]
KB_DIR = str(ROOT) + "/data/demo_knowledge_base"






class MDPreprocessor:
    def __init__(self, kb_dir = KB_DIR): # default is the demo kb 
        self.kb_dir = kb_dir
    
    def chunker(self):
        """
        split markdown documents into semantic chunks using header hierarchy.
        Loads all files from disk (for full mode).
        """
        # load all markdown files from knowledge base directory
        md_loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader)
        loaded_md_docs = md_loader.load()
        
        return self._chunk_documents(loaded_md_docs)
    
    def _chunk_documents(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Split provided documents into semantic chunks using header hierarchy.
        Used for incremental mode when only specific documents need processing.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            Tuple of (chunked documents, document map)
        """
        # configure splitter to chunk on h2/h3/h4 headers
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on= [
                ("##", "category"),
                ("###", "section"),
                ("####", "subsection")
            ],
            strip_headers=False  # keep headers in text for context
        )
        
        # split each document and preserve source metadata
        final_docs = []
        doc_map = {}  # store original doc text for metadata extraction
        
        for doc in documents:
            # extract document-level metadata once per file (not per chunk)
            doc_metadata = self.extract_file_metadata(doc.metadata.get('source', ''))
            
            chunks = md_splitter.split_text(doc.page_content)
            for chunk in chunks: # stamps all the chunks with the source tag
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(doc_metadata)  # apply doc-level metadata to all chunks
                
                # create dual representation for hybrid search
                # page_content: original markdown (for dense embeddings)
                # normalized_text: plain text (for bm25 sparse search)
                chunk.metadata['normalized_text'] = self._normalise(chunk.page_content)
                
                final_docs.append(chunk)
                
            # store original doc text for later metadata extraction
            doc_map[doc.metadata.get('source', '')] = doc.page_content
        
        return final_docs, doc_map

    def _normalise(self, text: str) -> str:
        """
        normalise markdown text using unicode fixes and strip formatting.
        """
        # fix unicode issues (mojibake, encoding errors, etc.)
        normalized_text = ftfy.fix_text(text)
        
        # strip markdown formatting to plain text
        md = markdown.Markdown(extensions=['extra'])
        plain_text = md.convert(normalized_text)
        
        # remove html tags that markdown converter might leave
        plain_text = re.sub(r'<[^>]+>', '', plain_text)
        
        return plain_text.strip()
    
    def extract_chunk_metadata(self, chunk, full_text: str = "", tfidf_vectorizer: Optional[TfidfVectorizer] = None):
        """
        extract rich metadata from markdown chunks for enhanced retrieval.
        """
        metadata = chunk.metadata.copy()
        
        # chunk-level metadata
        metadata['section_hierarchy'] = self._build_hierarchy(chunk.metadata)
        metadata['parent_context'] = chunk.metadata.get('category', '')
        
        # code block handling - delegate to specialized function
        code_metadata = self._handle_code_blocks(chunk)
        metadata.update(code_metadata)
        
        # extract service/module references
        chunk_text = chunk.page_content
        metadata['mentioned_services'] = self._extract_services(chunk_text)
        metadata['mentioned_files'] = self._extract_file_references(chunk_text)
        
        # extract domain-specific entities
        metadata['transaction_ids'] = self._extract_transaction_ids(chunk_text)
        metadata['config_keys'] = self._extract_config_keys(chunk_text)
        
        # structural metadata from full document
        metadata['total_headers'] = len(re.findall(r'^#{2,4}\s', full_text, re.MULTILINE))
        
        # optional: extract distinctive keywords using tf-idf (for scale)
        if tfidf_vectorizer:
            metadata['distinctive_keywords'] = self._extract_tfidf_keywords(chunk, tfidf_vectorizer)
        
        return metadata
    
    def extract_file_metadata(self, source_path: str) -> dict:
        """extract document-level metadata once per file (not per chunk)."""
        doc_metadata = {}
        
        if not source_path:
            return doc_metadata
        
        file_path = Path(source_path)
        if not file_path.exists():
            return doc_metadata
        
        # classification metadata
        doc_metadata['doc_type'] = self._classify_document_type(file_path.name)
        doc_metadata['priority'] = self._assign_priority(file_path.name)
        
        # temporal metadata from filesystem
        stat = file_path.stat()
        doc_metadata['created_at'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        doc_metadata['modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return doc_metadata
    
    def _classify_document_type(self, filename: str) -> str:
        """classify document based on filename."""
        if 'architecture' in filename.lower():
            return 'architecture'
        elif 'spec' in filename.lower() or 'constraint' in filename.lower():
            return 'specifications'
        elif 'troubleshoot' in filename.lower():
            return 'troubleshooting'
        else:
            return 'general'
    
    def _assign_priority(self, filename: str) -> str:
        """assign priority level for constraint queries."""
        if 'spec' in filename.lower() or 'constraint' in filename.lower():
            return 'high'
        elif 'architecture' in filename.lower():
            return 'medium'
        else:
            return 'normal'
    
    def _build_hierarchy(self, chunk_metadata: dict) -> str:
        """build breadcrumb trail from header metadata."""
        parts = []
        for level in ['category', 'section', 'subsection']:
            if level in chunk_metadata and chunk_metadata[level]:
                parts.append(chunk_metadata[level])
        return ' > '.join(parts) if parts else ''
    
    def _extract_services(self, text: str) -> list:
        """extract mentioned service/module names."""
        services = set()
        patterns = [
            r'`(gmail|database|printer|query_service|parser|vision|generator)\.py`',
            r'(GmailService|DatabaseService|PrinterService|QueryService)',
            r'(gmail|database|printer|docuflow)\s+(?:service|module)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            services.update(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
        return list(services)
    
    def _extract_file_references(self, text: str) -> list:
        """extract file path references."""
        # match paths like src/services/gmail.py or config/settings.yaml
        pattern = r'`?([a-z_]+/[a-z_/]+\.(?:py|yaml|yml|json|md))`?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))
    
    def _extract_transaction_ids(self, text: str) -> list:
        """extract vinted transaction ids (10+ digits)."""
        pattern = r'\b(\d{10,})\b'
        matches = re.findall(pattern, text)
        return list(set(matches))[:5]  # limit to first 5 unique ids
    
    def _extract_config_keys(self, text: str) -> list:
        """extract configuration keys from yaml/code examples."""
        keys = set()
        # yaml-style keys
        yaml_pattern = r'^\s*([a-z_]+):\s*'
        keys.update(re.findall(yaml_pattern, text, re.MULTILINE | re.IGNORECASE))
        # python dict keys
        dict_pattern = r'["\']([a-z_]+)["\']\s*:'
        keys.update(re.findall(dict_pattern, text, re.IGNORECASE))
        return list(keys)[:10]  # limit to top 10
    
    def _handle_code_blocks(self, chunk) -> dict:
        """
        special handling for code blocks to enable dual representation and syntax-aware search.
        """
        chunk_text = chunk.page_content
        code_metadata = {}
        
        # extract all code blocks with their languages
        code_blocks = self._extract_code_blocks(chunk_text)
        
        if not code_blocks:
            code_metadata['has_code'] = False
            return code_metadata
        
        # dual representation: raw code + semantic context
        code_metadata['has_code'] = True
        code_metadata['code_languages'] = list(set(cb['language'] for cb in code_blocks))
        code_metadata['code_block_count'] = len(code_blocks)
        
        # extract syntax elements from python code blocks
        python_blocks = [cb for cb in code_blocks if cb['language'] == 'python']
        if python_blocks:
            code_metadata['function_names'] = self._extract_function_names(python_blocks)
            code_metadata['class_names'] = self._extract_class_names(python_blocks)
            code_metadata['imports'] = self._extract_imports(python_blocks)
        
        # extract yaml config blocks
        yaml_blocks = [cb for cb in code_blocks if cb['language'] in ['yaml', 'yml']]
        if yaml_blocks:
            code_metadata['has_config_example'] = True
        
        # create semantic summary for code blocks
        code_metadata['code_purpose'] = self._infer_code_purpose(chunk_text, code_blocks)
        
        return code_metadata
    
    def _extract_code_blocks(self, text: str) -> list:
        """extract all code blocks with language tags."""
        # pattern matches ```language\ncode\n```
        pattern = r'```(\w+)?\n([\s\S]*?)```'
        matches = re.findall(pattern, text)
        
        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                'language': lang.lower() if lang else 'text',
                'code': code.strip()
            })
        return code_blocks
    
    def _extract_function_names(self, python_blocks: list) -> list:
        """extract function definitions from python code."""
        functions = set()
        for block in python_blocks:
            # match def function_name( pattern
            pattern = r'def\s+([a-z_]\w*)\s*\('
            matches = re.findall(pattern, block['code'], re.IGNORECASE)
            functions.update(matches)
        return list(functions)
    
    def _extract_class_names(self, python_blocks: list) -> list:
        """extract class definitions from python code."""
        classes = set()
        for block in python_blocks:
            # match class ClassName: or class ClassName( pattern
            pattern = r'class\s+([A-Z]\w*)[\s:(]'
            matches = re.findall(pattern, block['code'])
            classes.update(matches)
        return list(classes)
    
    def _extract_imports(self, python_blocks: list) -> list:
        """extract import statements from python code."""
        imports = set()
        for block in python_blocks:
            # match import statements
            import_pattern = r'(?:from\s+([a-z_.][\w.]*)|import\s+([a-z_][\w.]*))' 
            matches = re.findall(import_pattern, block['code'], re.IGNORECASE)
            for match in matches:
                # match returns tuple (from_import, direct_import)
                imports.add(match[0] if match[0] else match[1])
        return list(imports)[:10]  # limit to top 10
    
    def _infer_code_purpose(self, full_text: str, code_blocks: list) -> str:
        """create semantic description of what the code demonstrates."""
        # extract text immediately before first code block as context
        if not code_blocks:
            return ""
        
        # find the prose before the first code block
        first_block_start = full_text.find('```')
        if first_block_start > 0:
            context_text = full_text[:first_block_start].strip()
            # get last sentence or line before code
            sentences = context_text.split('\n')
            relevant_context = sentences[-1] if sentences else ""
            return relevant_context[:200]  # limit to 200 chars
        
        return ""
    
    def build_graph(self, chunks: List[Document]) -> Tuple[List[Document], nx.DiGraph]:
        """
        build networkx graph of document relationships via mentions and cross-references.
        creates bidirectional edges and cross-document bridges for enhanced traversal.
        """
        # initialize directed graph
        graph = nx.DiGraph()
        
        # add all chunks as nodes with their metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            chunk.metadata['chunk_id'] = chunk_id
            graph.add_node(chunk_id, **chunk.metadata)
        
        # extract relationships and add edges
        for i, source_chunk in enumerate(chunks):
            source_id = source_chunk.metadata['chunk_id']
            source_text = source_chunk.page_content
            source_services = set(source_chunk.metadata.get('mentioned_services', []))
            source_files = set(source_chunk.metadata.get('mentioned_files', []))
            source_doc_type = source_chunk.metadata.get('doc_type', 'general')
            source_parent = source_chunk.metadata.get('category', '')
            source_transaction_ids = set(source_chunk.metadata.get('transaction_ids', []))
            source_config_keys = set(source_chunk.metadata.get('config_keys', []))
            
            # find target chunks with matching entities
            for j, target_chunk in enumerate(chunks):
                if i == j:  # skip self-references
                    continue
                
                target_id = target_chunk.metadata['chunk_id']
                target_services = set(target_chunk.metadata.get('mentioned_services', []))
                target_files = set(target_chunk.metadata.get('mentioned_files', []))
                target_doc_type = target_chunk.metadata.get('doc_type', 'general')
                target_parent = target_chunk.metadata.get('category', '')
                target_transaction_ids = set(target_chunk.metadata.get('transaction_ids', []))
                target_config_keys = set(target_chunk.metadata.get('config_keys', []))
                
                # relationship type 1: shared service mentions (bidirectional)
                shared_services = source_services & target_services
                if shared_services:
                    # add edge in both directions for symmetric relationship
                    graph.add_edge(source_id, target_id, 
                                 relationship_type='shared_service',
                                 shared_entities=list(shared_services),
                                 weight=len(shared_services))
                    graph.add_edge(target_id, source_id,
                                 relationship_type='shared_service',
                                 shared_entities=list(shared_services),
                                 weight=len(shared_services))
                
                # relationship type 2: file path references (bidirectional)
                shared_files = source_files & target_files
                if shared_files:
                    graph.add_edge(source_id, target_id,
                                 relationship_type='shared_file_reference',
                                 shared_entities=list(shared_files),
                                 weight=len(shared_files))
                    graph.add_edge(target_id, source_id,
                                 relationship_type='shared_file_reference',
                                 shared_entities=list(shared_files),
                                 weight=len(shared_files))
                
                # relationship type 3: explicit mentions (directional - only source→target)
                target_section = target_chunk.metadata.get('section', '')
                if target_section and target_section.lower() in source_text.lower():
                    graph.add_edge(source_id, target_id,
                                 relationship_type='explicit_mention',
                                 mentioned_section=target_section,
                                 weight=2.0)  # higher weight for explicit mentions
                
                # relationship type 4: code-to-docs mapping (code in source, explanation in target)
                source_has_code = source_chunk.metadata.get('has_code', False)
                target_hierarchy = target_chunk.metadata.get('section_hierarchy', '')
                if source_has_code and any(
                    func in target_hierarchy 
                    for func in source_chunk.metadata.get('function_names', [])
                ):
                    graph.add_edge(source_id, target_id,
                                 relationship_type='code_implementation_link',
                                 weight=3.0)  # high weight for code-docs links
                
                # relationship type 5: doc-type bridges (cross-document entity links)
                # connects specs/troubleshooting/architecture via shared domain entities
                if source_doc_type != target_doc_type:
                    # bridge via transaction IDs
                    shared_tx_ids = source_transaction_ids & target_transaction_ids
                    if shared_tx_ids:
                        graph.add_edge(source_id, target_id,
                                     relationship_type='cross_doc_transaction_link',
                                     shared_entities=list(shared_tx_ids),
                                     weight=2.5)  # high weight for bridging doc types
                        graph.add_edge(target_id, source_id,
                                     relationship_type='cross_doc_transaction_link',
                                     shared_entities=list(shared_tx_ids),
                                     weight=2.5)
                    
                    # bridge via config keys
                    shared_config = source_config_keys & target_config_keys
                    if shared_config and len(shared_config) >= 2:  # require 2+ shared keys
                        graph.add_edge(source_id, target_id,
                                     relationship_type='cross_doc_config_link',
                                     shared_entities=list(shared_config),
                                     weight=2.0)
                        graph.add_edge(target_id, source_id,
                                     relationship_type='cross_doc_config_link',
                                     shared_entities=list(shared_config),
                                     weight=2.0)
                
                # relationship type 6: header hierarchy proximity (same parent section)
                # connects sibling chunks under same category (e.g., all "Service Module Map" sections)
                if source_parent and source_parent == target_parent and source_doc_type == target_doc_type:
                    # only connect if they don't already have a stronger relationship
                    if not graph.has_edge(source_id, target_id):
                        graph.add_edge(source_id, target_id,
                                     relationship_type='sibling_section',
                                     parent_section=source_parent,
                                     weight=1.0)
                        graph.add_edge(target_id, source_id,
                                     relationship_type='sibling_section',
                                     parent_section=source_parent,
                                     weight=1.0)
        
        # serialize graph relationships back to chunk metadata
        for chunk in chunks:
            chunk_id = chunk.metadata['chunk_id']
            
            # outgoing edges (this chunk references others)
            outgoing = list(graph.successors(chunk_id))
            chunk.metadata['related_chunks'] = outgoing
            
            # relationship types for outgoing edges
            relationship_types = [
                graph[chunk_id][target]['relationship_type'] 
                for target in outgoing
            ]
            chunk.metadata['relationship_types'] = relationship_types
            
            # incoming edges (other chunks reference this one)
            incoming = list(graph.predecessors(chunk_id))
            chunk.metadata['referenced_by'] = incoming
            
            # graph centrality metrics
            chunk.metadata['out_degree'] = graph.out_degree(chunk_id)
            chunk.metadata['in_degree'] = graph.in_degree(chunk_id)
        
        return chunks, graph
    
    def build_corpus_vocabulary(self, chunks: List[Document]) -> TfidfVectorizer:
        """
        build tf-idf model across entire corpus for distinctive keyword extraction.
        scales to hundreds of documents for automatic term discovery.
        """
        # use normalized text for statistical analysis
        corpus_texts = [c.metadata.get('normalized_text', c.page_content) for c in chunks]
        
        # configure tf-idf to capture technical terms and phrases
        vectorizer = TfidfVectorizer(
            max_features=500,              # top 500 distinctive terms
            min_df=2,                      # term must appear in at least 2 chunks
            max_df=0.8,                    # ignore terms in >80% of corpus (too common)
            ngram_range=(1, 3),            # capture phrases like "database locked error"
            token_pattern=r'\b[a-z_]{2,}\b',  # match technical identifiers
            stop_words='english',          # remove common words
            lowercase=True
        )
        
        vectorizer.fit(corpus_texts)
        return vectorizer
    
    def _extract_tfidf_keywords(self, chunk: Document, vectorizer: TfidfVectorizer, top_n: int = 10) -> List[Dict[str, float]]:
        """
        extract top-n distinctive keywords from chunk using tf-idf scores.
        """
        # transform chunk text to tf-idf vector
        normalized_text = chunk.metadata.get('normalized_text', chunk.page_content)
        tfidf_vector = vectorizer.transform([normalized_text])
        
        # get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_vector.toarray()[0]
        
        # get top-n terms by score
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        keywords = [
            {'term': feature_names[idx], 'score': float(scores[idx])}
            for idx in top_indices
            if scores[idx] > 0  # only include non-zero scores
        ]
        
        return keywords
    
    def run(self, documents: Optional[List[Document]] = None, use_tfidf: bool = False, use_graph: bool = False):
        """
        orchestrate the full preprocessing pipeline for markdown documents.
        
        args:
            documents: optional list of Document objects to process. If None, loads all files from disk.
            use_tfidf: enable tf-idf keyword extraction (recommended for 50+ documents)
            use_graph: enable knowledge graph construction (optional enhancement for large KBs)
        
        returns:
            enriched_chunks: list of chunks with extracted metadata
            graph: networkx DiGraph (None if use_graph=False)
            tfidf_vectorizer: fitted TfidfVectorizer (None if use_tfidf=False)
        """
        # step 1 & 2: load and chunk documents with doc-level metadata
        if documents is None:
            # Full mode: load all files from disk
            chunks, doc_map = self.chunker()
        else:
            # Incremental mode: chunk provided documents
            chunks, doc_map = self._chunk_documents(documents)
        
        # step 3: build corpus-level tf-idf vocabulary (optional, for scale)
        tfidf_vectorizer = None
        if use_tfidf:
            tfidf_vectorizer = self.build_corpus_vocabulary(chunks)
        
        # step 4: extract chunk-level metadata for each chunk
        enriched_chunks = []
        for chunk in chunks:
            source = chunk.metadata.get('source', '')
            full_text = doc_map.get(source, '')
            
            # extract rich chunk-level metadata (with optional tf-idf)
            enriched_metadata = self.extract_chunk_metadata(chunk, full_text, tfidf_vectorizer)
            chunk.metadata = enriched_metadata
            
            enriched_chunks.append(chunk)
        
        # step 5: build cross-reference graph (optional enhancement)
        graph = None
        if use_graph:
            enriched_chunks, graph = self.build_graph(enriched_chunks)
        
        # return vectorizer for query-time keyword expansion
        return enriched_chunks, graph, tfidf_vectorizer

# ============================================================
# LOG FILE PREPROCESSING
# ============================================================

class LogPreprocessor:
    """
    Process operational log files into searchable, structured chunks.
    Handles large files, extracts metadata, builds temporal relationships.
    """
    
    def __init__(self, kb_dir: str = KB_DIR):
        """
        Initialize log preprocessor.
        
        Args:
            kb_dir: Directory containing log files to process (same as knowledge base)
        """
        self.kb_dir = kb_dir
        self.chunk_size = 1000  # characters per chunk (configurable)
        self.chunk_overlap = 200  # overlap for context preservation
        
        # State machine transitions for timeline inference
        self.STATE_TRANSITIONS = {
            'initiated': ['processing', 'error', 'failed'],  # Add immediate failure path
            'processing': ['validating', 'completed', 'error', 'processing'],  # Add re-processing
            'validating': ['storing', 'error', 'failed'],  # Add validation failure
            'storing': ['completed', 'error', 'failed'],  # Add storage failure
            'error': ['retrying', 'failed', 'cleanup'],
            'retrying': ['processing', 'failed', 'error', 'retrying'],  # Add retry loops
            'cleanup': ['failed', 'completed'],
            'failed': [],  # terminal state
            'completed': []  # terminal state
        }
    
    def load_logs(self, file_pattern: str = "*.log") -> List[Dict[str, any]]:
        """
        Stream large log files from directory.
        
        Args:
            file_pattern: Glob pattern for log files (e.g., "*.log", "print_debug/*.log")
        
        Returns:
            List of raw log entries with file metadata
        """
        log_dir = Path(self.kb_dir)
        all_entries = []
        
        # find all matching log files using glob pattern
        matching_files = list(log_dir.glob(file_pattern))
        
        if not matching_files:
            print(f"Warning: No log files found matching pattern '{file_pattern}' in {log_dir}")
            return all_entries
        
        print(f"Found {len(matching_files)} log file(s) to process")
        
        # stream each log file line-by-line (memory-efficient for large files)
        for log_file in matching_files:
            print(f"  Streaming: {log_file.name}")
            
            try:
                # open file and iterate line-by-line (automatic streaming)
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, raw_line in enumerate(f, start=1):
                        # skip empty lines
                        if not raw_line.strip():
                            continue
                        
                        # parse the log line into structured data
                        parsed_entry = self.parse_log_entry(raw_line.rstrip('\n'))
                        
                        # merge with file metadata
                        parsed_entry['source_file'] = str(log_file)
                        parsed_entry['source_filename'] = log_file.name
                        parsed_entry['line_number'] = line_num
                        parsed_entry['raw_line'] = raw_line.rstrip('\n')  # keep original for reference
                        
                        all_entries.append(parsed_entry)
            
            except Exception as e:
                print(f"  Error reading {log_file.name}: {e}")
                continue
        
        print(f"Loaded and parsed {len(all_entries)} log entries from {len(matching_files)} file(s)")
        return all_entries
    
    def parse_log_entry(self, raw_line: str) -> Dict[str, any]:
        """
        Extract structured data from a single log line using regex.
        
        Expected format:
            2026-02-01 14:23:45 | INFO | services.printer | TID:12345 | Label printed successfully
        
        Extracts:
            - timestamp: datetime object
            - level: INFO/WARNING/ERROR/DEBUG
            - module: services.printer
            - transaction_id: 12345 (if present)
            - message: raw message text
            - error_type: extracted from error messages
        
        Returns:
            Parsed log entry as dictionary
        """
        parsed = {
            'timestamp': None,
            'level': None,
            'module': None,
            'transaction_id': None,
            'message': raw_line,  # fallback to raw line
            'error_type': None
        }
        
        # regex pattern for structured logs: timestamp | level | module | [TID:xxx |] message
        # example: 2026-02-01 14:23:45 | INFO | services.printer | TID:12345 | Label printed
        pattern = r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*([^\|]+?)\s*\|(?:\s*TID:(\d+)\s*\|)?\s*(.+)$'
        
        match = re.match(pattern, raw_line.strip())
        
        if match:
            timestamp_str, level, module, tid, message = match.groups()
            
            # parse timestamp
            try:
                parsed['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # fallback: store as string if parsing fails
                parsed['timestamp'] = timestamp_str
            
            # extract level (uppercase)
            parsed['level'] = level.upper()
            
            # extract module (strip whitespace)
            parsed['module'] = module.strip()
            
            # extract transaction ID (if present)
            if tid:
                parsed['transaction_id'] = tid
            
            # extract message
            parsed['message'] = message.strip()
            
            # extract error type from error messages
            if parsed['level'] in ['ERROR', 'CRITICAL']:
                error_type = self._extract_error_type(message)
                if error_type:
                    parsed['error_type'] = error_type
        
        else:
            # fallback: try simpler patterns or unstructured logs
            # pattern 2: just timestamp and message
            simple_pattern = r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.+)$'
            simple_match = re.match(simple_pattern, raw_line.strip())
            
            if simple_match:
                timestamp_str, message = simple_match.groups()
                try:
                    parsed['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    parsed['timestamp'] = timestamp_str
                parsed['message'] = message.strip()
            
            # try to extract transaction ID from anywhere in the line
            tid_pattern = r'TID:(\d+)'
            tid_match = re.search(tid_pattern, raw_line)
            if tid_match:
                parsed['transaction_id'] = tid_match.group(1)
        
        return parsed
    
    def _extract_error_type(self, message: str) -> Optional[str]:
        """
        Extract error type/exception name from error message.
        
        Examples:
            "FileNotFoundError: file not found" -> "FileNotFoundError"
            "Database connection failed: timeout" -> "ConnectionError"
            "Permission denied" -> "PermissionError"
        
        Returns:
            Error type string or None
        """
        # pattern 1: explicit exception name (e.g., "ValueError: invalid input")
        exception_pattern = r'^([A-Z]\w*Error|[A-Z]\w*Exception):'
        match = re.match(exception_pattern, message)
        if match:
            return match.group(1)
        
        # pattern 2: common error keywords
        error_keywords = {
            'permission denied': 'PermissionError',
            'file not found': 'FileNotFoundError',
            'connection refused': 'ConnectionRefusedError',
            'timeout': 'TimeoutError',
            'database locked': 'DatabaseLockError',
            'failed to connect': 'ConnectionError',
            'invalid': 'ValueError',
            'not found': 'NotFoundError'
        }
        
        message_lower = message.lower()
        for keyword, error_type in error_keywords.items():
            if keyword in message_lower:
                return error_type
        
        return None
    
    def chunk_logs(self, parsed_entries: List[Dict]) -> List[Document]:
        """
        Group log entries into semantic chunks.
        
        Chunking strategies:
            1. By transaction ID: Group all logs for same TID together
            2. By time window: Group logs within N-second intervals
            3. By size: Use RecursiveCharacterTextSplitter for normalization
        
        Returns:
            List of Document objects with chunk metadata
        """
        if not parsed_entries:
            return []
        
        all_chunks = []
        
        # strategy 1: group by transaction ID
        tid_groups = self._group_by_transaction(parsed_entries)
        
        # strategy 2: identify entries without TIDs (O(n) set-based filtering)
        entries_without_tid = [
            entry for entry in parsed_entries
            if not entry.get('transaction_id')
        ]
        
        # strategy 3: group remaining entries by time window
        time_groups = self._group_by_time_window(entries_without_tid, window_seconds=60)
        
        # combine all groups (TID groups + time groups)
        all_groups = list(tid_groups.values()) + time_groups
        
        # convert each group to a Document
        for group in all_groups:
            if not group:
                continue
            
            # reconstruct log text from entries (structured format)
            log_lines = []
            for entry in group:
                # format: timestamp | level | module | TID:xxx | message
                ts = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry['timestamp'], datetime) else str(entry['timestamp'])
                level = entry.get('level', 'INFO')
                module = entry.get('module', 'unknown')
                tid = entry.get('transaction_id', '')
                message = entry.get('message', '')
                
                if tid:
                    log_line = f"{ts} | {level} | {module} | TID:{tid} | {message}"
                else:
                    log_line = f"{ts} | {level} | {module} | {message}"
                
                log_lines.append(log_line)
            
            chunk_text = '\n'.join(log_lines)
            
            # strategy 3: apply size normalization if chunk too large
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=['\n', ' ', ''],  # split on newlines first, then spaces
                length_function=len
            )
            
            # split if needed
            if len(chunk_text) > self.chunk_size:
                split_texts = text_splitter.split_text(chunk_text)
            else:
                split_texts = [chunk_text]
            
            # create Document for each split
            for split_text in split_texts:
                # recalculate metadata from the actual split content (not the whole group)
                split_lines = split_text.strip().split('\n')
                
                # parse metadata from split_text itself
                split_transaction_ids = set()
                split_log_levels = set()
                split_modules = set()
                split_error_types = set()
                split_timestamps = []
                has_errors = False
                
                for line in split_lines:
                    # extract TID from line
                    tid_match = re.search(r'TID:(\d+)', line)
                    if tid_match:
                        split_transaction_ids.add(tid_match.group(1))
                    
                    # extract level from structured format: | LEVEL |
                    level_match = re.search(r'\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|', line)
                    if level_match:
                        level = level_match.group(1)
                        split_log_levels.add(level)
                        if level in ['ERROR', 'CRITICAL']:
                            has_errors = True
                    
                    # extract module from structured format: | module |
                    # pattern: timestamp | level | module | ...
                    parts = line.split('|')
                    if len(parts) >= 3:
                        module = parts[2].strip()
                        if module and module != 'unknown':
                            split_modules.add(module)
                    
                    # extract timestamp from line start
                    ts_match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                    if ts_match:
                        try:
                            ts = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                            split_timestamps.append(ts)
                        except ValueError:
                            pass
                
                # find matching entries from group to get error_types
                for entry in group:
                    if entry.get('raw_line') in split_text or entry.get('message') in split_text:
                        if entry.get('error_type'):
                            split_error_types.add(entry.get('error_type'))
                
                # get source file from first entry
                first_entry = group[0]
                
                # build metadata from split content
                metadata = {
                    'source_file': first_entry.get('source_file', ''),
                    'source_filename': first_entry.get('source_filename', ''),
                    'entry_count': len(split_lines),
                    'start_timestamp': min(split_timestamps) if split_timestamps else None,
                    'end_timestamp': max(split_timestamps) if split_timestamps else None,
                    'transaction_ids': list(split_transaction_ids),
                    'log_levels': list(split_log_levels),
                    'modules': list(split_modules),
                    'has_errors': has_errors,
                    'error_types': list(split_error_types)
                }
                
                # create Document with structured text
                doc = Document(
                    page_content=split_text,
                    metadata=metadata
                )
                
                # assign stable chunk_id before any processing
                doc.metadata['chunk_id'] = f"log_chunk_{len(all_chunks)}"
                
                # apply dual representation (normalized_text for BM25)
                doc = self.normalize_log_chunk(doc)
                
                # extract rich metadata (stack traces, metrics, error signatures)
                enriched_metadata = self.extract_log_metadata(doc)
                doc.metadata.update(enriched_metadata)
                
                all_chunks.append(doc)
        
        return all_chunks
    
    def _group_by_transaction(self, entries: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group log entries by transaction ID.
        
        Returns:
            Dictionary mapping transaction_id -> list of log entries
        """
        groups = defaultdict(list)
        
        for entry in entries:
            tid = entry.get('transaction_id')
            if tid:
                groups[tid].append(entry)
        
        return dict(groups)
    
    def _group_by_time_window(self, entries: List[Dict], window_seconds: int = 60) -> List[List[Dict]]:
        """
        Group log entries into time-based windows.
        
        Args:
            window_seconds: Size of time window in seconds
        
        Returns:
            List of time-windowed entry groups
        """
        if not entries:
            return []
        
        # sort entries by timestamp (handle None timestamps)
        sorted_entries = sorted(
            entries,
            key=lambda e: e['timestamp'] if isinstance(e['timestamp'], datetime) else datetime.min
        )
        
        groups = []
        current_group = [sorted_entries[0]]
        
        for entry in sorted_entries[1:]:
            current_ts = entry['timestamp']
            prev_ts = current_group[-1]['timestamp']
            
            # skip entries with invalid timestamps
            if not isinstance(current_ts, datetime) or not isinstance(prev_ts, datetime):
                current_group.append(entry)
                continue
            
            # calculate time difference
            time_diff = (current_ts - prev_ts).total_seconds()
            
            # start new group if time gap exceeds window
            if time_diff > window_seconds:
                groups.append(current_group)
                current_group = [entry]
            else:
                current_group.append(entry)
        
        # add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def normalize_log_chunk(self, chunk: Document) -> Document:
        """
        Create dual representation for hybrid search.
        
        - page_content: Structured log format (for dense embeddings)
        - normalized_text: Plain message text (for BM25 sparse search)
        
        Returns:
            Document with normalized_text metadata field
        """
        # extract just the messages from structured log lines
        log_lines = chunk.page_content.split('\n')
        messages = []
        
        for line in log_lines:
            # split on pipe and take the last part (message)
            parts = line.split('|')
            if parts:
                message = parts[-1].strip()
                messages.append(message)
        
        # join messages as plain text for BM25 search
        normalized_text = ' '.join(messages)
        
        # add to metadata
        chunk.metadata['normalized_text'] = normalized_text
        
        return chunk
    
    def extract_log_metadata(self, chunk: Document) -> Dict[str, any]:
        """
        Extract rich metadata from log chunks.
        
        Extracts:
            - transaction_ids: List of unique TIDs in chunk
            - error_signatures: Unique error patterns
            - module_names: Services/modules mentioned
            - log_levels: Distribution of levels (INFO:10, ERROR:2, etc.)
            - time_range: Start and end timestamps
            - stack_traces: Extracted stack trace snippets
            - metrics: Numerical values (latency, counts, etc.)
        
        Returns:
            Metadata dictionary
        """
        metadata = {}
        chunk_text = chunk.page_content
        
        # log level distribution (position-aware to avoid false positives)
        level_counts = defaultdict(int)
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            # Match level only in column 2: timestamp | LEVEL | module | ...
            # Pattern: line start -> timestamp -> | -> LEVEL -> |
            pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*' + level + r'\s*\|'
            matches = re.findall(pattern, chunk_text, re.MULTILINE)
            count = len(matches)
            if count > 0:
                level_counts[level] = count
        metadata['log_level_distribution'] = dict(level_counts)
        
        # extract error signatures from error messages
        error_signatures = []
        error_lines = [line for line in chunk_text.split('\n') if '| ERROR |' in line or '| CRITICAL |' in line]
        for line in error_lines:
            parts = line.split('|')
            if parts:
                message = parts[-1].strip()
                signature = self._extract_error_signature(message)
                if signature:
                    error_signatures.append(signature)
        metadata['error_signatures'] = list(set(error_signatures))  # unique signatures
        
        # extract stack traces
        stack_traces = self._extract_stack_trace(chunk_text)
        if stack_traces:
            metadata['has_stack_trace'] = True
            metadata['stack_trace_frames'] = stack_traces
        else:
            metadata['has_stack_trace'] = False
        
        # extract metrics from messages
        all_metrics = {}
        for line in chunk_text.split('\n'):
            parts = line.split('|')
            if parts:
                message = parts[-1].strip()
                metrics = self._extract_metrics(message)
                all_metrics.update(metrics)
        metadata['extracted_metrics'] = all_metrics
        
        # calculate time range duration
        if chunk.metadata.get('start_timestamp') and chunk.metadata.get('end_timestamp'):
            start = chunk.metadata['start_timestamp']
            end = chunk.metadata['end_timestamp']
            if isinstance(start, datetime) and isinstance(end, datetime):
                duration = (end - start).total_seconds()
                metadata['duration_seconds'] = duration
        
        return metadata
    
    def _extract_error_signature(self, message: str) -> Optional[str]:
        """
        Extract canonical error pattern from error message.
        
        Example:
            "FileNotFoundError: /path/to/file.txt not found" 
            -> "FileNotFoundError: file not found"
        
        Returns:
            Normalized error signature
        """
        # pattern 1: explicit exception name
        exception_pattern = r'^([A-Z]\w*(?:Error|Exception)):\s*(.+)$'
        match = re.match(exception_pattern, message)
        
        if match:
            exception_type, error_msg = match.groups()
            
            # normalize the message: remove specific paths/IDs/numbers
            normalized_msg = re.sub(r'/[^\s]+', '<path>', error_msg)  # replace paths
            normalized_msg = re.sub(r'\b\d{10,}\b', '<transaction_id>', normalized_msg)  # replace long numbers
            normalized_msg = re.sub(r'\b\d+\.\d+\b', '<number>', normalized_msg)  # replace decimals
            normalized_msg = re.sub(r'\b\d+\b', '<number>', normalized_msg)  # replace integers
            
            return f"{exception_type}: {normalized_msg[:100]}"  # limit to 100 chars
        
        # pattern 2: common error patterns without exception name
        error_patterns = [
            (r'failed to connect to (.+)', 'connection failed'),
            (r'database locked', 'database locked'),
            (r'permission denied', 'permission denied'),
            (r'timeout', 'timeout'),
            (r'file not found', 'file not found'),
        ]
        
        message_lower = message.lower()
        for pattern, signature in error_patterns:
            if re.search(pattern, message_lower):
                return signature
        
        return None
    
    def _extract_stack_trace(self, chunk_text: str) -> List[str]:
        """
        Extract stack trace snippets from log chunk.
        
        Returns:
            List of stack trace frames
        """
        stack_traces = []
        
        # pattern: python stack traces (File "...", line N, in function_name)
        trace_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        matches = re.findall(trace_pattern, chunk_text)
        
        for file_path, line_num, function_name in matches:
            # extract just the filename (not full path)
            filename = file_path.split('/')[-1] if '/' in file_path else file_path
            stack_traces.append(f"{filename}:{line_num} in {function_name}")
        
        # limit to first 10 frames to avoid metadata bloat
        return stack_traces[:10]
    
    def _extract_metrics(self, message: str) -> Dict[str, float]:
        """
        Extract numerical metrics from log messages.
        
        Examples:
            "Processing completed in 1.23s" -> {'duration_seconds': 1.23}
            "Processed 45 items" -> {'items_count': 45}
        
        Returns:
            Dictionary of metric_name -> value
        """
        metrics = {}
        
        # pattern 1: duration in seconds (e.g., "took 1.23s", "in 0.5s")
        duration_pattern = r'(?:took|in|completed in)\s+([\d.]+)\s*s(?:ec|econds)?'
        duration_match = re.search(duration_pattern, message, re.IGNORECASE)
        if duration_match:
            try:
                metrics['duration_seconds'] = float(duration_match.group(1))
            except ValueError:
                pass
        
        # pattern 2: item counts (e.g., "processed 45 items", "found 3 files")
        count_pattern = r'(?:processed|found|created|deleted)\s+(\d+)\s+(?:items|files|records|rows)'
        count_match = re.search(count_pattern, message, re.IGNORECASE)
        if count_match:
            try:
                metrics['item_count'] = int(count_match.group(1))
            except ValueError:
                pass
        
        # pattern 3: file sizes (e.g., "1.5MB", "2048KB")
        size_pattern = r'([\d.]+)\s*(MB|KB|GB)'
        size_match = re.search(size_pattern, message, re.IGNORECASE)
        if size_match:
            try:
                size_value = float(size_match.group(1))
                unit = size_match.group(2).upper()
                # convert to MB
                if unit == 'KB':
                    size_value /= 1024
                elif unit == 'GB':
                    size_value *= 1024
                metrics['file_size_mb'] = size_value
            except ValueError:
                pass
        
        # pattern 4: retry counts (e.g., "retry 3/5", "attempt 2")
        retry_pattern = r'(?:retry|attempt)\s+(\d+)'
        retry_match = re.search(retry_pattern, message, re.IGNORECASE)
        if retry_match:
            try:
                metrics['retry_count'] = int(retry_match.group(1))
            except ValueError:
                pass
        
        return metrics
    
    def _chunk_to_entry(self, chunk: Document) -> Dict:
        """
        Convert a chunk back to entry-like dict for relationship inference.
        
        Args:
            chunk: Document chunk
        
        Returns:
            Entry dict with timestamp, level, message, transaction_id
        """
        # Extract first log line from chunk
        first_line = chunk.page_content.split('\n')[0] if chunk.page_content else ''
        
        # Parse the first line (structured format: timestamp | level | module | message)
        parsed = self.parse_log_entry(first_line)
        
        return {
            'timestamp': chunk.metadata.get('start_timestamp'),
            'level': chunk.metadata.get('log_levels', ['INFO'])[0] if chunk.metadata.get('log_levels') else 'INFO',
            'message': parsed.get('message', first_line),
            'transaction_id': chunk.metadata.get('transaction_ids', [None])[0] if chunk.metadata.get('transaction_ids') else None
        }
    
    def build_timeline(self, chunks: List[Document]) -> Tuple[List[Document], Optional[anytree.Node]]:
        """
        Build temporal/causal tree of log events using anytree.
        
        Creates one timeline tree per transaction ID. Each tree represents
        the causal flow of events within that transaction.
        
        Relationships:
            - Parent-child: Transaction start -> steps -> completion
            - Temporal ordering: Chronological event sequences
            - Causal links: Error -> retry -> success/failure
        
        Note:
            If multiple transactions exist, only the FIRST timeline root is returned.
            All chunks are still enriched with timeline metadata regardless of which
            root is returned. To access all timeline roots, modify this method to
            return the full timeline_roots list.
        
        Returns:
            Tuple of (enriched chunks, first timeline root or None)
                - enriched chunks: All chunks with timeline metadata added
                - timeline root: Root node of first transaction timeline, or None if no timelines built
        """
        from anytree import Node
        
        # Group chunks by transaction ID
        tid_groups = defaultdict(list)
        chunks_without_tid = []
        assigned_chunks = set()  # Track which chunks are already assigned to prevent duplicates
        
        for chunk in chunks:
            tids = chunk.metadata.get('transaction_ids', [])
            if tids:
                # Chunk may have multiple TIDs - only assign to first to prevent metadata overwrites
                chunk_id = chunk.metadata.get('chunk_id')
                if chunk_id not in assigned_chunks:
                    tid_groups[tids[0]].append(chunk)
                    assigned_chunks.add(chunk_id)
            else:
                chunks_without_tid.append(chunk)
        
        # Build tree for each transaction
        timeline_roots = []
        
        for tid, tid_chunks in tid_groups.items():
            # Sort by timestamp
            sorted_chunks = sorted(
                tid_chunks,
                key=lambda c: c.metadata.get('start_timestamp', datetime.min)
            )
            
            if not sorted_chunks:
                continue
            
            # Create root node (first event in transaction)
            first_chunk = sorted_chunks[0]
            first_line = first_chunk.page_content.split('\n')[0] if first_chunk.page_content else f"TID:{tid}"
            
            root = Node(
                name=first_line[:80],  # Truncate for readability
                chunk_id=first_chunk.metadata.get('chunk_id', f'chunk_{id(first_chunk)}'),
                timestamp=first_chunk.metadata.get('start_timestamp'),
                tid=tid
            )
            
            # Build tree incrementally
            current_parent = root
            previous_chunks = [first_chunk]
            
            for chunk in sorted_chunks[1:]:
                # Convert chunk to entry for relationship inference
                curr_entry = self._chunk_to_entry(chunk)
                
                # Try to find causal parent by checking recent chunks
                parent_node = current_parent
                relationship_type = None
                
                # Check last 3 chunks for causal relationship (sliding window)
                for prev_chunk in reversed(previous_chunks[-3:]):
                    prev_entry = self._chunk_to_entry(prev_chunk)
                    relationship = self._infer_causal_relationship(prev_entry, curr_entry)
                    
                    if relationship:
                        # Find the node for this chunk
                        for node in [root] + list(root.descendants):
                            if node.chunk_id == prev_chunk.metadata.get('chunk_id', f'chunk_{id(prev_chunk)}'):
                                parent_node = node
                                relationship_type = relationship
                                break
                        break
                
                # Create node for current chunk
                chunk_line = chunk.page_content.split('\n')[0] if chunk.page_content else ''
                node = Node(
                    name=chunk_line[:80],
                    parent=parent_node,
                    chunk_id=chunk.metadata.get('chunk_id', f'chunk_{id(chunk)}'),
                    relationship_type=relationship_type,
                    timestamp=chunk.metadata.get('start_timestamp'),
                    tid=tid
                )
                
                # Update parent pointer based on relationship type
                if relationship_type in ['triggered_next_step', 'sequential_step', 'initiated_to_processing', 
                                         'processing_to_validating', 'validating_to_storing', 'storing_to_completed']:
                    # Linear chain - next event becomes parent
                    current_parent = node
                elif relationship_type in ['caused_retry', 'triggered_recovery', 'error_to_retrying', 
                                           'retrying_to_retrying', 'error_to_cleanup']:
                    # Branch - keep same parent (multiple children)
                    pass
                else:
                    # Unknown relationship - attach to current parent
                    pass
                
                # Add to sliding window
                previous_chunks.append(chunk)
            
            timeline_roots.append(root)
        
        # Enrich chunks with timeline metadata
        for chunk in chunks:
            chunk_id = chunk.metadata.get('chunk_id', f'chunk_{id(chunk)}')
            
            # Find this chunk in timeline trees
            for root in timeline_roots:
                for node in [root] + list(root.descendants):
                    if node.chunk_id == chunk_id:
                        # Add timeline metadata
                        chunk.metadata['timeline_position'] = self._get_position_type(node)
                        chunk.metadata['causal_parent'] = node.parent.chunk_id if node.parent else None
                        chunk.metadata['causal_children'] = [c.chunk_id for c in node.children]
                        chunk.metadata['tree_depth'] = node.depth
                        chunk.metadata['relationship_to_parent'] = getattr(node, 'relationship_type', None)
                        chunk.metadata['in_timeline'] = True
                        break
        
        # Mark chunks not in timeline
        for chunk in chunks_without_tid:
            chunk.metadata['in_timeline'] = False
        
        # Return first root (or None if no timelines built)
        return chunks, timeline_roots[0] if timeline_roots else None
    
    def _get_position_type(self, node: anytree.Node) -> str:
        """
        Determine position type of node in timeline.
        
        Args:
            node: anytree Node
        
        Returns:
            'root' | 'intermediate' | 'terminal'
        """
        if node.parent is None:
            return 'root'
        elif len(node.children) == 0:
            return 'terminal'
        else:
            return 'intermediate'
    
    def _infer_state(self, message: str) -> Optional[str]:
        """
        Extract implicit state from log message.
        
        Args:
            message: Log message text
        
        Returns:
            State string or None
        """
        # Only examine first line for state keywords (avoid false matches in stack traces)
        first_line = message.split('\n')[0].lower()
        
        # State detection rules (order matters - more specific first)
        
        # Check for compound states first (e.g., "completed with errors")
        if 'completed' in first_line or 'finished' in first_line or 'success' in first_line:
            # Check for error markers even in completion messages
            if any(err in first_line for err in ['error', 'failed', 'exception', 'with errors']):
                return 'error'  # Completed with errors = error state
            return 'completed'
        
        # Terminal states
        if any(w in first_line for w in ['failed', 'failure', 'aborted', 'terminated']):
            return 'failed'
        
        # Recovery states (check before "error" to avoid ambiguity)
        if any(w in first_line for w in ['retrying', 'retry', 'attempting again', 'reconnecting']):
            return 'retrying'
        
        if any(w in first_line for w in ['cleanup', 'cleaning up', 'rolling back', 'reverting']):
            return 'cleanup'
        
        # Error state
        if any(w in first_line for w in ['error', 'exception', 'critical', 'fatal']):
            return 'error'
        
        # Initial state (check BEFORE processing to prioritize "started processing")
        if any(w in first_line for w in ['started', 'starting', 'beginning', 'initiating', 'initiated']):
            return 'initiated'
        
        # Processing states (after checking for "started" keywords)
        if any(w in first_line for w in ['validating', 'checking', 'verifying', 'validated']):
            return 'validating'
        
        if any(w in first_line for w in ['saving', 'storing', 'updating', 'writing', 'persisting']):
            return 'storing'
        
        if any(w in first_line for w in ['processing', 'executing', 'running', 'handling']):
            return 'processing'
        
        return None
    
    def _are_temporally_proximate(self, entry1: Dict, entry2: Dict, max_seconds: int = 10) -> bool:
        """
        Check if two log entries are temporally close.
        
        Args:
            entry1: First log entry
            entry2: Second log entry
            max_seconds: Maximum time gap to consider proximate
        
        Returns:
            True if entries are within max_seconds of each other
        """
        ts1 = entry1.get('timestamp')
        ts2 = entry2.get('timestamp')
        
        # Both must have valid timestamps
        if not isinstance(ts1, datetime) or not isinstance(ts2, datetime):
            return False
        
        # Calculate time difference
        time_diff = abs((ts2 - ts1).total_seconds())
        
        return time_diff <= max_seconds
    
    def _infer_causal_relationship(self, entry1: Dict, entry2: Dict) -> Optional[str]:
        """
        Detect causal relationship between two log entries.
        
        Uses hybrid strategy:
            Strategy 1: Temporal proximity + keyword signals
            Strategy 2: Transaction-scoped state machine
        
        Patterns:
            - "Database locked" -> "Retrying connection" (caused_retry)
            - "PDF downloaded" -> "Parsing started" (triggered_next_step)
            - "Error occurred" -> "Cleanup initiated" (triggered_recovery)
        
        Returns:
            Relationship type string or None
        """
        # Prerequisite: Events must be temporally close (within 10 seconds)
        if not self._are_temporally_proximate(entry1, entry2, max_seconds=10):
            return None
        
        # Strategy 2: State machine (high confidence)
        state1 = self._infer_state(entry1.get('message', ''))
        state2 = self._infer_state(entry2.get('message', ''))
        
        if state1 and state2:
            # Check if transition is valid in state machine
            valid_transitions = self.STATE_TRANSITIONS.get(state1, [])
            if state2 in valid_transitions:
                return f'{state1}_to_{state2}'
        
        # Strategy 1: Keyword-based patterns (fallback)
        # Extract semantic signals from messages
        msg1 = entry1.get('message', '').lower()
        msg2 = entry2.get('message', '').lower()
        level1 = entry1.get('level')
        level2 = entry2.get('level')
        
        # Pattern 1: Error → Retry (verb-based detection)
        if level1 in ['ERROR', 'CRITICAL']:
            retry_verbs = ['retrying', 'retry', 'attempting', 'reconnecting', 'attempting again']
            if any(verb in msg2 for verb in retry_verbs):
                return 'caused_retry'
        
        # Pattern 2: Sequential workflow steps (verb transitions)
        workflow_transitions = [
            (['started', 'beginning', 'initiating'], ['processing', 'executing', 'running']),
            (['downloading', 'fetching', 'retrieving'], ['parsing', 'extracting', 'analyzing']),
            (['parsing', 'extracted', 'analyzed'], ['validating', 'checking', 'verifying']),
            (['validated', 'checked', 'verified'], ['saving', 'storing', 'updating', 'writing']),
            (['saved', 'stored', 'updated'], ['completed', 'finished', 'success', 'done'])
        ]
        
        for start_verbs, next_verbs in workflow_transitions:
            if any(v in msg1 for v in start_verbs) and any(v in msg2 for v in next_verbs):
                return 'triggered_next_step'
        
        # Pattern 3: Same transaction ID = likely sequential step
        tid1 = entry1.get('transaction_id')
        tid2 = entry2.get('transaction_id')
        if tid1 and tid2 and tid1 == tid2:
            if level1 == 'INFO' and level2 == 'INFO':
                return 'sequential_step'
        
        # Pattern 4: Error → Cleanup/Recovery
        if level1 in ['ERROR', 'CRITICAL']:
            recovery_keywords = ['cleanup', 'cleaning up', 'rolling back', 'reverting', 'aborting', 'canceling']
            if any(kw in msg2 for kw in recovery_keywords):
                return 'triggered_recovery'
        
        return None
    
    def deduplicate_logs(self, chunks: List[Document]) -> List[Document]:
        """
        Handle repetitive log patterns.
        
        Strategy:
            1. Hash exact duplicate messages
            2. Count occurrences
            3. Sample: Keep first + last occurrence, store count
            4. Store signature for common errors
        
        Updates chunk metadata with:
            - is_duplicate: bool
            - occurrence_count: int
            - first_seen: timestamp
            - last_seen: timestamp
        
        Returns:
            Deduplicated chunk list (with occurrence metadata)
        """
        # Track message signatures and their occurrences
        message_signatures = defaultdict(list)  # hash -> list of (chunk_idx, timestamp)
        
        # Phase 1: Hash all chunks and group by signature
        for idx, chunk in enumerate(chunks):
            # Get normalized text (plain messages without formatting)
            normalized_text = chunk.metadata.get('normalized_text', chunk.page_content)
            
            # Create hash signature
            message_hash = self._hash_log_message(normalized_text)
            
            # Store chunk index and timestamp
            timestamp = chunk.metadata.get('start_timestamp')
            message_signatures[message_hash].append((idx, timestamp, chunk))
        
        # Phase 2: Mark duplicates and add occurrence metadata
        chunks_to_keep = []
        
        for message_hash, occurrences in message_signatures.items():
            occurrence_count = len(occurrences)
            
            if occurrence_count == 1:
                # Unique message - keep as is
                idx, timestamp, chunk = occurrences[0]
                chunk.metadata['is_duplicate'] = False
                chunk.metadata['occurrence_count'] = 1
                chunks_to_keep.append(chunk)
            
            else:
                # Duplicate messages - apply sampling strategy
                # Sort by timestamp
                sorted_occurrences = sorted(
                    occurrences,
                    key=lambda x: x[1] if isinstance(x[1], datetime) else datetime.min
                )
                
                # Keep first occurrence (with full context)
                first_idx, first_ts, first_chunk = sorted_occurrences[0]
                first_chunk.metadata['is_duplicate'] = True
                first_chunk.metadata['is_first_occurrence'] = True
                first_chunk.metadata['occurrence_count'] = occurrence_count
                first_chunk.metadata['first_seen'] = first_ts
                first_chunk.metadata['last_seen'] = sorted_occurrences[-1][1]
                first_chunk.metadata['message_signature'] = message_hash
                chunks_to_keep.append(first_chunk)
                
                # Keep last occurrence (if different from first)
                if occurrence_count > 1:
                    last_idx, last_ts, last_chunk = sorted_occurrences[-1]
                    last_chunk.metadata['is_duplicate'] = True
                    last_chunk.metadata['is_last_occurrence'] = True
                    last_chunk.metadata['occurrence_count'] = occurrence_count
                    last_chunk.metadata['first_seen'] = first_ts
                    last_chunk.metadata['last_seen'] = last_ts
                    last_chunk.metadata['message_signature'] = message_hash
                    chunks_to_keep.append(last_chunk)
                
                # Store summary of dropped occurrences in first chunk
                if occurrence_count > 2:
                    first_chunk.metadata['dropped_occurrences'] = occurrence_count - 2
                    first_chunk.metadata['sample_note'] = f'Showing first and last of {occurrence_count} identical messages'
        
        return chunks_to_keep
    
    def _hash_log_message(self, message: str) -> str:
        """
        Create hash for deduplication.
        
        Normalizes message by removing timestamps, transaction IDs, and numbers
        to create a stable signature for detecting repetitive patterns.
        
        Args:
            message: Raw log message
        
        Returns:
            MD5 hash of normalized message
        """
        # Normalize message for hashing
        normalized = message.lower().strip()
        
        # Remove variable parts that shouldn't affect deduplication
        # Replace transaction IDs with placeholder
        normalized = re.sub(r'\b\d{10,}\b', '<TID>', normalized)
        # Replace numbers with placeholder
        normalized = re.sub(r'\b\d+\.\d+\b', '<NUM>', normalized)
        normalized = re.sub(r'\b\d+\b', '<NUM>', normalized)
        # Replace file paths with placeholder
        normalized = re.sub(r'/[^\s]+', '<PATH>', normalized)
        # Replace timestamps
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', normalized)
        
        # Create MD5 hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def run(self, 
            file_pattern: str = "*.log",
            use_timeline: bool = False,
            deduplicate: bool = True) -> Tuple[List[Document], Optional[anytree.Node]]:
        """
        Orchestrate the full log preprocessing pipeline.
        
        Pipeline steps:
            1. Load: Stream log files and parse entries
            2. Chunk: Group by transaction ID and time windows
            3. Normalize: Create dual representation (structured + plain text)
            4. Extract Metadata: Error signatures, metrics, stack traces
            5. Build Timeline (optional): Create causal event trees
            6. Deduplicate (optional): Sample repetitive log patterns
        
        Args:
            file_pattern: Glob pattern for log files (e.g., "*.log", "logs/**/*.log")
            use_timeline: Enable temporal/causal tree construction
            deduplicate: Enable deduplication of repetitive logs
        
        Returns:
            Tuple of (enriched_chunks, timeline_tree)
                - enriched_chunks: List of processed Document chunks
                - timeline_tree: Root of first timeline (if use_timeline=True), else None
        
        Example:
            >>> preprocessor = LogPreprocessor(kb_dir="/path/to/logs")
            >>> chunks, timeline = preprocessor.run(
            ...     file_pattern="**/*.log",
            ...     use_timeline=True,
            ...     deduplicate=True
            ... )
            >>> print(f"Processed {len(chunks)} log chunks")
        """
        print("="*80)
        print("LOG PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load and parse log files
        print(f"\n[Step 1/6] Loading logs with pattern: {file_pattern}")
        parsed_entries = self.load_logs(file_pattern=file_pattern)
        
        if not parsed_entries:
            print("No log entries found. Exiting pipeline.")
            return [], None
        
        # Step 2: Chunk logs (groups by TID/time + size normalization)
        print(f"\n[Step 2/6] Chunking {len(parsed_entries)} log entries")
        chunks = self.chunk_logs(parsed_entries)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 3-4: Normalize and extract metadata (already done in chunk_logs)
        print(f"\n[Step 3-4/6] Normalization and metadata extraction completed during chunking")
        
        # Step 5: Build timeline (optional)
        timeline_root = None
        if use_timeline:
            print(f"\n[Step 5/6] Building temporal/causal timelines")
            chunks, timeline_root = self.build_timeline(chunks)
            
            # Count chunks in timelines
            chunks_in_timeline = sum(1 for c in chunks if c.metadata.get('in_timeline', False))
            print(f"  {chunks_in_timeline}/{len(chunks)} chunks added to timelines")
            
            if timeline_root:
                print(f"  Timeline root: {timeline_root.name[:60]}...")
        else:
            print(f"\n[Step 5/6] Timeline building: SKIPPED (use_timeline=False)")
        
        # Step 6: Deduplicate (optional)
        if deduplicate:
            print(f"\n[Step 6/6] Deduplicating repetitive log patterns")
            original_count = len(chunks)
            chunks = self.deduplicate_logs(chunks)
            removed_count = original_count - len(chunks)
            print(f"  Removed {removed_count} duplicate chunks ({len(chunks)} remaining)")
        else:
            print(f"\n[Step 6/6] Deduplication: SKIPPED (deduplicate=False)")
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Final output: {len(chunks)} enriched log chunks")
        
        # Show sample metadata
        if chunks:
            sample = chunks[0]
            print(f"\nSample chunk metadata keys:")
            print(f"  {list(sample.metadata.keys())[:10]}...")
        
        return chunks, timeline_root


# ============================================================
# USAGE & TESTING
# ============================================================


# usage
if __name__ == "__main__":
    preprocessor = MDPreprocessor()
    
    # BASIC MODE: core metadata extraction only (production default)
    # final_docs, knowledge_graph, vectorizer = preprocessor.run(use_tfidf=False, use_graph=False)
    
    # PROOF OF CONCEPT MODE: enable graph for large KB demonstration
    final_docs, knowledge_graph, vectorizer = preprocessor.run(use_tfidf=False, use_graph=True)
    
    # ADVANCED MODE: enable all features for 50+ document knowledge bases
    # final_docs, knowledge_graph, vectorizer = preprocessor.run(use_tfidf=True, use_graph=True)
    
    print(f"Processed {len(final_docs)} chunks")
    print(f"Graph: {'enabled' if knowledge_graph else 'disabled'}")
    if knowledge_graph:
        print(f"  Nodes: {knowledge_graph.number_of_nodes()}")
        print(f"  Edges: {knowledge_graph.number_of_edges()}")
    print(f"TF-IDF vectorizer: {'enabled' if vectorizer else 'disabled'}")
    
    # show sample chunk with metadata
    if final_docs:
        sample = final_docs[0]
        print(f"\nSample chunk metadata:")
        print(f"  section_hierarchy: {sample.metadata.get('section_hierarchy', 'N/A')}")
        print(f"  doc_type: {sample.metadata.get('doc_type')}")
        print(f"  mentioned_services: {sample.metadata.get('mentioned_services', [])}")
        
        # graph metadata (only if enabled)
        if knowledge_graph:
            print(f"  chunk_id: {sample.metadata.get('chunk_id')}")
            print(f"  related_chunks: {sample.metadata.get('related_chunks', [])[:3]}...")
            print(f"  relationship_types: {sample.metadata.get('relationship_types', [])[:3]}...")
        
        # tfidf keywords (only if enabled)
        if 'distinctive_keywords' in sample.metadata:
            print(f"  distinctive_keywords: {sample.metadata['distinctive_keywords'][:5]}...")
    
    # show graph analysis (only if enabled)
    if knowledge_graph:
        print("\n" + "="*80)
        print("Graph Connectivity Analysis:")
        print("="*80)
        
        connected_chunks = [c for c in final_docs if c.metadata.get('related_chunks', [])]
        print(f"\nFound {len(connected_chunks)} chunks with relationships (out of {len(final_docs)} total)\n")
        
        for chunk in connected_chunks[:5]:  # show first 5 connected chunks
            print(f"{chunk.metadata.get('chunk_id')}:")
            print(f"   Section: {chunk.metadata.get('section_hierarchy', 'N/A')[:60]}...")
            print(f"   Doc type: {chunk.metadata.get('doc_type')} | Priority: {chunk.metadata.get('priority')}")
            print(f"   Related to: {chunk.metadata.get('related_chunks', [])[:3]}")
            print(f"   Relationship types: {chunk.metadata.get('relationship_types', [])[:3]}")
            print(f"   Mentioned services: {chunk.metadata.get('mentioned_services', [])}")
            print(f"   Graph degree: {chunk.metadata.get('out_degree')} out, {chunk.metadata.get('in_degree')} in")
            print()