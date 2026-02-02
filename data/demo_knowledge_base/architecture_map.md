# VintedOS Architecture & Data Flow

**Local-first ETL Pipeline for P2P Commerce Transaction Processing**

This document provides a technical overview of the VintedOS system architecture, module responsibilities, and data flow orchestration.

---

## Service Module Map

### Core Services Layer (`src/services/`)

#### `gmail.py` - Gmail API Connector
**Responsibility:** Email ingestion and attachment retrieval.

- Establishes OAuth 2.0 authenticated connections to Gmail API (not IMAP)
- Polls inbox using configurable query filters (e.g., `has:attachment filename:pdf`)
- Downloads PDF attachments as raw bytes
- Applies regex patterns to filter relevant files:
  - Return forms: `^Order-return-form`
  - Shipping labels: `^Vinted-(?:Label|Digital-Label)`
- Ignores non-Vinted files (invoices, spam, forwarded emails)
- Provides batch download capabilities to minimize API calls
- Supports message trashing for cleanup operations

**Key Methods:**
- `fetch_attachments_with_pattern()` - Query-based attachment discovery
- `download_attachment()` - Binary content retrieval
- `trash_messages()` - Post-processing cleanup

---

#### `database.py` - SQLite Persistence Layer
**Responsibility:** All database write operations and transaction management.

- Manages SQLite connection with WAL (Write-Ahead Logging) mode for concurrent access
- Handles CRUD operations for transactions, items, print jobs, processing logs
- Implements "fire-and-forget" telemetry pattern (pipeline continues on DB failures)
- Uses context managers to prevent database locks
- Provides transaction lifecycle tracking (PENDING → PARSED → MATCHED → PRINTED → COMPLETED)

**Key Methods:**
- `create_transaction()` - Initial transaction record creation
- `update_transaction_status()` - State machine progression
- `log_step()` - Audit trail for debugging
- `add_print_job()` - Print attempt recording

---

#### `query_service.py` - Read-Only Analytics Layer
**Responsibility:** Complex database queries for reporting and AI agents.

- Provides structured data access for dashboards and LLM tools
- Implements aggregation queries (revenue, top sellers, failure analysis)
- Supports safe SQL execution for LLM-generated queries
- Returns serialized dictionaries (JSON-compatible)
- No write operations (enforces separation of concerns)

**Key Methods:**
- `get_pipeline_stats()` - Comprehensive metrics
- `get_failed_transactions()` - Error analysis
- `get_top_selling_items()` - Business intelligence
- `execute_safe_query()` - LLM-safe SQL execution

---

#### `printer.py` - CUPS Thermal Printer Interface
**Responsibility:** Physical label printing via CUPS (Common Unix Printing System).

- Sends PDFs to Zebra GK420d thermal printer using `lp` command
- Implements retry logic with exponential backoff
- Supports dry-run mode (saves to debug folder instead of printing)
- Batch processing with configurable inter-job delays
- Returns structured PrintJob results for telemetry

**Configuration:**
- Printer: Zebra GK420d (203 DPI thermal)
- Options: `fit-to-page`, `Darkness=30`
- Network: IP-based or USB connection via CUPS

---

### DocuFlow Library (`src/docuflow/`)

**Purpose:** Reusable PDF processing toolkit (extraction, transformation, generation)

#### `parser.py` - Text Extraction Engine
**Responsibility:** PDF text parsing and structured data extraction.

- Uses `pdfplumber` library for text extraction from PDFs
- Parses Return Form PDFs to extract:
  - Transaction ID (regex pattern: `Transaction ID:\s*(\d+)`)
  - Item names and prices (formatted as `Item Name £12.50`)
  - Multi-page document support
- Filters out non-product items (shipping fees, buyer protection)
- Returns structured dictionaries:
  ```python
  {
    "transaction_id": "18273645901",
    "items": [
      {"item": "Nike Air Max 90", "price": 29.99},
      {"item": "Adidas Jacket", "price": 35.00}
    ]
  }
  ```

**Key Methods:**
- `parse_pdf()` - Main parsing orchestrator
- `extract_transaction_id_from_text()` - Regex-based ID extraction
- `extract_items_and_prices_structured()` - Line-by-line item parsing

---

#### `vision.py` - Computer Vision Engine
**Responsibility:** Image preprocessing for optimal thermal printing.

- Converts PDFs to images using `pdf2image` (Poppler backend)
- Uses **PIL/Pillow** for image manipulation 
- Implements adaptive image processing pipeline:
  1. **Grayscale conversion** - Reduces color complexity
  2. **Binarization** - Converts to pure black/white using threshold
  3. **Auto-crop** - Detects and removes whitespace borders
  4. **Rotation** - Landscape → Portrait (if needed for 4×6 format)

**Critical Methods:**
- `crop_and_resize_to_4x6()` - Main processing pipeline
- `_apply_binarization()` - Threshold-based conversion (default: 200)
- `_auto_crop_whitespace()` - Border detection using pixel scanning

**Why this matters:**
- Thermal printers only print black (no grayscale)
- Binarization prevents "gray smudges" and faded text
- Auto-cropping maximizes usable label space

---

#### `generator.py` - PDF Composition Engine
**Responsibility:** Creating enhanced labels with overlays.

- Uses `PyPDF2` for PDF manipulation and `ReportLab` for text rendering
- Generates final 4×6 inch labels by:
  1. Scaling original shipping label to 3.8×5.5 inches
  2. Creating text overlay with transaction ID and item list
  3. Merging overlay onto scaled label
  4. Centering content on 4×6 page with margins

**Output Types:**
- **Single-item labels**: Item name + price at bottom
- **Multi-item labels**: "MULTI-ITEM ORDER (X items) - CHECK LIST" + separate itemized PDF

**Key Methods:**
- `add_items_to_pdf()` - Main label enhancement
- `create_items_list_pdf()` - Multi-page item manifests
- `_create_text_overlay()` - Text layer generation

---

## Pipeline Orchestration (`src/main.py`)

### The Matching & Coupling Logic

#### Overview
The `_match_and_generate()` method implements a **synchronous pairing algorithm** that matches return forms with shipping labels based on Transaction IDs.

#### How It Works

**Step 1: Separate Ingestion**
The pipeline fetches documents in two independent steps:
```python
# Step 1: Fetch all return forms
return_forms = _fetch_and_parse_return_forms()
# Returns: {"18273645901": TransactionData(...), "18273645902": ...}

# Step 2: Fetch all shipping labels  
labels = _fetch_and_process_labels()
# Returns: {"18273645901": AttachmentData(...), "18273645902": ...}
```

**Step 2: Pairing Algorithm**
```python
def _match_and_generate(return_forms, labels):
    for tid, label_data in labels.items():
        # Check if matching return form exists
        if tid not in return_forms:
            logger.warning(f"No return form found for label TID {tid}, skipping")
            continue  # Label is orphaned - no generation
        
        # BOTH assets exist - proceed with generation
        transaction = return_forms[tid]
        enhanced_pdf = generate_label(label_data, transaction.items)
        save_to_disk(enhanced_pdf)
```

#### Critical Matching Rules

**Rule 1: Both Assets Required**
- Label generation ONLY occurs when **both** return form AND shipping label exist for the same Transaction ID
- If only one asset is present, it is **skipped** (not queued)

**Rule 2: No Persistent Queue**
- The system does NOT maintain a "pending queue" across pipeline runs
- Orphaned documents (label without form, or form without label) are logged and ignored
- Each pipeline execution is stateless - only processes complete pairs

**Rule 3: Transaction ID is the Natural Key**
- Matching is performed by exact string comparison of the extracted Transaction ID
- IDs are extracted from filenames (labels) and PDF text (return forms)
- Pattern: 10+ digit number (e.g., `18273645901`)

#### What Happens to Orphaned Documents?

**Scenario A: Label exists, no return form**
```
Found label: Vinted-Label-18273645901.pdf
No return form found for TID 18273645901
→ Skipped (label not generated)
→ Gmail message remains in inbox
→ Will be reprocessed in next pipeline run
```

**Scenario B: Return form exists, no label**
```
Found return form: Order-return-form-18273645902.pdf
No matching label in labels dictionary
→ Skipped during matching phase
→ Transaction recorded in database with status PENDING
→ Will complete when label arrives in future run
```

#### Why This Architecture?

**Advantages:**
- **Simple**: No complex queue management or state persistence
- **Fault-tolerant**: Missing documents don't crash the pipeline
- **Eventually consistent**: Re-running the pipeline will catch late-arriving documents
- **Observable**: All skipped pairs are logged for debugging

**Trade-offs:**
- Requires re-processing emails until pairs are complete
- No guaranteed ordering (labels may arrive before forms)
- Relies on Gmail API for persistence (inbox as temporary storage)

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  1. EMAIL INGESTION (gmail.py)                          │
│  • Poll Gmail API                                       │
│  • Download return forms + shipping labels              │
│  Output: Raw PDF bytes                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │                               │
    Return Forms                   Shipping Labels
         │                               │
         ↓                               ↓
┌─────────────────────┐      ┌─────────────────────┐
│  2a. TEXT PARSING   │      │  2b. IMAGE PROCESSING│
│  (parser.py)        │      │  (vision.py)        │
│  • Extract TID      │      │  • Binarize         │
│  • Parse items      │      │  • Crop borders     │
│  Output: Dict       │      │  Output: Clean PDF  │
└─────────────────────┘      └─────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  3. MATCHING & GENERATION (main.py)                     │
│  • Match by Transaction ID                              │
│  • Pair form data + label image                         │
│  • Generate enhanced 4x6 label (generator.py)           │
│  Output: Final PDF with item overlay                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  4. PRINTING (printer.py)                               │
│  • Send PDF to Zebra GK420d via CUPS                    │
│  • Record print job status                              │
│  Output: Physical thermal label                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  5. PERSISTENCE (database.py)                           │
│  • Write transaction records                            │
│  • Log processing steps                                 │
│  • Update status: COMPLETED                             │
│  Output: Queryable history in SQLite                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  6. CLEANUP (main.py)                                   │
│  • Trash processed Gmail messages                       │
│  • Delete local PDF files                               │
│  Output: Clean inbox, no file clutter                   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Patterns

### 1. Service-Oriented Architecture
- Each module has a single, well-defined responsibility
- Services communicate via clean data object interfaces
- No circular dependencies

### 2. Fire-and-Forget Telemetry
- Database writes are wrapped in try/except blocks
- Pipeline continues even if DB operations fail
- Prevents a locked database from stopping order fulfillment

### 3. Immutable Data Objects
- `TransactionData`, `AttachmentData` are dataclasses
- Services return new objects rather than mutating inputs
- Makes the flow easier to reason about and test

### 4. Configuration-Driven Behavior
- All magic numbers live in `config/settings.yaml`
- Regex patterns, DPI, thresholds all externalized
- No hardcoded values in business logic

---

## Execution Sequence

When `python etl_pipeline.py` is executed:

```
1. Initialize services (Gmail, Database, Printer, PDF processors)
2. Start pipeline run tracking (get run_id from DB)
3. Fetch & parse return forms → Dict[TID, TransactionData]
4. Fetch & process shipping labels → Dict[TID, AttachmentData]
5. Match pairs by TID → Generate enhanced labels → Save to disk
6. Print all generated labels → Record print jobs
7. Trash Gmail messages → Delete local files
8. End pipeline run tracking → Log statistics
```

**Total runtime:** ~30-60 seconds for 10 transactions (depends on Gmail API latency)

---

## Error Handling Strategy

### Graceful Degradation
- **Gmail fails**: Pipeline logs error, exits cleanly
- **PDF parsing fails**: Skip that document, continue with others
- **Image processing fails**: Return original PDF without enhancement
- **Printing fails**: Record failure in DB, continue with other labels
- **Database fails**: Log warning, pipeline continues (data lost but orders printed)

### Fault Isolation
Each transaction is processed independently - one failure doesn't affect others.

---

## Future Architecture (AI Agent Layer)

The system is being extended with a "Chief of Staff" AI agent that provides natural language querying:

**Query Service** (already built)
- Exposes structured data access methods
- Used by both dashboards and AI agents

**RAG Retrieval** (in development)
- Vector store (Qdrant) for semantic search
- BM25 index for keyword matching
- Hybrid retrieval for optimal relevance

**Function Calling** (planned)
- LLM calls QueryService methods as tools
- Natural language → SQL queries
- Automated troubleshooting and insights

This maintains the separation: ETL pipeline handles operations, AI agent handles intelligence.
