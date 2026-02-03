# VintedOS - Intelligent ETL Pipeline for P2P Commerce

**Automated label printing system with RAG-powered analytics**

VintedOS automates the processing of Vinted shipping labels: fetching emails from Gmail, parsing PDFs, generating enhanced labels, printing to Zebra printers, and providing intelligent query capabilities through a RAG (Retrieval-Augmented Generation) agent.

---

## Dependency Installation

**Recommended: Install from source**
```bash
pip install -e .
```

**Alternative methods:**
```bash
# Using pip with requirements file
pip install -r install/requirements.txt

# Using conda
conda env create -f install/environment.yml
conda activate vinted-os
```


## Work in Progress

**Note:** This repository is currently being prepared for public release. The original production version contained proprietary data and a web-based frontend that have been removed. Additionally, the orchestration layer that synchronized the ETL pipeline with the RAG agent has been extracted, so these components currently operate independently. A future update will restore the real-time integration between the pipeline and agent.


---

## Quick Start for Reviewers

### Step 1: Set Up API Key (Required)

**Quick setup (recommended):**
```bash
python configure.py --api-key YOUR_GEMINI_API_KEY
```

**Or interactive mode:**
```bash
python configure.py
```

**Manual setup:**
```bash
cp .env.example .env
# Then edit .env and add your key
```

Get a free API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)

**Optional: Change LLM Model**
```bash
python configure.py --model gemini-2.5-flash   # Switch to different model
```

Default model is `gemini-3-flash-preview`. Other options: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, etc.

### Step 2: Choose Your Demo

**Option 1: Interactive Agent (Recommended)**
```bash
python agent.py --demo
```

Try these queries:
- "Show me recent transactions"
- "What's the total revenue from completed orders?"
- "Are there any failed transactions?"
- "Give me a dashboard summary"

**Option 2: Jupyter Notebook**
```bash
jupyter notebook demos/demo_rag_showcase.ipynb
```

The notebook provides a comprehensive guided tour of the RAG system's capabilities.

**Option 3: ETL Pipeline // WIP: DEMO NEEDS NON-PROPRIETARY LABELS; NOT SOURCED YET**
```bash
# Initialize demo database (first time only)
python tests/init_demo_db.py

# Run the ETL pipeline in demo mode (DOES NOT WORK YET)
python etl_pipeline.py --demo
```

---

## Gmail API Setup (Optional - for ETL pipeline)

To run the ETL pipeline with real Gmail data, you need to set up Gmail API credentials:

1. **Upload credentials** - Place your `credentials.json` file in the project root
   - Get credentials from [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Gmail API for your project
   
2. **Generate token** - Run the quickstart script to authenticate:
   ```bash
   python scripts/gmailquickstart.py
   ```
   - This opens a browser for OAuth authorization
   - Creates `token.json` for subsequent API calls
   
3. **Run production pipeline**:
   ```bash
   python etl_pipeline.py
   ```


---

## Demo Database

The demo database contains **15 comprehensive transactions** with:
- **33 items** ranging from £3.50 to £285.00
- All transaction statuses (PENDING, PARSED, MATCHED, PRINTED, COMPLETED, FAILED)
- **3 pipeline runs** with execution history
- Complete schema: Gmail messages, attachments, print jobs, processing logs
- **Total revenue**: £1,100+

All data is safe, isolated, and production-identical in structure.

### Inspect the Database Yourself

Want to explore the demo database directly? You can use any SQLite viewer:

**DBeaver Community (Recommended)**
- Download: [dbeaver.io/download](https://dbeaver.io/download)
- Open the database file at `data/demo_knowledge_base/demo.db`
- Browse all tables, relationships, and data
- Execute custom SQL queries

**Alternative Tools**
- **SQLite Online**: [sqliteonline.com](https://sqliteonline.com) - No installation needed
- **VS Code**: Install "SQLite Viewer" extension
- **Command Line**: `sqlite3 data/demo_knowledge_base/demo.db`

This allows you to verify the data structure, review transaction details, and understand the complete schema independent of the RAG agent.

---

## System Architecture

### Core Pipeline
1. **Gmail Integration** - Fetches emails with shipping labels
2. **PDF Processing** - Extracts transaction data and items
3. **Label Generation** - Creates enhanced labels with overlays
4. **Printer Service** - Sends to Zebra thermal printer
5. **Database Logging** - Tracks all operations and relationships

### RAG Agent
- **Natural Language Interface** - Ask questions in plain English
- **Intelligent Routing** - Automatically selects appropriate tools
- **Multi-turn Conversations** - Maintains context across queries
- **12 Specialized Tools** - Transaction queries, analytics, revenue analysis
- **LangChain + Gemini** - Powered by Google's latest models

---

## Project Structure

```
vinted-os-db/
├── README.md                   # This file
├── agent.py                    # Interactive RAG agent CLI
├── etl_pipeline.py             # ETL pipeline entry point
├── .env.example                # Environment template
├── pyproject.toml              # Package configuration & dependencies
├── config/
│   └── settings.yaml           # All configuration
├── demos/
│   └── demo_rag_showcase.ipynb # Professional demo notebook
├── docs/
│   ├── DEMO_MODE.md            # Demo mode guide
│   ├── QUICKSTART_DEMO.md      # Quick reference
│   └── AGENT_DEMO_SUMMARY.md   # RAG agent details
├── install/
│   ├── requirements.txt        # Pip dependencies (alternative)
│   └── environment.yml         # Conda environment (alternative)
├── scripts/
│   └── gmailquickstart.py      # Gmail OAuth setup
├── src/
│   ├── agent/                  # RAG agent implementation
│   │   ├── core/              # LLM client, ReAct loop, memory
│   │   ├── indexing/          # LangGraph pipeline
│   │   ├── retrieval/         # Vector store, hybrid search
│   │   └── tools/             # Query tools for function calling
│   ├── services/              # Database, Gmail, Printer
│   └── docuflow/              # PDF parsing, label generation
├── tests/
│   ├── init_demo_db.py        # Demo database initialization
│   └── check_demo_schema.py   # Schema verification
└── data/
    └── demo_knowledge_base/   # Documentation for RAG
```

---

## Features

### ETL Pipeline
- Gmail API integration with batch processing
- PDF parsing with OCR fallback
- Multi-item order detection and handling
- Label enhancement with transaction overlays
- Zebra printer support (GK420d)
- Comprehensive error handling and logging

### RAG Agent
- 12 specialized query tools
- Revenue analytics and business intelligence
- Transaction status tracking
- Print job monitoring
- Multi-turn contextual conversations
- Hybrid search (vector + BM25)

### Database
- SQLite with WAL mode for concurrency
- Complete relationship schema
- Automatic migrations
- Telemetry and audit trails

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **LLM** | Google Gemini 3 Flash (Preview); Can be configured differently|
| **Frameworks** | LangChain, LangGraph |
| **Database** | SQLite|
| **Vector Store** | Qdrant |
| **PDF Processing** | PyMuPDF, Pillow, ReportLab |
| **APIs** | Gmail API, Google AI |

---

## Documentation

- **[DEMO_MODE.md](docs/DEMO_MODE.md)** - Comprehensive demo mode guide
- **[QUICKSTART_DEMO.md](docs/QUICKSTART_DEMO.md)** - Quick reference card
- **[AGENT_DEMO_SUMMARY.md](docs/AGENT_DEMO_SUMMARY.md)** - RAG agent deep dive
- **[config/settings.yaml](config/settings.yaml)** - All settings explained

---

## Command Reference

```bash
# Demo Mode (Safe Testing)
python agent.py --demo              # Interactive agent
python etl_pipeline.py --demo       # ETL pipeline
python tests/init_demo_db.py        # Reset demo database

# Production Mode
python agent.py                     # Production agent
python etl_pipeline.py              # Production ETL

# Help & Info
python agent.py --help              # Agent usage
python etl_pipeline.py --help       # Pipeline usage
python tests/verify_demo.py         # Verify demo setup
python tests/check_demo_schema.py   # Check database schema
```

---

## Example Queries

The RAG agent understands natural language:

```
"Show me the 5 most recent transactions"
"What are the details of transaction 9876543210?"
"Are there any failed transactions?"
"What's the total revenue from completed orders?"
"Give me a complete dashboard summary"
"Show me transactions with more than 3 items"
"What's the print job success rate?"
"Which customer spent the most?"
```

---

## Key Highlights

### For Technical Reviewers
- LangGraph-powered indexing pipeline for reliable data ingestion
- Clean architecture with separation of concerns
- Comprehensive type hints and documentation
- Extensive error handling and logging
- Production-ready database schema
- WAL mode for concurrent access
- Proper resource cleanup and context managers

### For Business Reviewers
- Fully automated label processing
- Real-time analytics and insights
- Natural language query interface
- Complete audit trail
- Multi-item order handling
- Error detection and recovery

---

## System Statistics (Demo)

| Metric | Value |
|--------|-------|
| Transactions Processed | 15 |
| Items Tracked | 33 |
| Total Revenue | £1,100+ |
| Success Rate | 86.7% |
| Print Jobs | 10 |
| Processing Logs | 52+ |

---

## Testing

All components are tested and verified:
- Database schema matches production
- All relationships properly configured
- Tool execution verified
- Multi-turn conversations working
- Error handling tested
- Demo mode fully isolated

---

## Learning Path

1. **Start here**: Run `python agent.py --demo` and explore
2. **Deep dive**: Open `demos/demo_rag_showcase.ipynb` for guided tour
3. **Understand**: Read `docs/DEMO_MODE.md` for architecture
4. **Customize**: Check `config/settings.yaml` for all options
5. **Extend**: Review `src/agent/tools/` for adding new capabilities

---

## License & Contact

This is a portfolio project demonstrating full-stack AI integration with production-ready code quality.

**Key Technologies**: Python, LangChain, LangGraph, Gemini, SQLModel, Qdrant, Gmail API

---


