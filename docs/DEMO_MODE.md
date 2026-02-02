# VintedOS Demo Mode

This guide explains how to test VintedOS using the demo mode, which allows you to explore the system without connecting to Gmail or a physical printer.

## What is Demo Mode?

Demo mode is a testing feature that:
- Uses a separate demo database (`demo_db.db`) instead of your production database
- Automatically enables dry-run mode (saves PDFs to debug folder instead of printing)
- Allows you to test the pipeline with sample data
- **Works with both the ETL pipeline AND the RAG agent**
- Keeps your production data safe and isolated

## Quick Start

### 1. Initialize Demo Database

First, create the demo database with sample transactions:

```bash
python tests/init_demo_db.py
```

This creates:
- A demo database file (`demo_db.db`)
- Sample transactions with test data
- Example print jobs

### 2. Run in Demo Mode

Execute the pipeline in demo mode:

```bash
python etl_pipeline.py --demo
```

You should see:
```
============================================================
VintedOS - P2P Commerce ETL Pipeline
============================================================
Mode: DEMO (using demo database)
============================================================
```

## Demo vs Dry-Run vs Production

| Mode | Database | Printer | Use Case |
|------|----------|---------|----------|
| **Production** | `vinted_os.db` | Physical printer | Live operations |
| **Dry-Run** | `vinted_os.db` | Save to debug folder | Test without printing |
| **Demo** | `demo_db.db` | Save to debug folder | Test with sample data |

### Commands

```bash
# Production mode (uses real database and printer)
python etl_pipeline.py

# Dry-run mode (uses real database, saves PDFs to debug folder)
python etl_pipeline.py --dry-run

# Demo mode (uses demo database, saves PDFs to debug folder)
python etl_pipeline.py --demo

# RAG Agent in production mode
python agent.py

# RAG Agent in demo mode
python agent.py --demo
```

## Demo Database Contents

The demo database includes **15 complete transactions** with all schema relationships:

### Full Schema Coverage

Each transaction includes:
- **Transaction record** with status, timestamps, customer info
- **Items** with titles and prices
- **Gmail messages** with subjects and received dates
- **Attachments** (PDFs) with filenames and content types
- **Print jobs** (where applicable) with success/failure status
- **Processing logs** tracking each pipeline step

### Transaction Breakdown

| Order ID | Customer | Items | Total | Status | Use Case |
|----------|----------|-------|-------|--------|----------|
| 1234567890 | Alice Johnson | 1 | £25.00 | COMPLETED | Single item, fully processed |
| 9876543210 | Bob Smith | 3 | £33.00 | COMPLETED | Multi-item order |
| 5555555555 | Carol Davis | 1 | £30.00 | PRINTED | Printed, awaiting completion |
| 7777777777 | David Wilson | 2 | £34.00 | PENDING | Just fetched from Gmail |
| 8888888888 | Eve Martinez | 1 | £35.00 | FAILED | Failed print job |
| 1111222233 | Frank Thompson | 5 | £131.00 | COMPLETED | Large multi-item order |
| 3333444455 | Grace Lee | 1 | £15.00 | PARSED | Parsed but not matched |
| 6666777788 | Henry Brown | 2 | £127.00 | COMPLETED | Standard order |
| 2222333344 | Isabel Garcia | 3 | £155.00 | PENDING | Outdoor gear bundle |
| 4444555566 | Jack O'Connor | 1 | £22.00 | MATCHED | Matched status |
| 7788990011 | Karen Mitchell | 4 | £123.00 | COMPLETED | Vintage clothing bundle |
| 9999000011 | Laura Chen | 1 | £285.00 | COMPLETED | High-value item |
| 1212343456 | Mike Rodriguez | 3 | £14.00 | PRINTED | Budget items |
| 5566778899 | Nancy Patel | 1 | £145.00 | FAILED | Parse error scenario |
| 3344556677 | Oscar Kim | 4 | £91.00 | COMPLETED | Sportswear collection |

### Database Statistics

- **Total Transactions**: 15
- **Total Items**: 33
- **Pipeline Runs**: 3 (simulating different execution times)
- **Gmail Messages**: 15 (one per transaction)
- **Attachments**: 21 (labels and return forms)
- **Print Jobs**: 10 (with 2 failures)
- **Processing Logs**: 52 (tracking all pipeline steps)
- **Total Revenue**: £1,100+
- **All Status Types**: PENDING, PARSED, MATCHED, PRINTED, COMPLETED, FAILED

## Testing Scenarios

### Basic Pipeline Test
```bash
# Initialize and run
python tests/init_demo_db.py
python etl_pipeline.py --demo
```

### RAG Agent Test
```bash
# Initialize demo database (if not already done)
python tests/init_demo_db.py

# Start agent in demo mode
python agent.py --demo

# Try queries like:
# - "Show me recent transactions"
# - "What items have been sold?"
# - "Get transaction details for order 1234567890"
# - "What are the system statistics?"
```

### Custom Demo Data

You can modify `tests/init_demo_db.py` to add your own test transactions:

```python
# Add a custom transaction
tx_id = "1111111111"
db.create_transaction(
    transaction_id=tx_id,
    status=TransactionStatus.PENDING,
    message_ids=["custom_msg_001"]
)

db.add_item_to_transaction(
    transaction_id=tx_id,
    item_name="Your Custom Item",
    item_price=20.00
)
```

### Reset Demo Database

To start fresh, delete the demo database and re-initialize:

```bash
rm demo_db.db demo_db.db-wal demo_db.db-shm  # Remove database files
python tests/init_demo_db.py                  # Recreate demo database
```

## Configuration

Demo mode uses settings from `config/settings.yaml`:

```yaml
database:
  filename: "vinted_os.db"      # Production database
  demo_filename: "demo_db.db"   # Demo database
```

You can customize the demo database filename by editing this setting.

## Limitations

Demo mode is designed for testing and development:

- ❌ Does not connect to Gmail (no email fetching)
- ❌ Does not print to physical printer
- ❌ Uses mock/sample data only
- ✅ Tests database operations
- ✅ Tests PDF processing logic
- ✅ Tests label generation
- ✅ Safe to experiment with

## Troubleshooting

### "Database is locked" error

If you see this error:
1. Close any database browser tools (DB Browser for SQLite)
2. Delete WAL files: `rm demo_db.db-wal demo_db.db-shm`
3. Try again

### Demo database not found

Run the initialization script:
```bash
python tests/init_demo_db.py
```

### Changes not persisting

Make sure you're running with `--demo` flag:
```bash
python etl_pipeline.py --demo  # ✓ Uses demo database
python etl_pipeline.py         # ✗ Uses production database
```

## Next Steps

After testing with demo mode:

1. Review the generated PDFs in the debug folder
2. Check the demo database contents with a SQLite browser
3. Examine logs in `logs/` directory
4. When ready, configure production mode with real Gmail credentials
5. Test with `--dry-run` before enabling actual printing

## See Also

- [Main README](../README.md) - Full project documentation
- [Configuration Guide](config/settings.yaml) - All settings explained
- [Database Schema](src/models.py) - Database models and structure
