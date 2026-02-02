# Demo Mode Quick Reference

## 🎯 Quick Start (30 seconds)

```bash
# 1. Create demo database with sample data
python tests/init_demo_db.py

# 2. Run the pipeline in demo mode
python etl_pipeline.py --demo
```

## 📋 Commands Cheat Sheet

| Command | Description |
|---------|-------------|
| `python etl_pipeline.py` | **Production** - Live ETL pipeline with Gmail & printer |
| `python etl_pipeline.py --dry-run` | **Dry Run** - Production DB, no printing |
| `python etl_pipeline.py --demo` | **Demo** - Demo DB, no printing (safe!) |
| `python agent.py` | **Agent Production** - RAG agent with production DB |
| `python agent.py --demo` | **Agent Demo** - RAG agent with demo DB |
| `python etl_pipeline.py --help` | Show ETL pipeline help |
| `python agent.py --help` | Show agent help |
| `python tests/init_demo_db.py` | Initialize demo database |

## 🔍 What Each Mode Does

### Production Mode
- ✅ Connects to Gmail
- ✅ Uses `vinted_os.db`
- ✅ Prints to physical printer
- ⚠️ Use with caution

### Dry-Run Mode (`--dry-run`)
- ✅ Connects to Gmail
- ✅ Uses `vinted_os.db`
- ❌ Saves PDFs instead of printing
- 💡 Good for testing without printer

### Demo Mode (`--demo`)
- ❌ No Gmail connection needed
- ✅ Uses `demo_db.db`
- ❌ Saves PDFs instead of printing
- ✅ Uses sample data
- 💡 **Perfect for testing and demos!**

## 📁 Files Created

```
demo_db.db           # Demo database (SQLite)
demo_db.db-wal       # Write-ahead log (temporary)
demo_db.db-shm       # Shared memory (temporary)
logs/print_debug/    # Generated PDFs (in demo mode)
```

## 🔄 Reset Demo Database

```bash
# Delete all demo database files
rm demo_db.db demo_db.db-wal demo_db.db-shm

# Recreate with fresh data
python tests/init_demo_db.py
```

## 📦 Demo Database Contents

After initialization, you'll have **15 comprehensive transactions**:

| Transaction ID | Items | Total | Status | Notes |
|----------------|-------|-------|--------|-------|
| 1234567890 | 1 | £25.00 | Completed | Single item |
| 9876543210 | 3 | £33.00 | Completed | Multi-item |
| 5555555555 | 1 | £30.00 | Printed | Ready for completion |
| 7777777777 | 2 | £34.00 | Pending | Just fetched |
| 8888888888 | 1 | £35.00 | Failed | Print failure |
| 1111222233 | 5 | £131.00 | Completed | Large order |
| 3333444455 | 1 | £15.00 | Parsed | Parsed only |
| 6666777788 | 2 | £127.00 | Completed | Standard |
| 2222333344 | 3 | £155.00 | Pending | Outdoor gear |
| 4444555566 | 1 | £22.00 | Matched | Matched status |
| 7788990011 | 4 | £123.00 | Completed | Vintage clothing |
| 9999000011 | 1 | £285.00 | Completed | High-value |
| 1212343456 | 3 | £14.00 | Printed | Budget items |
| 5566778899 | 1 | £145.00 | Failed | Parse error |
| 3344556677 | 4 | £91.00 | Completed | Sportswear |

**Total: 33 items, £1,100+ revenue, all schema relationships included!**

## 💡 Pro Tips

1. **Always use demo mode first** when testing new features
2. **Demo mode is safe** - it won't affect your production data
3. **PDFs are saved** to `logs/print_debug/` for inspection
4. **Customize demo data** by editing `tests/init_demo_db.py`
5. **Check logs** in `logs/` for detailed execution info

## 🚀 Next Steps

1. ✅ Run `python tests/init_demo_db.py`
2. ✅ Run `python etl_pipeline.py --demo` (test ETL pipeline)
3. ✅ Run `python agent.py --demo` (test RAG agent)
4. ✅ Check generated PDFs in `logs/print_debug/`
5. ✅ Review [DEMO_MODE.md](DEMO_MODE.md) for full documentation
6. ⏭️ Configure Gmail when ready for production

## 💡 RAG Agent Demo Examples

After running `python agent.py --demo`, try these queries:

```
👤 You: Show me recent transactions
👤 You: What items are in transaction 1234567890?
👤 You: Get statistics for the pipeline
👤 You: What are the total sales?
```

## ❓ Troubleshooting

**"Database is locked"**
```bash
rm demo_db.db-wal demo_db.db-shm
```

**"Demo database not found"**
```bash
python tests/init_demo_db.py
```

**Changes not saving**
- Make sure you're using `--demo` flag!

---

For detailed documentation, see [DEMO_MODE.md](DEMO_MODE.md)
