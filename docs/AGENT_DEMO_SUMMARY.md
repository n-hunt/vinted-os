# RAG Agent Demo Mode - Summary

## ✅ What Was Implemented

The RAG agent now fully supports demo mode, allowing you to test database queries and analytics with the demo database instead of production data.

### Changes Made

1. **[src/services/query_service.py](src/services/query_service.py#L38)**
   - Added `demo_mode` parameter to `QueryService.__init__()`
   - Passes demo mode to underlying `DatabaseService`

2. **[src/agent/tools/query_tools.py](src/agent/tools/query_tools.py#L41)**
   - Added `set_demo_mode()` function to configure agent tools
   - Updated `get_query_service()` to respect demo mode setting
   - Global state management for demo mode

3. **[src/agent/tools/__init__.py](src/agent/tools/__init__.py)**
   - Exported `set_demo_mode()` function for external use
   - Made agent configuration accessible

4. **[agent.py](agent.py)** (NEW FILE)
   - Created CLI entry point for the RAG agent
   - Supports `--demo` flag
   - Interactive query interface

5. **Documentation Updates**
   - [DEMO_MODE.md](DEMO_MODE.md) - Added agent usage instructions
   - [QUICKSTART_DEMO.md](QUICKSTART_DEMO.md) - Added agent commands
   - [tests/verify_demo.py](tests/verify_demo.py) - Added agent verification

## 🚀 How to Use

### ETL Pipeline Demo Mode
```bash
python etl_pipeline.py --demo
```

### RAG Agent Demo Mode
```bash
python agent.py --demo
```

Both commands will:
- ✅ Use the demo database (`demo_db.db`)
- ✅ Keep production data safe
- ✅ Allow full testing and exploration

## 🔍 How It Works

### Initialization Sequence

1. **User runs:** `python agent.py --demo`
2. **Before imports:** `set_demo_mode(True)` is called
3. **On first query:** `get_query_service()` creates `QueryService(demo_mode=True)`
4. **QueryService creates:** `DatabaseService(demo_mode=True)`
5. **DatabaseService connects to:** `demo_db.db` instead of `vinted_os.db`

### Key Design Decisions

- **Global state management:** Demo mode is set globally before any tools are initialized
- **Singleton pattern:** `QueryService` is reused across tool calls for efficiency
- **Early configuration:** Demo mode must be set BEFORE importing agent tools
- **Explicit flag:** Both pipeline and agent use consistent `--demo` flag

## 📊 Comparison: Production vs Demo

| Component | Production Mode | Demo Mode |
|-----------|----------------|-----------|
| **Database** | `vinted_os.db` | `demo_db.db` |
| **ETL Pipeline** | `python etl_pipeline.py` | `python etl_pipeline.py --demo` |
| **RAG Agent** | `python agent.py` | `python agent.py --demo` |
| **QueryService** | Production DB | Demo DB |
| **All Tools** | Production data | Demo data |

## 🧪 Testing

```bash
# 1. Verify setup
python tests/verify_demo.py

# 2. Test ETL pipeline
python etl_pipeline.py --demo

# 3. Test RAG agent
python agent.py --demo

# Try queries:
# - "Show me recent transactions"
# - "What items are in transaction 1234567890?"
# - "Get pipeline statistics"
```

## 💡 Example Agent Session

```
$ python agent.py --demo

🎯 Running in DEMO mode (using demo database)

============================================================
VintedOS RAG Agent
============================================================

Initializing agent components...
✓ Agent initialized successfully

============================================================
Type your questions below. Type 'quit' or 'exit' to stop.
============================================================

👤 You: Show me recent transactions

🤖 Agent: Here are the recent transactions:

1. Transaction 1234567890
   - Customer: Demo Customer 1
   - Items: 1
   - Total: £25.00
   - Status: Pending

2. Transaction 9876543210
   - Customer: Demo Customer 2
   - Items: 3
   - Total: £33.00
   - Status: Pending

3. Transaction 5555555555
   - Customer: Demo Customer 3
   - Items: 1
   - Total: £30.00
   - Status: Pending

👤 You: quit

👋 Goodbye!
💾 Conversation saved to: data/conversations/cli_session.json
```

## 🎯 Benefits

1. **Safe Testing:** Never risk production data
2. **Consistent Interface:** Same commands for pipeline and agent
3. **Easy Switching:** Just add `--demo` flag
4. **Isolated State:** Demo and production completely separate
5. **Full Functionality:** All features work in demo mode

## 📝 Notes

- Demo mode must be set **before** initializing agent tools
- The `set_demo_mode()` function resets the QueryService singleton
- Both ETL pipeline and RAG agent share the same demo database
- Demo database can be reset by running `python tests/init_demo_db.py`

## ✅ Verification

All checks pass:
- ✓ Configuration supports demo mode
- ✓ QueryService accepts demo_mode parameter
- ✓ Agent tools support set_demo_mode()
- ✓ CLI entry point (agent.py) exists
- ✓ Documentation updated
- ✓ Verification script validates setup

---

**You can now easily test the entire VintedOS system (ETL + RAG agent) using demo mode!** 🎉
