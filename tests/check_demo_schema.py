#!/usr/bin/env python3
"""Quick script to verify demo database schema and content."""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.services.database import DatabaseService
from sqlmodel import Session, select, text
from src.models import Transaction, Item, GmailMessage, Attachment, PrintJob, ProcessingLog, PipelineRun

db = DatabaseService(demo_mode=True)

print("=" * 60)
print("Demo Database Schema Verification")
print("=" * 60)

with Session(db.engine) as session:
    # Check all tables exist
    tables = session.exec(text("SELECT name FROM sqlite_master WHERE type='table'")).all()
    table_names = [t[0] for t in tables]
    print("\n✓ Tables created:")
    for name in sorted(table_names):
        print(f"  - {name}")
    
    # Check counts
    print("\n✓ Record counts:")
    print(f"  Pipeline Runs: {len(session.exec(select(PipelineRun)).all())}")
    print(f"  Transactions: {len(session.exec(select(Transaction)).all())}")
    print(f"  Items: {len(session.exec(select(Item)).all())}")
    print(f"  Gmail Messages: {len(session.exec(select(GmailMessage)).all())}")
    print(f"  Attachments: {len(session.exec(select(Attachment)).all())}")
    print(f"  Print Jobs: {len(session.exec(select(PrintJob)).all())}")
    print(f"  Processing Logs: {len(session.exec(select(ProcessingLog)).all())}")
    
    # Check a sample transaction with all relationships
    tx = session.exec(select(Transaction).where(Transaction.vinted_order_id == '9876543210')).first()
    print(f"\n✓ Sample Transaction (9876543210):")
    print(f"  Customer: {tx.customer_name}")
    print(f"  Status: {tx.status.value}")
    print(f"  Items: {len(tx.items)}")
    for item in tx.items:
        print(f"    • {item.title}: £{item.price}")
    print(f"  Gmail Messages: {len(tx.gmail_messages)}")
    print(f"  Attachments: {sum(len(msg.attachments) for msg in tx.gmail_messages)}")
    print(f"  Print Jobs: {len(tx.print_jobs)}")
    print(f"  Processing Logs: {len(tx.logs)}")
    
    # Show status breakdown
    print("\n✓ Transaction Status Breakdown:")
    for status in ["PENDING", "PARSED", "MATCHED", "PRINTED", "COMPLETED", "FAILED"]:
        count = len(session.exec(
            select(Transaction).where(Transaction.status == status)
        ).all())
        if count > 0:
            print(f"  {status}: {count}")

print("\n" + "=" * 60)
print("✅ Schema verification complete!")
print("=" * 60)
