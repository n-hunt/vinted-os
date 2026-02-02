#!/usr/bin/env python3
"""
Demo Database Initialization Script

Creates a comprehensive demo database with sample transactions and data for testing.
Fully matches the production schema with all relationships and fields.

Usage:
    python tests/init_demo_db.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.services.database import DatabaseService
from src.models import (
    TransactionStatus,
    PrintStatus,
    Transaction,
    Item,
    GmailMessage,
    Attachment,
    PrintJob,
    ProcessingLog,
    PipelineRun
)
from sqlmodel import Session


def create_pipeline_run(db: DatabaseService, start_offset_hours: int = 0) -> int:
    """Create a pipeline run entry."""
    with Session(db.engine) as session:
        run = PipelineRun(
            start_time=datetime.now(timezone.utc) - timedelta(hours=start_offset_hours),
            end_time=datetime.now(timezone.utc) - timedelta(hours=start_offset_hours - 1) if start_offset_hours > 0 else None,
            items_processed=0,  # Will be updated as we create transactions
            items_failed=0,
            status="completed" if start_offset_hours > 0 else "running"
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id


def create_full_transaction(
    db: DatabaseService,
    vinted_order_id: str,
    customer_name: str,
    items: List[Dict[str, Any]],
    status: TransactionStatus,
    pipeline_run_id: int,
    gmail_id: str,
    gmail_subject: str,
    attachments: List[Dict[str, str]],
    has_print_job: bool = False,
    print_success: bool = True,
    processing_logs: List[Dict[str, Any]] = None
) -> int:
    """
    Create a complete transaction with all relationships matching the schema.
    
    This ensures the demo database exactly matches the production schema.
    """
    with Session(db.engine) as session:
        # 1. Create Transaction
        transaction = Transaction(
            vinted_order_id=vinted_order_id,
            customer_name=customer_name,
            status=status,
            label_path=f"static/services/labels/{vinted_order_id}_label.pdf" if status in [TransactionStatus.PRINTED, TransactionStatus.COMPLETED] else None,
            pipeline_run_id=pipeline_run_id,
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            updated_at=datetime.now(timezone.utc)
        )
        session.add(transaction)
        session.commit()
        session.refresh(transaction)
        tx_id = transaction.id
        
        # 2. Create Items
        for item_data in items:
            item = Item(
                transaction_id=tx_id,
                title=item_data["title"],
                price=item_data["price"],
                currency=item_data.get("currency", "GBP")
            )
            session.add(item)
        
        # 3. Create GmailMessage
        gmail_msg = GmailMessage(
            transaction_id=tx_id,
            gmail_id=gmail_id,
            subject=gmail_subject,
            received_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        session.add(gmail_msg)
        session.commit()
        session.refresh(gmail_msg)
        
        # 4. Create Attachments
        for att_data in attachments:
            attachment = Attachment(
                message_id=gmail_msg.id,
                filename=att_data["filename"],
                content_type=att_data.get("content_type", "application/pdf"),
                local_path=att_data.get("local_path")
            )
            session.add(attachment)
        
        # 5. Create Print Jobs (if applicable)
        if has_print_job:
            print_job = PrintJob(
                transaction_id=tx_id,
                printer_name="Zebra_GK420d",
                status=PrintStatus.SUCCESS if print_success else PrintStatus.FAILED,
                error_message=None if print_success else "Demo: Simulated print failure",
                attempted_at=datetime.now(timezone.utc)
            )
            session.add(print_job)
        
        # 6. Create Processing Logs
        if processing_logs:
            for log_data in processing_logs:
                log = ProcessingLog(
                    transaction_id=tx_id,
                    step=log_data["step"],
                    status=log_data["status"],
                    details=log_data.get("details"),
                    timestamp=datetime.now(timezone.utc) - timedelta(minutes=log_data.get("offset_minutes", 0))
                )
                session.add(log)
        
        session.commit()
        return tx_id


def init_demo_database():
    """Initialize demo database with comprehensive sample data."""
    print("\n" + "="*60)
    print("VintedOS Demo Database Initialization")
    print("="*60 + "\n")
    
    # Initialize demo database service
    db = DatabaseService(demo_mode=True)
    
    print("Creating pipeline runs...")
    
    # Create pipeline runs
    run1_id = create_pipeline_run(db, start_offset_hours=48)  # Completed run from 2 days ago
    run2_id = create_pipeline_run(db, start_offset_hours=24)  # Completed run from yesterday
    run3_id = create_pipeline_run(db, start_offset_hours=0)   # Current run
    
    print(f"  ✓ Created {3} pipeline runs")
    
    print("\nCreating demo transactions with full schema...")
    
    # Transaction 1: Single item, completed successfully
    tx1_id = create_full_transaction(
        db,
        vinted_order_id="1234567890",
        customer_name="Alice Johnson",
        items=[
            {"title": "Vintage Nike Hoodie - Grey Size L", "price": 25.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run1_id,
        gmail_id="demo_gmail_001",
        gmail_subject="Vinted: Order #1234567890 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-1234567890.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/1234567890_Vinted-Label.pdf"
            },
            {
                "filename": "Order-return-form-1234567890.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/1234567890_return-form.pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 60},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 59},
            {"step": "label_generation", "status": "success", "offset_minutes": 58},
            {"step": "printing", "status": "success", "offset_minutes": 57}
        ]
    )
    print(f"  ✓ Created transaction 1234567890 (COMPLETED, 1 item)")
    
    # Transaction 2: Multi-item order, completed
    tx2_id = create_full_transaction(
        db,
        vinted_order_id="9876543210",
        customer_name="Bob Smith",
        items=[
            {"title": "Adidas Track Pants - Black Size M", "price": 15.00},
            {"title": "Puma T-Shirt - White Size S", "price": 10.00},
            {"title": "Reebok Cap - Navy One Size", "price": 8.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run1_id,
        gmail_id="demo_gmail_002",
        gmail_subject="Vinted: Order #9876543210 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-9876543210.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/9876543210_Vinted-Label.pdf"
            },
            {
                "filename": "Order-return-form-9876543210.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 120},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 119},
            {"step": "label_generation", "status": "success", "offset_minutes": 118},
            {"step": "printing", "status": "success", "offset_minutes": 117}
        ]
    )
    print(f"  ✓ Created transaction 9876543210 (COMPLETED, 3 items)")
    
    # Transaction 3: Printed but not completed
    tx3_id = create_full_transaction(
        db,
        vinted_order_id="5555555555",
        customer_name="Carol Davis",
        items=[
            {"title": "Levi's 501 Jeans - Blue Size 32", "price": 30.00}
        ],
        status=TransactionStatus.PRINTED,
        pipeline_run_id=run2_id,
        gmail_id="demo_gmail_003",
        gmail_subject="Vinted: Order #5555555555 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Digital-Label-5555555555.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/5555555555_label.pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 90},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 89},
            {"step": "label_generation", "status": "success", "offset_minutes": 88},
            {"step": "printing", "status": "success", "offset_minutes": 87}
        ]
    )
    print(f"  ✓ Created transaction 5555555555 (PRINTED, 1 item)")
    
    # Transaction 4: Pending - just fetched
    tx4_id = create_full_transaction(
        db,
        vinted_order_id="7777777777",
        customer_name="David Wilson",
        items=[
            {"title": "H&M Denim Jacket - Blue Size L", "price": 22.00},
            {"title": "Zara Cotton Shirt - White Size M", "price": 12.00}
        ],
        status=TransactionStatus.PENDING,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_004",
        gmail_subject="Vinted: Order #7777777777 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-7777777777.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 5}
        ]
    )
    print(f"  ✓ Created transaction 7777777777 (PENDING, 2 items)")
    
    # Transaction 5: Failed print job
    tx5_id = create_full_transaction(
        db,
        vinted_order_id="8888888888",
        customer_name="Eve Martinez",
        items=[
            {"title": "North Face Fleece - Red Size S", "price": 35.00}
        ],
        status=TransactionStatus.FAILED,
        pipeline_run_id=run2_id,
        gmail_id="demo_gmail_005",
        gmail_subject="Vinted: Order #8888888888 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-8888888888.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=True,
        print_success=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 45},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 44},
            {"step": "label_generation", "status": "success", "offset_minutes": 43},
            {"step": "printing", "status": "error", "details": "Printer offline", "offset_minutes": 42}
        ]
    )
    print(f"  ✓ Created transaction 8888888888 (FAILED, 1 item)")
    
    # Transaction 6: Large multi-item order
    tx6_id = create_full_transaction(
        db,
        vinted_order_id="1111222233",
        customer_name="Frank Thompson",
        items=[
            {"title": "Nike Air Max 90 - White Size 10", "price": 65.00},
            {"title": "Adidas Socks Pack - Mixed Colors", "price": 8.00},
            {"title": "Under Armour Gym Shorts - Black Size L", "price": 18.00},
            {"title": "Champion Hoodie - Grey Size XL", "price": 28.00},
            {"title": "New Balance Cap - Navy", "price": 12.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run2_id,
        gmail_id="demo_gmail_006",
        gmail_subject="Vinted: Order #1111222233 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-1111222233.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/1111222233_label.pdf"
            },
            {
                "filename": "Order-return-form-1111222233.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 150},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 149},
            {"step": "label_generation", "status": "success", "offset_minutes": 148},
            {"step": "multi_item_list", "status": "success", "details": "Generated item list PDF", "offset_minutes": 147},
            {"step": "printing", "status": "success", "offset_minutes": 146}
        ]
    )
    print(f"  ✓ Created transaction 1111222233 (COMPLETED, 5 items)")
    
    # Transaction 7: Parsed but not matched
    tx7_id = create_full_transaction(
        db,
        vinted_order_id="3333444455",
        customer_name="Grace Lee",
        items=[
            {"title": "Vintage Band T-Shirt - Black Size M", "price": 15.00}
        ],
        status=TransactionStatus.PARSED,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_007",
        gmail_subject="Vinted: Order #3333444455 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-3333444455.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 15},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 14}
        ]
    )
    print(f"  ✓ Created transaction 3333444455 (PARSED, 1 item)")
    
    # Transaction 8: Another completed with different customer
    tx8_id = create_full_transaction(
        db,
        vinted_order_id="6666777788",
        customer_name="Henry Brown",
        items=[
            {"title": "Carhartt Work Pants - Tan Size 34", "price": 42.00},
            {"title": "Timberland Boots - Brown Size 11", "price": 85.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run1_id,
        gmail_id="demo_gmail_008",
        gmail_subject="Vinted: Order #6666777788 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-6666777788.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/6666777788_label.pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 180},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 179},
            {"step": "label_generation", "status": "success", "offset_minutes": 178},
            {"step": "printing", "status": "success", "offset_minutes": 177}
        ]
    )
    print(f"  ✓ Created transaction 6666777788 (COMPLETED, 2 items)")
    
    # Transaction 9: Pending multi-item
    tx9_id = create_full_transaction(
        db,
        vinted_order_id="2222333344",
        customer_name="Isabel Garcia",
        items=[
            {"title": "Patagonia Fleece Jacket - Navy Size M", "price": 48.00},
            {"title": "Columbia Hiking Boots - Brown Size 9", "price": 72.00},
            {"title": "North Face Backpack - Green", "price": 35.00}
        ],
        status=TransactionStatus.PENDING,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_009",
        gmail_subject="Vinted: Order #2222333344 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-2222333344.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 2}
        ]
    )
    print(f"  ✓ Created transaction 2222333344 (PENDING, 3 items)")
    
    # Transaction 10: Matched status
    tx10_id = create_full_transaction(
        db,
        vinted_order_id="4444555566",
        customer_name="Jack O'Connor",
        items=[
            {"title": "Ralph Lauren Polo Shirt - White Size L", "price": 22.00}
        ],
        status=TransactionStatus.MATCHED,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_010",
        gmail_subject="Vinted: Order #4444555566 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-4444555566.pdf",
                "content_type": "application/pdf"
            },
            {
                "filename": "Order-return-form-4444555566.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 20},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 19},
            {"step": "matching", "status": "success", "offset_minutes": 18}
        ]
    )
    print(f"  ✓ Created transaction 4444555566 (MATCHED, 1 item)")
    
    # Transaction 11: Large vintage clothing bundle
    tx11_id = create_full_transaction(
        db,
        vinted_order_id="7788990011",
        customer_name="Karen Mitchell",
        items=[
            {"title": "Vintage Levi's 501 Jeans - Blue Size 30", "price": 38.00},
            {"title": "Vintage Band T-Shirt Collection (3 items)", "price": 25.00},
            {"title": "Vintage Nike Windbreaker - Red Size M", "price": 32.00},
            {"title": "Vintage Adidas Track Jacket - Black Size L", "price": 28.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run2_id,
        gmail_id="demo_gmail_011",
        gmail_subject="Vinted: Order #7788990011 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-7788990011.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/7788990011_label.pdf"
            },
            {
                "filename": "Order-return-form-7788990011.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 200},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 199},
            {"step": "label_generation", "status": "success", "offset_minutes": 198},
            {"step": "multi_item_list", "status": "success", "details": "4 items listed", "offset_minutes": 197},
            {"step": "printing", "status": "success", "offset_minutes": 196}
        ]
    )
    print(f"  ✓ Created transaction 7788990011 (COMPLETED, 4 items)")
    
    # Transaction 12: Single expensive item
    tx12_id = create_full_transaction(
        db,
        vinted_order_id="9999000011",
        customer_name="Laura Chen",
        items=[
            {"title": "Canada Goose Parka - Black Size M", "price": 285.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run1_id,
        gmail_id="demo_gmail_012",
        gmail_subject="Vinted: Order #9999000011 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Digital-Label-9999000011.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/9999000011_label.pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 240},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 239},
            {"step": "label_generation", "status": "success", "offset_minutes": 238},
            {"step": "printing", "status": "success", "offset_minutes": 237}
        ]
    )
    print(f"  ✓ Created transaction 9999000011 (COMPLETED, 1 item - high value)")
    
    # Transaction 13: Budget items bundle
    tx13_id = create_full_transaction(
        db,
        vinted_order_id="1212343456",
        customer_name="Mike Rodriguez",
        items=[
            {"title": "Basic White T-Shirt - Size M", "price": 3.50},
            {"title": "Black Socks (3 pairs)", "price": 4.00},
            {"title": "Grey Cotton Shorts - Size L", "price": 6.50}
        ],
        status=TransactionStatus.PRINTED,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_013",
        gmail_subject="Vinted: Order #1212343456 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-1212343456.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/1212343456_label.pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 30},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 29},
            {"step": "label_generation", "status": "success", "offset_minutes": 28},
            {"step": "printing", "status": "success", "offset_minutes": 27}
        ]
    )
    print(f"  ✓ Created transaction 1212343456 (PRINTED, 3 items - budget)")
    
    # Transaction 14: Failed parsing scenario
    tx14_id = create_full_transaction(
        db,
        vinted_order_id="5566778899",
        customer_name="Nancy Patel",
        items=[
            {"title": "Designer Handbag - Brown Leather", "price": 145.00}
        ],
        status=TransactionStatus.FAILED,
        pipeline_run_id=run3_id,
        gmail_id="demo_gmail_014",
        gmail_subject="Vinted: Order #5566778899 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-5566778899.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=False,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 10},
            {"step": "pdf_parse", "status": "error", "details": "Corrupted PDF file", "offset_minutes": 9}
        ]
    )
    print(f"  ✓ Created transaction 5566778899 (FAILED, 1 item - parse error)")
    
    # Transaction 15: Sportswear collection
    tx15_id = create_full_transaction(
        db,
        vinted_order_id="3344556677",
        customer_name="Oscar Kim",
        items=[
            {"title": "Nike Running Shoes - Black Size 10.5", "price": 52.00},
            {"title": "Under Armour Compression Shirt - Blue Size M", "price": 18.00},
            {"title": "Adidas Running Shorts - Black Size L", "price": 16.00},
            {"title": "Sports Water Bottle - Blue", "price": 5.00}
        ],
        status=TransactionStatus.COMPLETED,
        pipeline_run_id=run2_id,
        gmail_id="demo_gmail_015",
        gmail_subject="Vinted: Order #3344556677 - Shipping Label",
        attachments=[
            {
                "filename": "Vinted-Label-3344556677.pdf",
                "content_type": "application/pdf",
                "local_path": "static/services/labels/3344556677_label.pdf"
            },
            {
                "filename": "Order-return-form-3344556677.pdf",
                "content_type": "application/pdf"
            }
        ],
        has_print_job=True,
        print_success=True,
        processing_logs=[
            {"step": "gmail_fetch", "status": "success", "offset_minutes": 100},
            {"step": "pdf_parse", "status": "success", "offset_minutes": 99},
            {"step": "label_generation", "status": "success", "offset_minutes": 98},
            {"step": "multi_item_list", "status": "success", "details": "4 items - sportswear", "offset_minutes": 97},
            {"step": "printing", "status": "success", "offset_minutes": 96}
        ]
    )
    print(f"  ✓ Created transaction 3344556677 (COMPLETED, 4 items - sportswear)")
    
    # Update pipeline run statistics
    with Session(db.engine) as session:
        for run_id in [run1_id, run2_id, run3_id]:
            from sqlmodel import select
            run = session.exec(select(PipelineRun).where(PipelineRun.id == run_id)).first()
            if run:
                transactions = session.exec(
                    select(Transaction).where(Transaction.pipeline_run_id == run_id)
                ).all()
                run.items_processed = sum(len(tx.items) for tx in transactions)
                run.items_failed = sum(1 for tx in transactions if tx.status == TransactionStatus.FAILED)
        session.commit()
    
    print("\n" + "="*60)
    print("Demo database initialized successfully!")
    print("="*60)
    print("\nStatistics:")
    print(f"  • Pipeline Runs: 3")
    print(f"  • Transactions: 15")
    print(f"  • Total Items: 33")
    print(f"  • Completed: 7")
    print(f"  • Printed: 2")
    print(f"  • Matched: 1")
    print(f"  • Parsed: 1")
    print(f"  • Pending: 2")
    print(f"  • Failed: 2")
    print("\nYou can now test the system with:")
    print("  python run.py --demo")
    print("  python agent.py --demo")
    print("\n")


if __name__ == "__main__":
    try:
        init_demo_database()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error initializing demo database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
