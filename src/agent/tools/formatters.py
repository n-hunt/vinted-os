"""
Tool Result Formatters for LLM Consumption

Converts tool execution results into LLM-friendly formats:
- Natural language summaries
- Markdown tables
- Token-efficient representations
- Highlighted key information

Design Philosophy:
- Information density (max info, min tokens)
- Scannable structure (tables, bullets, headers)
- Contextual formatting (different tools need different formats)
- Truncation awareness (show when data is truncated)
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


class ToolResultFormatter:
    """
    Formats tool results for LLM consumption.
    
    Provides multiple output formats:
    - text: Natural language summary
    - markdown: Structured markdown with tables
    - compact: Token-efficient representation
    """
    
    def __init__(self, max_items: int = 10, max_text_length: int = 500):
        """
        Initialize formatter.
        
        Args:
            max_items: Maximum number of items to display in lists
            max_text_length: Maximum length for text fields
        """
        self.max_items = max_items
        self.max_text_length = max_text_length
    
    # ============================================================
    # MAIN FORMATTING ENTRY POINT
    # ============================================================
    
    def format(
        self,
        tool_result: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """
        Format tool result for LLM consumption.
        
        Args:
            tool_result: ToolResult dict from query tool
            format_type: Output format ('text', 'markdown', 'compact')
            
        Returns:
            Formatted string
        """
        # Handle error results
        if not tool_result.get("success", False):
            return self._format_error(tool_result, format_type)
        
        # Get data and metadata
        data = tool_result.get("data")
        metadata = tool_result.get("metadata", {})
        
        # Route to appropriate formatter based on data structure
        if data is None:
            return self._format_empty(metadata, format_type)
        elif isinstance(data, list):
            return self._format_list(data, metadata, format_type)
        elif isinstance(data, dict):
            return self._format_dict(data, metadata, format_type)
        else:
            return str(data)
    
    # ============================================================
    # ERROR & EMPTY FORMATTERS
    # ============================================================
    
    def _format_error(self, tool_result: Dict[str, Any], format_type: str) -> str:
        """Format error result."""
        error = tool_result.get("error", "Unknown error")
        metadata = tool_result.get("metadata", {})
        exec_time = metadata.get("execution_time_seconds", 0)
        
        if format_type == "compact":
            return f"ERROR: {error}"
        
        return f"""ERROR: **Tool Execution Failed**

**Error:** {error}

**Execution Time:** {exec_time:.3f}s
"""
    
    def _format_empty(self, metadata: Dict[str, Any], format_type: str) -> str:
        """Format empty/no results."""
        count = metadata.get("count", 0)
        exec_time = metadata.get("execution_time_seconds", 0)
        
        if format_type == "compact":
            return f"SUCCESS: No results found ({exec_time:.3f}s)"
        
        return f"""SUCCESS: **Query Successful**

**Results:** {count} items found

**Execution Time:** {exec_time:.3f}s
"""
    
    # ============================================================
    # LIST FORMATTERS (multiple records)
    # ============================================================
    
    def _format_list(
        self,
        data: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format list of records."""
        if not data:
            return self._format_empty(metadata, format_type)
        
        # Detect data type from first record
        first_record = data[0]
        
        # Transaction summaries
        if "vinted_order_id" in first_record and "items_count" in first_record:
            return self._format_transaction_list(data, metadata, format_type)
        
        # Print failures
        elif "printer_name" in first_record and "error_message" in first_record:
            return self._format_print_failures(data, metadata, format_type)
        
        # Revenue periods
        elif "period" in first_record and "revenue" in first_record:
            return self._format_revenue_periods(data, metadata, format_type)
        
        # Top selling items
        elif "times_sold" in first_record and "total_revenue" in first_record:
            return self._format_top_items(data, metadata, format_type)
        
        # Pipeline runs
        elif "start_time" in first_record and "items_processed" in first_record:
            return self._format_pipeline_runs(data, metadata, format_type)
        
        # Generic list
        else:
            return self._format_generic_list(data, metadata, format_type)
    
    def _format_transaction_list(
        self,
        transactions: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format transaction list."""
        count = len(transactions)
        total_count = metadata.get("count", count)
        truncated = total_count > self.max_items
        display_items = transactions[:self.max_items]
        
        if format_type == "compact":
            lines = [f"SUCCESS: Found {total_count} transaction(s)"]
            for tx in display_items[:3]:  # Show max 3 in compact mode
                lines.append(
                    f"  • #{tx['vinted_order_id']}: {tx['customer_name']} "
                    f"({tx['status']}, ${tx['total_value']:.2f})"
                )
            if truncated:
                lines.append(f"  ... and {total_count - 3} more")
            return "\n".join(lines)
        
        # Markdown table format
        output = [f"SUCCESS: **Found {total_count} Transaction(s)**\n"]
        
        # Build table
        output.append("| Order ID | Customer | Status | Items | Value | Created |")
        output.append("|----------|----------|--------|-------|-------|---------|")
        
        for tx in display_items:
            created = self._format_datetime(tx.get("created_at", ""))
            has_errors = "WARNING: " if tx.get("has_print_errors") else ""
            
            output.append(
                f"| {tx['vinted_order_id']} | "
                f"{self._truncate(tx['customer_name'], 20)} | "
                f"{has_errors}{tx['status']} | "
                f"{tx['items_count']} | "
                f"${tx['total_value']:.2f} | "
                f"{created} |"
            )
        
        if truncated:
            output.append(f"\n*Showing {len(display_items)} of {total_count} results*")
        
        # Add execution time
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_print_failures(
        self,
        failures: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format print failure list."""
        count = len(failures)
        display_items = failures[:self.max_items]
        
        if format_type == "compact":
            lines = [f"WARNING: Found {count} print failure(s)"]
            for fail in display_items[:3]:
                lines.append(
                    f"  • {fail['vinted_order_id']} on {fail['printer_name']}: "
                    f"{self._truncate(fail.get('error_message', 'Unknown error'), 50)}"
                )
            return "\n".join(lines)
        
        # Detailed format
        output = [f"WARNING: **Found {count} Print Failure(s)**\n"]
        
        for fail in display_items:
            attempted = self._format_datetime(fail.get("attempted_at", ""))
            error_msg = fail.get("error_message") or "Unknown error"
            
            output.append(f"**Order {fail['vinted_order_id']}** ({fail['customer_name']})")
            output.append(f"- **Printer:** {fail['printer_name']}")
            output.append(f"- **Error:** {self._truncate(error_msg, 100)}")
            output.append(f"- **Attempted:** {attempted}")
            output.append(f"- **Retry Count:** {fail.get('retry_count', 0)}")
            output.append("")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_revenue_periods(
        self,
        periods: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format revenue by period."""
        total_revenue = metadata.get("total_revenue", 0)
        total_transactions = metadata.get("total_transactions", 0)
        grouping = metadata.get("grouping", "period")
        
        if format_type == "compact":
            return (
                f"SUCCESS: Revenue: ${total_revenue:.2f} "
                f"across {total_transactions} transactions ({grouping}ly)"
            )
        
        # Markdown table
        output = [
            f"SUCCESS: **Revenue Report** (grouped by {grouping})\n",
            f"**Total Revenue:** ${total_revenue:.2f}",
            f"**Total Transactions:** {total_transactions}\n",
            "| Period | Revenue | Transactions | Items | Avg/Transaction |",
            "|--------|---------|--------------|-------|-----------------|"
        ]
        
        for period in periods[:self.max_items]:
            # QueryService may not return transaction_count and item_count
            tx_count = period.get('transaction_count', 'N/A')
            item_count = period.get('item_count', 'N/A')
            avg_revenue = (
                period['revenue'] / period['transaction_count']
                if period.get('transaction_count', 0) > 0 else 0
            )
            output.append(
                f"| {period['period']} | "
                f"${period['revenue']:.2f} | "
                f"{tx_count} | "
                f"{item_count} | "
                f"${avg_revenue:.2f} |"
            )
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_top_items(
        self,
        items: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format top-selling items."""
        count = len(items)
        
        if format_type == "compact":
            lines = [f"SUCCESS: Top {count} selling items:"]
            for i, item in enumerate(items[:5], 1):
                title = item.get('item_title', item.get('title', 'Unknown'))
                lines.append(
                    f"  {i}. {self._truncate(title, 40)}: "
                    f"{item['times_sold']}× (${item['total_revenue']:.2f})"
                )
            return "\n".join(lines)
        
        # Markdown table
        output = [
            f"SUCCESS: **Top {count} Selling Items**\n",
            "| Rank | Item | Times Sold | Total Revenue | Avg Price |",
            "|------|------|------------|---------------|-----------|"
        ]
        
        for i, item in enumerate(items[:self.max_items], 1):
            title = item.get('item_title', item.get('title', 'Unknown'))
            avg_price = item.get('average_price', item.get('avg_price', 0))
            output.append(
                f"| {i} | "
                f"{self._truncate(title, 40)} | "
                f"{item['times_sold']} | "
                f"${item['total_revenue']:.2f} | "
                f"${avg_price:.2f} |"
            )
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_pipeline_runs(
        self,
        runs: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format pipeline run history."""
        count = len(runs)
        
        if format_type == "compact":
            recent = runs[0] if runs else None
            if recent:
                items = recent.get('items_processed', 0)
                duration = recent.get('duration_seconds', 0) or 0
                return (
                    f"SUCCESS: Last run: {recent['status']} "
                    f"({items} items, "
                    f"{duration:.1f}s)"
                )
            return "SUCCESS: No pipeline runs found"
        
        # Detailed format
        output = [f"SUCCESS: **Pipeline Run History** ({count} runs)\n"]
        
        for run in runs[:self.max_items]:
            started = self._format_datetime(run.get("start_time", ""))
            duration = run.get("duration_seconds", 0) or 0
            status_icon = "SUCCESS" if run["status"] == "completed" else "WARNING"
            
            output.append(f"{status_icon} **Run #{run['id']}** - {run['status']}")
            output.append(f"- **Started:** {started}")
            output.append(f"- **Duration:** {duration:.1f}s")
            output.append(f"- **Processed:** {run.get('items_processed', 0)} items")
            if run.get('items_failed', 0) > 0:
                output.append(f"- **Failed:** {run['items_failed']} items")
            output.append("")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_generic_list(
        self,
        data: List[Dict],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format generic list of records."""
        count = len(data)
        
        output = [f"SUCCESS: **Found {count} Record(s)**\n"]
        
        for i, record in enumerate(data[:self.max_items], 1):
            output.append(f"**Record {i}:**")
            for key, value in record.items():
                if isinstance(value, (list, dict)):
                    output.append(f"- **{key}:** {len(value)} items")
                else:
                    output.append(f"- **{key}:** {self._truncate(str(value), 100)}")
            output.append("")
        
        if count > self.max_items:
            output.append(f"*Showing {self.max_items} of {count} results*")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    # ============================================================
    # DICT FORMATTERS (single record)
    # ============================================================
    
    def _format_dict(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format single record or structured data."""
        # Transaction full details
        if "vinted_order_id" in data and "items" in data:
            return self._format_transaction_full(data, metadata, format_type)
        
        # Pipeline stats
        elif "total_runs" in data and "success_rate" in data:
            return self._format_pipeline_stats(data, metadata, format_type)
        
        # Dashboard summary
        elif "total_transactions" in data and "system_health" in data:
            return self._format_dashboard(data, metadata, format_type)
        
        # Processing errors by step
        elif all(isinstance(v, list) for v in data.values()):
            return self._format_errors_by_step(data, metadata, format_type)
        
        # Generic dict
        else:
            return self._format_generic_dict(data, metadata, format_type)
    
    def _format_transaction_full(
        self,
        transaction: Dict[str, Any],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format full transaction details."""
        if format_type == "compact":
            return (
                f"SUCCESS: Transaction {transaction['vinted_order_id']}: "
                f"{transaction['status']}, {len(transaction['items'])} items, "
                f"${transaction['total_value']:.2f}"
            )
        
        # Detailed format
        output = [
            f"SUCCESS: **Transaction Details**\n",
            f"**Order ID:** {transaction['vinted_order_id']}",
            f"**Customer:** {transaction['customer_name']}",
            f"**Status:** {transaction['status']}",
            f"**Total Value:** ${transaction['total_value']:.2f}",
            f"**Created:** {self._format_datetime(transaction['created_at'])}",
            f"**Updated:** {self._format_datetime(transaction['updated_at'])}",
        ]
        
        if transaction.get("label_path"):
            output.append(f"**Label:** {transaction['label_path']}")
        
        # Items
        items = transaction.get("items", [])
        output.append(f"\n**Items:** ({len(items)})")
        for item in items[:5]:
            output.append(
                f"- {self._truncate(item['title'], 50)}: "
                f"${item.get('price', 0):.2f} {item.get('currency', 'EUR')}"
            )
        if len(items) > 5:
            output.append(f"  ... and {len(items) - 5} more items")
        
        # Print jobs
        print_jobs = transaction.get("print_jobs", [])
        if print_jobs:
            output.append(f"\n**Print Jobs:** ({len(print_jobs)})")
            for pj in print_jobs[:3]:
                status_icon = "SUCCESS" if pj["status"] == "completed" else "WARNING"
                output.append(
                    f"- {status_icon} {pj['printer_name']}: {pj['status']}"
                )
                if pj.get("error_message"):
                    output.append(f"  Error: {self._truncate(pj['error_message'], 80)}")
        
        # Logs
        logs = transaction.get("logs", [])
        if logs:
            output.append(f"\n**Processing Logs:** ({len(logs)})")
            for log in logs[-5:]:  # Show last 5
                output.append(f"- [{log['step']}] {log['status']}")
                if log.get("details"):
                    output.append(f"  {self._truncate(log['details'], 60)}")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_pipeline_stats(
        self,
        stats: Dict[str, Any],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format pipeline statistics."""
        if format_type == "compact":
            return (
                f"SUCCESS: Pipeline: {stats['success_rate']:.1f}% success rate, "
                f"{stats['total_runs']} runs, "
                f"{stats['total_transactions_processed']} transactions"
            )
        
        # Detailed format
        avg_duration = stats.get("avg_duration_seconds") or 0
        
        output = [
            "SUCCESS: **Pipeline Performance Statistics**\n",
            "**Run Summary:**",
            f"- Total Runs: {stats['total_runs']}",
            f"- Successful: {stats['successful_runs']}",
            f"- Failed: {stats['failed_runs']}",
            f"- Success Rate: {stats['success_rate']:.1f}%",
            f"- Avg Duration: {avg_duration:.1f}s",
            "\n**Processing Summary:**",
            f"- Transactions Processed: {stats['total_transactions_processed']}",
            f"- Items Processed: {stats['total_items_processed']}",
            f"- Labels Printed: {stats['total_labels_printed']}",
            f"- Failed Prints: {stats['failed_prints']}",
            f"- Print Success Rate: {stats['print_success_rate']:.1f}%",
        ]
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_dashboard(
        self,
        dashboard: Dict[str, Any],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format dashboard summary."""
        if format_type == "compact":
            return (
                f"SUCCESS: System: {dashboard['system_health']}, "
                f"{dashboard['total_transactions']} transactions, "
                f"${dashboard['total_revenue']:.2f} revenue"
            )
        
        # Visual dashboard
        health_icon = "SUCCESS" if dashboard["system_health"] == "healthy" else "WARNING"
        
        output = [
            f"{health_icon} **System Dashboard**\n",
            "**Transaction Summary:**",
            f"- Total: {dashboard['total_transactions']}",
            f"- Pending: {dashboard['pending_transactions']}",
            f"- Completed: {dashboard['completed_transactions']}",
            f"- Failed: {dashboard['failed_transactions']}",
            "\n**Revenue & Items:**",
            f"- Total Revenue: ${dashboard['total_revenue']:.2f}",
            f"- Total Items: {dashboard['total_items']}",
            "\n**System Health:**",
            f"- Status: {dashboard['system_health']}",
            f"- Recent Print Failures: {dashboard['recent_print_failures']}",
        ]
        
        if dashboard.get("last_pipeline_run"):
            last_run = self._format_datetime(dashboard["last_pipeline_run"])
            output.append(f"- Last Pipeline Run: {last_run}")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_errors_by_step(
        self,
        errors_dict: Dict[str, List],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format processing errors grouped by step."""
        total_errors = metadata.get("total_errors", 0)
        
        if format_type == "compact":
            steps = list(errors_dict.keys())
            return f"WARNING: {total_errors} errors across {len(steps)} steps: {', '.join(steps)}"
        
        # Detailed format
        output = [f"WARNING: **Processing Errors by Step** ({total_errors} total)\n"]
        
        for step, errors in errors_dict.items():
            output.append(f"**{step}:** ({len(errors)} errors)")
            for error in errors[:3]:  # Show max 3 per step
                details = error.get("details", "No details")
                timestamp = self._format_datetime(error.get("timestamp", ""))
                output.append(f"- [{timestamp}] {self._truncate(details, 80)}")
            if len(errors) > 3:
                output.append(f"  ... and {len(errors) - 3} more errors")
            output.append("")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    def _format_generic_dict(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        format_type: str
    ) -> str:
        """Format generic dictionary data."""
        output = ["SUCCESS: **Query Result**\n"]
        
        for key, value in data.items():
            if isinstance(value, list):
                output.append(f"**{key}:** ({len(value)} items)")
                for item in value[:3]:
                    output.append(f"- {self._truncate(str(item), 80)}")
                if len(value) > 3:
                    output.append(f"  ... and {len(value) - 3} more")
            elif isinstance(value, dict):
                output.append(f"**{key}:** {len(value)} fields")
            else:
                output.append(f"**{key}:** {self._truncate(str(value), 100)}")
        
        exec_time = metadata.get("execution_time_seconds", 0)
        output.append(f"\n*Execution time: {exec_time:.3f}s*")
        
        return "\n".join(output)
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def _format_datetime(self, dt_str: str) -> str:
        """Format ISO datetime string to readable format."""
        if not dt_str:
            return "N/A"
        
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return dt_str[:16]  # Just truncate if parsing fails


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

# Global formatter instance
_formatter: Optional[ToolResultFormatter] = None


def get_formatter() -> ToolResultFormatter:
    """Get or create ToolResultFormatter singleton."""
    global _formatter
    if _formatter is None:
        _formatter = ToolResultFormatter()
    return _formatter


def format_for_llm(
    tool_result: Dict[str, Any],
    format_type: str = "markdown"
) -> str:
    """
    Format tool result for LLM consumption (convenience function).
    
    Args:
        tool_result: ToolResult dict
        format_type: 'text', 'markdown', or 'compact'
        
    Returns:
        Formatted string
    """
    formatter = get_formatter()
    return formatter.format(tool_result, format_type)
