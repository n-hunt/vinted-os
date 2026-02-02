"""
Service Adapters

I/O layer components that wrap external systems:
- Gmail API
- Printer (CUPS)
- Database (future)

These adapters provide clean interfaces and isolate external dependencies.
"""

__all__ = ['gmail', 'printer']
