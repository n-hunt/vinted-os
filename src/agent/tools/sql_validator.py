"""
SQL Safety Validator

Validates SQL queries for safe execution in read-only contexts.
Prevents dangerous operations and SQL injection attacks.

Design Philosophy:
- Whitelist approach (only allow SELECT)
- Multiple validation layers
- Detailed error reporting
- No false negatives (better to block safe query than allow unsafe one)
"""

import re
import logging
from typing import List, Optional, Set
from .schemas import SQLValidationResult

logger = logging.getLogger(__name__)


class SQLValidator:
    """
    SQL query safety validator for read-only database access.
    
    Validation checks:
    1. Query type (only SELECT allowed)
    2. Dangerous keywords (DROP, DELETE, UPDATE, etc.)
    3. SQL injection patterns
    4. Statement separators (no multi-statement)
    5. Table extraction for auditing
    """
    
    # Dangerous SQL keywords that should never appear
    FORBIDDEN_KEYWORDS = {
        # DDL (Data Definition Language)
        'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'RENAME',
        
        # DML (Data Manipulation Language)
        'INSERT', 'UPDATE', 'DELETE', 'REPLACE', 'MERGE',
        
        # DCL (Data Control Language)
        'GRANT', 'REVOKE',
        
        # TCL (Transaction Control Language)
        'COMMIT', 'ROLLBACK', 'SAVEPOINT',
        
        # Other dangerous operations
        'EXEC', 'EXECUTE', 'CALL',
        'PRAGMA',  # SQLite-specific settings
        'ATTACH', 'DETACH',  # SQLite database attachment
    }
    
    # Allowed query types (only SELECT)
    ALLOWED_QUERY_TYPES = {'SELECT'}
    
    # SQL injection patterns (common attack vectors)
    INJECTION_PATTERNS = [
        r';\s*DROP',  # Statement separator + DROP
        r';\s*DELETE',  # Statement separator + DELETE
        r'--',  # SQL comment (can be used to bypass conditions)
        r'/\*',  # Block comment start
        r'\*/',  # Block comment end
        r'xp_',  # SQL Server extended procedures
        r'sp_',  # SQL Server stored procedures
        r'\bOR\b.*=.*',  # OR-based injection (e.g., OR 1=1)
        r'\bAND\b.*=.*',  # AND-based injection
        r'UNION\s+SELECT',  # UNION-based injection
        r'CONCAT\s*\(',  # String concatenation (can build dynamic SQL)
        r'CHAR\s*\(',  # Character conversion (obfuscation)
        r'0x[0-9A-Fa-f]+',  # Hex literals (obfuscation)
    ]
    
    def __init__(self):
        """Initialize SQL validator."""
        self.injection_regex = re.compile(
            '|'.join(self.INJECTION_PATTERNS),
            re.IGNORECASE
        )
    
    def validate(self, query: str) -> SQLValidationResult:
        """
        Validate SQL query for safety.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            SQLValidationResult with safety status and details
        """
        if not query or not query.strip():
            return SQLValidationResult(
                is_safe=False,
                issues=["Empty query"],
                query_type=None,
                affected_tables=[]
            )
        
        # Normalize query (remove extra whitespace, convert to uppercase for checks)
        normalized = self._normalize_query(query)
        issues: List[str] = []
        
        # Check 1: Detect query type
        query_type = self._detect_query_type(normalized)
        if query_type not in self.ALLOWED_QUERY_TYPES:
            issues.append(f"Query type '{query_type}' not allowed (only SELECT permitted)")
        
        # Check 2: Scan for forbidden keywords
        forbidden_found = self._check_forbidden_keywords(normalized)
        if forbidden_found:
            issues.append(f"Forbidden keywords detected: {', '.join(forbidden_found)}")
        
        # Check 3: Check for SQL injection patterns
        injection_issues = self._check_injection_patterns(query)
        if injection_issues:
            issues.extend(injection_issues)
        
        # Check 4: Ensure single statement (no semicolons except at end)
        if self._has_multiple_statements(query):
            issues.append("Multiple statements detected (only single SELECT allowed)")
        
        # Check 5: Validate parentheses balance
        if not self._check_parentheses_balance(query):
            issues.append("Unbalanced parentheses (possible syntax error)")
        
        # Extract tables (even if query is unsafe, useful for logging)
        tables = self._extract_tables(normalized)
        
        # Determine if safe
        is_safe = len(issues) == 0 and query_type in self.ALLOWED_QUERY_TYPES
        
        result = SQLValidationResult(
            is_safe=is_safe,
            issues=issues,
            query_type=query_type,
            affected_tables=tables
        )
        
        # Log validation result
        if not is_safe:
            logger.warning(
                f"Unsafe SQL query blocked: {issues}. Query: {query[:100]}..."
            )
        else:
            logger.debug(f"SQL query validated successfully: {tables}")
        
        return result
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for analysis.
        
        Args:
            query: Raw SQL query
            
        Returns:
            Normalized query (uppercase, single spaces)
        """
        # Remove leading/trailing whitespace
        normalized = query.strip()
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Convert to uppercase for keyword matching
        normalized = normalized.upper()
        
        return normalized
    
    def _detect_query_type(self, normalized_query: str) -> Optional[str]:
        """
        Detect the type of SQL query.
        
        Args:
            normalized_query: Uppercase, normalized query
            
        Returns:
            Query type (SELECT, INSERT, etc.) or None
        """
        # Match first keyword in query
        match = re.match(r'^(\w+)', normalized_query)
        if match:
            return match.group(1)
        return None
    
    def _check_forbidden_keywords(self, normalized_query: str) -> Set[str]:
        """
        Check for forbidden SQL keywords.
        
        Args:
            normalized_query: Uppercase, normalized query
            
        Returns:
            Set of forbidden keywords found
        """
        found = set()
        
        for keyword in self.FORBIDDEN_KEYWORDS:
            # Use word boundaries to avoid false positives
            # (e.g., "UPDATE" in a column name "LAST_UPDATE")
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, normalized_query):
                found.add(keyword)
        
        return found
    
    def _check_injection_patterns(self, query: str) -> List[str]:
        """
        Check for common SQL injection patterns.
        
        Args:
            query: Raw SQL query (not normalized, case-sensitive)
            
        Returns:
            List of injection issues found
        """
        issues = []
        
        # Check for comments (-- and /* */)
        if '--' in query:
            issues.append("SQL comment detected (--)")
        
        if '/*' in query or '*/' in query:
            issues.append("SQL block comment detected (/* */)")
        
        # Check for suspicious OR/AND patterns
        # Match: OR 1=1, OR '1'='1', etc.
        or_injection = re.search(
            r"\bOR\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?",
            query,
            re.IGNORECASE
        )
        if or_injection:
            issues.append(f"Suspicious OR pattern detected: {or_injection.group()}")
        
        # Check for UNION SELECT (UNION-based injection)
        if re.search(r'\bUNION\b.*\bSELECT\b', query, re.IGNORECASE):
            issues.append("UNION SELECT detected (possible injection)")
        
        # Check for hex literals (obfuscation technique)
        if re.search(r'0x[0-9A-Fa-f]+', query):
            issues.append("Hex literal detected (possible obfuscation)")
        
        # Check for CHAR/CONCAT (dynamic SQL building)
        if re.search(r'\b(CHAR|CONCAT)\s*\(', query, re.IGNORECASE):
            issues.append("String manipulation function detected (CHAR/CONCAT)")
        
        return issues
    
    def _has_multiple_statements(self, query: str) -> bool:
        """
        Check if query contains multiple statements.
        
        Args:
            query: Raw SQL query
            
        Returns:
            True if multiple statements detected
        """
        # Remove string literals (to avoid false positives from semicolons in strings)
        query_no_strings = re.sub(r"'[^']*'", '', query)
        query_no_strings = re.sub(r'"[^"]*"', '', query_no_strings)
        
        # Count semicolons
        semicolons = query_no_strings.count(';')
        
        # Allow one semicolon at the end (common practice)
        if semicolons == 0:
            return False
        elif semicolons == 1 and query_no_strings.rstrip().endswith(';'):
            return False
        else:
            return True
    
    def _check_parentheses_balance(self, query: str) -> bool:
        """
        Check if parentheses are balanced in query.
        
        Args:
            query: Raw SQL query
            
        Returns:
            True if balanced, False otherwise
        """
        count = 0
        for char in query:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:  # More closing than opening
                    return False
        
        return count == 0  # Should end with 0 (all opened are closed)
    
    def _extract_tables(self, normalized_query: str) -> List[str]:
        """
        Extract table names from SELECT query.
        
        Args:
            normalized_query: Uppercase, normalized query
            
        Returns:
            List of table names found
        """
        tables = []
        
        # Pattern: FROM table_name or JOIN table_name
        # Handles: FROM table, FROM table alias, JOIN table ON ...
        from_pattern = r'\bFROM\s+(\w+)'
        join_pattern = r'\bJOIN\s+(\w+)'
        
        # Find all FROM clauses
        from_matches = re.findall(from_pattern, normalized_query)
        tables.extend(from_matches)
        
        # Find all JOIN clauses
        join_matches = re.findall(join_pattern, normalized_query)
        tables.extend(join_matches)
        
        # Remove duplicates and return
        return list(set(tables))


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

# Global validator instance (singleton)
_validator: Optional[SQLValidator] = None


def get_validator() -> SQLValidator:
    """Get or create SQLValidator singleton."""
    global _validator
    if _validator is None:
        _validator = SQLValidator()
        logger.info("Initialized SQL validator")
    return _validator


def validate_sql(query: str) -> SQLValidationResult:
    """
    Validate SQL query for safety (convenience function).
    
    Args:
        query: SQL query string
        
    Returns:
        SQLValidationResult with safety status
    """
    validator = get_validator()
    return validator.validate(query)


def is_query_safe(query: str) -> bool:
    """
    Quick check if SQL query is safe to execute.
    
    Args:
        query: SQL query string
        
    Returns:
        True if safe, False otherwise
    """
    result = validate_sql(query)
    return result.is_safe


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Test cases
    test_queries = [
        # Safe queries
        ("SELECT * FROM transactions", True),
        ("SELECT id, name FROM customers WHERE status = 'active'", True),
        ("SELECT COUNT(*) FROM items", True),
        ("SELECT t.id, c.name FROM transactions t JOIN customers c ON t.customer_id = c.id", True),
        
        # Unsafe queries
        ("DROP TABLE transactions", False),
        ("DELETE FROM customers WHERE id = 1", False),
        ("SELECT * FROM users; DROP TABLE users;", False),
        ("SELECT * FROM users WHERE name = '' OR 1=1 --'", False),
        ("INSERT INTO logs VALUES ('test')", False),
        ("UPDATE transactions SET status = 'completed'", False),
        ("SELECT * FROM users UNION SELECT * FROM admin", False),
        
        # Edge cases
        ("", False),
        ("   ", False),
        ("SELECT * FROM (SELECT * FROM nested)", True),  # Subquery is okay
    ]
    
    print("SQL Validator Test Results")
    print("=" * 60)
    
    for query, expected_safe in test_queries:
        result = validate_sql(query)
        status = "SUCCESS: PASS" if result.is_safe == expected_safe else "ERROR: FAIL"
        
        print(f"\n{status}")
        print(f"Query: {query[:60]}...")
        print(f"Safe: {result.is_safe} (expected: {expected_safe})")
        print(f"Type: {result.query_type}")
        print(f"Tables: {result.affected_tables}")
        if result.issues:
            print(f"Issues: {result.issues}")
