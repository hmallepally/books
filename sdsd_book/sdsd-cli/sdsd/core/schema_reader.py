"""Schema Reader.

Implements the "Schema Retrieval" step from Chapter 7, §7.2:
queries the project for database schema (DDL) definitions and
injects the exact table definitions into the assembled prompt.

Supports:
- Raw .sql files in a `migrations/`, `schema/`, or `db/` directory
- Alembic-style Python migration files (extracts op.create_table calls)
"""

from __future__ import annotations

import re
from pathlib import Path

from sdsd.models.spec import SchemaDefinition


# Regex to find CREATE TABLE statements in SQL files
_CREATE_TABLE_RE = re.compile(
    r"(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w.`\"\[\]]+\s*\([^;]+\);?)",
    re.IGNORECASE | re.DOTALL,
)

# Regex to extract table name from CREATE TABLE
_TABLE_NAME_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[\]]?([\w.]+)[`\"\]\]]?",
    re.IGNORECASE,
)


def find_schemas(target: Path) -> list[SchemaDefinition]:
    """Discover and extract database schemas relevant to a target path.

    Searches for SQL migration files and raw schema definitions in
    conventional directories. This implements the "Schema Retrieval"
    described in Ch.7, §7.2, step 2.

    Args:
        target: The target file or directory being worked on.

    Returns:
        List of discovered schema definitions.
    """
    if target.is_file():
        scan_root = target.parent
    else:
        scan_root = target

    schemas = []

    # Strategy 1: Walk up to find migrations/ or schema/ directories
    search_dirs = _find_schema_directories(scan_root)

    for schema_dir in search_dirs:
        for sql_file in sorted(schema_dir.rglob("*.sql")):
            schemas.extend(_extract_from_sql(sql_file))

        # Also check Alembic-style Python migrations
        for py_file in sorted(schema_dir.rglob("*.py")):
            schemas.extend(_extract_from_alembic(py_file))

    # Strategy 2: Check for .sql files directly in the target directory
    if target.is_dir():
        for sql_file in sorted(target.glob("*.sql")):
            schemas.extend(_extract_from_sql(sql_file))

    return schemas


def _find_schema_directories(start: Path) -> list[Path]:
    """Search for conventional schema/migration directories."""
    schema_dir_names = {"migrations", "schema", "db", "sql", "database"}
    found = []

    current = start.resolve()
    for _ in range(10):
        for name in schema_dir_names:
            candidate = current / name
            if candidate.is_dir():
                found.append(candidate)

        parent = current.parent
        if parent == current:
            break
        current = parent

    return found


def _extract_from_sql(filepath: Path) -> list[SchemaDefinition]:
    """Extract CREATE TABLE statements from a .sql file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    schemas = []
    for match in _CREATE_TABLE_RE.finditer(content):
        ddl = match.group(1).strip()
        table_match = _TABLE_NAME_RE.search(ddl)
        table_name = table_match.group(1) if table_match else "unknown"

        schemas.append(SchemaDefinition(
            table_name=table_name,
            ddl=ddl,
            source_file=str(filepath),
        ))

    return schemas


def _extract_from_alembic(filepath: Path) -> list[SchemaDefinition]:
    """Extract table schemas from Alembic-style Python migration files.

    Looks for `op.create_table(...)` calls and extracts the table name.
    Provides a simplified DDL representation.
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    # Quick check — skip files without create_table
    if "create_table" not in content:
        return []

    schemas = []
    # Find op.create_table('table_name', ...) calls
    pattern = re.compile(r"op\.create_table\(\s*['\"](\w+)['\"]", re.MULTILINE)
    for match in pattern.finditer(content):
        table_name = match.group(1)
        schemas.append(SchemaDefinition(
            table_name=table_name,
            ddl=f"-- Alembic migration defines table: {table_name}\n-- See: {filepath.name}",
            source_file=str(filepath),
        ))

    return schemas
