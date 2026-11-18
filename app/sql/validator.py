from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.errors import SQLValidationError

ALLOWED_TABLES = frozenset({"pedidos", "ventas_mes_pais", "top10_clientes", "clientes_inactivos_90d"})

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable", "TruncateTable",
        "Command", "Copy", "Attach", "Detach", "Pragma", "Set", "Use", "Grant", "Revoke",
        "Transaction", "Commit", "Rollback", "Into", "LoadData", "Install", "Export",
    ) if hasattr(exp, name)
)

FORBIDDEN_FUNCTION_PREFIXES = (
    "read_", "parquet_", "glob", "getenv", "current_setting", "duckdb_", "pragma_", "sqlite_",
    "postgres_", "mysql_", "query", "load", "install", "httpfs", "iceberg_", "delta_",
    "json_execute", "sniff_csv", "which_secret", "pg_",
)


@dataclass
class ValidatedSQL:
    sql: str
    tables: list[str]
    limit_applied: int


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() if hasattr(node, "sql_name") else type(node).__name__).lower()


def validate_sql(sql: str, *, max_rows: int, allowed_tables: frozenset[str] = ALLOWED_TABLES) -> ValidatedSQL:
    if not sql or not sql.strip():
        raise SQLValidationError("SQL vacío")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except ParseError as e:
        raise SQLValidationError("SQL no parseable", details={"error": str(e)[:300]}) from e
    if len(statements) != 1:
        raise SQLValidationError(f"Se esperaba una única sentencia y hay {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        raise SQLValidationError(f"Solo se permiten consultas SELECT (recibido: {type(tree).__name__})")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLValidationError(f"Operación no permitida: {type(node).__name__}")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                raise SQLValidationError(f"Función no permitida: {name}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables: list[str] = []
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func) or not isinstance(table.this, exp.Identifier):
            raise SQLValidationError("No se permiten funciones de tabla en FROM")
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLValidationError(f"No se permiten esquemas/catálogos explícitos: {table.sql()}")
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in allowed_tables:
            raise SQLValidationError(f"Tabla no permitida: {name}",
                                     details={"permitidas": sorted(allowed_tables)})
        tables.append(name)
    if not tables:
        raise SQLValidationError("La consulta no lee de ninguna tabla permitida")

    limit = max_rows
    existing = tree.args.get("limit")
    if isinstance(tree, exp.Select) and existing is not None:
        try:
            limit = min(max_rows, int(existing.expression.name))
        except (AttributeError, ValueError, TypeError):
            limit = max_rows
    inner = tree.sql(dialect="duckdb")
    safe_sql = f"SELECT * FROM ({inner}) AS q LIMIT {limit}"
    return ValidatedSQL(sql=safe_sql, tables=sorted(set(tables)), limit_applied=limit)
