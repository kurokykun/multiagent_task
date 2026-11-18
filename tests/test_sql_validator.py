import pytest

from app.errors import SQLValidationError
from app.sql.executor import ReadOnlyExecutor
from app.sql.validator import validate_sql


@pytest.mark.parametrize("q", [
    "SELECT pais, SUM(importe) FROM pedidos GROUP BY pais",
    "SELECT * FROM top10_clientes",
    "WITH t AS (SELECT cliente, COUNT(*) n FROM pedidos GROUP BY 1) SELECT * FROM t ORDER BY n DESC",
    "SELECT mes FROM ventas_mes_pais UNION SELECT ultima_compra FROM clientes_inactivos_90d",
    "SELECT * FROM pedidos WHERE pais = 'España' -- comentario inocente",
])
def test_valid_queries_pass(q):
    v = validate_sql(q, max_rows=100)
    assert v.sql.endswith("LIMIT 100")


def test_limit_is_enforced_and_capped():
    assert validate_sql("SELECT * FROM pedidos", max_rows=50).limit_applied == 50
    assert validate_sql("SELECT * FROM pedidos LIMIT 10", max_rows=50).limit_applied == 10
    assert validate_sql("SELECT * FROM pedidos LIMIT 100000", max_rows=50).limit_applied == 50


@pytest.mark.parametrize("q", [
    "DELETE FROM pedidos",
    "UPDATE pedidos SET importe = 0",
    "INSERT INTO pedidos VALUES ('x')",
    "DROP TABLE pedidos",
    "CREATE TABLE x AS SELECT * FROM pedidos",
    "SELECT * FROM pedidos; DROP TABLE pedidos",
    "SELECT * FROM pedidos WHERE pais = 'x'; DELETE FROM pedidos; --'",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM duckdb_tables()",
    "SELECT * FROM secretos",
    "SELECT * FROM pedidos p JOIN usuarios u ON p.cliente = u.id",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM '/etc/passwd'",
    "SELECT getenv('LLM_API_KEY')",
    "SELECT * FROM pedidos WHERE cliente IN (SELECT * FROM read_parquet('s3://x/y'))",
    "ATTACH 'otra.db' AS o",
    "COPY pedidos TO '/tmp/fuga.csv'",
    "PRAGMA database_list",
    "SET enable_external_access = true",
    "INSTALL httpfs",
    "SELECT 1",
    "",
    "esto no es sql",
])
def test_injection_and_forbidden_queries_are_rejected(q):
    with pytest.raises(SQLValidationError):
        validate_sql(q, max_rows=100)


def test_executor_is_read_only_even_if_validator_is_bypassed(sample_db):
    ex = ReadOnlyExecutor(sample_db)
    assert ex.run(validate_sql("SELECT COUNT(*) AS n FROM pedidos", max_rows=5).sql).rows[0][0] > 0
    with pytest.raises(Exception):
        ex.run("DELETE FROM pedidos")
    with pytest.raises(Exception):
        ex.run("SELECT * FROM read_csv('/etc/passwd')")
