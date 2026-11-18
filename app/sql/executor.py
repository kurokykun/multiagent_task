from __future__ import annotations

import datetime as dt
import decimal
import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb

SCHEMA_DESCRIPTION = """\
Tabla pedidos(pedido_id VARCHAR, fecha DATE, cliente VARCHAR, pais VARCHAR, producto VARCHAR,
              categoria VARCHAR, importe DECIMAL(12,2), estado VARCHAR)
  - pais ∈ {España, México, Argentina, Colombia, Chile, Portugal}
  - categoria ∈ {Electrónica, Hogar, Moda, Deportes, Libros}
  - estado ∈ {completado, enviado, pendiente, cancelado, devuelto}
  - Una "venta" es un pedido con estado 'completado' o 'enviado'.
  - Datos desde 2024-01-01 hasta 2025-12-31.
Vista ventas_mes_pais(mes DATE, pais VARCHAR, num_pedidos BIGINT, importe_total DECIMAL, ticket_medio DECIMAL)
Vista top10_clientes(cliente VARCHAR, pais VARCHAR, num_pedidos BIGINT, importe_total DECIMAL, ultima_compra DATE)
Vista clientes_inactivos_90d(cliente VARCHAR, pais VARCHAR, ultima_compra DATE, num_pedidos BIGINT,
                             importe_historico DECIMAL, dias_sin_comprar INTEGER)
  - Clientes cuya última compra es anterior a 90 días antes de la fecha más reciente del dataset.
"""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return v


class ReadOnlyExecutor:
    def __init__(self, db_path: Path, timeout_s: float = 10.0):
        self.db_path = db_path
        self.timeout_s = timeout_s

    def run(self, sql: str) -> QueryResult:
        con = duckdb.connect(str(self.db_path), read_only=True,
                             config={"enable_external_access": False})
        timer = threading.Timer(self.timeout_s, con.interrupt)
        try:
            con.execute("SET lock_configuration = true")
            timer.start()
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = [[_jsonable(v) for v in r] for r in cur.fetchall()]
            return QueryResult(columns=columns, rows=rows)
        finally:
            timer.cancel()
            con.close()
