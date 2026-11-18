from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from app.data.clean import QualityReport, clean_orders
from app.data.generate import generate_rows, write_csv

VIEWS_SQL = Path(__file__).with_name("views.sql")


def load_to_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute("""
            CREATE TABLE pedidos (
                pedido_id VARCHAR PRIMARY KEY,
                fecha     DATE NOT NULL,
                cliente   VARCHAR NOT NULL,
                pais      VARCHAR NOT NULL,
                producto  VARCHAR NOT NULL,
                categoria VARCHAR NOT NULL,
                importe   DECIMAL(12, 2) NOT NULL CHECK (importe > 0),
                estado    VARCHAR NOT NULL
            )
        """)
        con.register("clean_df", df)
        con.execute("INSERT INTO pedidos SELECT pedido_id, CAST(fecha AS DATE), cliente, pais, producto, "
                    "categoria, importe, estado FROM clean_df")
        con.execute(VIEWS_SQL.read_text(encoding="utf-8"))
    finally:
        con.close()
    tmp.replace(db_path)


def write_report(report: QualityReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "quality_report.md").write_text(report.to_markdown(), encoding="utf-8")


def run(raw_csv: Path, db_path: Path, reports_dir: Path, regenerate: bool = False) -> QualityReport:
    if regenerate or not raw_csv.exists():
        write_csv(raw_csv, generate_rows())
    raw = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    clean, report = clean_orders(raw)
    load_to_duckdb(clean, db_path)
    write_report(report, reports_dir)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/pedidos_raw.csv"))
    p.add_argument("--db", type=Path, default=Path("data/pedidos.duckdb"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--regenerate", action="store_true")
    a = p.parse_args()
    report = run(a.raw, a.db, a.reports, a.regenerate)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
