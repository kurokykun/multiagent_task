import pandas as pd
import pytest

from app.data.clean import clean_orders, normalize_country, parse_date

COLS = ["pedido_id", "fecha", "cliente", "pais", "producto", "categoria", "importe", "estado"]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_cleaning_discards_with_reasons_and_keeps_valid_rows():
    raw = _df([
        ["P1", "2025-03-04", "CLI-1", "España", "Camiseta", "Moda", "19.90", "completado"],
        ["P1", "2025-03-04", " CLI-1 ", "ESPAÑA", "Camiseta", "Moda", "19.90", "completado"],
        ["P2", "04/03/2025", "CLI-2", "spain", "Novela", "Libros", "12,50", "Enviado"],
        ["P3", "2025-03-04", "CLI-3", "MX", "Novela", "Libros", "", "completado"],
        ["P4", "2025-03-04", "CLI-4", "MX", "Novela", "Libros", "-5.00", "completado"],
        ["P5", "31/02/2025", "CLI-5", "MX", "Novela", "Libros", "5.00", "completado"],
        ["P6", "2025-03-04", "CLI-6", "Atlantis", "Novela", "Libros", "5.00", "completado"],
    ])
    clean, report = clean_orders(raw)

    assert list(clean["pedido_id"]) == ["P1", "P2"]
    assert report.discarded == {"duplicado": 1, "importe_vacio": 1, "importe_no_positivo": 1,
                                "fecha_invalida": 1, "pais_desconocido": 1}
    assert report.input_rows == report.output_rows + report.discarded_total

    p2 = clean.set_index("pedido_id").loc["P2"]
    assert p2["fecha"] == pd.Timestamp("2025-03-04")
    assert p2["importe"] == 12.5
    assert p2["pais"] == "España"
    assert p2["estado"] == "enviado"


@pytest.mark.parametrize("raw,expected", [
    ("España", "España"), (" españa ", "España"), ("ESP", "España"), ("Spain", "España"),
    ("Mexico", "México"), ("MX", "México"), ("prt", "Portugal"), ("Atlantis", None), ("", None),
])
def test_normalize_country(raw, expected):
    assert normalize_country(raw) == expected


@pytest.mark.parametrize("raw,iso", [
    ("2025-03-04", "2025-03-04"), ("04/03/2025", "2025-03-04"),
    ("2025/03/04", "2025-03-04"), ("04-Mar-2025", "2025-03-04"),
])
def test_parse_date_formats(raw, iso):
    assert parse_date(raw).date().isoformat() == iso


def test_full_generated_dataset_is_reproducible():
    from app.data.generate import generate_rows

    a, ra = clean_orders(pd.DataFrame(generate_rows(n_orders=800, seed=1)))
    b, rb = clean_orders(pd.DataFrame(generate_rows(n_orders=800, seed=1)))
    assert ra.to_dict() == rb.to_dict()
    assert a.equals(b)
    assert a["pedido_id"].is_unique
    assert (a["importe"] > 0).all()
    assert {"duplicado", "importe_vacio", "importe_no_positivo"} <= set(ra.discarded)
