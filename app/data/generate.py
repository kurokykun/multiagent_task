from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_ORDERS = 5_500
START = date(2024, 1, 1)
END = date(2025, 12, 31)

COUNTRY_VARIANTS: dict[str, list[str]] = {
    "España": ["España", "españa", "ESPAÑA", "Espana", "Spain", "ES", "ESP", " España "],
    "México": ["México", "Mexico", "MX", "méxico", "MEX"],
    "Argentina": ["Argentina", "argentina", "AR", "ARG"],
    "Colombia": ["Colombia", "colombia", "CO", "COL"],
    "Chile": ["Chile", "chile", "CL", "CHL"],
    "Portugal": ["Portugal", "portugal", "PT", "PRT"],
}
UNKNOWN_COUNTRIES = ["Atlantis", "N/A", "", "??"]

CATALOG: dict[str, list[tuple[str, float]]] = {
    "Electrónica": [("Auriculares BT", 59.9), ("Teclado mecánico", 89.0), ("Monitor 27", 249.0),
                    ("Ratón inalámbrico", 29.9), ("Webcam HD", 49.5)],
    "Hogar": [("Lámpara LED", 34.0), ("Cafetera", 79.0), ("Juego de sábanas", 45.0),
              ("Aspiradora", 159.0)],
    "Moda": [("Zapatillas running", 95.0), ("Chaqueta", 120.0), ("Camiseta", 19.9),
             ("Mochila", 39.9)],
    "Deportes": [("Esterilla yoga", 25.0), ("Mancuernas 10kg", 49.0), ("Bicicleta estática", 299.0)],
    "Libros": [("Novela", 18.5), ("Manual Python", 39.0), ("Cómic", 12.9)],
}
STATUSES = ["completado"] * 70 + ["enviado"] * 12 + ["pendiente"] * 6 + ["cancelado"] * 7 + ["devuelto"] * 5
STATUS_VARIANTS = {"completado": ["Completado", "COMPLETADO", " completado"], "enviado": ["Enviado"]}

HEADER = ["pedido_id", "fecha", "cliente", "pais", "producto", "categoria", "importe", "estado"]


def _fmt_date(d: date, rng: random.Random) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    r = rng.random()
    if r < 0.55:
        return d.isoformat()
    if r < 0.80:
        return d.strftime("%d/%m/%Y")
    if r < 0.92:
        return d.strftime("%Y/%m/%d")
    return f"{d.day:02d}-{months[d.month - 1]}-{d.year}"


def generate_rows(n_orders: int = N_ORDERS, seed: int = SEED) -> list[dict[str, str]]:
    rng = random.Random(seed)
    n_customers = 300
    customers = []
    for i in range(1, n_customers + 1):
        country = rng.choice(list(COUNTRY_VARIANTS))
        churn_day = END - timedelta(days=rng.randint(91, 400)) if rng.random() < 0.2 else END
        weight = 0.5 + rng.paretovariate(2.0)
        customers.append((f"CLI-{i:04d}", country, churn_day, weight))
    weights = [c[3] for c in customers]

    rows: list[dict[str, str]] = []
    span = (END - START).days
    for n in range(1, n_orders + 1):
        cid, country, churn_day, _ = rng.choices(customers, weights=weights)[0]
        max_span = (churn_day - START).days
        d = START + timedelta(days=rng.randint(0, max(0, min(span, max_span))))
        category = rng.choice(list(CATALOG))
        product, price = rng.choice(CATALOG[category])
        qty = rng.choice([1, 1, 1, 2, 2, 3])
        amount = round(price * qty * rng.uniform(0.9, 1.1), 2)
        status = rng.choice(STATUSES)

        row = {
            "pedido_id": f"P{n:06d}",
            "fecha": _fmt_date(d, rng),
            "cliente": cid,
            "pais": rng.choice(COUNTRY_VARIANTS[country]),
            "producto": product,
            "categoria": category,
            "importe": f"{amount:.2f}",
            "estado": rng.choice(STATUS_VARIANTS.get(status, [status])) if rng.random() < 0.1 else status,
        }

        r = rng.random()
        if r < 0.02:
            row["importe"] = ""
        elif r < 0.035:
            row["importe"] = f"-{amount:.2f}"
        elif r < 0.04:
            row["importe"] = "0"
        elif r < 0.08:
            row["importe"] = f"{amount:.2f}".replace(".", ",")
        if rng.random() < 0.008:
            row["fecha"] = rng.choice(["31/02/2025", "2025-13-01", "ayer", ""])
        if rng.random() < 0.006:
            row["pais"] = rng.choice(UNKNOWN_COUNTRIES)
        rows.append(row)

    for src in rng.sample(rows, k=180):
        dup = dict(src)
        if rng.random() < 0.4:
            dup["cliente"] = f" {dup['cliente']} "
            dup["pais"] = dup["pais"].upper()
        rows.insert(rng.randint(0, len(rows)), dup)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/pedidos_raw.csv"))
    parser.add_argument("--n", type=int, default=N_ORDERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rows = generate_rows(args.n, args.seed)
    write_csv(args.out, rows)
    print(f"Generadas {len(rows)} filas en {args.out}")


if __name__ == "__main__":
    main()
