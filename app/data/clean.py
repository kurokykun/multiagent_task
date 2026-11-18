from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y"]

_COUNTRY_ALIASES: dict[str, list[str]] = {
    "España": ["espana", "spain", "es", "esp"],
    "México": ["mexico", "mx", "mex"],
    "Argentina": ["argentina", "ar", "arg"],
    "Colombia": ["colombia", "co", "col"],
    "Chile": ["chile", "cl", "chl"],
    "Portugal": ["portugal", "pt", "prt"],
}
COUNTRY_MAP = {alias: canon for canon, aliases in _COUNTRY_ALIASES.items() for alias in aliases}
VALID_STATUSES = {"completado", "enviado", "pendiente", "cancelado", "devuelto"}
REQUIRED = ["pedido_id", "cliente", "producto", "categoria"]


@dataclass
class QualityReport:
    input_rows: int = 0
    output_rows: int = 0
    discarded: dict[str, int] = field(default_factory=dict)
    fixed: dict[str, int] = field(default_factory=dict)

    @property
    def discarded_total(self) -> int:
        return sum(self.discarded.values())

    def to_dict(self) -> dict:
        return {
            "filas_entrada": self.input_rows,
            "filas_salida": self.output_rows,
            "filas_descartadas": self.discarded_total,
            "descartes_por_motivo": dict(sorted(self.discarded.items(), key=lambda kv: -kv[1])),
            "correcciones_aplicadas": self.fixed,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Informe de calidad de datos",
            "",
            f"- Filas de entrada: **{self.input_rows}**",
            f"- Filas cargadas: **{self.output_rows}**",
            f"- Filas descartadas: **{self.discarded_total}**",
            "",
            "## Filas descartadas por motivo",
            "",
            "| Motivo | Filas |",
            "|---|---:|",
        ]
        for reason, n in sorted(self.discarded.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {n} |")
        lines += ["", "## Correcciones aplicadas (filas que se conservan)", "", "| Corrección | Filas |", "|---|---:|"]
        for fix, n in sorted(self.fixed.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {fix} | {n} |")
        return "\n".join(lines) + "\n"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_country(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = _strip_accents(value.strip().lower())
    return COUNTRY_MAP.get(key)


def parse_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def parse_amount(value: object) -> tuple[float | None, str | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, "importe_vacio"
    v = str(value).strip()
    if not v:
        return None, "importe_vacio"
    v = v.replace(",", ".")
    try:
        amount = float(v)
    except ValueError:
        return None, "importe_no_numerico"
    if amount <= 0:
        return None, "importe_no_positivo"
    return round(amount, 2), None


def clean_orders(raw: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    report = QualityReport(input_rows=len(raw))
    df = raw.copy().astype("object")
    for col in df.columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    fixed: dict[str, int] = {}

    def bump(d: dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    records = []
    for row in df.to_dict(orient="records"):
        reason = None
        if any(not isinstance(row.get(c), str) or not row.get(c) for c in REQUIRED):
            reason = "campos_obligatorios_vacios"

        fecha = parse_date(row.get("fecha"))
        if reason is None and fecha is None:
            reason = "fecha_invalida"

        amount, amount_err = parse_amount(row.get("importe"))
        if reason is None and amount_err:
            reason = amount_err

        country = normalize_country(row.get("pais"))
        if reason is None and country is None:
            reason = "pais_desconocido"

        status = str(row.get("estado") or "").strip().lower()
        if reason is None and status not in VALID_STATUSES:
            reason = "estado_desconocido"

        if reason:
            bump(report.discarded, reason)
            continue

        if not str(row["fecha"]).startswith(fecha.strftime("%Y-%m-%d")):
            bump(fixed, "fecha_normalizada_a_iso")
        if "," in str(row["importe"]):
            bump(fixed, "importe_coma_decimal")
        if row["pais"] != country:
            bump(fixed, "pais_normalizado")
        if row["estado"] != status:
            bump(fixed, "estado_normalizado")

        records.append({
            "pedido_id": row["pedido_id"],
            "fecha": fecha.date(),
            "cliente": row["cliente"].upper(),
            "pais": country,
            "producto": row["producto"],
            "categoria": row["categoria"],
            "importe": amount,
            "estado": status,
        })

    clean = pd.DataFrame.from_records(
        records, columns=["pedido_id", "fecha", "cliente", "pais", "producto", "categoria", "importe", "estado"]
    )
    before = len(clean)
    clean = clean.drop_duplicates(subset=["pedido_id"], keep="first").reset_index(drop=True)
    if before - len(clean):
        report.discarded["duplicado"] = before - len(clean)

    clean["fecha"] = pd.to_datetime(clean["fecha"])
    report.fixed = fixed
    report.output_rows = len(clean)
    assert report.input_rows == report.output_rows + report.discarded_total
    return clean, report
