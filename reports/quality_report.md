# Informe de calidad de datos

- Filas de entrada: **5680**
- Filas cargadas: **5227**
- Filas descartadas: **453**

## Filas descartadas por motivo

| Motivo | Filas |
|---|---:|
| duplicado | 173 |
| importe_vacio | 105 |
| importe_no_positivo | 95 |
| fecha_invalida | 45 |
| pais_desconocido | 35 |

## Correcciones aplicadas (filas que se conservan)

| Corrección | Filas |
|---|---:|
| pais_normalizado | 4130 |
| fecha_normalizada_a_iso | 2444 |
| estado_normalizado | 308 |
| importe_coma_decimal | 227 |
