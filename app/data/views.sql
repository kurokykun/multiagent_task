CREATE OR REPLACE VIEW ventas_mes_pais AS
SELECT
    CAST(date_trunc('month', fecha) AS DATE) AS mes,
    pais,
    COUNT(*)                                AS num_pedidos,
    ROUND(SUM(importe), 2)                  AS importe_total,
    ROUND(AVG(importe), 2)                  AS ticket_medio
FROM pedidos
WHERE estado IN ('completado', 'enviado')
GROUP BY 1, 2;

CREATE OR REPLACE VIEW top10_clientes AS
SELECT
    cliente,
    ANY_VALUE(pais)            AS pais,
    COUNT(*)                   AS num_pedidos,
    ROUND(SUM(importe), 2)     AS importe_total,
    MAX(fecha)                 AS ultima_compra
FROM pedidos
WHERE estado IN ('completado', 'enviado')
GROUP BY cliente
ORDER BY importe_total DESC, cliente
LIMIT 10;

CREATE OR REPLACE VIEW clientes_inactivos_90d AS
WITH ref AS (SELECT MAX(fecha) AS fecha_ref FROM pedidos),
ult AS (
    SELECT cliente, ANY_VALUE(pais) AS pais, MAX(fecha) AS ultima_compra,
           COUNT(*) AS num_pedidos, ROUND(SUM(importe), 2) AS importe_historico
    FROM pedidos
    WHERE estado IN ('completado', 'enviado')
    GROUP BY cliente
)
SELECT ult.cliente, ult.pais, ult.ultima_compra, ult.num_pedidos, ult.importe_historico,
       CAST(ref.fecha_ref - ult.ultima_compra AS INTEGER) AS dias_sin_comprar
FROM ult, ref
WHERE ult.ultima_compra < ref.fecha_ref - INTERVAL 90 DAY
ORDER BY dias_sin_comprar DESC;
