# Análisis de Complejidad Computacional — Similarity Search API

## Endpoints Públicos

### `POST /similarity/score` — Score par a par
**Temporal:** O(d + n log n) donde `d` = dimensiones del embedding (cosine en O(d)) y `n` = tamaño del vocabulario de tokens para cálculo de H(X), H(Y), H(X,Y) vía histograma discretizado. **Espacial:** O(n) para tablas de frecuencia conjunta. Mejor caso: embeddings pre-normalizados con vocabulario reducido (n < 1k), O(d). Promedio: O(d + n log n) con n ≈ 5k–20k tokens únicos. Peor caso: corpus denso con n → 50k tokens y d = 1536, donde el cálculo de entropía conjunta domina por factor ~8x sobre cosine. **Cuello de botella:** construcción de la distribución conjunta P(X,Y) para H(X,Y) — requiere doble iteración sobre tokens co-ocurrentes; no paralelizable trivialmente en CPU single-core.

### `POST /similarity/batch` — Score sobre corpus completo (hasta 500k ítems)
**Temporal:** O(N · (d + n log n)) donde N = número de ítems del corpus. Sin índice persistente, cada request recomputa cosine y NMI para cada ítem contra la query — complejidad lineal en N, no sublineal. **Espacial:** O(N · d) para materializar embeddings en memoria por request, más O(n) auxiliar para entropía. Mejor caso: N < 1k, ejecuta en < 50ms. Promedio: N ≈ 50k, ~800ms–1.2s. Peor caso: N = 500k con d = 1536, ~12–18s sin optimización vectorizada — inaceptable sin numpy batching. **Cuello de botella:** ausencia de índice invierte la ventaja stateless en penalización de latencia a N > 100k; el cálculo de alpha por entropía marginal del corpus completo añade O(N · n) adicional sobre el ya costoso paso de scoring.

### `GET /similarity/calibrate` — Calibración de alpha por entropía del corpus
**Temporal:** O(N · n log n) para calcular H(corpus) como entropía marginal agregada sobre todos los ítems — suma de entropías individuales ponderadas. **Espacial:** O(n) — solo acumula distribución de frecuencia global, no almacena corpus. Mejor caso: corpus homogéneo donde entropías individuales convergen rápido (baja varianza), O(N · n). Peor caso: corpus semánticamente disperso con n_max por ítem, requiere recalcular tabla de frecuencias por ítem sin reutilización entre llamadas. **Cuello de botella:** H_max requiere conocer el soporte completo del vocabulario; si el corpus llega fragmentado en múltiples requests, la estimación de H_max es aproximada y degrada la calibración de alpha.

---

## Saturación y Escalado

Con un servidor FastAPI de 4 workers en una instancia c5.xlarge (4 vCPU), el punto de saturación estimado es **~35–50 req/s para `/similarity/score`** (N=1, d=768) y **~2–4 req/s para `/similarity/batch`** con N=50k. El cuello de botella no es I/O sino CPU en el cálculo de entropía conjunta. La estrategia de escala prioritaria es vectorizar la construcción de P(X,Y) con `numpy.histogramdd` sobre batches de pares, precalcular y cachear H(corpus) con TTL por hash de corpus cuando el cliente envía el mismo dataset en requests sucesivos, y exponer `calibrate` como llamada separada para que el cliente amortice su costo O(N · n log n) una sola vez por corpus estable.