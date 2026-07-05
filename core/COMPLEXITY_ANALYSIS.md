# Análisis de Complejidad Computacional — Similarity Search API (NMI+Cosine Hybrid)

## Endpoints Públicos

### `POST /similarity/search`

**Temporal:** O(N · (F_cat² + F_cont)) por query, donde N = tamaño del corpus, F_cat = features categóricas, F_cont = features continuas. El cálculo de NMI por par de features categóricas requiere construir la tabla de contingencia en O(F_cat²·n_bins²) y computar entropías conjuntas; coseno sobre el bloque continuo es O(F_cont) por candidato.

**Casos:** Mejor O(N·F_cont) cuando el payload es 100% continuo (NMI se omite). Promedio O(N·(F_cat·log F_cat + F_cont)) con cardinalidad moderada. Peor O(N·F_cat²·C²) donde C = cardinalidad máxima de una variable categórica — explota con variables nominales de alta cardinalidad sin binning previo.

**Cuello de botella:** Construcción de tablas de contingencia para NMI cuando F_cat > 10 y C > 50; la complejidad cuadrática en features categóricas domina sobre el producto escalar coseno, que es trivialmente vectorizable con NumPy.

---

### `POST /similarity/explain`

**Temporal:** O(F_total) después de que `/search` cachea los scores intermedios; si se invoca sin caché, hereda O(N·(F_cat² + F_cont)). El desglose por componente (w_nmi, w_cos, entropía marginal por feature) es un postproceso O(F_total) sobre resultados ya computados.

**Casos:** Mejor/promedio O(F_total) con caché activo (hits esperados >80% en corpus estático). Peor igual que `/search` si el TTL expiró o el corpus mutó.

**Cuello de botella:** Serialización del breakdown por feature a JSON cuando F_total > 200; considerar respuesta paginada o proyección de features por relevancia.

---

### `POST /similarity/calibrate` (weight flywheel — DuckDB)

**Temporal:** O(D·log D) donde D = registros históricos de distribución de pesos por dominio almacenados en DuckDB. La actualización de umbrales de entropía es una aggregación SQL sobre la tabla de pesos — DuckDB la ejecuta vectorizado en columnar.

**Casos:** Mejor O(1) si el dominio no tiene registros previos (devuelve umbrales default). Promedio O(D·log D) con D ~ 10⁵ registros. Peor O(D²) solo si se solicita recalibración con clustering jerárquico sobre distribuciones de pesos (opcional).

**Cuello de botella:** Escritura concurrente en DuckDB bajo alta QPS — DuckDB no es OLTP; el cuello real es el lock de escritura si `/search` registra pesos síncronamente.

---

## Saturación y Estrategia de Escala

Con corpus N=10k, F_cat=5, F_cont=10, un worker Uvicorn en CPU moderna procesa ~180–240 req/s antes de que la latencia p95 supere 200ms — el factor limitante es NMI, no E/S. Para escalar: (1) precalcular y cachear las tablas de contingencia por corpus hash (Redis TTL = vida del corpus), reduciendo el inner loop a O(F_cont·N) puro; (2) separar el registro de pesos en DuckDB a un worker async desacoplado con cola en memoria para eliminar el write-lock del hot path; (3) paralelizar el loop sobre N con `numpy.einsum` vectorizado para el bloque coseno y `numba.prange` para el bloque NMI — combinación que permite escalar a ~1,200 req/s sin cambiar la arquitectura stateless.