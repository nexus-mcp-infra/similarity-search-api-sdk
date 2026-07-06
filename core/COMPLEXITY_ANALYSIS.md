# Análisis de Complejidad Computacional — Similarity Search API (NMI+Cosine Fusion)

## Parámetros de referencia

- `n` = número de items en el corpus del payload
- `f` = número de features por item
- `c` = cardinalidad promedio de features categóricas
- `k` = resultados solicitados (top-k)

---

## Endpoints públicos

### `POST /similarity/search`

**Temporal:** `O(n · f · c)` dominado por el cálculo de NMI sobre features categóricas: por cada par (query, item) se estima la entropía conjunta con tabla de contingencia de tamaño `c²`, dando `O(n · f · c²)` en el peor caso con alta cardinalidad categórica. El cosine sobre TF-IDF es `O(n · f)` y queda subordinado.
**Espacial:** `O(n · f)` para materializar las representaciones vectoriales del corpus completo en memoria por request — sin índice persistente, todo vive en el heap de la invocación.
**Mejor caso:** `O(n · f)` cuando todas las features son numéricas continuas (NMI no se activa, solo cosine). **Promedio:** `O(n · f · c)` con `c ≈ 10–20`. **Peor caso:** `O(n · f · c²)` con features categóricas de alta cardinalidad (`c > 100`, e.g. códigos postales, SKUs).
**Cuello de botella:** la construcción de tablas de contingencia para NMI es `O(n · c²)` por feature categórica — con `f_cat` features categóricas de alta cardinalidad el costo crece cuadráticamente en `c` y domina completamente.

---

### `POST /similarity/score`

**Temporal:** `O(f · c)` — opera sobre un par único (query, item) sin iterar corpus. NMI se calcula feature a feature con corrección de Laplace en `O(c log c)` por feature categórica; cosine en `O(f)`.
**Espacial:** `O(f)` — solo dos vectores en memoria.
**Mejor / promedio / peor:** `O(f)` / `O(f · c)` / `O(f · c log c)` según proporción de features categóricas y su cardinalidad.
**Cuello de botella:** ninguno relevante a escala de par único; el overhead dominante es la serialización/deserialización JSON del payload, no el cálculo.

---

### `POST /similarity/rank`

**Temporal:** `O(n · f · c + n log k)` — fusión NMI+cosine sobre todo el corpus seguida de un heap parcial de tamaño `k` para extraer top-k en `O(n log k)` en lugar de sort completo `O(n log n)`.
**Espacial:** `O(n · f + k)` — corpus materializado más el heap de resultados.
**Mejor caso:** `O(n · f + n log k)` con features puramente numéricas. **Promedio/peor:** igual que `/search` con el heap añadido, que es despreciable frente al costo NMI.
**Cuello de botella:** idéntico a `/search`: la fase NMI domina; el heap top-k es `O(n log k)` y no compite.

---

## Saturación y estrategia de escala

Con `n = 500` items, `f = 20` features y `c = 15`, cada request ejecuta ~150 000 operaciones elementales; en serverless Python 3.11 con Uvicorn el throughput empírico estimado ronda **40–80 req/s por instancia** antes de que la CPU sature (el paso NMI no libera el GIL al ser puro Python/NumPy en loop categórico). Para escalar más allá: (1) vectorizar la construcción de tablas de contingencia con `numpy.histogramdd` eliminando el loop por item — reduce la constante de NMI ~4x; (2) aplicar early-exit cosine pre-filter descartando items con cosine < umbral antes de computar NMI, reduciendo `n` efectivo en 60–80% en corpus con distribución natural; (3) escalar horizontalmente con concurrencia stateless pura — cada instancia es independiente, sin coordinación.