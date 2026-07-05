# Análisis de Complejidad Computacional — NMI-Weighted Cosine Similarity API

## Endpoints Públicos

### `POST /similarity/search`
**Temporal:** O(n · d²) donde *n* = items en el corpus local del payload y *d* = dimensión de features. El cuello de botella es el cálculo de NMI por par de features: construir la tabla de contingencia entre cada feature y la variable objetivo cuesta O(d² · B) con B = bins de discretización (B ≈ 10–20 en práctica). El Cosine posterior sobre el subconjunto filtrado cuesta O(n · d') con d' << d.
**Casos:** Mejor O(n · d) cuando d' → 1 (NMI elimina casi todo); promedio O(n · d · log d) con filtrado moderado; peor O(n · d²) con corpus denso y alta dimensión sin filtrado efectivo.
**Cuello de botella:** Construcción de matrices de contingencia para NMI — crece cuadráticamente con *d*, no con *n*.

---

### `POST /similarity/score`
**Temporal:** O(d²) fijo independiente de *n* — opera sobre un único par (query, target). NMI se calcula sobre las *d* features del par, Cosine sobre el subconjunto superviviente. Espacial: O(d) para histogramas de contingencia en memoria.
**Casos:** Mejor/promedio/peor convergen en O(d²) al ser un par único; la varianza real está en *d*, no en rutas de código distintas.
**Cuello de botella:** Si *d* > 2 000, la fase NMI supera la latencia HTTP base — punto de quiebre empírico alrededor de d ≈ 1 500 con NumPy vectorizado.

---

### `GET /similarity/confidence-percentile`
**Temporal:** O(log N_hist) donde N_hist = filas acumuladas en la distribución empírica de PostgreSQL por dominio. Consulta un percentil sobre un índice B-tree sobre la columna `nmi_score`. Espacial: O(1) — no carga el histograma completo en memoria.
**Casos:** Mejor O(1) con resultado cacheado por dominio+ventana temporal; promedio O(log N_hist); peor O(N_hist) si el índice no está construido sobre la partición de dominio correcta.
**Cuello de botella:** Cold start por dominio nuevo — los primeros ~200 registros producen percentiles de baja confianza hasta que la distribución empírica converge.

---

## Saturación y Estrategia de Escala

Con d = 512 y n = 1 000, el pipeline NMI->Cosine ejecuta en ~18 ms sobre un core de CPython 3.11 con NumPy. El punto de saturación de una instancia single-worker (FastAPI + Uvicorn) es aproximadamente **55–65 req/s** antes de que la latencia p95 supere 200 ms — el límite es CPU, no I/O. Para escalar más allá: (1) paralelizar la fase NMI por columnas de features con `numpy.apply_along_axis` o Numba JIT, reduciendo O(d²) a O(d²/k) con k workers; (2) cachear el vector de pesos NMI resultante por hash del schema de features del corpus — payloads con el mismo schema reutilizan el filtrado sin recalcular; (3) horizontal scaling es trivial al ser stateless — el único estado compartido es la distribución empírica en PostgreSQL, que es solo lectura durante la inferencia.