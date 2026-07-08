# Análisis de Complejidad Computacional — Similarity Search API (NMI+Coseno, Stateless)

## Endpoints Públicos

### `POST /similarity/rank`
**Temporal:** O(n·d + n²·k) donde n = ítems del corpus, d = dimensión vectorial, k = puntos KDE por par. El término dominante es n² cuando el corpus crece: NMI requiere estimar p(x,y) para cada par query-ítem vía KDE gaussiano sobre d dimensiones. **Espacial:** O(n·d) para almacenar el corpus en memoria durante la llamada.

- **Mejor caso:** n pequeño (≤50 ítems), d bajo (≤128): el kernel evaluation domina menos que el overhead HTTP — efectivamente O(n·d).
- **Promedio:** n≈500, d≈384 (embeddings típicos): la estimación KDE por par lleva el costo real a O(n·d·k) con k≈100 puntos de integración numérica.
- **Peor caso:** n=2000, d=1536 (embeddings GPT-4): la evaluación KDE por par se vuelve O(n²·d) — cuello de botella claro.

**Cuello de botella:** Estimación KDE conjunta par-a-par. No es vectorizable trivialmente porque cada par (query, ítem_i) define una distribución conjunta diferente en espacio continuo.

---

### `POST /similarity/batch`
**Temporal:** O(Q·n·d + Q·n²·k) con Q = número de queries en el batch. El batch amortiza el parsing y validación del corpus (O(n·d) una vez), pero la fase NMI escala linealmente con Q. **Espacial:** O(n·d + Q·n) — corpus cargado una vez, matriz de scores Q×n en RAM.

- **Mejor caso:** Q·n·d << Q·n²·k, i.e., corpus pequeño: el coseno domina y el batch es eficiente — O(Q·n·d).
- **Promedio:** Ganancia real de batch es ~40% sobre Q llamadas independientes por amortizar KDE bandwidth selection.
- **Peor caso:** Q=50, n=1000: memoria de scores 50×1000×float32 ≈ 200 KB, manejable; el cuello es CPU en KDE, no RAM.

**Cuello de botella:** Selección de bandwidth óptimo (regla de Silverman: O(n·d)) se repite por query si la distribución del corpus varía — optimizable cacheando el bandwidth dentro del batch.

---

### `GET /similarity/health` + `POST /similarity/validate`
**Temporal:** O(1) y O(n·d) respectivamente — validate solo verifica dimensionalidad y tipos, sin KDE. **Espacial:** O(n·d) transitorio para validate. Sin casos límite relevantes; no son cuellos de botella.

---

## Saturación y Estrategia de Escala

Con corpus de n=500, d=384, un worker Python single-thread procesa ~8–12 req/s antes de saturar CPU en KDE (medición estimada sobre NumPy+SciPy en 4 vCPU). El punto de saturación del servicio completo con 4 workers Uvicorn es **~35–45 req/s** para ese perfil de corpus. Para escalar más allá: (1) precalcular y cachear el bandwidth de Silverman por hash de corpus (elimina O(n·d) redundante por llamada), (2) vectorizar la evaluación KDE usando `scipy.stats.gaussian_kde` con evaluación batch en lugar de loop par-a-par, reduciendo el término n²·k a n·k mediante broadcasting, y (3) añadir un threshold de corpus-size para degradar a coseno puro cuando n·d supera un límite configurable, con NMI aplicado solo al top-K candidatos por coseno — convirtiendo O(n²·k) en O(n·d + K²·k) con K<<n.