# Análisis de Complejidad Computacional — Similarity Search API (NMI+Cosine Hybrid)

---

## Endpoints Públicos

### `POST /search` — Búsqueda híbrida sobre corpus en memoria

**Complejidad temporal:** O(n · d + n · b²) donde n = tamaño del corpus, d = dimensión del vector, b = número de bins adaptativos. El término dominante es la estimación de distribución conjunta por binning: para cada par (query, candidato) en modo denso, el histograma 2D tiene coste O(d) para construirse y O(b²) para normalizarse y calcular la entropía conjunta H(X,Y). La fusión lineal alpha·NMI + (1-alpha)·Cosine es O(d) y no cambia el orden.

**Casos:** Mejor O(n·d) cuando `input_type=discrete` (NMI directo sin binning). Promedio O(n·d·b) con b ≈ √d para bins adaptativos. Peor O(n·d²) si b crece proporcional a d en vectores de alta dimensión sin cap explícito.

**Cuello de botella:** La construcción del histograma 2D con corrección Strehl-Ghosh — `H_corrected = H(X,Y) - (b²-1)/(2n·ln2)` — itera sobre todos los bins incluso vacíos. Para n < 200 la corrección es crítica pero el coste de normalizar la distribución conjunta domina sobre el producto punto.

---

### `POST /rank` — Re-ranking de resultados precomputados por NMI

**Complejidad temporal:** O(k · b²) donde k = número de candidatos a re-rankear (k ≤ n). El re-ranking no reconstruye distribuciones desde cero si los vectores ya vienen normalizados: el coste se reduce al cálculo de entropía marginal H(X), H(Y) y conjunta H(X,Y) sobre el histograma ya binneado.

**Casos:** Mejor O(k·d) para distribuciones discretas nativas pasadas como input. Promedio/peor O(k·b²) idéntico al endpoint `/search` pero acotado por k << n en el caso de uso típico (re-rank top-50 sobre corpus de 10k).

**Cuello de botella:** Sincronización en el paso de re-normalización cuando alpha != 0.5 — el peso asimétrico obliga a recalcular ambas ramas del scoring antes de fusionar, duplicando efectivamente las operaciones de entropía.

---

### `POST /validate-distribution` — Verificación de inputs discretos

**Complejidad temporal:** O(v) donde v = longitud del vector de probabilidades. Validación que la distribución suma 1.0 dentro de tolerancia float64, que no existen probabilidades negativas, y que el soporte no es trivial (entropía > 0). Espacial O(1) — no requiere copia del input.

**Casos:** Mejor/promedio/peor todos O(v) — el scan es lineal e incondicional.

**Cuello de botella:** Ninguno estructural; el riesgo real es propagación silenciosa de distribuciones degeneradas si este endpoint se omite en el flujo, produciendo NMI = 0 o NaN downstream.

---

## Saturación y Escalabilidad

Con corpus n = 500, d = 768 (embeddings BERT estándar) y b = 28 (√d redondeado), cada llamada a `/search` ejecuta aproximadamente 500 · 28² ≈ 392,000 operaciones de histograma. En CPython 3.11 con NumPy vectorizado, el throughput estimado es **~40–80 req/s por worker single-core** antes de saturar CPU. Para escalar más allá: (1) cap de bins a b ≤ 20 independientemente de d — el impacto en precisión NMI es < 2% para n > 100 según la cota de bias Strehl-Ghosh; (2) paralelizar la construcción de histogramas por batch via `numpy.histogramdd` sobre matriz (n, 2, d) en lugar de loop explícito; (3) desplegar múltiples workers stateless detrás de un load balancer — el diseño sin índice persistente es intrínsecamente horizontal.