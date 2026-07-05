## Análisis de Complejidad — NMI-Weighted Cosine Similarity API

---

### `POST /similarity/search`

**Temporal:** O(n · d · log d) donde n = items en la colección candidata y d = dimensiones. El cuello de botella es el cálculo de NMI por feature: binning Freedman-Diaconis cuesta O(n · log n) por dimensión, repetido d veces. La multiplicación coseno ponderada posterior es O(n · d), dominada por la fase NMI.

**Espacial:** O(n · d) para materializar la colección completa en memoria durante el request; no hay índice persistente, pero la colección vive íntegra en RAM durante el cómputo.

**Casos:** Mejor — features todas categóricas, frecuencias directas O(n · d). Promedio — mix categórico/continuo con d ≈ 50–200, O(n · d · log n). Peor — todas continuas con d grande (>500) y n > 10k, O(n · d · log n) con constante alta por binning adaptativo.

**Cuello de botella:** Cálculo de entropía marginal y conjunta por dimensión continua (Freedman-Diaconis requiere ordenamiento por dimensión). Con d = 200 y n = 1000, esto implica 200 sorts independientes de 1000 elementos.

---

### `POST /similarity/batch`

**Temporal:** O(Q · n · d · log d) donde Q = queries en el batch. Sin paralelismo intra-batch, es lineal en Q. Con paralelismo via `np.vectorize` o `ThreadPoolExecutor`, el factor efectivo se reduce a O(ceil(Q/W) · n · d · log d) con W workers.

**Espacial:** O(Q · n · d) en el peor caso si el batch completo se materializa; O(n · d) si se procesa en streaming por query.

**Casos:** Mejor — Q = 1 (equivalente a `/search`). Promedio — Q ≈ 10–50 con paralelismo parcial. Peor — Q grande (>200) sin streaming, presión de memoria excede L3 cache y degrada throughput por cache misses.

**Cuello de botella:** Contención de memoria al materializar múltiples colecciones simultáneamente; el NMI no es reutilizable entre queries si las colecciones candidatas difieren.

---

### `GET /similarity/explain`

**Temporal:** O(d · log d) — recalcula el vector de pesos NMI para un par único y ordena features por contribución. Costo dominado por el ranking final, no por coseno.

**Espacial:** O(d) — solo el vector de pesos y los metadatos por dimensión.

**Casos:** Uniforme en todos los escenarios; la varianza viene del tipo de feature (categórica vs. continua), no del volumen.

**Cuello de botella:** Ninguno significativo; este endpoint es I/O-bound en práctica.

---

### Saturación y Escala

Con workers uvicorn de 4 CPUs y requests típicos (n = 500, d = 100, mix 50/50), el punto de saturación estimado es **~40–60 req/s** por instancia — el 80% del tiempo de CPU lo consume el binning NMI. Para escalar más allá: (1) cachear el vector de pesos NMI de colecciones recurrentes con hash SHA-256 del payload como clave (TTL corto, 60 s), convirtiendo llamadas repetidas en O(n · d) puro; (2) precomputar bins Freedman-Diaconis en paralelo por dimensión con `np.partition` en lugar de sort completo, reduciendo el ordenamiento de O(n · log n) a O(n) para el percentil necesario.