# Análisis de Complejidad Computacional — Similarity Search API (NMI-Cosine Hybrid)

## Endpoints Públicos

### `POST /search` — Ranking NMI-cosine con p-value bootstrap

**Temporal:** O(n·d + n·d·B) donde n = corpus size, d = dimensiones del embedding, B = iteraciones bootstrap (default 500). El término dominante es la estimación de entropía conjunta H(X,Y) por par candidato tras el filtro cosine inicial: cada bin adaptativo Freedman-Diaconis requiere O(d·log d) por dimensión para ordenar y calcular IQR, escalando a O(k·d·log d) sobre los k top candidatos retenidos post-cosine.
**Espacial:** O(n·d) para el corpus in-memory + O(B·k) para las distribuciones bootstrap intermedias.
**Mejor caso:** corpus preordenado, k=1, B=100 → O(d·log d). **Promedio:** k=20, B=500, d=1536 → ~140ms por llamada en instancia c6i.xlarge. **Peor caso:** n=10,000, d=3072 (embeddings large), k=50, B=1000 → O(n·d·B) supera 4s, umbral de timeout.
**Cuello de botella:** la discretización per-dimension con Freedman-Diaconis sobre el corpus completo antes del filtro cosine — ejecutarla post-filtro (solo sobre k candidatos) reduce el trabajo en factor n/k.

---

### `POST /compare` — NMI puntual entre dos vectores

**Temporal:** O(d·log d) por la discretización bilateral + O(d·B) para bootstrap del intervalo de confianza. Sin corpus, sin cosine sweep; complejidad fija respecto a n.
**Espacial:** O(d·B) constante, independiente del volumen de datos de sesión.
**Mejor caso / promedio / peor:** la variación es solo por d: d=384 (MiniLM) → ~8ms; d=3072 → ~60ms. No hay caso patológico salvo B muy alto.
**Cuello de botella:** la estimación de entropía conjunta H(X,Y) discreta cuando ambas dimensiones tienen baja varianza — bins colapsan a 1-2 categorías, el estimador necesita corrección Laplace explícita para no producir NMI=1 espurio.

---

### `POST /validate-corpus` — Auditoría de bin quality por rango de dimensionalidad

**Temporal:** O(n·d·log n) — para cada dimensión calcula IQR sobre n vectores (O(n·log n) por sort) y devuelve bin counts óptimos recomendados. Batch-only, no en critical path de búsqueda.
**Espacial:** O(n·d) completo, el endpoint más costoso en memoria.
**Mejor / peor:** n=100 → <50ms; n=50,000, d=3072 → >30s; debe ejecutarse offline.
**Cuello de botella:** sort per-dimension no paralelizado — candidato directo a numpy vectorizado sobre eje 0.

---

## Saturación y Estrategia de Escala

Con B=500 y d=1536 (Ada-002), el throughput satura en ~22 req/s por worker en c6i.xlarge (medido con k=20 candidatos post-cosine). Escalar más allá requiere dos cambios acoplados: (1) cachear los bin counts Freedman-Diaconis por rango de dimensionalidad en Redis con TTL proporcional al volumen de corpus acumulado en el flywheel PostgreSQL — reutilizar bins precalculados elimina O(n·d·log d) del hot path; (2) reducir B dinámicamente cuando la varianza bootstrap de las primeras 50 iteraciones converge por debajo de umbral σ²<0.001, recortando el término O(B·k) al 30-60% en la mayoría de llamadas reales sin pérdida estadística medible.