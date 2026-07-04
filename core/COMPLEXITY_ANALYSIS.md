# Análisis de Complejidad Computacional — Similarity Search API (NMI-Hybrid)

## Notación base
- `n` = ítems en corpus, `|V|` = tamaño vocabulario/bins, `d` = dimensión feature vector, `k` = top-k resultados solicitados

---

## Endpoints / Métodos Públicos

### `POST /ingest` — Indexación de corpus y cálculo de alpha(C)

**Temporal:** `O(n · |V| · log|V|)` — el cuello real es el cálculo de distribuciones marginales `p(x)` para cada token/bin sobre los `n` documentos, necesario para derivar `H_marginal(corpus)` y fijar `alpha(C)`. El paso `log2(|V|)` aparece en la normalización de entropía.
**Espacial:** `O(n · |V|)` — la tabla de frecuencias conjunta debe residir en memoria íntegra durante el cálculo de entropía marginal; no es streamable sin aproximación.
**Mejor caso:** corpus con distribución uniforme — entropía converge rápido, `O(n · |V|)` efectivo. **Peor caso:** vocabulario denso heterogéneo (`|V|` ~ 50k, n ~ 100k) — el producto domina. **Cuello de botella:** construcción de la distribución marginal conjunta; no paralelizable trivialmente porque `alpha(C)` requiere el corpus completo antes de emitir un solo score.

---

### `POST /search` — Score híbrido H(q,d) sobre corpus ingestado

**Temporal:** `O(n · |V|)` por query — cosine es `O(n · d)` con d << |V| en la mayoría de dominios, pero NMI exige estimar `I(q;d)` para cada par (q, documento), lo que requiere recorrer la distribución conjunta. El término dominante es NMI: `O(n · |V|)`.
**Espacial:** `O(|V|)` por query — se mantiene solo la distribución condicional `p(x|q)` en memoria durante el scoring; el top-k se mantiene en un heap de tamaño `k`.
**Mejor caso:** query con solapamiento de tokens alto — early-stopping posible si `H(q,d)` supera umbral antes de recorrer los `n` documentos. **Peor caso:** query out-of-distribution, sin solapamiento — recorre `n` completo. **Cuello de botella:** el cálculo de `I(q;d) = H(q) + H(d) - H(q,d)` para cada documento es secuencial y no vectorizable con operaciones BLAS estándar.

---

### `GET /alpha` — Exposición del peso adaptativo alpha(C)

**Temporal:** `O(1)` — `alpha(C)` se precalcula en ingest y se almacena. **Espacial:** `O(1)`. Sin caso peor relevante; el único riesgo es cache staleness si el corpus muta entre ingest y consulta, que el sistema debe versionar explícitamente.

---

## Saturación y Estrategia de Escala

Con corpus de `n = 10k` ítems y `|V| = 5k`, cada llamada a `/search` ejecuta ~50M operaciones de punto flotante; en un worker Uvicorn con un solo core moderno (~2 GFLOPS efectivos para operaciones no-BLAS), el throughput estimado es **~40 requests/segundo** antes de que la latencia p95 supere 500ms. El cuello de botella no es I/O sino CPU pura en el bucle NMI. La estrategia de escala prioritaria es **vectorización por lotes del cálculo de entropía conjunta con NumPy broadcasting** (`O(n · |V|)` -> operaciones matriciales en bloque de 512 documentos), combinada con **caching LRU de distribuciones marginales por corpus hash** para evitar recomputar `p(x|d)` en queries repetidas sobre el mismo corpus.