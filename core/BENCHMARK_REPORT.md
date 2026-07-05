## Metodología

Tests ejecutados sobre instancia c6i.2xlarge (8 vCPU, 16 GB RAM, AWS us-east-1). Colecciones sintéticas con mezcla real texto+numérico+categórico, tamaños n={50, 500, 2000} items, d={10, 50, 200} dimensiones. Latencia medida extremo a extremo (HTTP round-trip) con wrk2 a tasa controlada; 10,000 requests por celda, percentil 99 reportado. Throughput sostenido medido a 5 minutos continuos sin warmup artificial.

---

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput (n=500, d=50) | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (esta)** | < 5 min | 8 | 340 req/s | 47 ms |
| Pinecone (cosine nativo) | ~45 min | 120 | 420 req/s | 38 ms |
| Weaviate (cosine nativo) | ~90 min | 310 | 380 req/s | 41 ms |
| scikit-learn NMI manual | N/A (lib, no API) | 85 | N/A | N/A |
| Qdrant (cosine nativo) | ~60 min | 180 | 390 req/s | 40 ms |

*Nota de estimación: throughput de competidores asume índice ya cargado — costo de ingestión excluido. Similarity Search API no tiene fase de ingestión.*

---

## Análisis estadístico

Intervalos de confianza al 95% sobre latencia p99: ±2.1 ms para esta API (bootstrap n=10,000), calculados con distribución empírica sin asumir normalidad. Diferencia en tiempo de integración respecto a Pinecone es estadísticamente dominante — no hay solapamiento entre distribuciones de setup time (mínimo Pinecone: 28 min vs. máximo esta API: 7 min, p < 0.001 sobre 20 trials con developers independientes). La penalización de throughput respecto a cosine puro (~19% menos req/s) es atribuible al cómputo O(n·d·log d) del binning Freedman-Diaconis, no a overhead de red.

---

## Interpretación

**Cuándo es superior:** Datasets heterogéneos (features categóricas + numéricas + embeddings de texto mezclados) donde cosine puro degrada precisión al ignorar dependencias no lineales; casos de uso sin estado persistente como comparación en carrito, deduplicación batch ad-hoc, o pipelines serverless donde levantar un vector DB es over-engineering no justificable.

**Cuándo NO usarla:** Colecciones estáticas grandes (n > 50,000 items) consultadas repetidamente — sin índice persistente, recalcular NMI en cada request a esa escala supera el punto de equilibrio donde Pinecone/Qdrant amortizan su setup cost. Tampoco sustituye a ANN (Approximate Nearest Neighbors) cuando la latencia sub-10 ms es requisito duro: el cómputo exacto O(n·d·log d) no es negociable en esta arquitectura.