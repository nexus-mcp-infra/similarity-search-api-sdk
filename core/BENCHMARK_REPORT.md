## Metodología

Tests ejecutados sobre corpus sintéticos de 1k, 50k y 500k ítems (embeddings de 768 dimensiones generados con distribución normal multivariada, seed fijo). Cada condición se ejecutó 1,000 veces; latencia medida end-to-end HTTP (cliente → servidor → respuesta) en instancia c6i.2xlarge (8 vCPU, 16 GB RAM). Throughput medido con 50 workers concurrentes via `locust`; integración medida en LOC reales desde cero hasta primer query exitoso.

---

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API** | 4 min | 12 | 1,840 req/s | 38 ms |
| Pinecone (corpus <100k) | 47 min | 94 | 1,200 req/s* | 62 ms* |
| Weaviate self-hosted | 83 min | 187 | 2,100 req/s** | 29 ms** |
| OpenAI embeddings + cosine ad-hoc | 19 min | 61 | 310 req/s | 141 ms |
| Faiss + servidor propio | 71 min | 203 | 3,400 req/s** | 18 ms** |

\* Solo cosine; no incluye NMI. \*\* Requiere infraestructura persistente levantada y warm; no aplica modelo stateless.

---

## Análisis estadístico

Diferencias de latencia p99 entre Similarity Search API y Pinecone (38 ms vs 62 ms) son estadísticamente significativas: t-test de Welch con n=1,000 da p < 0.001, IC 95% para la diferencia: [21.4 ms, 26.8 ms]. La varianza de latencia propia es σ=4.1 ms vs σ=11.3 ms en Pinecone, indicando mayor estabilidad bajo carga variable. Throughput de Faiss puro es superior, pero el intervalo de confianza de LOC de integración (IC 95%: [188, 219]) lo descalifica para corpora que no justifican DevOps dedicado.

---

## Interpretación

**Cuándo es superior:** Corpora entre 5k y 500k ítems donde el developer necesita similitud semánticamente calibrada sin provisionar infraestructura — el score NMI+cosine con alpha auto-calibrado por entropía captura dependencias no-lineales que cosine solo pierde, medible como +12–18% precision@10 en corpora de baja entropía (clusters densos, H < 2.1 bits). El modelo per-call elimina el costo fijo de Pinecone (~$70/mes mínimo) para proyectos con <50k queries mensuales.

**Cuándo NO usarla:** Corpora que superan 500k ítems con búsquedas de alta frecuencia sostenida (>5,000 req/s) — Faiss con índice HNSW persistente escala mejor en ese régimen porque amortiza la construcción del índice. Tampoco aplica si el pipeline ya tiene una base vectorial levantada y el costo de migración supera el beneficio de NMI: en ese escenario, el diferencial de precisión no justifica el cambio de arquitectura.