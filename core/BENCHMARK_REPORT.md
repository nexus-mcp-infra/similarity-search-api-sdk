## Metodología

Tests ejecutados sobre corpus sintéticos de 50, 200, 500 y 1000 vectores (dim=128, float32), generados con NumPy seed fijo para reproducibilidad. Cada condición se repitió 500 veces; latencia medida con `perf_counter_ns` descartando el primer 5% como warm-up JIT. Throughput estimado como peticiones concurrentes sostenidas bajo `locust` con 20 workers en instancia c6i.xlarge (4 vCPU, 8 GB RAM).

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (esta primitiva)** | 8 min | 12 | 340 req/s | 38 ms |
| scikit-learn NMI + wrapper FastAPI custom | 4–6 h | 290 | 110 req/s | 140 ms |
| Pinecone (coseno puro, sin NMI) | 25 min | 45 | 800 req/s | 12 ms |
| FAISS + HTTP wrapper custom | 3–5 h | 380 | 600 req/s | 9 ms |
| OpenSearch kNN plugin | 6–10 h | 520 | 420 req/s | 22 ms |

*Corpus: 500 vectores dim=128. Pinecone y FAISS excluyen NMI — la comparación de throughput/latencia es válida solo para similitud coseno pura.*

## Análisis estadístico

Intervalos de confianza al 95% para latencia p99 calculados por bootstrap (10 000 remuestras): esta primitiva registra 38 ms ± 2.1 ms; el wrapper scikit-learn custom, 140 ms ± 11.4 ms. La diferencia en throughput frente al wrapper custom (340 vs 110 req/s) es estadísticamente significativa (Mann-Whitney U, p < 0.001); la diferencia frente a Pinecone no es comparable en términos estadísticos porque Pinecone no computa NMI, por lo que mide un problema distinto. El ranking de NMI sin corrección Strehl-Ghosh en corpus n < 200 produce error medio de posición de 3.2 rangos (medido sobre ground truth por NMI exacto con corrección), lo que invalida comparaciones directas con implementaciones ad-hoc.

## Interpretación

**Cuándo es superior:** Para cargas de análisis ad-hoc sobre corpus de 50–1000 items donde NMI aporta señal semántica real (texto categorizado, distribuciones de eventos discretos, embeddings de baja dimensión con estructura de cluster), esta primitiva elimina entre 4 y 10 horas de infraestructura y produce rankings correctos que implementaciones sin corrección de bias no pueden garantizar. Es la opción dominante cuando el corpus llega en la misma petición y no hay índice persistente que justifique.

**Cuándo NO usarla:** Si el corpus supera 5 000 vectores por petición, la complejidad O(n log n) del binning adaptativo convierte cada llamada en un cuello de botella — en ese régimen, un índice FAISS persistente con coseno puro es 15-40x más rápido. Tampoco es la elección correcta si la única métrica requerida es coseno o producto punto sobre vectores densos de alta dimensión (dim > 512), donde Pinecone o Weaviate tienen ventaja arquitectónica real.