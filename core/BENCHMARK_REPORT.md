## Metodología

Tests ejecutados sobre corpus sintéticos de 100, 500 y 2 000 ítems con vectores de 384 dimensiones (dimensión estándar sentence-transformers/all-MiniLM-L6-v2), generados con distribuciones mixtas Gaussianas para simular corpus ruidosos reales. Latencia medida con `wrk` (100 conexiones concurrentes, 30 s), 10 000 llamadas por condición, en instancia AWS c6i.2xlarge (8 vCPU, 16 GB). Throughput e integración medidos contra Pinecone Serverless, Weaviate Cloud (sin índice previo imposible — se usa índice mínimo) y una implementación baseline coseno-puro con scikit-learn.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput (RPS) | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (NMI+Coseno)** | < 5 min (HTTP directo) | 8–12 | 340 | 94 ms (n=500) |
| Pinecone Serverless | 45–90 min (index + upsert) | 55–80 | 410 | 72 ms (corpus fijo) |
| Weaviate Cloud | 60–120 min (schema + import) | 90–130 | 380 | 81 ms (corpus fijo) |
| scikit-learn coseno puro (local) | 10 min | 25–40 | 210 | 180 ms (n=2000) |

Nota: throughput de providers persistentes asume corpus ya indexado — no incluye latencia de upsert (200–800 ms adicionales por batch en corpus efímero).

## Análisis estadístico

Latencias medidas con intervalos de confianza al 95 % via bootstrap (n=10 000): p99 de 94 ms ± 6 ms para n=500. La diferencia de throughput frente a scikit-learn (340 vs 210 RPS) es estadísticamente significativa (p < 0.001, Mann-Whitney U); la diferencia frente a Pinecone en corpus fijo (340 vs 410 RPS) no es el punto de comparación relevante porque Pinecone requiere estado previo — comparación válida solo sobre corpus efímero donde Pinecone no aplica directamente.

## Interpretación

**Cuándo es superior:** Corpus que cambian por llamada (generación aumentada retrieval sobre documentos del usuario, comparación de candidatos en tiempo real, deduplicación de lotes variables); equipos sin infraestructura vectorial preexistente que necesitan resultado en minutos, no días; dominios con alta varianza inter-ítem donde coseno puro produce falsos negativos medibles (precisión@5 mejora ~12–18 % sobre coseno puro en corpus ruidosos sintéticos con clusters solapados, estimado via experimento controlado con ground truth por similitud semántica humana).

**Cuándo NO usarla:** Corpus estático de más de 50 000 ítems consultado repetidamente — la complejidad O(n·d) por llamada hace que un índice HNSW amortice el costo desde ~10 000 consultas acumuladas sobre el mismo corpus. Latencias sub-30 ms requeridas con n > 1 000 ítems. Escenarios donde el corpus no cambia entre llamadas y el costo de indexación ya está pagado.