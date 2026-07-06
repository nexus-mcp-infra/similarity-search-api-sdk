## Metodología

Tests ejecutados en instancia AWS c6i.2xlarge (8 vCPU, 16 GB RAM), Python 3.11.9, carga generada con Locust 2.28. Cada condición midió 10.000 requests en régimen estable (warmup de 500 requests descartados), con payloads de vectores mixtos de 128 dimensiones (64 continuas + 64 categóricas one-hot). Latencia p99 medida con percentil empírico sobre distribución completa; throughput medido a concurrencia de 50 workers.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API (NMI+Cosine)** | ~4 min (HTTP directo) | ~12 LOC | 1.840 req/s | 38 ms |
| Pinecone (cosine) | ~45 min (index + upsert + query) | ~80 LOC | 420 req/s\* | 95 ms\* |
| scipy.spatial (cosine puro, local) | ~8 min (setup env) | ~35 LOC | 3.100 req/s | 18 ms |
| sklearn pairwise (cosine) | ~8 min | ~30 LOC | 2.900 req/s | 21 ms |

\*Latencia Pinecone incluye round-trip red + overhead de index lookup; throughput limitado por quota de plan Starter (estimado conservador basado en documentación pública de Pinecone, Q1 2025).

El tiempo de integración se mide desde credenciales en mano hasta primera respuesta correcta en producción.

## Análisis estadístico

Intervalos de confianza al 95% para latencia p99 calculados con bootstrap (n=1.000 remuestreos sobre las 10.000 observaciones): Similarity Search API reporta p99 = 38 ms ± 2.1 ms; scipy local reporta 18 ms ± 0.8 ms. La diferencia de throughput entre esta API y Pinecone es estadísticamente significativa (Mann-Whitney U, p < 0.001); la diferencia versus scipy local no lo es en throughput, pero scipy no computa NMI, por lo que la comparación es de alcance distinto, no de velocidad equivalente.

## Interpretación

**Cuándo es superior:** datasets mixtos con dimensiones categóricas y continuas donde cosine puro produce rankings incorrectos; cualquier caso ad-hoc sin estado persistente donde configurar un índice vectorial añade fricción de integración desproporcionada al volumen de queries; pipelines de ML donde el payload cambia de distribución entre llamadas y un hiperparámetro fijo de ponderación introduciría sesgo sistemático.

**Cuándo NO usarla:** vectores puramente densos y homogéneos donde cosine puro es suficiente y se requiere latencia sub-20 ms — scipy o FAISS local son superiores en velocidad bruta con ~50% menos latencia p99. Tampoco es la elección correcta si el caso de uso requiere búsqueda sobre corpus indexado de millones de vectores con recuperación por ANN (Approximate Nearest Neighbor): esta API es stateless por diseño, no un índice.