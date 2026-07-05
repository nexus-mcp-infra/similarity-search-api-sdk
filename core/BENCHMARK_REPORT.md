## Metodología

Tests ejecutados sobre corpus sintético de 10k, 50k y 100k items con payloads mixtos (3–8 features categóricas + 2–4 continuas), representativos de catálogos e-commerce y datasets de texto estructurado. Latencia medida con `wrk` (16 conexiones concurrentes, 30s por run, 5 runs por condición); throughput como RPS sostenido sin degradación de p99 > 20% respecto a p50. Integración medida en minutos wall-clock desde `pip install` hasta primera respuesta válida con payload real.

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **Similarity Search API** | 4 min | 12 LOC | 1,840 RPS | 38 ms |
| Pinecone + NMI custom pipeline | 47 min | 310 LOC | 2,100 RPS | 29 ms |
| Weaviate local + custom scorer | 61 min | 480 LOC | 1,950 RPS | 31 ms |
| scikit-learn pipeline manual | 18 min | 190 LOC | 620 RPS | 112 ms |
| Redis HNSW (solo coseno) | 22 min | 145 LOC | 2,400 RPS | 22 ms |

*Throughput medido en corpus 50k items, payload promedio 6 features. RPS de Pinecone/Weaviate excluye tiempo de indexación previa (Pinecone: ~8 min adicionales para 50k items).*

## Análisis estadístico

Intervalos de confianza al 95% para latencia p99: Similarity Search API [34 ms, 42 ms]; Pinecone pipeline [25 ms, 33 ms]; scikit-learn [98 ms, 127 ms]. Diferencia en throughput entre esta API y scikit-learn es estadísticamente significativa (Mann-Whitney U, p < 0.001, n=150 muestras por condición); diferencia respecto a Pinecone no es significativa en throughput bruto (p = 0.21), confirmando que el gap real es de complejidad operacional, no de rendimiento crudo. Varianza de latencia p99 en esta API es la más baja del grupo (CV = 0.06 vs 0.14 en scikit-learn), indicando comportamiento predecible bajo carga.

## Interpretación

**Esta API es superior cuando:** el corpus es menor de 100k items y no existe infraestructura de índices previa; cuando el payload combina features categóricas y continuas heterogéneas sin esquema fijo; y cuando se necesita explicabilidad del score por componente (qué fracción aporta NMI vs coseno) sin instrumentación adicional. El cálculo automático de `w_nmi` por entropía marginal elimina el parámetro de tuning más costoso en tiempo de desarrollo de cualquier pipeline híbrido equivalente.

**No usar esta API cuando:** el corpus supera 500k items con actualizaciones frecuentes (>1k inserts/min), donde índices persistentes como HNSW recuperan ventaja neta en latencia; cuando el payload es exclusivamente vectorial denso (embeddings de 768+ dimensiones sin features categóricas), caso en que Redis HNSW ofrece p99 un 42% menor con menor overhead; o cuando el SLA exige latencia p99 < 25 ms de forma estricta.