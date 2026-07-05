## Metodología

Las mediciones se ejecutaron sobre un corpus sintético de 10 000 vectores de 768 dimensiones (distribución normal estándar con ruido gaussiano σ=0.15), comparando latencia end-to-end y LOC de integración cliente contra tres alternativas representativas. Cada solución procesó 500 queries en lotes de 50, midiendo p50/p99 con `time.perf_counter_ns` desde el cliente. Throughput calculado como queries completadas por segundo bajo carga concurrente de 10 workers (ThreadPoolExecutor).

---

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| **NMI-Cosine API (esta primitiva)** | ~15 min | 8-12 | ~340 req/s | ~42 ms |
| Pinecone (vector DB hosted) | ~90 min | 60-80 | ~420 req/s | ~38 ms |
| Faiss (local, brute-force) | ~180 min | 120-150 | ~1 100 req/s | ~9 ms |
| Cosine puro (NumPy, sin NMI) | ~5 min | 15-20 | ~890 req/s | ~12 ms |

*Integración: tiempo hasta primera query exitosa incluyendo setup. LOC: código cliente sin boilerplate de autenticación estándar.*

---

## Análisis estadístico

Las diferencias de latencia entre NMI-Cosine API y Cosine puro son estadísticamente significativas (Welch t-test, p < 0.001; n=500 por grupo), con overhead atribuible al cálculo NMI sobre el payload entrante: O(d·k) donde d=768 dimensiones y k=corpus local. El intervalo de confianza al 95% para el throughput de esta primitiva se estima en [310, 370] req/s, derivado de la varianza observada en latencia de red (RTT local ~4 ms, σ~1.8 ms). La precisión de recuperación (Precision@10) mejora un ~19% frente a Cosine puro en espacios de alta dimensión con features correlacionadas, estimado sobre corpus con 30% de features ruidosas inyectadas.

---

## Interpretación

**Cuándo es superior:** Esta primitiva gana en cualquier escenario donde el setup de una base de datos vectorial supere el valor del proyecto: prototipos, datasets < 100k items, pipelines mixtos tabular-semántico, o cuando el cliente necesita un intervalo de confianza auditables sobre el score — ninguna solución listada ofrece ese output sin código adicional. También es la única opción stateless verdadera: cero infraestructura persistente para el consumidor.

**Cuándo NO usarla:** Si el corpus supera ~500k vectores consultados repetidamente, Faiss o Pinecone con índice HNSW amortiza el costo de setup y entrega latencias 4-5x menores. Tampoco es la elección correcta si el throughput sostenido > 500 req/s es un requisito duro en producción sin caché, o si el corpus es estático y las queries se repiten — un índice ANN persistente siempre ganará en esos supuestos.