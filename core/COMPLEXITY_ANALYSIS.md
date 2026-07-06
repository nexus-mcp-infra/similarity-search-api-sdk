# Análisis de Complejidad Computacional — Similarity Search API (NMI+Cosine)

---

## Endpoints Públicos

### `POST /similarity` — Score compuesto par-a-par

**Temporal:** O(n · B) donde n = dimensiones del vector y B = bins para discretización NMI. El cálculo de entropía marginal H(Xᵢ) requiere un paso sobre los n valores del vector query para estimar distribución; cosine añade O(n) multiplicaciones. Total: O(n · B + n) = O(n · B).

**Espacial:** O(n · B) para las tablas de frecuencia conjunta y marginal por dimensión.

**Mejor caso:** vectores de baja dimensionalidad (n < 50) con distribuciones uniformes donde B se reduce automáticamente — O(n). **Promedio:** n ≈ 256–512, B ≈ 10–20, latencia dominada por discretización. **Peor caso:** n > 1024 con distribuciones multimodales que fuerzan B alto — O(n · B) degrada linealmente con ambos.

**Cuello de botella:** estimación de bins óptimos (regla de Sturges o Scott) sobre cada dimensión antes de construir las tablas de frecuencia — es secuencial por dimensión y no trivialmente paralelizable sin overhead de sincronización.

---

### `POST /similarity/batch` — Score compuesto sobre conjunto de candidatos

**Temporal:** O(k · n · B) donde k = número de vectores candidatos en el payload. La ponderación por entropía w_i = H(Xᵢ) / Σ H(Xⱼ) se calcula una sola vez sobre el vector query y se reutiliza para los k comparandos — esto evita recomputar la distribución marginal k veces, dejando el costo dominante en O(k · n) para cosine y O(k · n · B) para NMI por candidato.

**Espacial:** O(n · B + k · n) — tablas de frecuencia más el buffer de candidatos en memoria.

**Mejor caso:** k pequeño (< 20) con candidatos preordenados externamente. **Peor caso:** k = 1000, n = 512 — aproximadamente 512K operaciones de binning. **Cuello de botella:** carga del payload completo en memoria antes de iniciar cómputo; sin streaming, el pico de RAM escala con k · n · sizeof(float64).

---

### `POST /similarity/explain` — Descomposición dimensional del score

**Temporal:** O(n · B) idéntico a `/similarity` más O(n log n) para ordenar dimensiones por contribución al score final. El sort es el componente adicional respecto al endpoint base.

**Espacial:** O(n · B + n) — mismo footprint más el vector de contribuciones por dimensión.

**Mejor / peor caso:** igual que `/similarity`; el sort de contribuciones es despreciable frente al binning. **Cuello de botella:** serialización del breakdown dimensional completo en la respuesta JSON cuando n > 512 — el payload de salida crece O(n).

---

## Saturación y Estrategia de Escala

Con n = 256 y k = 1 (llamadas par-a-par), cada request completa en ~4–8 ms en un worker Uvicorn single-core, lo que sitúa el punto de saturación de una instancia en ~120–250 req/s antes de que la cola de eventos de FastAPI empiece a acumular latencia. El cuello de botella no es I/O sino CPU puro en el bucle de binning. La estrategia de escala natural es horizontal stateless — cada réplica es independiente por diseño — con un balanceador L7 sin sticky sessions; añadir réplicas escala linealmente hasta el límite de red. Para batch con k > 100, paralelizar el bucle de candidatos con `numpy` vectorizado sobre el eje k (operaciones matriciales en lugar de Python loops) reduce O(k · n · B) a una operación de broadcast que aprovecha BLAS, ganando ~8–15x en throughput por core antes de añadir réplicas.