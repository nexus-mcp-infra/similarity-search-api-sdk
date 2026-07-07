# Análisis de Complejidad Computacional — Similarity Search API

## Endpoints Públicos

### `POST /similarity` — Score híbrido NMI+Cosine entre dos items

**Temporal:** O(F_cat · H + F_cont · D + B) donde F_cat = features categoriales, H = cardinalidad máxima por feature (cálculo NMI via tabla de contingencia), F_cont = dimensión del embedding Cosine, B = n_bootstrap = 500 remuestreos fijos. **Mejor caso:** features 100% continuas, sin paso NMI → O(F_cont · D + B); **peor caso:** features mixtas con cardinalidad H elevada → O(F_cat · H² + F_cont · D + B), dominado por la tabla de contingencia NMI cuando H > 50. **Espacial:** O(H²) por la matriz de contingencia por feature categorial + O(B) para la distribución bootstrap. **Cuello de botella:** el bootstrap CI con n=500 es O(B · F) constante pero añade ~15–25ms fijos por llamada independientemente del tamaño del input; no escala con datos, pero tampoco se puede eliminar sin perder los intervalos de confianza.

---

### `POST /similarity/batch` — Score híbrido sobre N pares simultáneos

**Temporal:** O(N · (F_cat · H² + F_cont · D + B)) en el caso general; la inferencia de tipo de feature se ejecuta una vez por par, no una vez por batch, porque cada par puede tener schema distinto. **Mejor caso:** N pares homogéneos con features continuas → reutilización de embeddings Cosine reduce a O(N · D + B); **peor caso:** N pares con schemas heterogéneos y H variable → sin posibilidad de vectorizar el paso NMI entre pares. **Espacial:** O(N · H²) — matrices de contingencia no se comparten entre pares; con N=100 y H=100, esto equivale a ~4MB en float32. **Cuello de botella:** la ausencia de schema declarado fuerza inferencia de tipo en cada par; si el cliente puede declarar schema, este paso O(F) se elimina del hot path.

---

### `GET /similarity/calibration` — Consulta de pesos NMI/Cosine recalibrados por dominio

**Temporal:** O(1) — lectura de tabla de pesos pre-computados desde ClickHouse con query puntual por `domain_id`. **Mejor / promedio / peor:** O(1) en los tres casos; la varianza es de latencia de red al cluster ClickHouse, no de cómputo. **Espacial:** O(1) respuesta. **Cuello de botella:** dependencia de latencia externa a ClickHouse (~2–5ms p50, ~20ms p99); un cache LRU por `domain_id` con TTL de 60s lo convierte en O(1) local en el 95% de llamadas.

---

## Saturación y Estrategia de Escala

Con Uvicorn en modo async y workers = 2 × CPU_cores, el punto de saturación se estima en **~120 req/s por instancia** para el endpoint `/similarity` con configuración mixta típica (F_cat ≈ 5, H ≈ 20, F_cont ≈ 384 dimensiones), limitado por el bootstrap fijo de 500 remuestreos que ocupa ~18ms de CPU por llamada. Para escalar más allá: (1) mover el bootstrap a un worker thread pool para no bloquear el event loop, (2) reducir n_bootstrap adaptativamente a 200 cuando el score híbrido supera 0.85 de confianza preliminar — los módulos `src/math/statistics` soportan early stopping por convergencia de CI — y (3) cachear la inferencia de tipo de feature por hash de schema para eliminar O(F) redundante en batches homogéneos.