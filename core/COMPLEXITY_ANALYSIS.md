# Análisis de Complejidad Computacional — Hybrid Similarity Search API

## Endpoints Públicos

### `POST /score` — Score híbrido NMI + Cosine para un par de items

**Complejidad temporal:** O(F_cat · N_cat + F_cont · d) donde F_cat es el número de features categóricas, N_cat el número de categorías únicas por feature, y d la dimensión del espacio continuo. El cálculo de NMI requiere construir la tabla de contingencia por feature (O(N_cat²) en el peor caso con cardinalidad máxima), mientras que Cosine opera en O(d). **Mejor caso:** features continuas puras, degrada a O(d) puro. **Peor caso:** features categóricas de alta cardinalidad con N_cat >> d, donde la entropía conjunta domina. **Cuello de botella:** construcción de la tabla de contingencia por feature categórica — no vectorizable con BLAS, rompe el pipeline de operaciones matriciales.

**Complejidad espacial:** O(F_cat · N_cat²) para las tablas de contingencia + O(d) para los vectores continuos. Stateless: ninguna estructura persiste entre llamadas.

---

### `POST /batch_score` — Score híbrido sobre corpus de hasta N items contra una query

**Complejidad temporal:** O(N · (F_cat · N_cat + F_cont · d)) — lineal en el tamaño del corpus. La detección automática de tipos y calibración de pesos w_cat, w_cont se ejecuta una vez sobre el corpus completo en O(N · F) y se reutiliza para todos los pares; sin este memoizado interno por request, el coste sería O(N² · F). **Mejor caso:** corpus homogéneo continuo, O(N · d). **Promedio:** datasets mixtos reales, O(N · 500 · 50) para F_cat=10, N_cat=50, d=384. **Peor caso:** corpus de N=500 items con features categóricas de cardinalidad 200+, donde el coste de NMI escala cuadráticamente en N_cat. **Cuello de botella:** la calibración de pesos w sobre el corpus completo — es el único paso no paralelizable dentro del request porque requiere estadísticas globales del corpus antes de puntuar cada par.

**Complejidad espacial:** O(N · F) para materializar el corpus en memoria + O(F_cat · N_cat²) para las tablas de contingencia — el caso límite real es ~50 MB para N=500, F=50, N_cat=100.

---

### `POST /detect_feature_types` — Inferencia automática de tipos y pesos w

**Complejidad temporal:** O(N · F) — un pase lineal sobre el corpus para calcular entropía por columna y ratio continuo/categórico. Sin caso patológico conocido dado que no hay estructuras cuadráticas. **Cuello de botella:** I/O de deserialización JSON del corpus, no el cálculo estadístico.

**Complejidad espacial:** O(F) — solo acumula contadores por columna.

---

## Punto de Saturación y Estrategia de Escala

Con un corpus de N=500 y F=30 features mixtas, el coste dominante por request es ~12–40 ms de CPU pura (tabla de contingencia NMI), lo que sitúa el punto de saturación en **25–80 req/s por instancia single-core** antes de que la cola de Uvicorn empiece a crecer. Para escalar más allá: (1) paralelizar el cálculo de NMI por feature categórica con `concurrent.futures.ProcessPoolExecutor` dado que cada tabla de contingencia es independiente, reduciendo el cuello de botella a O(max_F_cat · N_cat) en vez de O(sum); (2) cachear los vectores de pesos w_cat/w_cont por hash del esquema de features detectado — en pipelines de CI donde el schema es estable, esto convierte `batch_score` en O(N · d) efectivo desde la segunda llamada.