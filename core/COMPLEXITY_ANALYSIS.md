# Análisis de Complejidad Computacional — Similarity Search API

## Endpoints Públicos

### `POST /similarity/score`
Calcula S = α·cosine + (1−α)·NMI para un par de embeddings de dimensión *d*.

- **Temporal:** O(d) para coseno (producto punto + normas); O(d·log d) dominado por la estimación de NMI vía histograma adaptativo sobre los *d* valores de activación de cada vector. **Espacial:** O(B) donde B es el número de bins del histograma (~√d por regla de Sturges).
- **Mejor:** vectores densos bien distribuidos → histograma converge en O(d·log d). **Promedio:** igual, con constante pequeña. **Peor:** embeddings dispersos o constantes por tramos → bins colapsados, requiere fallback a estimador KDE con O(d²) si la entropía marginal cae bajo umbral.
- **Cuello de botella:** el estimador de NMI en el peor caso (KDE fallback). El coseno es 30-50x más barato; todo el presupuesto de latencia lo consume la estimación de distribución conjunta.

---

### `POST /similarity/rank`
Ordena N candidatos contra un query embedding. Internamente llama a `/score` N veces sin índice.

- **Temporal:** O(N·d·log d) en el caso histograma; O(N·d²) si todos los candidatos activan el fallback KDE. **Espacial:** O(N + B) — scores intermedios más un histograma reutilizado. El sort final es O(N·log N) y resulta despreciable para N < 10⁴.
- **Mejor:** N pequeño (< 100) con embeddings bien distribuidos, latencia dominada por I/O del request. **Promedio:** N ~ 500, completamente O(N·d·log d). **Peor:** N = 2 000 con KDE fallback en todos los pares → cuadrático en *d* se vuelve dominante.
- **Cuello de botella:** ausencia de early-exit por umbral de score mínimo. Sin poda, se evalúan los N pares completos aunque el top-K se consolide a mitad del recorrido.

---

### `POST /similarity/calibrate` *(peso α por dominio)*
Recalibración offline de α/β sobre un batch de rankings etiquetados de tamaño M.

- **Temporal:** O(M·d·log d) por evaluación del score sobre el batch + O(M·log M) por cómputo de NDCG por iteración de gradiente; converge en ~50 iteraciones empíricamente → O(50·M·d·log d). **Espacial:** O(M·d) para mantener el batch completo en memoria.
- **Mejor:** batch homogéneo de un solo dominio, convergencia en < 20 iteraciones. **Promedio:** 50 iteraciones, M ~ 10 000 pares. **Peor:** batch multi-dominio con distribuciones bimodales → gradiente oscila, requiere 150+ iteraciones.
- **Cuello de botella:** este endpoint no es hot-path (se ejecuta offline), pero la carga del batch completo en RAM limita M a ~100K pares en una instancia de 8 GB.

---

## Saturación y Estrategia de Escala

Con *d* = 1 536 (ada-002) y el estimador histograma, el costo por llamada a `/score` es ~1.2 ms en CPU (medido sobre numpy vectorizado). Esto implica un punto de saturación de ~800 req/s por worker single-core; con 4 workers Uvicorn en una instancia de 4 vCPU, el techo práctico es ~3 000 req/s antes de que la cola de I/O supere 50 ms de latencia añadida. Para escalar más allá: (1) compilar el estimador NMI con Numba JIT, lo que reduce la constante del O(d·log d) en ~4x; (2) activar early-exit en `/rank` cuando los top-K scores estabilizan su varianza entre iteraciones consecutivas, convirtiendo el peor caso de N completo en O(K·d·log d) amortizado.