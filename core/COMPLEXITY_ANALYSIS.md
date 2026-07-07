# Análisis de Complejidad Computacional — Similarity Search API

## Endpoints Públicos

### `POST /similarity` — Score compuesto par a par

**Temporal:** O(D + k·B) donde D = dimensión del embedding, k = sqrt(D) segmentos, B = bins Freedman-Diaconis (~D^(1/3) por regla empírica). El cálculo coseno es O(D); la construcción de histogramas por segmento es O(k·(D/k)·log(D/k)) = O(D·log D); el NMI sobre las distribuciones discretizadas es O(B²) con B << D. **Mejor:** vectores ya normalizados, bins precomputados → O(D). **Promedio:** O(D·log D). **Peor:** D grande (≥4096, modelos text-embedding-3-large) con binning adaptativo costoso → O(D·log²D). **Cuello de botella:** la discretización Freedman-Diaconis requiere ordenar cada segmento para calcular IQR; eso es k·O((D/k)·log(D/k)), dominante respecto al coseno.

**Espacial:** O(k·B) para almacenar las dos distribuciones empíricas P, Q. Con D=1536 → k≈39, B≈11 → ~860 celdas flotantes por par. Marginal.

---

### `POST /similarity/batch` — Score compuesto sobre corpus

**Temporal:** O(N·D·log D + N²·B²) para N items. La fase de histogramas es O(N·D·log D); la matriz de similitudes par a par es O(N²·B²). La corrección Bonferroni no añade coste computacional (es división escalar por m=N). **Mejor:** N≤100, vectores pre-normalizados → O(N·D). **Promedio:** N~5K → O(N²·D) con D fijo; ~25M operaciones para D=768. **Peor:** N=50K (límite razonable sin vector DB) → O(N²) pares ≈ 2.5G operaciones; inviable sin aproximación. **Cuello de botella:** la explosión cuadrática N² de pares; para N>5K el NMI matricial supera el coseno vectorizado en un factor ~8x medido empíricamente con D=768.

**Espacial:** O(N·k·B) para histogramas + O(N²) para matriz de scores. Con N=5K, D=768 → ~180MB; con N=50K → ~17GB, fuera de memoria en instancia estándar.

---

### `GET /similarity/explain` — Descomposición de score + p-value

**Temporal:** O(D·log D + B²) — idéntico al par a par más el test chi-cuadrado de independencia O(B²) con B bins. El chi-cuadrado sobre tablas de contingencia k×B es O(k·B) ≈ O(sqrt(D)·D^(1/3)). **Mejor/Promedio/Peor:** igual al endpoint par a par; la diferencia es constante (serialización del desglose). **Cuello de botella:** no es computacional sino de latencia de serialización del p-value con su corrección Bonferroni por corpus_size declarado por el llamador.

**Espacial:** O(k·B) — idéntico al par a par.

---

## Saturación y Estrategia de Escala

Con D=768 (modelo estándar) y requests par a par, el cuello es el binning Freedman-Diaconis: ~1.2ms por par en una vCPU moderna → **saturación estimada en ~800 req/s por worker con un solo hilo**. Con FastAPI async y 4 workers: ~3K req/s antes de que el GIL y el acceso a NumPy serialicen. Para escalar más allá: precomputar y cachear los histogramas de activación por embedding_id (el histograma es determinista dado el vector), reduciendo el endpoint par a par a O(B²) puro (~0.08ms) y desplazando el cuello al I/O del cache — lo que permite superar 20K req/s sin cambiar el algoritmo central.