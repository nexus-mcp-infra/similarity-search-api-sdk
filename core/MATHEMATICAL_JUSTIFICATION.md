# Justificación Matemática: Similarity Search API con NMI Compuesto

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de integración sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints el término logarítmico vale $\approx 2.58b$; duplicar la superficie a $n=10$ lo eleva a $\approx 3.46b$, un 34% de fricción cognitiva adicional sin valor semántico nuevo. Para esta primitiva, las operaciones atómicas son exhaustivas con cinco: comparar par, buscar en corpus, calibrar alpha, consultar p-value y describir corpus — cualquier endpoint adicional sería composición de estos, no primitiva nueva.

## 2. Pricing Per-Call vs. Por Asiento (Elasticidad)

La elasticidad precio-demanda para datasets menores a 500K items es altamente elástica ($|\epsilon| > 1$) porque el desarrollador tiene sustituto viable: implementar coseno crudo. El pricing por índice/namespace impone costo fijo $C_f > 0$ incluso con demanda $q=0$, desplazando la curva de demanda hacia abajo y eliminando casos de uso esporádicos o por lotes. El modelo per-call convierte $C_f = 0$ y cobra solo $C_v \cdot q$, capturando el excedente del consumidor en el segmento de uso irregular donde la primitiva tiene mayor ventaja comparativa frente a una vector DB sobreingeniada.

## 3. Estructura de Datos: Histogramas con Binning Freedman-Diaconis

El ancho de bin óptimo $h = 2 \cdot \text{IQR}(x) \cdot n^{-1/3}$ minimiza el error cuadrático medio integrado (MISE) del estimador de densidad, produciendo $k = \lceil (x_{max} - x_{min}) / h \rceil$ bins adaptativos al rango real de magnitudes del embedding. Esto es crítico porque los embeddings de transformers exhiben distribuciones de magnitud por segmento fuertemente no uniformes — bins fijos de ancho constante subestimarían $H(P)$ en colas pesadas, sesgando $\text{NMI}(P,Q)$ hacia cero espuriamente. La partición del embedding en $k = \lfloor\sqrt{D}\rfloor$ segmentos garantiza que cada segmento capture una región del espacio de activación con dimensionalidad $O(\sqrt{D})$, balanceando resolución local contra varianza estadística del histograma empírico.

## 4. Invariante Matemático de Corrección

El invariante es que $\text{NMI}(P,Q) \in [0,1]$ es invariante a transformaciones monotónas de escala en las magnitudes, mientras que la similitud coseno no lo es cuando el embedding space tiene anisotropía dimensional (subespacios más densos que otros). El score compuesto $S = \alpha \cdot \cos(\theta) + (1-\alpha) \cdot \text{NMI}(P,Q)$ con $\alpha=0.6$ mantiene el invariante de que $S=1$ sii los vectores son idénticos bajo ambas métricas simultáneamente, y el p-value con corrección Bonferroni $p_{adj} = \min(1, p_{raw} \cdot m)$ donde $m = \text{corpus\_size}$ garantiza control de FWER: la probabilidad de al menos un falso positivo en el corpus completo se mantiene $\leq 0.05$, haciendo el threshold interpretable globalmente.

## 5. Límites Teóricos del Sistema

El test $\chi^2$ de independencia requiere frecuencias esperadas $\geq 5$ por celda para que la aproximación asintótica sea válida; con embeddings de $D < 25$ (partición $k < 5$) o corpus $< 50$ items, el p-value pierde calibración y la API lo señaliza explícitamente. El NMI captura dependencia de primer orden entre distribuciones marginales de segmentos — no captura dependencias de orden superior entre segmentos cruzados ($O(k^2)$ comparaciones), lo que significa que embeddings con correlación inter-segmento alta pero NMI marginal bajo seguirán siendo indistinguibles de independencia real. Finalmente, la complejidad de búsqueda lineal en corpus es $O(n \cdot D)$ sin indexación aproximada; para $n > 500\text{K}$ el SLA de latencia colapsa y el producto explícitamente no es sustituto de HNSW o FAISS a esa escala.