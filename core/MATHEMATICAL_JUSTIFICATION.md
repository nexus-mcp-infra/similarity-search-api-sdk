# Justificación Matemática: NMI-Weighted Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

Hick's Law establece $T = b \cdot \log_2(n+1)$: el tiempo de decisión crece logarítmicamente con el número de alternativas. Con $n=5$ endpoints el tiempo de selección es $b \cdot \log_2(6) \approx 2.58b$; con $n=10$ sube a $3.46b$ — un 34% más de fricción cognitiva sin ganancia funcional. Para esta primitiva, la superficie mínima viable es `POST /similarity`, `POST /batch`, `GET /confidence-percentiles`, `GET /health`, `POST /explain` — cada uno resuelve una responsabilidad no solapada y el espacio de decisión permanece en $O(\log n)$.

## 2. Pricing Per-Call vs. Por Asiento (Elasticidad)

La elasticidad precio-demanda $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ para infraestructura de búsqueda es altamente elástica en el eje de adopción inicial ($|\varepsilon| > 1$): un coste fijo mensual eleva la barrera de entrada para datasets $< 100$k items, que es exactamente el segmento objetivo. El pricing per-call traslada el coste marginal al valor marginal generado: una búsqueda sobre 50k vectores con NMI-filtering cuesta $O(d \cdot k \cdot \log k)$ operaciones, donde $d$ es la dimensión original y $k$ el número de features seleccionadas por NMI — ese coste computacional real se refleja directamente en el precio, alineando incentivos sin subsidiar uso inactivo.

## 3. Estructura de Datos y Complejidad Algorítmica

El pipeline NMI -> Cosine opera en dos fases con complejidades distintas: NMI sobre $d$ features con $n$ observaciones cuesta $O(n \cdot d \cdot \log d)$ usando histogramas de frecuencia conjunta; la proyección al subespacio de $k$ features seleccionadas ($k \ll d$) reduce el Cosine posterior a $O(n \cdot k)$. La representación interna usa arrays NumPy contigüos en memoria (C-order) para aprovechar BLAS Level-2 en el producto punto normalizado — frente a representaciones dispersas, el acceso secuencial reduce cache misses en $\sim 4\times$ para $d \leq 2048$. La clave es que la selección NMI actúa como proyección de rango: $\text{rank}(X_{\text{filtered}}) \leq k < d$, haciendo la similitud posterior numéricamente más estable.

## 4. Invariante Matemático de Corrección

El invariante central es que $\text{NMI}(X_i, Y) \in [0, 1]$ es una medida de información normalizada y simétrica: $\text{NMI} = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)}$, donde $I(X;Y)$ es la información mutua y $H(\cdot)$ la entropía de Shannon. Esto garantiza que el peso asignado a cada feature es **invariante a la escala y a la cardinalidad** del dominio — no puede volverse negativo ni amplificar ruido. El intervalo de confianza devuelto es $\hat{\mu}_{\text{NMI}} \pm z_{\alpha/2} \cdot \hat{\sigma} / \sqrt{n_{\text{corpus}}}$, donde $\hat{\sigma}$ proviene de la distribución empírica acumulada en PostgreSQL: a medida que $n_{\text{corpus}} \to \infty$, el intervalo colapsa hacia cero por la ley de los grandes números, haciendo el sistema auto-calibrante.

## 5. Límites Teóricos del Sistema

El pipeline es stateless por diseño, lo que implica que **no puede aprender dependencias temporales ni ordinales** entre queries sucesivas — no es un sustituto de embedding stores con índices HNSW ($O(\log n)$ búsqueda aproximada) cuando el corpus supera $\sim 500$k items, donde la ventaja de indexación persistente domina al coste de NMI. Adicionalmente, NMI asume que la dependencia estadística entre features es capturável con distribuciones marginales discretizadas: para features con distribuciones multimodales de cola pesada ($\kappa > 6$), la estimación de entropía por histograma introduce sesgo $O(1/n)$ que no desaparece con corpus pequeños — ese es el régimen donde el intervalo de confianza se ensancha correctamente como señal de advertencia, no se oculta.