# Justificación Matemática: Similarity Search API con NMI+Cosine Híbrido

## 1. Por qué máximo 5 endpoints (Hick's Law)

Hick's Law: $T = b \cdot \log_2(n+1)$ predice que el tiempo de decisión de integración crece logarítmicamente con el número de opciones. Con 5 endpoints el desarrollador toma decisiones de routing en $T = b \cdot \log_2(6) \approx 2.58b$; con 10 endpoints ese costo escala a $3.46b$ — un 34% de fricción cognitiva adicional que se traduce directamente en tiempo hasta primera llamada exitosa. Para una primitiva que compite contra scikit-learn (sin HTTP) y FAISS (sin NMI), reducir ese tiempo es más importante que la exhaustividad funcional.

## 2. Por qué pricing per-call vs por asiento (elasticidad precio-demanda)

La elasticidad precio-demanda para herramientas de análisis ad-hoc es alta en magnitud ($|\varepsilon| > 1$): un developer que compara 500 items una vez por semana no justifica una suscripción fija, pero sí paga $\Delta C \propto n \cdot O(n \log n)$ por operación donde el valor percibido escala con el tamaño del corpus. El modelo per-call alinea $P(q) = c \cdot q$ con la utilidad marginal real del usuario — que es proporcional al corpus procesado, no al número de personas en su equipo. Suscripción fija introduce riesgo de infrautilización que destruye disposición a pagar.

## 3. Por qué esta estructura de datos (complejidad algorítmica)

El corpus llega en memoria por petición como matriz $X \in \mathbb{R}^{n \times d}$, evitando índice persistente. Para la ruta de vectores densos, el binning adaptativo sobre $d$ dimensiones produce histogramas conjuntos en $O(n \log n)$ por par vía sort-based binning — frente a $O(n^2)$ de un kernel de densidad ingenuo. La corrección de Strehl-Ghosh añade $O(k)$ overhead constante donde $k$ es el número de bins, asintóticamente despreciable. Mantener el corpus en RAM por petición elimina el costo $O(\log N)$ de lookup en índice persistente para corpus $n < 10^4$, que es exactamente el rango del dolor declarado.

## 4. El invariante matemático que hace esta solución correcta

El invariante es que $\text{NMI}_{\text{SG}}(X,Y) \in [0,1]$ se mantiene bajo muestras pequeñas solo con la corrección de Strehl-Ghosh: $\text{NMI}_{\text{SG}} = \frac{MI(X;Y)}{\sqrt{H(X) \cdot H(Y)}}$ donde $H$ se estima con corrección de Miller-Madow $\hat{H} = H_{\text{plug-in}} + \frac{m-1}{2n}$ para reducir el bias $O(1/n)$. Sin esta corrección, para $n < 200$ el estimador plug-in sobreestima $MI(X;Y)$ sistemáticamente, produciendo rankings con inversiones de orden no aleatorias. El scoring híbrido $S = \alpha \cdot \text{NMI}_{\text{SG}} + (1-\alpha) \cdot \cos\theta$ preserva el invariante $S \in [0,1]$ para todo $\alpha \in [0,1]$ porque ambas métricas componentes están normalizadas en el mismo intervalo.

## 5. Límites teóricos del sistema

El diseño es correcto pero acotado: la estimación de distribución conjunta por binning asume que las dimensiones de $X$ son suficientemente estacionarias para que un histograma discreto sea una aproximación válida — esto falla para distribuciones multimodales con alta varianza en $d > 50$ donde el número de bins necesarios crece exponencialmente ($k^d$) y la corrección de Miller-Madow ya no controla el bias. El sistema tampoco puede garantizar ordenamiento total determinista cuando $|S_i - S_j| < \varepsilon_{\text{float64}} \approx 10^{-15}$, caso que ocurre con vectores casi idénticos; y por diseño no escala a $n > 10^5$ en una sola llamada sin degradar de $O(n \log n)$ a comportamiento I/O-bound por serialización HTTP.