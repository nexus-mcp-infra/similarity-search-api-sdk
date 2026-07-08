# Justificación Matemática — Ephemeral Similarity Search API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de un integrador sigue $T = b \cdot \log_2(n+1)$. Con $n=5$ endpoints, $T \approx 2.58b$; con $n=10$, $T \approx 3.46b$ — un 34% de fricción cognitiva adicional sin ganancia funcional. Para una primitiva stateless cuya propuesta es reducir setup a cero, la superficie de API es parte del producto: cada endpoint extra contradice el invariante de simplicidad y aumenta la tasa de abandono en onboarding, que es exactamente el dolor que se está monetizando.

## 2. Pricing Per-Call vs Por Asiento (Elasticidad Precio-Demanda)

La elasticidad precio-demanda $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$ es distinta por segmento: equipos pequeños con corpus efímeros tienen $|\varepsilon| > 1$ (demanda elástica) — una suscripción fija los expulsa del mercado. El modelo per-call alinea el precio con la unidad de valor entregado: una ejecución $O(n \cdot d)$ sobre corpus de tamaño $n$ y dimensión $d$, donde el coste marginal crece con la carga computacional real. Esto también elimina el coste de almacenamiento vectorial persistente para el cliente, convirtiendo CAPEX en OPEX puro y reduciendo la barrera de entrada a cero.

## 3. Estructura de Datos y Complejidad Algorítmica

El corpus se recibe como matriz $X \in \mathbb{R}^{n \times d}$ por llamada, sin índice previo. La similitud coseno se calcula en $O(n \cdot d)$ mediante producto matricial normalizado $S_{\cos}(q, x_i) = \frac{q \cdot x_i}{\|q\| \|x_i\|}$. La estimación KDE para NMI evalúa la densidad conjunta $\hat{p}(x,y)$ en $O(n^2)$ en el caso general, pero con kernel gaussiano isotrópico y anchura de banda $h = 1.06 \hat{\sigma} n^{-1/5}$ (regla de Silverman) la convergencia es suficiente para $n \leq 10^4$ sin discretización. Elegir KDE sobre histograma elimina el sesgo $O(h^2)$ que hace inestable NMI en alta dimensión con bins fijos.

## 4. Invariante Matemático de Corrección

El score compuesto $\mathcal{F}(\alpha) = \alpha \cdot \widetilde{\text{NMI}}(q, x_i) + (1-\alpha) \cdot S_{\cos}(q, x_i)$ es correcto si y solo si $\alpha$ es función monótona de la varianza inter-ítem del corpus: $\alpha = \sigma^2_{\text{inter}} / (\sigma^2_{\text{inter}} + \sigma^2_{\text{intra}})$. Este peso adaptativo garantiza que en corpus de alta varianza (ruidosos), NMI domina — capturando dependencia estadística no lineal que coseno pierde — mientras que en corpus homogéneos, coseno es suficiente y computacionalmente más barato. El invariante es: $\mathcal{F}$ es una métrica de similitud válida (simétrica, acotada en $[0,1]$) para cualquier $\alpha \in [0,1]$, lo que garantiza que el ranking es total y reproducible.

## 5. Límites Teóricos del Sistema

La estimación KDE de la distribución conjunta sufre la maldición de la dimensionalidad: para $d > 50$, el estimador necesita $n \gg d^5$ muestras para que $h$ óptimo sea informativo, haciendo que NMI converja lentamente y aporte ruido en lugar de señal. La API no puede sustituir a un índice ANN (HNSW, IVF) para $n > 10^4$: la complejidad $O(n^2)$ de KDE y $O(n \cdot d)$ de coseno hacen la latencia prohibitiva sin aproximación. El sistema es stateless por diseño — no puede aprender representaciones ni refinar embeddings; asume que los vectores de entrada ya codifican semántica útil.