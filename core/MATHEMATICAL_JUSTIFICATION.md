# Justificación Matemática: Similarity Search API (NMI+Cosine Stateless)

## 1. Máximo 5 Endpoints — Hick's Law

$$T = b \cdot \log_2(n+1)$$

Con $n=5$ opciones, el tiempo de decisión del integrador es $T = b \cdot \log_2(6) \approx 2.58b$, frente a $3.46b$ con $n=10$. La API expone exactamente las operaciones atómicas no solapadas que el problema requiere: comparar, rankear, y calibrar — nada más. Cada endpoint adicional sin responsabilidad única aumenta la carga cognitiva sin aumentar la superficie de valor, lo que eleva el tiempo de integración y reduce la tasa de conversión comercial.

## 2. Pricing Per-Call — Elasticidad Precio-Demanda

$$E_d = \frac{\partial Q / Q}{\partial P / P}$$

El caso de uso central es comparación ad-hoc sin estado persistente: un developer que resuelve un problema puntual no tolera suscripción fija con coste amortizado sobre volumen incierto. La elasticidad por operación es negativa y alta en magnitud ($|E_d| > 1$) para uso esporádico, lo que significa que una suscripción fija destruye demanda en la cola larga. Per-call convierte el coste marginal del usuario en función lineal del valor recibido, alineando incentivos sin fricción de commit.

## 3. Estructura de Datos — Complejidad Algorítmica

El payload ad-hoc se representa como matriz $X \in \mathbb{R}^{n \times d}$ procesada en memoria sin persistencia. El cálculo de entropía marginal $H(X_i) = -\sum_k p_k \log_2 p_k$ por dimensión es $O(n)$ con estimación por histograma discretizado; la ponderación $w_i = H(X_i) / \sum_j H(X_j)$ es $O(d)$; cosine similarity sobre vectores densos es $O(d)$; NMI entre dos dimensiones es $O(n \log n)$ por el sort implícito en la estimación conjunta. El coste total por llamada es $O(n \cdot d \cdot \log n)$ — dominado por NMI, no por cosine, lo que justifica que el cuello de botella esté en la estimación de distribución conjunta, no en el álgebra lineal.

## 4. Invariante Matemático

El score compuesto garantiza el siguiente invariante: para cualquier payload donde las dimensiones sean estadísticamente independientes ($I(X_i; X_j) = 0$ para todo $i \neq j$), el score colapsa a cosine puro ponderado por entropía uniforme. Formalmente:

$$S = \sum_i w_i \cdot \left[\alpha \cdot \text{NMI}(X_i, Y_i) + (1-\alpha) \cdot \cos(x_i, y_i)\right], \quad w_i = \frac{H(X_i)}{\sum_j H(X_j)}$$

Este invariante garantiza que el score nunca amplifica dimensiones de entropía cero (constantes), que no aportarían información mutua ni dirección vectorial — la ponderación las anula automáticamente sin intervención del usuario.

## 5. Límites Teóricos

La estimación de $H(X_i)$ asume que el histograma discretizado converge a la distribución real con $n \geq 30$ observaciones por dimensión (teorema central del límite sobre frecuencias relativas); con $n < 30$, la entropía marginal está sesgada hacia arriba y los pesos $w_i$ pierden calibración. El sistema no puede operar en streaming incremental sin acumular el payload completo, porque NMI requiere la distribución conjunta global — esto es una consecuencia directa del principio de procesamiento de información de Shannon: la información mutua entre variables no se puede estimar localmente sobre una submuestra sin sesgo garantizado. Por el mismo motivo, el score no es aditivo entre llamadas parciales: $I(X;Y|Z) \neq I(X;Y) + I(Z;Y)$ en el caso general.