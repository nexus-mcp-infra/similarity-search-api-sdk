# Pricing

El modelo de suscripción fijo penaliza exactamente al desarrollador que más valor obtiene de una primitiva stateless: el que tiene carga variable. Un MVP que procesa quinientas búsquedas un martes y cero el fin de semana no debería pagar por capacidad ociosa, y un sistema de recomendación que escala durante eventos no debería negociar un tier superior antes de saber si el pico se repite. El precio decreciente por volumen refleja la estructura de costos real de la primitiva: el cálculo de entropía marginal sobre el corpus entrante tiene un costo computacional que se amortiza a medida que los batches crecen en tamaño y frecuencia, y esa eficiencia se traslada directamente al precio marginal sin que el usuario tenga que predecir su uso con semanas de antelación.

La ausencia de compromiso mínimo elimina la fricción de adopción que destruye el tiempo-a-valor en herramientas de infraestructura. El desarrollador que evalúa si la fusión NMI-cosine mejora su pipeline respecto a similitud coseno simple no puede justificar internamente un contrato mensual antes de ver los resultados en sus propios datos. Con tarifa por llamada puede hacer esa comparación en producción real, con su distribución real, y la decisión de escalar se toma sola cuando los números de calidad de ranking lo confirman. Un tier fijo convierte esa decisión técnica en una decisión presupuestaria prematura, y la mayoría de los equipos la evitan eligiendo la alternativa más barata aunque sea inferior.

El techo inexistente no es generosidad comercial, es consistencia con la naturaleza stateless de la primitiva. Un vector database tiene costos de almacenamiento que crecen con el corpus indexado; esta API no almacena nada entre requests, por lo que no existe un costo de infraestructura acumulado que justifique un precio plano una vez superado cierto umbral. Cobrar más por el primer millar de llamadas que por el décimo refleja que el cálculo del peso adaptativo sobre un corpus de alta entropía en un batch grande es marginalmente más barato por vector que sobre un corpus pequeño, porque los estimadores de entropía convergen con más muestras y la vectorización NumPy amortiza el overhead fijo de la llamada HTTP. El precio sigue la física del cómputo, no una convención de ventas.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |