# Pricing

El modelo de suscripción fija penaliza exactamente el patrón de uso que define a los desarrolladores que más se benefician de esta primitiva: cargas de trabajo batch esporádicas, pipelines de evaluación que se ejecutan una vez por sprint, o búsquedas de similitud sobre corpus que crecen de forma no lineal. Pagar capacidad reservada mensual por un servicio que se invoca intensamente durante cuarenta y ocho horas y luego no se toca durante tres semanas es el mismo antipatrón que esta API elimina en el lado de la infraestructura de vectores. El pricing usage-based con tarifa decreciente por volumen alinea el coste del llamador con el valor extraído en cada sesión: quien hace una búsqueda puntual paga por esa búsqueda; quien procesa un corpus completo en batch recibe el descuento marginal que refleja las economías de escala reales del cálculo de NMI a volumen.

La tarifa decreciente sin tiers discretos es técnicamente honesta respecto a cómo funciona el cálculo internamente. El coste computacional del score compuesto no salta en escalones: el binning Freedman-Diaconis, el cálculo de entropía de Shannon y la corrección Bonferroni tienen una curva de coste que baja por llamada a medida que el overhead de inicialización se amortiza entre más pares. Un sistema de tiers con precios fijos por banda introduciría discontinuidades de precio que no corresponden a ninguna discontinuidad real en el coste de producción, lo que genera arbitraje irracional donde el llamador evita cruzar un umbral o lo cruza artificialmente para saltar al tier siguiente. La función de tarifa decreciente continua elimina ese incentivo perverso y hace que el precio señalice información real sobre el coste marginal de cada par adicional calculado.

Sin compromiso mínimo es la consecuencia directa de que esta primitiva resuelve un problema que hoy nadie tiene resuelto con infraestructura propia: si el llamador ya tuviese un pipeline de NMI calibrado con Bonferroni corriendo en producción, no estaría aquí. Eso significa que el primer contacto con el activo es invariablemente exploratorio — validar si la señal estadística mejora la precisión de sus rankings existentes antes de comprometer cualquier arquitectura downstream. Forzar un mínimo mensual en esa fase convierte la evaluación técnica en una decisión presupuestaria, lo que desplaza la adopción hacia competidores peores pero más baratos de probar. Sin piso de entrada, el desarrollador puede ejecutar la hipótesis, medir el delta de calidad frente a similitud coseno pura, y escalar desde cero si el p-value calibrado demuestra ser la señal que su sistema necesitaba.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |