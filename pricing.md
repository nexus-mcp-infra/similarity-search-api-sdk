# Pricing

El modelo de suscripción por tiers asume que el consumo de una primitiva de infraestructura es predecible y estable mes a mes — una suposición válida para herramientas de productividad, no para una API de similitud efímera cuya propuesta de valor central es precisamente que no requiere estado persistente ni compromiso previo. Quien usa esta primitiva no tiene un corpus fijo: tiene corpus que cambian por llamada, volúmenes que fluctúan según el flujo de su producto, y picos de uso imposibles de anticipar al momento de elegir un tier. Forzar ese patrón de uso dentro de una caja de suscripción mensual traslada el riesgo de variabilidad al developer, que paga capacidad ociosa en meses bajos o se queda bloqueado en meses de pico — exactamente el tipo de fricción que esta primitiva elimina en el plano técnico y que el pricing no debería reintroducir en el plano económico.

La tarifa decreciente por volumen resuelve además un problema de alineación de incentivos que los tiers no pueden resolver: en un modelo de tiers, el provider gana si el cliente sobre-estima su uso y elige un plan mayor; el cliente gana si subestima y queda en un plan menor. Esa asimetría destruye confianza. Con precio marginal decreciente sin techo, el incentivo es simétrico — cuanto más valor extrae el cliente de la primitiva, menor es su costo unitario, y el provider captura más valor absoluto en cada cliente de alto volumen sin que nadie tenga que adivinar el futuro. Esto es coherente con la naturaleza del activo: el cálculo NMI+Coseno tiene complejidad O(n·d) por llamada, y a escala, las eficiencias computacionales reales del sistema se transfieren parcialmente al cliente en forma de precio marginal más bajo, no se quedan como margen puro del provider.

Finalmente, la ausencia de compromiso mínimo es inseparable del diferenciador técnico. El argumento de venta de esta API es que elimina el onboarding de índices HNSW/IVF y el costo de almacenamiento vectorial persistente — si el pricing requiriese un mínimo mensual o un contrato anual, estaría reimponiendo una barrera de entrada análoga a la que el producto técnico promete destruir. Un developer de equipo pequeño que quiere probar similitud semántica híbrida sobre un corpus ad-hoc no debería tener que justificar internamente un gasto fijo antes de ejecutar su primera llamada. La primera llamada debe ser la justificación, y el modelo de precio debe permitir exactamente eso.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |