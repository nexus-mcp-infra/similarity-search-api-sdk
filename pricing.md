# Pricing

El modelo de suscripción fija asume que el desarrollador ya sabe que necesita búsqueda por similitud antes de usarla — y esa suposición es exactamente el problema. La barrera real no es el precio mensual, es la obligación de comprometerse con infraestructura antes de tener evidencia de que el enfoque resuelve el problema. Un tier de suscripción convierte la evaluación técnica en una decisión financiera prematura: el desarrollador no puede descubrir si NMI+cosine sobre payload raw supera su búsqueda actual sin primero pagar por el derecho de intentarlo. El modelo por llamada elimina esa fricción de raíz — la primera query tiene el mismo acceso completo que la décima mil, sin periodo de prueba artificialmente limitado ni feature gating diseñado para presionar el upgrade.

La tarifa decreciente por volumen refleja la estructura de costes real del sistema stateless: sin índice persistente que mantener, sin estado entre llamadas, sin infraestructura dedicada por cliente, el coste marginal de procesar la llamada número diez mil es genuinamente menor que el de la primera, y esa reducción se traslada directamente. Un tier fijo capturaría esa eficiencia como margen; el pricing decreciente la devuelve como incentivo al cliente que escala. Esto alinea los incentivos en la dirección correcta: cuanto más valor extrae el cliente del sistema, menos paga por unidad de valor, lo que hace que crecer dentro de la primitiva sea siempre preferible a migrar hacia infraestructura propia.

El caso de uso central — datasets mixtos procesados en ráfagas irregulares, sin volumen predecible — es estructuralmente incompatible con suscripciones fijas. Un pipeline de enriquecimiento de catálogo que corre una vez al día, una evaluación de similitud que se dispara con cada ingesta, un prototipo que procesa quinientas queries en un pico semanal: ninguno de estos patrones justifica pagar por capacidad reservada que permanece ociosa el noventa por ciento del tiempo. Sin techo y sin compromiso mínimo significa que el mismo modelo de precios sirve al desarrollador individual explorando el concepto y al sistema de producción procesando volumen industrial — sin necesidad de renegociar contrato cuando el uso cambia.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |