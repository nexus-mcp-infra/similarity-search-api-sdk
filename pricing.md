# Pricing

El modelo de suscripción por tiers presupone que el desarrollador sabe de antemano cuántas búsquedas va a necesitar al mes. Pero el caso de uso central de esta API es precisamente lo contrario: comparaciones efímeras, batch jobs esporádicos, pipelines de CI que se ejecutan en cada commit, experimentos de data science que no tienen cadencia predecible. Un tier fijo obliga al usuario a pagar por capacidad ociosa durante los ciclos bajos y a escalar de golpe al tier siguiente cuando llega un pico, aunque ese pico dure una hora. El resultado es que el modelo de suscripción transfiere al comprador el riesgo de estimar mal su propio uso, un riesgo que no debería existir en una primitiva stateless donde cada llamada es independiente y no genera coste de infraestructura remanente del lado del proveedor.

La tarifa decreciente por volumen refleja la realidad económica de la operación: el coste marginal de procesar el décimo millar de comparaciones en un mismo mes es inferior al de las primeras cien, porque los overheads fijos de autenticación, routing y cold start ya están amortizados. Trasladar esa curva de coste real al precio es lo que permite que un equipo de dos personas que hace un experimento puntual pague proporcional a lo que consume, mientras que un equipo enterprise que integra la API en producción con volúmenes altos no subsidia a los demás ni se ve penalizado por crecer. La ausencia de compromiso mínimo cierra el ciclo: elimina la fricción de adopción en la fase de evaluación, que es precisamente cuando el developer necesita ejecutar cientos de llamadas de prueba para medir el gap de calidad entre el hybrid scorer y Cosine puro en sus propios datos.

Finalmente, el modelo usage-based es coherente con la propuesta técnica de statelessness. Una API sin índice persistente, sin almacenamiento, sin estado entre llamadas no tiene razón de cobrar por "asientos" ni por "proyectos activos" ni por ninguna unidad que no sea la operación misma. Cobrar por suscripción en un producto que no retiene nada del usuario entre llamadas sería una inconsistencia conceptual: el precio estaría desconectado del valor entregado, que es el cómputo del score híbrido en el momento de la llamada. El usage-based alinea incentivos en ambos sentidos: el proveedor tiene interés en que la API sea suficientemente buena para que el desarrollador la llame más, y el desarrollador tiene libertad de escalar o parar sin coste de salida.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |