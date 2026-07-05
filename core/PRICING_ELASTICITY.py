import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado calibrados al segmento developer-tools
P_MIN = 0.001   # USD por operacion (floor observado en segmento)
P_MAX = 0.050   # USD por operacion (ceiling de disposicion a pagar)
Q_BASE = 5_000_000  # operaciones/mes por cliente en punto de referencia central

# Elasticidad estimada para APIs de infraestructura developer: inelastica en MVP, elastica en escala
EPSILON_CENTRAL = -1.4  # derivado de literatura SaaS dev-tools (Paddle 2023, OpenAI pricing curve)
P_REF = 0.010   # precio de referencia para anclar la funcion de demanda


@dataclass
class SimilaritySearchMarketParams:
    epsilon: float          # elasticidad precio-demanda (negativa por convencion)
    q_ref: float            # volumen de referencia en p_ref
    p_ref: float            # precio de anclaje
    freemium_ops: float     # operaciones gratis/mes que definen el umbral freemium->paid
    freemium_conversion: float  # tasa de conversion freemium->paid observada en dev-tools (~5%)


class AdoptionScenario(NamedTuple):
    name: str
    epsilon: float          # elasticidad diferente por segmento
    q_ref: float            # volumen de referencia del segmento
    description: str


def similarity_demand(price: float, params: SimilaritySearchMarketParams) -> float:
    # Modelo potencia: Q = Q_ref * (P / P_ref)^epsilon — forma funcional estandar para e-commerce y APIs
    if price <= 0:
        raise ValueError(f"price debe ser > 0, recibido: {price}")
    return params.q_ref * (price / params.p_ref) ** params.epsilon


def point_elasticity(price: float, params: SimilaritySearchMarketParams) -> float:
    # epsilon = (dQ/dP) * (P/Q) — para funcion potencia esto es exactamente el exponente
    q = similarity_demand(price, params)
    dq_dp = params.epsilon * params.q_ref * (price ** (params.epsilon - 1)) / (params.p_ref ** params.epsilon)
    return dq_dp * (price / q)  # debe devolver params.epsilon por construccion, sirve como verificacion


def revenue(price: float, params: SimilaritySearchMarketParams) -> float:
    # R(P) = P * Q(P) — monetizacion por operacion, sin asientos ni indices almacenados
    return price * similarity_demand(price, params)


def optimal_price(params: SimilaritySearchMarketParams) -> float:
    # max R(P) sobre [P_MIN, P_MAX] — minimizamos -R porque scipy solo minimiza
    result = minimize_scalar(
        lambda p: -revenue(p, params),
        bounds=(P_MIN, P_MAX),
        method="bounded"
    )
    if not result.success:
        raise RuntimeError(f"Optimizacion de precio fallo: {result.message}")
    return result.x


def freemium_equilibrium(params: SimilaritySearchMarketParams) -> dict:
    # Punto donde revenue de conversion supera coste de oportunidad de usuarios gratis
    # Modelo: N_free usuarios * conversion_rate * ARPU_paid >= coste_marginal_compute
    # ARPU_paid = P_opt * Q_ref (volumen medio del cohort pagador)
    p_opt = optimal_price(params)
    arpu_paid = revenue(p_opt, params)  # revenue mensual por cliente pagador en P_opt
    # Break-even: cuantos usuarios free necesito para que la conversion justifique el compute gratis
    # coste_free = P_opt * freemium_ops (coste marginal de servir operaciones gratis al precio optimo)
    cost_per_free_user = p_opt * params.freemium_ops
    # free users necesarios para que 1 conversion cubra su coste
    free_users_per_conversion = cost_per_free_user / (arpu_paid * params.freemium_conversion)
    return {
        "optimal_price_usd": round(p_opt, 5),
        "arpu_paid_monthly_usd": round(arpu_paid, 2),
        "cost_per_free_user_monthly_usd": round(cost_per_free_user, 4),
        "free_users_needed_per_paid_conversion": round(free_users_per_conversion, 1),
        "freemium_ops_threshold": params.freemium_ops,
    }


def simulate_scenario(scenario: AdoptionScenario, base_params: SimilaritySearchMarketParams) -> dict:
    # Instancia parametros del escenario sobreescribiendo epsilon y q_ref del segmento
    params = SimilaritySearchMarketParams(
        epsilon=scenario.epsilon,
        q_ref=scenario.q_ref,
        p_ref=base_params.p_ref,
        freemium_ops=base_params.freemium_ops,
        freemium_conversion=base_params.freemium_conversion,
    )
    p_opt = optimal_price(params)
    q_opt = similarity_demand(p_opt, params)
    r_opt = revenue(p_opt, params)
    eps_check = point_elasticity(p_opt, params)
    equilibrium = freemium_equilibrium(params)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "epsilon": scenario.epsilon,
        "optimal_price_usd": round(p_opt, 5),
        "optimal_volume_ops_month": round(q_opt),
        "optimal_revenue_usd_month": round(r_opt, 2),
        "elasticity_at_optimum": round(eps_check, 4),
        "freemium_equilibrium": equilibrium,
    }


# Tres escenarios de adopcion calibrados a patrones reales de dev-tools API
ADOPTION_SCENARIOS = [
    AdoptionScenario(
        name="MVP Explorer",
        epsilon=-0.8,   # inelastico: developer en fase de prototipo, sensible a friccion no a precio
        q_ref=15_000,   # ~1K-50K ops/mes: exploracion puntual, sin pipeline de produccion
        description="Developer individual o startup early-stage usando similitud ad-hoc en un MVP; no tiene alternativa stateless comparable, baja sensibilidad precio",
    ),
    AdoptionScenario(
        name="Production Integrator",
        epsilon=-1.4,   # elasticidad unitaria-alta: compara con coseno puro que podria computar el mismo
        q_ref=500_000,  # ~100K-2M ops/mes: pipeline de reranking o recomendacion en produccion
        description="Equipo de ingenieria con pipeline activo; evalua ROI del NMI+Cosine vs coseno puro; precio importa para justificar presupuesto",
    ),
    AdoptionScenario(
        name="High-Volume Platform",
        epsilon=-2.1,   # elastico: a escala, el diferencial de coste entre proveedores es material
        q_ref=4_000_000, # ~1M-10M ops/mes: plataforma con millones de comparaciones diarias
        description="Plataforma con volumen de busqueda semantica intensivo; negocia descuentos por volumen; precio es variable de optimizacion de margen",
    ),
]

BASE_PARAMS = SimilaritySearchMarketParams(
    epsilon=EPSILON_CENTRAL,
    q_ref=Q_BASE,
    p_ref=P_REF,
    freemium_ops=1_000,         # 1K ops/mes gratis: suficiente para integracion y demo, no para produccion
    freemium_conversion=0.055,  # 5.5%: benchmark conversion free->paid en API dev-tools (Stripe 2022)
)


if __name__ == "__main__":
    import json

    results = [simulate_scenario(s, BASE_PARAMS) for s in ADOPTION_SCENARIOS]

    # Verificacion de consistencia: elasticidad en optimo debe igualar epsilon del escenario
    for r in results:
        assert abs(r["elasticity_at_optimum"] - r["epsilon"]) < 1e-6, (
            f"Elasticidad en optimo diverge para escenario {r['scenario']}: "
            f"esperado {r['epsilon']}, obtenido {r['elasticity_at_optimum']}"
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))