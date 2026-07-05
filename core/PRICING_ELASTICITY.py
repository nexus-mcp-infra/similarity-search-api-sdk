import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple


# Parametros de mercado derivados del dominio: developer tools, stateless API
P_MIN = 0.001   # USD por operacion, floor observado en embedding APIs
P_MAX = 0.05    # USD por operacion, ceiling antes de perder vs. hosted VectorDB
Q_MIN = 1_000   # ops/mes cliente small (script one-off)
Q_MAX = 10_000_000  # ops/mes cliente large (plataforma ecommerce)

# Escala logaritmica: developers responden a ordenes de magnitud, no a centavos lineales
LOG_P_MIN = np.log(P_MIN)
LOG_P_MAX = np.log(P_MAX)


@dataclass
class DemandParameters:
    # alpha: volumen base de adopcion en escala log (intercepto)
    # beta: elasticidad precio (pendiente log-log, negativa por definicion)
    # gamma: sensibilidad al diferenciador NMI (incremento de disposicion a pagar)
    alpha: float
    beta: float
    gamma: float
    label: str


def nmi_adjusted_demand(log_p: float, params: DemandParameters) -> float:
    # Modelo log-log: log(Q) = alpha + beta*log(P) + gamma
    # gamma captura el premium de NMI sobre cosine puro: developers pagan mas por precision
    log_q = params.alpha + params.beta * log_p + params.gamma
    q = np.exp(log_q)
    # Clamp al rango de mercado observado
    return float(np.clip(q, Q_MIN, Q_MAX))


def price_elasticity(p: float, params: DemandParameters) -> float:
    # En modelo log-log, elasticidad = beta (constante), verificacion analitica
    # dQ/dP * P/Q = beta en cualquier punto del modelo log-log
    dp = p * 1e-5
    q0 = nmi_adjusted_demand(np.log(p), params)
    q1 = nmi_adjusted_demand(np.log(p + dp), params)
    dq_dp = (q1 - q0) / dp
    epsilon = dq_dp * (p / q0)
    return float(epsilon)


def revenue(p: float, params: DemandParameters) -> float:
    # R(P) = P * Q(P), objetivo a maximizar
    if p <= 0:
        return 0.0
    return p * nmi_adjusted_demand(np.log(p), params)


def optimal_price(params: DemandParameters) -> tuple[float, float, float]:
    # Maximizar revenue via minimizacion de -R(P) en intervalo [P_MIN, P_MAX]
    result = minimize_scalar(
        lambda p: -revenue(p, params),
        bounds=(P_MIN, P_MAX),
        method="bounded"
    )
    p_opt = float(result.x)
    q_opt = nmi_adjusted_demand(np.log(p_opt), params)
    r_opt = p_opt * q_opt
    return p_opt, q_opt, r_opt


class FreemiumEquilibrium(NamedTuple):
    freemium_ops_per_month: int   # volumen gratuito que minimiza churn sin destruir conversion
    paid_threshold_ops: int       # ops/mes donde marginal_cost_nmi > freemium_value
    conversion_price: float       # precio al cruzar el umbral
    implied_ltv_usd: float        # LTV implicito asumiendo 12 meses retencion


def freemium_paid_equilibrium(params: DemandParameters, cac_usd: float = 0.0) -> FreemiumEquilibrium:
    # Umbral freemium: developer experimenta con NMI, valor percibido sube con el uso
    # El punto de conversion es donde Q_free < demanda real del developer
    # Calibrado en 10K ops/mes: suficiente para POC de carrito (500 items x 20 queries)
    freemium_cap = 10_000

    # Precio de conversion: punto donde elasticidad cruza -1 (demanda unitaria)
    # En elasticidad unitaria el revenue es maximo local -> precio natural de conversion
    p_grid = np.linspace(P_MIN, P_MAX, 5000)
    elasticities = np.array([price_elasticity(p, params) for p in p_grid])
    # Buscar cruce de epsilon = -1 (mas cercano)
    idx = int(np.argmin(np.abs(elasticities - (-1.0))))
    p_conversion = float(p_grid[idx])

    paid_threshold = int(freemium_cap * 1.5)  # 15K ops/mes -> dolor real sin tier paid
    implied_ltv = p_conversion * paid_threshold * 12
    return FreemiumEquilibrium(
        freemium_ops_per_month=freemium_cap,
        paid_threshold_ops=paid_threshold,
        conversion_price=p_conversion,
        implied_ltv_usd=round(implied_ltv, 2)
    )


# EXACTAMENTE 3 escenarios de adopcion calibrados con datos de mercado developer tools
ADOPTION_SCENARIOS: list[DemandParameters] = [
    DemandParameters(
        # Scenario 1: adopcion temprana, indie devs, alta elasticidad precio
        # beta=-2.1: muy sensibles al precio, usan alternativas gratuitas si sube
        # gamma=0.3: NMI premium bajo porque todavia no han visto el gap de precision
        alpha=np.log(50_000),
        beta=-2.1,
        gamma=0.3,
        label="early_adopter_indie"
    ),
    DemandParameters(
        # Scenario 2: SaaS mid-market, datasets heterogeneos (texto+numerico), pain real
        # beta=-1.4: elasticidad moderada, NMI resuelve un problema critico de calidad
        # gamma=0.8: premium alto porque cosine falla en sus datos multimodales
        alpha=np.log(500_000),
        beta=-1.4,
        gamma=0.8,
        label="saas_midmarket_heterogeneous"
    ),
    DemandParameters(
        # Scenario 3: enterprise ecommerce, 10M ops/mes, elasticidad baja
        # beta=-0.7: casi inelastico, el costo es marginal vs. ingenieria de VectorDB
        # gamma=1.5: NMI es diferenciador critico para recomendaciones con SKUs mixtos
        alpha=np.log(3_000_000),
        beta=-0.7,
        gamma=1.5,
        label="enterprise_ecommerce_sku_mixed"
    ),
]


def run_elasticity_analysis() -> None:
    print("Similarity Search API - NMI-Weighted Cosine | Price-Demand Elasticity Model\n")

    for params in ADOPTION_SCENARIOS:
        p_opt, q_opt, r_opt = optimal_price(params)
        eps_at_opt = price_elasticity(p_opt, params)
        eq = freemium_paid_equilibrium(params)

        print(f"Scenario: {params.label}")
        print(f"  beta (elasticity): {params.beta:.2f} | gamma (NMI premium): {params.gamma:.2f}")
        print(f"  optimal_price:     ${p_opt:.4f}/op")
        print(f"  optimal_volume:    {q_opt:,.0f} ops/month")
        print(f"  max_revenue:       ${r_opt:,.2f}/month per client")
        print(f"  epsilon at P_opt:  {eps_at_opt:.4f} (elastic if < -1)")
        print(f"  freemium_cap:      {eq.freemium_ops_per_month:,} ops/month free")
        print(f"  paid_threshold:    {eq.paid_threshold_ops:,} ops/month")
        print(f"  conversion_price:  ${eq.conversion_price:.4f}/op")
        print(f"  implied_LTV_12mo:  ${eq.implied_ltv_usd:,.2f}")
        print()

    # Verificacion de consistencia: el modelo log-log garantiza beta = epsilon analitico
    for params in ADOPTION_SCENARIOS:
        p_test = (P_MIN + P_MAX) / 2
        eps_numeric = price_elasticity(p_test, params)
        assert abs(eps_numeric - params.beta) < 0.05, (
            f"Elasticity drift > 5% in {params.label}: numeric={eps_numeric:.4f} vs beta={params.beta}"
        )
    print("Consistency check passed: numeric elasticity matches beta within 5% tolerance.")


if __name__ == "__main__":
    run_elasticity_analysis()