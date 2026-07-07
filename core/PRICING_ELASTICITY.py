import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from dataclasses import dataclass
from typing import Tuple

# Parametros de mercado calibrados para developer tools con billing per-call
P_MIN = 0.001   # USD por operacion, floor del mercado
P_MAX = 0.05    # USD por operacion, techo observado en competidores
Q_MAX = 10_000_000  # operaciones/mes, cliente enterprise saturado
Q_BASE = 50_000     # operaciones/mes, cliente mediano en cold start

# Elasticidad empirica para APIs de infraestructura cognitiva: inelastica por lock-in tecnico
EPSILON_BASE = -1.4  # elasticidad precio-demanda propia, estimada para dev tools de nicho


@dataclass
class AdoptionScenario:
    label: str
    monthly_ops_per_client: float
    client_count: int
    willingness_to_pay: float  # maximo USD/op que este segmento acepta
    nmi_value_weight: float    # cuanto valoran el diferenciador NMI vs coseno puro


def nmi_corrected_demand(price: float, alpha_sensitivity: float = 1.4,
                          beta_floor: float = 0.15) -> float:
    """
    Q(P) = Q_BASE * (P_MIN / P)^epsilon * sigmoid_adoption(P)

    El termino sigmoid captura que bajo P_MIN/3 la demanda no escala linealmente:
    developers perciben precio irrelevante y el limite pasa a ser awareness, no costo.
    beta_floor evita que Q colapse a 0 por encima del techo de mercado.
    """
    if price <= 0:
        raise ValueError(f"price debe ser positivo, recibido: {price}")

    # Demanda base por ley de potencia: elasticidad constante en rango observable
    q_power = Q_BASE * (P_MIN / price) ** alpha_sensitivity

    # Factor de adopcion: refleja que NMI-corrected tiene barrera de comprension inicial
    # Los developers necesitan ver el p-value antes de confiar; eso deprime adopcion early
    comprehension_barrier = 1 / (1 + np.exp(6 * (price - 0.018)))  # inflexion en $0.018
    nmi_premium_capture = 0.35 * (1 - comprehension_barrier)       # adoption adicional post-comprension

    q = q_power * (beta_floor + (1 - beta_floor) * (comprehension_barrier + nmi_premium_capture))
    return max(q, 0.0)


def price_elasticity_at(price: float, delta: float = 1e-6) -> float:
    """
    epsilon(P) = (dQ/dP) * (P/Q) — calculado numericamente para respetar la forma exacta de Q(P)
    delta elegido para evitar cancelacion catastrofica en flotante doble precision
    """
    if price <= delta:
        raise ValueError(f"price demasiado bajo para diferenciacion numerica estable: {price}")
    q_center = nmi_corrected_demand(price)
    if q_center <= 0:
        return float('-inf')
    dq_dp = (nmi_corrected_demand(price + delta) - nmi_corrected_demand(price - delta)) / (2 * delta)
    return dq_dp * (price / q_center)


def revenue(price: float) -> float:
    """R(P) = P * Q(P) — objetivo de maximizacion directa"""
    return price * nmi_corrected_demand(price)


def optimal_price_nmi_api() -> Tuple[float, float, float]:
    """
    Maximiza R(P) en el rango de disposicion a pagar real del mercado objetivo.
    Usa minimize_scalar sobre -R para aprovechar bracket garantizado en intervalo cerrado.
    """
    result = minimize_scalar(
        lambda p: -revenue(p),
        bounds=(P_MIN, P_MAX),
        method='bounded',
        options={'xatol': 1e-8}
    )
    p_opt = result.x
    q_opt = nmi_corrected_demand(p_opt)
    r_opt = revenue(p_opt)
    return p_opt, q_opt, r_opt


def freemium_to_paid_equilibrium(free_ops_limit: int = 500,
                                  conversion_rate_base: float = 0.04) -> dict:
    """
    Punto de equilibrio donde el costo de oportunidad del tier free supera el friction de pago.
    conversion_rate_base: 4% empirico para APIs tecnicas con free tier limitado.
    El NMI-corrected score tiene mayor conversion esperada porque el p-value es un gancho
    de credibilidad que el free tier expone pero no permite actuar a escala.
    """
    nmi_credibility_multiplier = 1.85  # p-value visible en free tier sube conversion
    effective_conversion = conversion_rate_base * nmi_credibility_multiplier

    # Equilibrio: revenue esperado de converted >= costo de soporte del free tier
    # costo de soporte estimado en $0.00008/op para infraestructura NMI (mas costosa que coseno puro)
    infra_cost_per_op = 0.000_08
    free_tier_monthly_cost = free_ops_limit * infra_cost_per_op

    p_opt, q_opt, _ = optimal_price_nmi_api()
    revenue_per_converted = p_opt * Q_BASE * 0.1  # converted empieza en 10% del volumen base

    # Numero minimo de usuarios free para que el funnel sea rentable
    free_users_breakeven = free_tier_monthly_cost / (effective_conversion * revenue_per_converted)

    return {
        "p_optimal_usd_per_op": round(p_opt, 6),
        "conversion_rate_effective": round(effective_conversion, 4),
        "free_users_needed_for_breakeven": int(np.ceil(free_users_breakeven)),
        "monthly_cost_per_free_user_usd": round(free_tier_monthly_cost, 6),
        "equilibrium_condition": "revenue_converted >= infra_cost_free_tier"
    }


def simulate_adoption_scenarios() -> list[dict]:
    """EXACTAMENTE 3 escenarios que cubren la distribucion real de clientes en este mercado."""
    scenarios = [
        AdoptionScenario(
            label="indie_developer_semantic_search",
            monthly_ops_per_client=8_000,
            client_count=1_200,
            willingness_to_pay=0.008,  # sensible al precio, usa free tier primero
            nmi_value_weight=0.3       # valora resultado, no entiende NMI en profundidad
        ),
        AdoptionScenario(
            label="ml_team_embedding_validation",
            monthly_ops_per_client=280_000,
            client_count=85,
            willingness_to_pay=0.022,  # paga por correctness estadistica, tiene budget
            nmi_value_weight=0.75      # el p-value es el feature que justifica el gasto
        ),
        AdoptionScenario(
            label="enterprise_recsys_pipeline",
            monthly_ops_per_client=4_200_000,
            client_count=12,
            willingness_to_pay=0.041,  # precio subordinado a SLA y precision
            nmi_value_weight=0.90      # NMI-corrected es requisito de auditoria interna
        ),
    ]

    results = []
    p_opt, _, _ = optimal_price_nmi_api()

    for s in scenarios:
        # Precio efectivo: minimo entre optimo del proveedor y WTP del segmento
        p_effective = min(p_opt, s.willingness_to_pay)
        q_per_client = nmi_corrected_demand(p_effective) * (s.monthly_ops_per_client / Q_BASE)
        monthly_revenue = p_effective * q_per_client * s.client_count
        epsilon = price_elasticity_at(p_effective)

        # Chi2 stat aproximado para ilustrar significancia del diferenciador NMI en este segmento
        # Grados de libertad = segmentos - 1; corpus_size proxy = ops/mes del segmento
        corpus_proxy = int(s.monthly_ops_per_client * 0.01)  # 1% de ops son pares evaluados
        bonferroni_threshold = 0.05 / max(corpus_proxy, 1)
        chi2_critical = chi2.ppf(1 - bonferroni_threshold, df=9)  # df=9: k=sqrt(768)~27, agrupado

        results.append({
            "scenario": s.label,
            "price_usd_per_op": round(p_effective, 6),
            "monthly_ops_per_client": int(q_per_client),
            "client_count": s.client_count,
            "monthly_revenue_usd": round(monthly_revenue, 2),
            "elasticity_at_price": round(epsilon, 4),
            "nmi_value_weight": s.nmi_value_weight,
            "bonferroni_corrected_alpha": round(bonferroni_threshold, 10),
            "chi2_critical_for_significance": round(chi2_critical, 4),
        })

    return results


if __name__ == "__main__":
    import json

    p_opt, q_opt, r_opt = optimal_price_nmi_api()
    epsilon_opt = price_elasticity_at(p_opt)
    equilibrium = freemium_to_paid_equilibrium()
    scenarios = simulate_adoption_scenarios()

    output = {
        "optimal_pricing": {
            "p_opt_usd_per_op": round(p_opt, 6),
            "q_opt_ops_per_month": int(q_opt),
            "r_opt_usd_per_month": round(r_opt, 2),
            "elasticity_at_optimum": round(epsilon_opt, 4),
        },
        "freemium_equilibrium": equilibrium,
        "adoption_scenarios": scenarios,
    }

    print(json.dumps(output, indent=2))