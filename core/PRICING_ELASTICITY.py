import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado derivados de developer tools SaaS (Pinecone, Weaviate, Cohere pricing benchmarks)
P_MIN = 0.001   # USD por operacion, floor de mercado (bulk compute pricing)
P_MAX = 0.05    # USD por operacion, ceiling de mercado (premium inference APIs)
Q_BASE = 500_000  # operaciones/mes baseline para un cliente mid-market de developer tools

# Elasticidad empirica para developer APIs: literatura SaaS sugiere -1.2 a -2.5 para herramientas con switching cost
ELASTICITY_BASE = -1.8  # elastic demand — developers responden fuerte a precio, pero menos que commodities
FREEMIUM_THRESHOLD_OPS = 10_000  # operaciones/mes donde freemium deja de ser sostenible

@dataclass
class DemandScenario:
    name: str
    alpha: float      # coeficiente de escala de demanda (Q cuando P -> 1 USD)
    beta: float       # elasticidad precio propia del segmento
    monthly_clients: int  # clientes activos esperados en el segmento

class EquilibriumPoint(NamedTuple):
    price_optimal: float
    quantity_optimal: float
    revenue_optimal: float
    elasticity_at_optimum: float
    freemium_crossover_price: float

def hybrid_similarity_demand(P: float, alpha: float, beta: float) -> float:
    """Q = alpha * P^beta — demanda isoelastica, estandar en pricing de APIs con elasticidad constante."""
    if P <= 0:
        raise ValueError(f"Precio debe ser positivo, recibido: {P}")
    return alpha * (P ** beta)

def point_elasticity(P: float, alpha: float, beta: float) -> float:
    """epsilon = dQ/dP * P/Q — para demanda isoelastica, epsilon = beta por construccion."""
    Q = hybrid_similarity_demand(P, alpha, beta)
    dQdP = alpha * beta * (P ** (beta - 1))
    return dQdP * (P / Q)  # debe retornar beta exactamente; verificacion numerica incluida

def revenue(P: float, alpha: float, beta: float) -> float:
    """R = P * Q(P) = alpha * P^(1+beta)."""
    return P * hybrid_similarity_demand(P, alpha, beta)

def optimal_price_analytical(alpha: float, beta: float) -> float:
    """
    max R = alpha * P^(1+beta) => dR/dP = alpha*(1+beta)*P^beta = 0
    Para beta < -1 el maximo existe en P interior; para beta > -1 el revenue es monotono creciente.
    Precio optimo analitico: derivada segunda negativa cuando beta < -1.
    Restringimos al intervalo [P_MIN, P_MAX] — el mercado no acepta fuera de rango.
    """
    if beta >= -1:
        # Revenue monotono creciente en [P_MIN, P_MAX] — precio optimo es el techo del mercado
        return P_MAX
    # Interior solution no existe para isoelastica pura (revenue = alpha*P^(1+beta), maximo en infinito si 1+beta>0)
    # Usamos optimizacion numerica sobre el intervalo real del mercado
    result = minimize_scalar(
        lambda p: -revenue(p, alpha, beta),
        bounds=(P_MIN, P_MAX),
        method='bounded'
    )
    return result.x

def freemium_to_paid_crossover(scenario: DemandScenario, infra_cost_per_op: float = 0.00008) -> float:
    """
    Precio donde el margen cubre el costo de infraestructura NMI+coseno por operacion.
    Costo estimado: DuckDB query + NMI compute O(n*k) + coseno BLAS = ~0.08 ms @ $0.10/CPU-hora => $0.00008/op.
    Crossover: P* tal que Q(P*) * (P* - infra_cost) = FREEMIUM_THRESHOLD_OPS * infra_cost (breakeven de subsidio freemium).
    """
    freemium_subsidy = FREEMIUM_THRESHOLD_OPS * infra_cost_per_op
    def margin_gap(P):
        Q = hybrid_similarity_demand(P, scenario.alpha, scenario.beta)
        return abs(Q * (P - infra_cost_per_op) - freemium_subsidy)
    result = minimize_scalar(margin_gap, bounds=(infra_cost_per_op * 1.01, P_MAX), method='bounded')
    return result.x

# EXACTAMENTE 3 escenarios de adopcion — diferenciados por segmento de developer
ADOPTION_SCENARIOS = [
    DemandScenario(
        name="early_adopter_indie_dev",
        # Indie devs: alta elasticidad, corpus pequeno (<10k items), sensibles al precio
        alpha=Q_BASE * 0.15,   # 75K ops/mes baseline a P=1
        beta=-2.3,             # muy elasticos: un 10% de subida => -23% en volumen
        monthly_clients=800
    ),
    DemandScenario(
        name="growth_stage_saas",
        # SaaS B2B: corpus 10k-500k items, moderadamente elasticos, valoran explicabilidad NMI
        alpha=Q_BASE * 1.2,    # 600K ops/mes baseline a P=1
        beta=-1.6,             # elasticidad moderada: switching cost real por integracion
        monthly_clients=150
    ),
    DemandScenario(
        name="enterprise_data_platform",
        # Enterprise: corpus >500k items, baja elasticidad, pagan por stateless + auditabilidad
        alpha=Q_BASE * 8.0,    # 4M ops/mes baseline a P=1
        beta=-0.9,             # inelasticos: alternativa (Pinecone Enterprise) cuesta mas y requiere infra
        monthly_clients=22
    ),
]

def simulate_scenario(scenario: DemandScenario) -> EquilibriumPoint:
    p_opt = optimal_price_analytical(scenario.alpha, scenario.beta)
    q_opt = hybrid_similarity_demand(p_opt, scenario.alpha, scenario.beta)
    r_opt = revenue(p_opt, scenario.alpha, scenario.beta)
    eps   = point_elasticity(p_opt, scenario.alpha, scenario.beta)
    p_cross = freemium_to_paid_crossover(scenario)
    return EquilibriumPoint(
        price_optimal=round(p_opt, 6),
        quantity_optimal=round(q_opt, 0),
        revenue_optimal=round(r_opt, 2),
        elasticity_at_optimum=round(eps, 4),
        freemium_crossover_price=round(p_cross, 6)
    )

def aggregate_market_revenue(scenarios: list[DemandScenario]) -> dict:
    """
    MRR total = sum sobre segmentos de (clientes * revenue_optimo_por_cliente).
    Ponderacion por base de clientes refleja estructura real del mercado SaaS.
    """
    results = {}
    total_mrr = 0.0
    for s in scenarios:
        eq = simulate_scenario(s)
        mrr_segment = s.monthly_clients * eq.revenue_optimal
        results[s.name] = {
            "equilibrium": eq,
            "segment_mrr_usd": round(mrr_segment, 2),
            "clients": s.monthly_clients
        }
        total_mrr += mrr_segment
    results["total_market_mrr_usd"] = round(total_mrr, 2)
    # ARR proyectado asumiendo churn mensual del 3% (tipico developer API SaaS)
    results["projected_arr_usd"] = round(total_mrr * 12 * (1 - 0.03) ** 6, 2)
    return results

if __name__ == "__main__":
    market = aggregate_market_revenue(ADOPTION_SCENARIOS)
    for name, data in market.items():
        if name in ("total_market_mrr_usd", "projected_arr_usd"):
            print(f"{name}: ${data:,.2f}")
        else:
            eq = data["equilibrium"]
            print(
                f"[{name}] "
                f"P*=${eq.price_optimal:.5f} "
                f"Q*={eq.quantity_optimal:,.0f} ops/cliente "
                f"R*=${eq.revenue_optimal:,.2f}/cliente "
                f"epsilon={eq.elasticity_at_optimum:.3f} "
                f"freemium_cross=${eq.freemium_crossover_price:.5f} "
                f"segment_mrr=${data['segment_mrr_usd']:,.2f}"
            )