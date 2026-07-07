import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado derivados del segmento developer-tools / stateless similarity
P_MIN = 0.001   # USD por operacion (floor: competitivo con llamadas embedding standalone)
P_MAX = 0.05    # USD por operacion (ceiling: umbral de arbitraje vs vector DB managed)
Q_BASE = 500_000  # operaciones/mes referencia central del segmento (geometric mean 1K-10M)

# Elasticidad empirica para developer tools de infraestructura:
# -1.8 refleja mercado sensible al precio pero con lock-in moderado por stateless convenience
# Fuente estructural: elasticidades tipicas de API infra oscilan entre -1.5 y -2.2
ELASTICITY_BASE = -1.8

# Coeficiente NMI+Cosine: el diferenciador tecnico reduce elasticidad en ~0.3
# porque no hay sustituto directo que combine ambas metricas en una llamada stateless
MOAT_ELASTICITY_DISCOUNT = 0.3
ELASTICITY_EFFECTIVE = ELASTICITY_BASE + MOAT_ELASTICITY_DISCOUNT  # -> -1.5

# Parametro de escala para Q = Q_BASE * (P / P_REF)^epsilon
P_REF = 0.005  # precio de referencia: punto medio logaritmico del rango target


@dataclass(frozen=True)
class AdoptionScenario:
    name: str
    elasticity: float       # refleja sustituibilidad percibida en ese segmento
    q_base_scale: float     # multiplicador sobre Q_BASE segun tamano del cliente
    freemium_threshold: int # operaciones/mes donde el costo supera friccion de pago


class DemandCurve:
    """Demanda isoelastica: Q(P) = Q_BASE * scale * (P_REF / P)^|epsilon|"""

    def __init__(self, elasticity: float, q_scale: float):
        # epsilon negativo -> exponente positivo al invertir la base
        self.epsilon = abs(elasticity)
        self.q_scale = q_scale
        self.q_anchor = Q_BASE * q_scale

    def quantity(self, price: float) -> float:
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        # formula isoelastica calibrada en P_REF
        return self.q_anchor * (P_REF / price) ** self.epsilon

    def elasticity_at(self, price: float) -> float:
        # por definicion del modelo isoelastico la elasticidad es constante = -epsilon
        # pero verificamos analiticamenete: dQ/dP = -epsilon * Q_anchor * P_REF^epsilon * P^(-epsilon-1)
        # epsilon_punto = (dQ/dP) * (P/Q) = -epsilon (constante) -> confirmado
        return -self.epsilon

    def revenue(self, price: float) -> float:
        return price * self.quantity(price)

    def optimal_price(self) -> float:
        # Para Q isoelastica con epsilon != 1, revenue = P * A * (P_REF/P)^epsilon
        # = A * P_REF^epsilon * P^(1-epsilon)
        # d(revenue)/dP = (1-epsilon) * A * P_REF^epsilon * P^(-epsilon) = 0
        # Si epsilon > 1: revenue decrece con P -> optimo en P_MAX (constraint)
        # Si epsilon < 1: revenue crece con P -> optimo en P_MAX tambien (constraint)
        # El maximo real esta en la restriccion superior del mercado: P_MAX
        # Verificamos numericamente para capturar efectos de saturacion de volumen
        result = minimize_scalar(
            lambda p: -self.revenue(p),
            bounds=(P_MIN, P_MAX),
            method="bounded"
        )
        return float(result.x)

    def freemium_paid_crossover(self, threshold_ops: int, free_tier_ops: int = 1000) -> float:
        # precio donde el costo de pagar supera la friccion cognitiva del freemium
        # se estima como el precio tal que revenue(P) * (threshold_ops - free_tier_ops) = 10 USD
        # (10 USD/mes es el umbral psicologico de compra de tarjeta tipico en devtools)
        paid_ops = max(threshold_ops - free_tier_ops, 1)
        crossover_price = 10.0 / paid_ops
        # se clampea al rango de mercado
        return float(np.clip(crossover_price, P_MIN, P_MAX))


# EXACTAMENTE 3 escenarios de adopcion segun perfil de cliente
SCENARIOS: list[AdoptionScenario] = [
    AdoptionScenario(
        name="indie_developer",
        # alta sensibilidad: puede usar cosine simple sin NMI si el precio sube
        elasticity=-2.1,
        # volumen bajo: ~5K ops/mes, 1% de Q_BASE
        q_base_scale=0.01,
        # pasa a paid cuando el ahorro vs vector DB (>20 USD/mes) justifica la tarjeta
        freemium_threshold=5_000,
    ),
    AdoptionScenario(
        name="saas_platform_mid",
        # elasticidad media: dependencia operacional pero alternativas existen
        elasticity=-1.5,
        # volumen medio: ~200K ops/mes, 40% de Q_BASE
        q_base_scale=0.4,
        # adopcion paid cuando la busqueda semantica es feature critico del producto
        freemium_threshold=50_000,
    ),
    AdoptionScenario(
        name="enterprise_data_pipeline",
        # baja elasticidad: lock-in por bootstrap CI + auditabilidad de scores
        elasticity=-0.9,
        # volumen alto: ~3M ops/mes, 600% de Q_BASE
        q_base_scale=6.0,
        # enterprise entra directo a paid; threshold es el minimo de contrato
        freemium_threshold=100_000,
    ),
]


class SimilaritySearchPriceEquilibrium(NamedTuple):
    scenario_name: str
    optimal_price_usd: float
    quantity_at_optimal: float
    revenue_at_optimal_usd: float
    elasticity: float
    freemium_crossover_price_usd: float
    freemium_crossover_monthly_revenue_usd: float


def compute_equilibrium(scenario: AdoptionScenario) -> SimilaritySearchPriceEquilibrium:
    curve = DemandCurve(elasticity=scenario.elasticity, q_scale=scenario.q_base_scale)
    p_opt = curve.optimal_price()
    q_opt = curve.quantity(p_opt)
    rev_opt = curve.revenue(p_opt)
    eps = curve.elasticity_at(p_opt)
    p_cross = curve.freemium_paid_crossover(scenario.freemium_threshold)
    rev_cross = p_cross * (scenario.freemium_threshold - 1000)
    return SimilaritySearchPriceEquilibrium(
        scenario_name=scenario.name,
        optimal_price_usd=round(p_opt, 6),
        quantity_at_optimal=round(q_opt, 0),
        revenue_at_optimal_usd=round(rev_opt, 2),
        elasticity=round(eps, 2),
        freemium_crossover_price_usd=round(p_cross, 6),
        freemium_crossover_monthly_revenue_usd=round(rev_cross, 2),
    )


def run_similarity_search_pricing_model() -> list[SimilaritySearchPriceEquilibrium]:
    results = []
    for scenario in SCENARIOS:
        eq = compute_equilibrium(scenario)
        results.append(eq)
        print(
            f"[{eq.scenario_name}] "
            f"P_opt={eq.optimal_price_usd} USD | "
            f"Q={eq.quantity_at_optimal:.0f} ops | "
            f"Rev={eq.revenue_at_optimal_usd} USD/mo | "
            f"eps={eq.elasticity} | "
            f"freemium->paid at P={eq.freemium_crossover_price_usd} USD "
            f"({eq.freemium_crossover_monthly_revenue_usd} USD/mo)"
        )
    return results


if __name__ == "__main__":
    equilibria = run_similarity_search_pricing_model()