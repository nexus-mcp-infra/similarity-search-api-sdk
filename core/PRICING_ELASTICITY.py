import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import NamedTuple

# Parametros de mercado calibrados para developer tools con pago por operacion
P_MIN = 0.001
P_MAX = 0.050
Q_BASE = 500_000  # operaciones/mes cliente mediano en similarity search stateless
PRICE_SENSITIVITY = 180  # calibrado: developers abandonan a 5x precio de alternativa DIY

@dataclass
class DemandCurve:
    # Elasticidad constante log-log: ln(Q) = a - b*ln(P), tipica en APIs de infraestructura
    base_volume: float
    sensitivity: float
    adoption_ceiling: float  # saturacion por segmento de mercado

    def quantity(self, price: float) -> float:
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        # Demanda log-lineal con techo de adopcion: evita Q->inf cuando P->0
        raw = self.base_volume * (P_MIN / price) ** self.sensitivity
        return min(raw, self.adoption_ceiling)

    def elasticity(self, price: float) -> float:
        # e = dQ/dP * P/Q = -sensitivity cuando Q < ceiling, 0 en saturacion
        q = self.quantity(price)
        if q >= self.adoption_ceiling:
            return 0.0
        return -self.sensitivity

    def revenue(self, price: float) -> float:
        return price * self.quantity(price)


class OptimalPricingResult(NamedTuple):
    price: float
    quantity: float
    revenue: float
    elasticity_at_optimum: float
    is_elastic: bool  # True si |e| > 1: reducir precio aumenta revenue


def maximize_nmi_cosine_api_revenue(curve: DemandCurve) -> OptimalPricingResult:
    # Minimizamos -revenue porque scipy solo tiene minimize
    result = minimize_scalar(
        lambda p: -curve.revenue(p),
        bounds=(P_MIN, P_MAX),
        method='bounded'
    )
    p_opt = result.x
    q_opt = curve.quantity(p_opt)
    e_opt = curve.elasticity(p_opt)
    return OptimalPricingResult(
        price=round(p_opt, 5),
        quantity=round(q_opt),
        revenue=round(p_opt * q_opt, 2),
        elasticity_at_optimum=round(e_opt, 3),
        is_elastic=abs(e_opt) > 1
    )


# Exactamente 3 escenarios de adopcion: reflejan los tres perfiles de dolor descritos
ADOPTION_SCENARIOS = {
    # Cliente que actualmente paga $70+/mes Pinecone para 500 queries/dia (~15K/mes)
    "pinecone_refugee": DemandCurve(
        base_volume=15_000,
        sensitivity=0.9,      # menos elastico: ya tiene budget aprobado para vector DB
        adoption_ceiling=50_000
    ),
    # Developer que evalua si similarity search resuelve su problema antes de infraestructura
    "stateless_evaluator": DemandCurve(
        base_volume=250_000,
        sensitivity=2.1,      # muy elastico: el valor es no committearse, precio lo rompe
        adoption_ceiling=2_000_000
    ),
    # Plataforma que orquesta embedding + vector DB + ranking como tres servicios hoy
    "orchestration_consolidator": DemandCurve(
        base_volume=1_500_000,
        sensitivity=1.4,      # elasticidad media: ahorra en 2 servicios, negocia volumen
        adoption_ceiling=10_000_000
    ),
}


@dataclass
class FreemiumEquilibriumPoint:
    # Umbral donde el costo de oportunidad del tier free supera el ingreso marginal
    free_ops_limit: int
    paid_conversion_price: float
    monthly_ops_at_conversion: float
    monthly_revenue_at_conversion: float
    implied_arpu_vs_pinecone: float  # ratio vs $70/mes baseline


def compute_freemium_to_paid_threshold(
    curve: DemandCurve,
    free_ops_limit: int = 1_000,
    conversion_friction_factor: float = 1.35  # developers convierten 35% mas arriba del optimo teorico
) -> FreemiumEquilibriumPoint:
    # Precio al que el usuario free alcanza el limite y la friccion de conversion se justifica
    opt = maximize_nmi_cosine_api_revenue(curve)
    # Precio de conversion: optimo ajustado por friccion psicologica de pagar la primera vez
    p_conversion = min(opt.price * conversion_friction_factor, P_MAX)
    q_conversion = curve.quantity(p_conversion)
    monthly_revenue = p_conversion * q_conversion
    # Comparacion contra $70/mes Pinecone: el pain point mas concreto del ICP
    arpu_ratio = monthly_revenue / 70.0
    return FreemiumEquilibriumPoint(
        free_ops_limit=free_ops_limit,
        paid_conversion_price=round(p_conversion, 5),
        monthly_ops_at_conversion=round(q_conversion),
        monthly_revenue_at_conversion=round(monthly_revenue, 2),
        implied_arpu_vs_pinecone=round(arpu_ratio, 3)
    )


def run_similarity_search_pricing_model() -> dict:
    results = {}
    for scenario_name, curve in ADOPTION_SCENARIOS.items():
        opt = maximize_nmi_cosine_api_revenue(curve)
        freemium = compute_freemium_to_paid_threshold(curve)

        # Scan de revenue en rango de precios para analisis de sensibilidad
        price_grid = np.linspace(P_MIN, P_MAX, 200)
        revenues = [curve.revenue(p) for p in price_grid]
        revenue_weighted_avg_price = float(np.average(price_grid, weights=revenues))

        results[scenario_name] = {
            "optimal_pricing": {
                "price_per_op": opt.price,
                "monthly_ops": opt.quantity,
                "monthly_revenue_usd": opt.revenue,
                "elasticity": opt.elasticity_at_optimum,
                "demand_is_elastic": opt.is_elastic,
            },
            "freemium_threshold": {
                "free_ops_limit": freemium.free_ops_limit,
                "conversion_price": freemium.paid_conversion_price,
                "ops_at_conversion": freemium.monthly_ops_at_conversion,
                "revenue_at_conversion_usd": freemium.monthly_revenue_at_conversion,
                "arpu_ratio_vs_pinecone_70usd": freemium.implied_arpu_vs_pinecone,
            },
            "sensitivity": {
                "revenue_weighted_avg_price": round(revenue_weighted_avg_price, 5),
                "max_revenue_in_grid": round(max(revenues), 2),
                "price_at_max_revenue": round(float(price_grid[int(np.argmax(revenues))]), 5),
            }
        }
    return results


if __name__ == "__main__":
    import json
    output = run_similarity_search_pricing_model()
    print(json.dumps(output, indent=2))