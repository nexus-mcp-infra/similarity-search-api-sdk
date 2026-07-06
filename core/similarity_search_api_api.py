from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import hashlib
import os
import time
import math

app = FastAPI(
    title="Similarity Search API",
    description="Stateless NMI+cosine unified scoring over raw JSON payloads — no index, no embedding, no setup.",
    version="1.0.0",
)

security = HTTPBearer()

API_KEY = os.environ.get("SIMILARITY_API_KEY", "dev-insecure-key")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return credentials.credentials


class SimilarityItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    features: dict[str, Any] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v):
        if not v:
            raise ValueError("features dict must contain at least one key-value pair.")
        return v


class RankedSimilarityRequest(BaseModel):
    query: SimilarityItem
    corpus: list[SimilarityItem] = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=500)
    alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for cosine score. NMI weight = 1 - alpha. Use alpha closer to 1.0 for numeric-heavy payloads, closer to 0.0 for categorical-heavy.",
    )

    @model_validator(mode="after")
    def corpus_ids_unique(self):
        ids = [item.id for item in self.corpus]
        if len(ids) != len(set(ids)):
            raise ValueError("corpus item IDs must be unique.")
        return self


class ScoredItem(BaseModel):
    id: str
    score: float
    cosine_score: float
    nmi_score: float


class RankedSimilarityResponse(BaseModel):
    results: list[ScoredItem]
    query_id: str
    alpha: float
    corpus_size: int
    latency_ms: float


class BatchRequest(BaseModel):
    queries: list[SimilarityItem] = Field(..., min_length=1, max_length=50)
    corpus: list[SimilarityItem] = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=500)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def corpus_ids_unique(self):
        ids = [item.id for item in self.corpus]
        if len(ids) != len(set(ids)):
            raise ValueError("corpus item IDs must be unique.")
        return self


class BatchResponse(BaseModel):
    results: list[RankedSimilarityResponse]
    total_latency_ms: float


class InspectFeaturesRequest(BaseModel):
    items: list[SimilarityItem] = Field(..., min_length=2, max_length=5000)


class FeatureProfile(BaseModel):
    key: str
    inferred_type: str
    coverage_ratio: float
    unique_ratio: float
    recommended_alpha_adjustment: str


class InspectFeaturesResponse(BaseModel):
    profiles: list[FeatureProfile]
    recommended_alpha: float
    notes: str


def _laplace_entropy(values: list[Any], pseudocount: float = 1.0) -> float:
    from collections import Counter
    counts = Counter(str(v) for v in values)
    total = sum(counts.values()) + pseudocount * len(counts)
    probs = np.array([(c + pseudocount) / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _pairwise_nmi(query_features: dict[str, Any], corpus_features: list[dict[str, Any]]) -> np.ndarray:
    keys = list(query_features.keys())
    n = len(corpus_features)
    nmi_scores = np.zeros(n)

    for key in keys:
        query_val = query_features.get(key)
        if query_val is None:
            continue

        corpus_vals = [cf.get(key) for cf in corpus_features]
        non_null_indices = [i for i, v in enumerate(corpus_vals) if v is not None]
        if not non_null_indices:
            continue

        for i in non_null_indices:
            c_val = corpus_vals[i]
            joint_population = [str(query_val), str(c_val)]
            marginal_q = [str(query_val)]
            marginal_c = [str(c_val)]

            h_q = _laplace_entropy(marginal_q)
            h_c = _laplace_entropy(marginal_c)
            h_joint = _laplace_entropy(joint_population)

            mi = max(0.0, h_q + h_c - h_joint)
            denominator = math.sqrt(h_q * h_c) if h_q > 0 and h_c > 0 else 0.0
            nmi = mi / denominator if denominator > 1e-12 else 0.0
            nmi_scores[i] += min(nmi, 1.0)

    if keys:
        nmi_scores /= len(keys)

    return np.clip(nmi_scores, 0.0, 1.0)


def _feature_dict_to_text(features: dict[str, Any]) -> str:
    parts = []
    for k, v in sorted(features.items()):
        if isinstance(v, (int, float)):
            parts.append(f"{k} {v}")
        elif isinstance(v, list):
            parts.append(f"{k} {' '.join(str(x) for x in v)}")
        else:
            parts.append(f"{k} {v}")
    return " ".join(parts)


def _cosine_similarity_tfidf(query_features: dict[str, Any], corpus_features: list[dict[str, Any]]) -> np.ndarray:
    documents = [_feature_dict_to_text(query_features)] + [
        _feature_dict_to_text(cf) for cf in corpus_features
    ]
    try:
        vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\S+")
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return np.zeros(len(corpus_features))

    query_vec = tfidf_matrix[0]
    corpus_matrix = tfidf_matrix[1:]

    dot_products = (corpus_matrix @ query_vec.T).toarray().flatten()
    query_norm = np.sqrt(query_vec.multiply(query_vec).sum())
    corpus_norms = np.sqrt(np.array(corpus_matrix.multiply(corpus_matrix).sum(axis=1)).flatten())

    denominators = corpus_norms * query_norm
    cosine_scores = np.where(denominators > 1e-12, dot_products / denominators, 0.0)
    return np.clip(cosine_scores, 0.0, 1.0)


def _numeric_cosine(query_features: dict[str, Any], corpus_features: list[dict[str, Any]]) -> np.ndarray:
    numeric_keys = [
        k for k, v in query_features.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not numeric_keys:
        return np.zeros(len(corpus_features))

    q_vec = np.array([float(query_features[k]) for k in numeric_keys])
    c_matrix = np.array([
        [float(cf.get(k, 0.0)) if isinstance(cf.get(k), (int, float)) else 0.0 for k in numeric_keys]
        for cf in corpus_features
    ])

    q_norm = np.linalg.norm(q_vec)
    c_norms = np.linalg.norm(c_matrix, axis=1)
    if q_norm < 1e-12:
        return np.zeros(len(corpus_features))

    dots = c_matrix @ q_vec
    denominators = c_norms * q_norm
    return np.where(denominators > 1e-12, dots / denominators, 0.0)


def _unified_cosine(query_features: dict[str, Any], corpus_features: list[dict[str, Any]]) -> np.ndarray:
    tfidf_cos = _cosine_similarity_tfidf(query_features, corpus_features)
    numeric_cos = _numeric_cosine(query_features, corpus_features)

    has_numeric = any(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in query_features.values()
    )
    if has_numeric:
        return np.clip(0.5 * tfidf_cos + 0.5 * numeric_cos, 0.0, 1.0)
    return tfidf_cos


def _run_ranked_similarity(
    query: SimilarityItem,
    corpus: list[SimilarityItem],
    top_k: int,
    alpha: float,
) -> tuple[list[ScoredItem], float]:
    t0 = time.perf_counter()
    corpus_features = [item.features for item in corpus]

    cosine_scores = _unified_cosine(query.features, corpus_features)
    nmi_scores = _pairwise_nmi(query.features, corpus_features)

    unified_scores = alpha * cosine_scores + (1.0 - alpha) * nmi_scores

    top_k_actual = min(top_k, len(corpus))
    top_indices = np.argpartition(unified_scores, -top_k_actual)[-top_k_actual:]
    top_indices = top_indices[np.argsort(unified_scores[top_indices])[::-1]]

    results = [
        ScoredItem(
            id=corpus[i].id,
            score=float(round(unified_scores[i], 6)),
            cosine_score=float(round(cosine_scores[i], 6)),
            nmi_score=float(round(nmi_scores[i], 6)),
        )
        for i in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms


@app.post("/rank", response_model=RankedSimilarityResponse, summary="Rank corpus items by NMI+cosine similarity to a query item.")
def rank_by_nmi_cosine_similarity(
    request: RankedSimilarityRequest,
    _: str = Depends(verify_api_key),
) -> RankedSimilarityResponse:
    results, latency_ms = _run_ranked_similarity(
        request.query, request.corpus, request.top_k, request.alpha
    )
    return RankedSimilarityResponse(
        results=results,
        query_id=request.query.id,
        alpha=request.alpha,
        corpus_size=len(request.corpus),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/rank/batch", response_model=BatchResponse, summary="Rank corpus items for multiple queries in a single HTTP call.")
def batch_rank_by_nmi_cosine_similarity(
    request: BatchRequest,
    _: str = Depends(verify_api_key),
) -> BatchResponse:
    t0 = time.perf_counter()
    all_results = []
    for query in request.queries:
        results, latency_ms = _run_ranked_similarity(
            query, request.corpus, request.top_k, request.alpha
        )
        all_results.append(
            RankedSimilarityResponse(
                results=results,
                query_id=query.id,
                alpha=request.alpha,
                corpus_size=len(request.corpus),
                latency_ms=round(latency_ms, 3),
            )
        )
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return BatchResponse(results=all_results, total_latency_ms=round(total_latency_ms, 3))


@app.post("/inspect/features", response_model=InspectFeaturesResponse, summary="Profile feature types in a dataset and recommend alpha for NMI+cosine blend.")
def inspect_features_and_recommend_alpha(
    request: InspectFeaturesRequest,
    _: str = Depends(verify_api_key),
) -> InspectFeaturesResponse:
    all_keys: set[str] = set()
    for item in request.items:
        all_keys.update(item.features.keys())

    profiles = []
    categorical_count = 0
    numeric_count = 0
    n = len(request.items)

    for key in sorted(all_keys):
        values = [item.features.get(key) for item in request.items]
        present = [v for v in values if v is not None]
        coverage = len(present) / n

        numeric_vals = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
        is_numeric = len(numeric_vals) / len(present) > 0.7 if present else False

        unique_count = len(set(str(v) for v in present))
        unique_ratio = unique_count / len(present) if present else 0.0

        if is_numeric:
            inferred_type = "numeric"
            numeric_count += 1
            adjustment = "favors cosine (increase alpha)"
        elif unique_ratio < 0.2:
            inferred_type = "categorical_low_cardinality"
            categorical_count += 1
            adjustment = "strongly favors NMI (decrease alpha significantly)"
        elif unique_ratio < 0.6:
            inferred_type = "categorical_high_cardinality"
            categorical_count += 1
            adjustment = "favors NMI (decrease alpha moderately)"
        else:
            inferred_type = "text_or_id"
            adjustment = "favors cosine TF-IDF (increase alpha slightly)"

        profiles.append(
            FeatureProfile(
                key=key,
                inferred_type=inferred_type,
                coverage_ratio=round(coverage, 4),
                unique_ratio=round(unique_ratio, 4),
                recommended_alpha_adjustment=adjustment,
            )
        )

    total_typed = categorical_count + numeric_count
    if total_typed == 0:
        recommended_alpha = 0.6
    else:
        recommended_alpha = round(0.3 + 0.7 * (numeric_count / total_typed), 2)
        recommended_alpha = max(0.1, min(0.9, recommended_alpha))

    cat_pct = round(100 * categorical_count / max(total_typed, 1))
    num_pct = 100 - cat_pct
    notes = (
        f"Dataset has {cat_pct}% categorical features and {num_pct}% numeric features. "
        f"Recommended alpha={recommended_alpha} (cosine weight). "
        f"Pass this value as 'alpha' in /rank calls for optimal NMI+cosine balance."
    )

    return InspectFeaturesResponse(
        profiles=profiles,
        recommended_alpha=recommended_alpha,
        notes=notes,
    )


@app.get("/health", summary="Liveness probe — returns service status and version.")
def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0", "math_backend": "numpy+scipy+sklearn"}

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import AsyncExitStack as _NexusMcpExitStack

import httpx
from pydantic import Field
from mcp.server.fastmcp import FastMCP as _NexusFastMCP

_nexus_mcp = _NexusFastMCP('nexus-similarity-search-api', stateless_http=True)


async def _nexus_mcp_call_core(method: str, path: str, params: dict) -> Any:
    """
    Llama al endpoint real del core -- via ASGI in-process (sin red
    real, sin segundo proceso), no un HTTP call externo. `app` ya
    existe en este mismo modulo (es el FastAPI que FORGE genero).
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        else:
            resp = await client.post(path, json=params)
        resp.raise_for_status()
        return resp.json()


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_by_nmi_cosine_fusion', description='Ranks a corpus of mixed-type items (text, categorical, numeric) against a query item using a unified score that fuses Normalized Mutual Information (for categorical/discrete feature dependencies) with cosine similarity (for continuous/text features). Use when your dataset has heterogeneous feature types and you need a single ranked list without precomputed embeddings or a persistent index. Do NOT use when all features are purely continuous/dense vectors (use a pure ANN index instead), when corpus exceeds 10 000 items per call (latency will degrade), or when you need approximate results at scale.')
async def rank_by_nmi_cosine_fusion(query: Annotated[str, Field(..., description='JSON-serialized object representing the query item. Keys are feature names; values are strings (categorical/text) or numbers (numeric). Must share at least one key with corpus items.', min_length=2, max_length=8192)], corpus: Annotated[list[str], Field(..., description='Array of JSON-serialized objects, each representing a corpus item with the same feature schema as the query. Minimum 2 items required. Items with zero overlapping keys with the query are scored 0.', min_length=2, max_length=10000)], nmi_weight: Annotated[float, Field(0.5, description='Weight assigned to the NMI component in the fused score (0.0 to 1.0). Cosine weight is derived as 1 - nmi_weight. Set closer to 1.0 when categorical features dominate; closer to 0.0 for mostly numeric/text corpora.', ge=0.0, le=1.0)], top_k: Annotated[float, Field(10, description='Number of top-ranked results to return. Must be between 1 and the corpus size. Results beyond top_k are discarded before returning.', ge=1, le=500)], numeric_bins: Annotated[float, Field(10, description='Number of equal-width bins used to discretize numeric features before NMI computation. Higher values capture finer distinctions but require larger corpora to be statistically stable. Ignored when there are no numeric features.', ge=2, le=50)]) -> dict[str, Any]:
    """NMI+Cosine Fused Similarity Rank"""
    params = {"query": query, "corpus": corpus, "nmi_weight": nmi_weight, "top_k": top_k, "numeric_bins": numeric_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_nmi_cosine_pair', description='Computes the fused NMI+cosine similarity score between exactly two items and returns the decomposed components (nmi_score, cosine_score, fused_score) alongside per-feature contributions. Use for explainability, threshold calibration, or validating why two specific items are ranked close or far apart. Do NOT use as a substitute for rank_by_nmi_cosine_fusion when ranking a corpus — calling this in a loop over N items is O(N) round-trips and will be slower and costlier than a single ranking call.')
async def score_nmi_cosine_pair(item_a: Annotated[str, Field(..., description='JSON-serialized object for the first item. Keys are feature names; values are strings or numbers.', min_length=2, max_length=4096)], item_b: Annotated[str, Field(..., description='JSON-serialized object for the second item. Must share at least one feature key with item_a to produce a non-zero score.', min_length=2, max_length=4096)], nmi_weight: Annotated[float, Field(0.5, description='Weight of the NMI component in the fused score. Cosine weight = 1 - nmi_weight. Must match the weight used in ranking calls if this is used for threshold calibration.', ge=0.0, le=1.0)], numeric_bins: Annotated[float, Field(10, description='Bin count for numeric discretization, consistent with what would be used in corpus ranking. Default 10.', ge=2, le=50)]) -> dict[str, Any]:
    """Pairwise NMI+Cosine Score"""
    params = {"item_a": item_a, "item_b": item_b, "nmi_weight": nmi_weight, "numeric_bins": numeric_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/score-pair', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_detect_dominant_feature_type', description='Analyzes a corpus payload and returns a breakdown of feature types (categorical, numeric, text) with their relative dominance and the recommended nmi_weight for optimal fusion. Use this as a calibration step before the first ranking call when you are unsure what nmi_weight to use. Do NOT use on every call in a hot path — run once per dataset schema and cache the recommended weight; schema does not change per query.')
async def detect_dominant_feature_type(corpus_sample: Annotated[list[str], Field(..., description='Array of JSON-serialized items (a representative sample, not necessarily the full corpus). Minimum 10 items recommended for stable type inference. Maximum 500 items accepted in a single detection call.', min_length=2, max_length=500)], cardinality_threshold: Annotated[float, Field(20, description='Maximum number of distinct values a numeric feature may have before it is reclassified as categorical for NMI purposes. Features with unique-value count below this threshold are treated as categorical.', ge=2, le=200)]) -> dict[str, Any]:
    """Feature-Type Dominance Detector"""
    params = {"corpus_sample": corpus_sample, "cardinality_threshold": cardinality_threshold}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/detect-feature-types', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_batch_rank_nmi_cosine_multi_query', description='Executes multiple independent similarity ranking operations in a single HTTP call, each with its own query item and shared corpus. Returns one ranked list per query. Use when you need to rank the same corpus against several query items simultaneously (e.g., recommendation for a batch of users, or multi-anchor nearest-neighbor search). Do NOT use when queries share no features with the corpus, when corpus differs per query (submit separate rank calls instead), or when batch size exceeds 50 queries (memory pressure per request).')
async def batch_rank_nmi_cosine_multi_query(queries: Annotated[list[str], Field(..., description='Array of JSON-serialized query objects. Each query is ranked independently against the shared corpus. Maximum 50 queries per batch call.', min_length=1, max_length=50)], corpus: Annotated[list[str], Field(..., description='Shared corpus of JSON-serialized items ranked against every query. Minimum 2 items, maximum 5 000 items when used in batch mode (lower than single-query mode due to per-request memory constraints).', min_length=2, max_length=5000)], nmi_weight: Annotated[float, Field(0.5, description='Shared NMI weight applied uniformly across all queries in the batch. Per-query weight overrides are not supported in batch mode to keep the response schema deterministic.', ge=0.0, le=1.0)], top_k: Annotated[float, Field(10, description='Number of top results returned per query. Applied uniformly to all queries in the batch.', ge=1, le=200)], numeric_bins: Annotated[float, Field(10, description='Bin count for numeric discretization, applied uniformly across all queries.', ge=2, le=50)]) -> dict[str, Any]:
    """Multi-Query Batch NMI+Cosine Rank"""
    params = {"queries": queries, "corpus": corpus, "nmi_weight": nmi_weight, "top_k": top_k, "numeric_bins": numeric_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/batch-rank', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_explain_nmi_contribution_per_feature', description='Given a query and a single ranked result item, decomposes the fused score into per-feature NMI and cosine contributions, identifying which features drove similarity and which suppressed it. Use for debugging unexpected rankings, for building explainability layers in downstream products, or for feature importance analysis. Do NOT use as part of a ranking pipeline — it operates on a single (query, result) pair and does not scale to corpus-level ranking; it is a diagnostic tool only.')
async def explain_nmi_contribution_per_feature(query: Annotated[str, Field(..., description='JSON-serialized query item used in the original ranking call.', min_length=2, max_length=4096)], result_item: Annotated[str, Field(..., description='JSON-serialized corpus item whose score you want explained. Should be one of the items returned by a prior rank_by_nmi_cosine_fusion call.', min_length=2, max_length=4096)], nmi_weight: Annotated[float, Field(0.5, description='Must match the nmi_weight used in the ranking call that produced this result, otherwise the decomposed score will not match the original fused score.', ge=0.0, le=1.0)], numeric_bins: Annotated[float, Field(10, description='Must match the numeric_bins value used in the original ranking call for the NMI decomposition to be consistent.', ge=2, le=50)]) -> dict[str, Any]:
    """Per-Feature NMI Contribution Explainer"""
    params = {"query": query, "result_item": result_item, "nmi_weight": nmi_weight, "numeric_bins": numeric_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/explain-feature-nmi', params)


# Crea el sub-app ASGI de streamable HTTP -- DEBE llamarse antes de
# poder acceder a _nexus_mcp.session_manager (se crea de forma
# perezosa, ver docstring del modulo).
# Se monta en "/" (no en "/mcp"): streamable_http_app() YA expone su
# propia ruta interna en "/mcp" -- montarlo de nuevo en "/mcp" duplica
# el path a "/mcp/mcp" y da 404 (bug real encontrado probando esto en
# runtime con un cliente MCP de verdad, no algo teorico).
_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()
_nexus_mcp_stack = _NexusMcpExitStack()


@app.on_event("startup")
async def _nexus_mcp_startup():
    await _nexus_mcp_stack.enter_async_context(_nexus_mcp.session_manager.run())


@app.on_event("shutdown")
async def _nexus_mcp_shutdown():
    await _nexus_mcp_stack.aclose()


app.mount("/", _nexus_mcp_asgi_app)

# --- NEXUS: reporte de uso real a Stripe (inyectado por forge_output_saver_v6) ---
@app.middleware("http")
async def _nexus_usage_middleware(request, call_next):
    response = await call_next(request)
    try:
        if response.status_code < 400:
            import os as _nexus_os
            import stripe as _nexus_stripe
            _customer_id = _nexus_os.environ.get("STRIPE_CUSTOMER_ID")
            _event_name = _nexus_os.environ.get("STRIPE_EVENT_NAME")
            _secret_key = _nexus_os.environ.get("STRIPE_SECRET_KEY")
            if _customer_id and _event_name and _secret_key:
                _nexus_stripe.api_key = _secret_key
                _nexus_stripe.billing.MeterEvent.create(
                    event_name=_event_name,
                    payload={
                        "stripe_customer_id": _customer_id,
                        "value": "1",
                    },
                )
    except Exception:
        pass  # nunca romper la response real por un fallo de billing
    return response
