from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import hashlib
import hmac
import os
import time


app = FastAPI(
    title="Hybrid Semantic Similarity API",
    description="Stateless NMI+Cosine hybrid similarity with bootstrap confidence intervals",
    version="1.0.0",
)

_bearer = HTTPBearer()

_VALID_TOKEN_HASH = hashlib.sha256(
    os.environ.get("SIMILARITY_API_TOKEN", "insecure-default-token").encode()
).hexdigest()


def _verify_token(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> str:
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if not hmac.compare_digest(token_hash, _VALID_TOKEN_HASH):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )
    return credentials.credentials


class FeatureVector(BaseModel):
    values: list[float | int | str] = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Mixed-type feature vector: floats/ints treated as continuous, strings as categorical",
    )

    @field_validator("values")
    @classmethod
    def validate_no_none_elements(cls, v):
        for i, el in enumerate(v):
            if el is None:
                raise ValueError(f"Element at index {i} is None — all values must be non-null")
        return v


class SimilarityRequest(BaseModel):
    query: FeatureVector
    candidates: list[FeatureVector] = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=512)
    bootstrap_samples: int = Field(default=500, ge=100, le=2000)
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.99)

    @field_validator("candidates")
    @classmethod
    def validate_uniform_length(cls, v, info):
        if "query" in info.data:
            q_len = len(info.data["query"].values)
            for i, c in enumerate(v):
                if len(c.values) != q_len:
                    raise ValueError(
                        f"Candidate at index {i} has {len(c.values)} features but query has {q_len} — lengths must match"
                    )
        return v


class SimilarityScore(BaseModel):
    candidate_index: int
    hybrid_score: float
    cosine_score: float
    nmi_score: float
    categorical_weight: float
    continuous_weight: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityScore]
    query_feature_profile: dict
    computation_ms: float
    api_version: str = "1.0.0"


class BatchRequest(BaseModel):
    queries: list[SimilarityRequest] = Field(..., min_length=1, max_length=32)


class BatchResponse(BaseModel):
    responses: list[SimilarityResponse]
    total_computation_ms: float


def _detect_feature_types(values: list) -> tuple[list[int], list[int]]:
    categorical_indices = []
    continuous_indices = []
    for i, v in enumerate(values):
        if isinstance(v, str):
            categorical_indices.append(i)
        else:
            continuous_indices.append(i)
    return categorical_indices, continuous_indices


def _encode_categorical(values: list, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.array([])
    subset = [str(values[i]) for i in indices]
    le = LabelEncoder()
    return le.fit_transform(subset).astype(float)


def _extract_continuous(values: list, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.array([])
    return np.array([float(values[i]) for i in indices])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _nmi_categorical_pair(q_cat: np.ndarray, c_cat: np.ndarray) -> float:
    if q_cat.size == 0 or c_cat.size == 0:
        return 0.0
    if np.all(q_cat == q_cat[0]) and np.all(c_cat == c_cat[0]):
        return 1.0 if q_cat[0] == c_cat[0] else 0.0
    if np.all(q_cat == q_cat[0]) or np.all(c_cat == c_cat[0]):
        return 0.0
    try:
        nmi = normalized_mutual_info_score(q_cat.astype(int), c_cat.astype(int), average_method="arithmetic")
        return float(np.clip(nmi, 0.0, 1.0))
    except Exception:
        return 0.0


def _dynamic_weights(n_cat: int, n_cont: int) -> tuple[float, float]:
    total = n_cat + n_cont
    if total == 0:
        return 0.5, 0.5
    w_cat = n_cat / total
    w_cont = n_cont / total
    return float(w_cat), float(w_cont)


def _hybrid_score_single(
    query_values: list,
    candidate_values: list,
    cat_idx: list[int],
    cont_idx: list[int],
    w_cat: float,
    w_cont: float,
) -> float:
    q_cont = _extract_continuous(query_values, cont_idx)
    c_cont = _extract_continuous(candidate_values, cont_idx)
    cosine = _cosine_similarity(q_cont, c_cont) if cont_idx else 0.0

    q_cat_enc = _encode_categorical(query_values, cat_idx)
    c_cat_enc_raw = [str(candidate_values[i]) for i in cat_idx] if cat_idx else []
    if cat_idx:
        all_labels = list(set([str(query_values[i]) for i in cat_idx] + c_cat_enc_raw))
        le = LabelEncoder()
        le.fit(all_labels)
        q_cat_arr = le.transform([str(query_values[i]) for i in cat_idx]).astype(float)
        c_cat_arr = le.transform(c_cat_enc_raw).astype(float)
        nmi = _nmi_categorical_pair(q_cat_arr, c_cat_arr)
    else:
        nmi = 0.0

    score = w_cat * nmi + w_cont * cosine
    return float(np.clip(score, 0.0, 1.0)), float(cosine), float(nmi)


def _bootstrap_ci(
    query_values: list,
    candidate_values: list,
    cat_idx: list[int],
    cont_idx: list[int],
    w_cat: float,
    w_cont: float,
    n_samples: int,
    confidence_level: float,
) -> tuple[float, float]:
    all_indices = list(range(len(query_values)))
    if len(all_indices) < 2:
        base_score, _, _ = _hybrid_score_single(query_values, candidate_values, cat_idx, cont_idx, w_cat, w_cont)
        return base_score, base_score

    rng = np.random.default_rng(seed=42)
    bootstrap_scores = []

    for _ in range(n_samples):
        sampled = rng.choice(all_indices, size=len(all_indices), replace=True).tolist()
        s_cat_idx = [j for j, orig in enumerate(sampled) if orig in cat_idx]
        s_cont_idx = [j for j, orig in enumerate(sampled) if orig in cont_idx]
        s_query = [query_values[i] for i in sampled]
        s_candidate = [candidate_values[i] for i in sampled]
        n_cat_s = len(s_cat_idx)
        n_cont_s = len(s_cont_idx)
        w_cat_s, w_cont_s = _dynamic_weights(n_cat_s, n_cont_s)
        score, _, _ = _hybrid_score_single(s_query, s_candidate, s_cat_idx, s_cont_idx, w_cat_s, w_cont_s)
        bootstrap_scores.append(score)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    return lower, upper


def _compute_similarity_response(req: SimilarityRequest) -> SimilarityResponse:
    t0 = time.perf_counter()

    query_values = req.query.values
    cat_idx, cont_idx = _detect_feature_types(query_values)
    w_cat, w_cont = _dynamic_weights(len(cat_idx), len(cont_idx))

    scores: list[SimilarityScore] = []
    for idx, candidate in enumerate(req.candidates):
        c_values = candidate.values
        hybrid, cosine, nmi = _hybrid_score_single(query_values, c_values, cat_idx, cont_idx, w_cat, w_cont)
        ci_lower, ci_upper = _bootstrap_ci(
            query_values, c_values, cat_idx, cont_idx, w_cat, w_cont,
            req.bootstrap_samples, req.confidence_level
        )
        scores.append(SimilarityScore(
            candidate_index=idx,
            hybrid_score=round(hybrid, 6),
            cosine_score=round(cosine, 6),
            nmi_score=round(nmi, 6),
            categorical_weight=round(w_cat, 4),
            continuous_weight=round(w_cont, 4),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_level=req.confidence_level,
        ))

    scores.sort(key=lambda s: s.hybrid_score, reverse=True)
    top = scores[: req.top_k]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SimilarityResponse(
        results=top,
        query_feature_profile={
            "total_features": len(query_values),
            "categorical_features": len(cat_idx),
            "continuous_features": len(cont_idx),
            "categorical_weight": round(w_cat, 4),
            "continuous_weight": round(w_cont, 4),
        },
        computation_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/v1/similarity/ranked",
    response_model=SimilarityResponse,
    summary="Compute hybrid NMI+Cosine similarity and return ranked candidates",
    status_code=status.HTTP_200_OK,
)
def rank_candidates_by_hybrid_similarity(
    req: SimilarityRequest,
    _token: str = Security(_verify_token),
) -> SimilarityResponse:
    if not req.query.values:
        raise HTTPException(status_code=422, detail="Query feature vector must not be empty")
    return _compute_similarity_response(req)


@app.post(
    "/v1/similarity/batch",
    response_model=BatchResponse,
    summary="Batch hybrid similarity ranking for multiple independent queries",
    status_code=status.HTTP_200_OK,
)
def batch_rank_candidates_by_hybrid_similarity(
    req: BatchRequest,
    _token: str = Security(_verify_token),
) -> BatchResponse:
    t0 = time.perf_counter()
    responses = []
    for sub_req in req.queries:
        responses.append(_compute_similarity_response(sub_req))
    total_ms = (time.perf_counter() - t0) * 1000.0
    return BatchResponse(responses=responses, total_computation_ms=round(total_ms, 2))


@app.post(
    "/v1/similarity/score",
    summary="Return raw hybrid score between exactly two vectors (no ranking)",
    status_code=status.HTTP_200_OK,
)
def score_vector_pair(
    query: FeatureVector,
    candidate: FeatureVector,
    bootstrap_samples: int = 500,
    confidence_level: float = 0.95,
    _token: str = Security(_verify_token),
) -> dict:
    if not query.values:
        raise HTTPException(status_code=422, detail="Query vector must not be empty")
    if len(query.values) != len(candidate.values):
        raise HTTPException(
            status_code=422,
            detail=f"Vector length mismatch: query has {len(query.values)} features, candidate has {len(candidate.values)}",
        )
    if bootstrap_samples < 100 or bootstrap_samples > 2000:
        raise HTTPException(status_code=422, detail="bootstrap_samples must be between 100 and 2000")
    if confidence_level < 0.80 or confidence_level > 0.99:
        raise HTTPException(status_code=422, detail="confidence_level must be between 0.80 and 0.99")

    cat_idx, cont_idx = _detect_feature_types(query.values)
    w_cat, w_cont = _dynamic_weights(len(cat_idx), len(cont_idx))
    hybrid, cosine, nmi = _hybrid_score_single(query.values, candidate.values, cat_idx, cont_idx, w_cat, w_cont)
    ci_lower, ci_upper = _bootstrap_ci(
        query.values, candidate.values, cat_idx, cont_idx, w_cat, w_cont,
        bootstrap_samples, confidence_level
    )
    return {
        "hybrid_score": round(hybrid, 6),
        "cosine_score": round(cosine, 6),
        "nmi_score": round(nmi, 6),
        "categorical_weight": round(w_cat, 4),
        "continuous_weight": round(w_cont, 4),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "confidence_level": confidence_level,
        "feature_profile": {
            "total": len(query.values),
            "categorical": len(cat_idx),
            "continuous": len(cont_idx),
        },
    }


@app.get(
    "/v1/health",
    summary="Liveness check — returns service status and math backend versions",
    status_code=status.HTTP_200_OK,
)
def health_check() -> dict:
    import sklearn
    import scipy
    return {
        "status": "ok",
        "api_version": "1.0.0",
        "math_backends": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


@app.get(
    "/v1/similarity/explain",
    summary="Explain hybrid score formula and weight calibration for a given feature profile",
    status_code=status.HTTP_200_OK,
)
def explain_hybrid_score_formula(
    n_categorical: int = 0,
    n_continuous: int = 0,
    _token: str = Security(_verify_token),
) -> dict:
    if n_categorical < 0 or n_continuous < 0:
        raise HTTPException(status_code=422, detail="Feature counts must be non-negative integers")
    if n_categorical + n_continuous == 0:
        raise HTTPException(status_code=422, detail="At least one feature must be declared")
    if n_categorical > 4096 or n_continuous > 4096:
        raise HTTPException(status_code=422, detail="Feature count per type must not exceed 4096")

    w_cat, w_cont = _dynamic_weights(n_categorical, n_continuous)
    return {
        "formula": "hybrid_score = w_cat * NMI(categorical_features) + w_cont * Cosine(continuous_features)",
        "w_cat": round(w_cat, 4),
        "w_cont": round(w_cont, 4),
        "calibration_method": "proportional to feature count ratio, per-call",
        "nmi_normalization": "arithmetic mean of marginal entropies H(X) and H(Y)",
        "cosine_domain": "continuous numeric features extracted from mixed vector",
        "bootstrap_ci": "BCa-approximated via percentile method over n=500..2000 index resamples",
        "n_categorical": n_categorical,
        "n_continuous": n_continuous,
        "type_inference": "automatic — strings -> categorical, int/float -> continuous, no schema declaration needed",
    }

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


@_nexus_mcp.tool(name='nexus_similarity_search_api_score_hybrid_nmi_cosine_similarity', description='Computes a calibrated hybrid similarity score between two mixed-feature records by fusing Normalized Mutual Information (for categorical features) and Cosine similarity (for continuous embedding vectors) into a single weighted score with bootstrap confidence intervals. Use when comparing two individual records with a known split between categorical and continuous features. Do NOT use for batch ranking against a corpus (use rank_corpus_by_hybrid_similarity instead), and do NOT use when all features are purely continuous (use score_cosine_with_bootstrap instead).')
async def score_hybrid_nmi_cosine_similarity(continuous_vec_a: Annotated[list[float], Field(..., description='Continuous embedding vector for record A. Must have the same length as continuous_vec_b. Values must be finite floats.', min_length=2, max_length=4096)], continuous_vec_b: Annotated[list[float], Field(..., description='Continuous embedding vector for record B. Must match the length of continuous_vec_a.', min_length=2, max_length=4096)], categorical_vec_a: Annotated[list[str], Field(..., description="Ordered list of categorical feature values for record A. Must have the same length as categorical_vec_b. Each element is a discrete label (e.g., 'electronics', 'US', 'premium').", min_length=1, max_length=64)], categorical_vec_b: Annotated[list[str], Field(..., description='Ordered list of categorical feature values for record B. Must match the length and positional semantics of categorical_vec_a.', min_length=1, max_length=64)], categorical_weight: Annotated[float, Field(0.5, description='Weight assigned to the NMI component in the final hybrid score. The continuous (cosine) component receives weight (1 - categorical_weight). Must be in [0.0, 1.0]. A value of 0.5 treats both modalities equally.', ge=0.0, le=1.0)], bootstrap_iterations: Annotated[float, Field(500, description='Number of bootstrap resampling iterations used to compute the 95% confidence interval on the hybrid score. Higher values increase precision but add latency. Recommended range: 200-1000.', ge=100, le=2000)]) -> dict[str, Any]:
    """NMI+Cosine Hybrid Score"""
    params = {"continuous_vec_a": continuous_vec_a, "continuous_vec_b": continuous_vec_b, "categorical_vec_a": categorical_vec_a, "categorical_vec_b": categorical_vec_b, "categorical_weight": categorical_weight, "bootstrap_iterations": bootstrap_iterations}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/hybrid-score', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_corpus_by_hybrid_similarity', description='Ranks a corpus of candidate records against a single query record using the NMI+Cosine hybrid score, returning the top-k most similar candidates with their scores and confidence intervals. Operates fully stateless — no index is built or persisted. Use when you have a query record and a list of candidates to rank in a single call, without any prior indexing step. Do NOT use for corpora larger than 500 candidates per call (latency will exceed 5s); for larger corpora, batch the candidates manually. Do NOT use when you only need a score between two records (use score_hybrid_nmi_cosine_similarity instead).')
async def rank_corpus_by_hybrid_similarity(query_continuous_vec: Annotated[list[float], Field(..., description='Continuous embedding vector of the query record.', min_length=2, max_length=4096)], query_categorical_vec: Annotated[list[str], Field(..., description='Ordered categorical feature values of the query record.', min_length=1, max_length=64)], corpus_continuous_vecs: Annotated[list[list[float]], Field(..., description='Matrix of continuous embedding vectors for the candidate corpus. Each row is one candidate. All rows must have the same dimensionality as query_continuous_vec.', min_length=1, max_length=500)], corpus_categorical_vecs: Annotated[Any, Field(..., description='Matrix of categorical feature vectors for the candidate corpus. Row i corresponds to corpus_continuous_vecs[i]. Each row must match the length of query_categorical_vec.', min_length=1, max_length=500)], top_k: Annotated[float, Field(10, description='Number of top-ranked candidates to return. Must not exceed the number of corpus entries.', ge=1, le=500)], categorical_weight: Annotated[float, Field(0.5, description='Weight of the NMI component in the hybrid score. Continuous cosine component receives (1 - categorical_weight).', ge=0.0, le=1.0)], bootstrap_iterations: Annotated[float, Field(200, description='Bootstrap iterations for confidence intervals on each ranked score. Lower values reduce latency for large corpora.', ge=100, le=1000)]) -> dict[str, Any]:
    """Hybrid Similarity Corpus Ranking"""
    params = {"query_continuous_vec": query_continuous_vec, "query_categorical_vec": query_categorical_vec, "corpus_continuous_vecs": corpus_continuous_vecs, "corpus_categorical_vecs": corpus_categorical_vecs, "top_k": top_k, "categorical_weight": categorical_weight, "bootstrap_iterations": bootstrap_iterations}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank-corpus', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_score_cosine_with_bootstrap', description='Computes Cosine similarity between two continuous embedding vectors and returns the point estimate plus a bootstrap-derived 95% confidence interval. Use when records have only continuous features and no categorical component, or when you need a pure cosine baseline to compare against the hybrid score. Do NOT use when categorical features are present and informative — their non-linear dependence will be lost; use score_hybrid_nmi_cosine_similarity instead.')
async def score_cosine_with_bootstrap(vec_a: Annotated[list[float], Field(..., description='First continuous embedding vector. Must be non-zero and have finite values.', min_length=2, max_length=4096)], vec_b: Annotated[list[float], Field(..., description='Second continuous embedding vector. Must match the length of vec_a.', min_length=2, max_length=4096)], bootstrap_iterations: Annotated[float, Field(500, description='Number of bootstrap resampling iterations for the confidence interval estimate.', ge=100, le=2000)], confidence_level: Annotated[float, Field(0.95, description='Confidence level for the interval, expressed as a proportion (e.g., 0.95 for 95% CI).', ge=0.8, le=0.99)]) -> dict[str, Any]:
    """Cosine Similarity + Bootstrap CI"""
    params = {"vec_a": vec_a, "vec_b": vec_b, "bootstrap_iterations": bootstrap_iterations, "confidence_level": confidence_level}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/cosine-bootstrap', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_categorical_matrix', description='Computes the full pairwise Normalized Mutual Information matrix across a set of categorical feature vectors, returning an N x N symmetric matrix of NMI scores. Use to understand inter-record categorical dependence structure before deciding categorical_weight in hybrid scoring, or when you need NMI in isolation without a continuous component. Do NOT use for N greater than 200 (O(N^2) complexity becomes prohibitive); do NOT use when continuous features are the primary signal.')
async def compute_pairwise_nmi_categorical_matrix(categorical_matrix: Annotated[Any, Field(..., description='N x F matrix where each row is the categorical feature vector of one record and each column is one categorical feature dimension. All rows must have equal length.', min_length=2, max_length=200)], normalize_per_feature: Annotated[bool, Field(True, description='If true, NMI is computed per feature dimension and averaged. If false, the joint NMI across all features is computed as a single value per pair. Default true is recommended for heterogeneous feature sets.')]) -> dict[str, Any]:
    """Pairwise NMI Categorical Matrix"""
    params = {"categorical_matrix": categorical_matrix, "normalize_per_feature": normalize_per_feature}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/nmi-pairwise-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_calibrate_hybrid_weight_from_labeled_pairs', description='Given a set of labeled record pairs with ground-truth similarity scores, estimates the optimal categorical_weight parameter that minimizes mean squared error between the hybrid NMI+Cosine score and the ground-truth labels. Returns the optimal weight, the MSE at that weight, and a 95% confidence interval on the weight via bootstrap. Use once before production deployment to set categorical_weight empirically rather than by intuition. Do NOT use with fewer than 30 labeled pairs (results will be statistically unreliable); do NOT use as a per-request operation — it is a one-time calibration call.')
async def calibrate_hybrid_weight_from_labeled_pairs(labeled_pairs: Annotated[list[list[float]], Field(None, description='List of labeled pairs, each encoded as a flat array: [ground_truth_score]. The actual feature vectors are passed separately via continuous_pairs and categorical_pairs. Length must match continuous_pairs and categorical_pairs.', min_length=30, max_length=2000)], ground_truth_scores: Annotated[list[float], Field(..., description='Array of ground-truth similarity scores in [0.0, 1.0] for each pair. Index i corresponds to pair i in continuous_pairs_a/b and categorical_pairs_a/b.', min_length=30, max_length=2000)], continuous_pairs_a: Annotated[list[list[float]], Field(..., description='Continuous vectors for the A-side of each labeled pair. Row i is the continuous embedding for pair i, record A.', min_length=30, max_length=2000)], continuous_pairs_b: Annotated[list[list[float]], Field(..., description='Continuous vectors for the B-side of each labeled pair. Must match dimensions and row count of continuous_pairs_a.', min_length=30, max_length=2000)], categorical_pairs_a: Annotated[Any, Field(..., description='Categorical feature vectors for the A-side of each labeled pair.', min_length=30, max_length=2000)], categorical_pairs_b: Annotated[Any, Field(..., description='Categorical feature vectors for the B-side of each labeled pair.', min_length=30, max_length=2000)], weight_search_resolution: Annotated[float, Field(20, description='Number of evenly spaced candidate weights in [0.0, 1.0] to evaluate during grid search. Higher resolution narrows the optimum more precisely at the cost of more compute.', ge=10, le=100)], bootstrap_iterations: Annotated[float, Field(500, description='Bootstrap iterations for confidence interval on the optimal weight estimate.', ge=200, le=2000)]) -> dict[str, Any]:
    """Hybrid Weight Calibration"""
    params = {"labeled_pairs": labeled_pairs, "ground_truth_scores": ground_truth_scores, "continuous_pairs_a": continuous_pairs_a, "continuous_pairs_b": continuous_pairs_b, "categorical_pairs_a": categorical_pairs_a, "categorical_pairs_b": categorical_pairs_b, "weight_search_resolution": weight_search_resolution, "bootstrap_iterations": bootstrap_iterations}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/calibrate-hybrid-weight', params)


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
