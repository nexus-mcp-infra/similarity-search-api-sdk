from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import os
import time
import hashlib
import hmac

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_API_KEY = os.environ.get("SIMILARITY_API_KEY", "")

app = FastAPI(
    title="Stateless Similarity Search API",
    description=(
        "Fuses NMI and cosine similarity into a single entropy-weighted score. "
        "No indexing, no persistent state, no prior upload required."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)


class VectorPayload(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=16384)
    candidates: list[list[float]] = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=10, ge=1, le=512)
    nmi_bins: int = Field(
        default=10,
        ge=3,
        le=64,
        description=(
            "Number of bins for NMI discretization. "
            "Increase for larger candidate sets; reduce for sparse data."
        ),
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fixed NMI weight override [0,1]. If None (recommended), "
            "weight is derived dynamically from marginal entropy of query dimensions."
        ),
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "VectorPayload":
        dim = len(self.query)
        if dim == 0:
            raise ValueError("query vector must have at least one dimension")
        for i, cand in enumerate(self.candidates):
            if len(cand) != dim:
                raise ValueError(
                    f"candidate[{i}] has {len(cand)} dimensions, expected {dim}"
                )
        return self


class BatchPayload(BaseModel):
    queries: list[list[float]] = Field(..., min_length=1, max_length=256)
    candidates: list[list[float]] = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=10, ge=1, le=512)
    nmi_bins: int = Field(default=10, ge=3, le=64)
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_batch_dimensions(self) -> "BatchPayload":
        dim = len(self.queries[0])
        for i, q in enumerate(self.queries):
            if len(q) != dim:
                raise ValueError(
                    f"queries[{i}] has {len(q)} dimensions, expected {dim}"
                )
        for i, c in enumerate(self.candidates):
            if len(c) != dim:
                raise ValueError(
                    f"candidates[{i}] has {len(c)} dimensions, expected {dim}"
                )
        return self


class SimilarityResult(BaseModel):
    index: int
    composite_score: float
    nmi_score: float
    cosine_score: float
    nmi_weight: float


class SearchResponse(BaseModel):
    results: list[SimilarityResult]
    query_dim: int
    candidate_count: int
    nmi_weight_used: float
    latency_ms: float


class BatchSearchResponse(BaseModel):
    results: list[list[SimilarityResult]]
    query_count: int
    candidate_count: int
    latency_ms: float


class MetricsResponse(BaseModel):
    query: list[float]
    reference: list[float]
    nmi_score: float
    cosine_score: float
    composite_score: float
    nmi_weight: float
    marginal_entropies: list[float]


class MetricsPayload(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=16384)
    reference: list[float] = Field(..., min_length=1, max_length=16384)
    nmi_bins: int = Field(default=10, ge=3, le=64)
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_same_dim(self) -> "MetricsPayload":
        if len(self.query) != len(self.reference):
            raise ValueError(
                f"query and reference must have equal dimensions: "
                f"{len(self.query)} != {len(self.reference)}"
            )
        return self


def _require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SIMILARITY_API_KEY environment variable is not set on this server.",
        )
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if not hmac.compare_digest(
        hashlib.sha256(api_key.encode()).digest(),
        hashlib.sha256(_VALID_API_KEY.encode()).digest(),
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key


def _marginal_entropy_per_dim(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Estimate marginal Shannon entropy H(X_i) for each column of matrix
    using equal-width histogram binning.

    H(X_i) = -sum_k p_k * log2(p_k)

    Returns array of shape (D,) with entropy in bits for each dimension.
    Dimensions with zero variance receive H=0 (fully deterministic).
    """
    n_samples, n_dims = matrix.shape
    entropies = np.zeros(n_dims, dtype=np.float64)

    for d in range(n_dims):
        col = matrix[:, d]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            entropies[d] = 0.0
            continue
        counts, _ = np.histogram(col, bins=n_bins, range=(col_min, col_max))
        probs = counts / counts.sum()
        entropies[d] = scipy_entropy(probs, base=2)

    return entropies


def _entropy_weighted_nmi_alpha(query: np.ndarray, n_bins: int) -> tuple[float, np.ndarray]:
    """
    Derive dynamic NMI weight alpha from query marginal entropy.

    alpha = mean(H(X_i)) / log2(n_bins)

    This maps alpha into [0,1]: high-entropy queries (spread, informative)
    push alpha toward 1 (NMI dominates); low-entropy queries (concentrated,
    near-constant) push alpha toward 0 (cosine dominates).

    Returns (alpha, marginal_entropies_per_dim).
    """
    entropies = _marginal_entropy_per_dim(query.reshape(1, -1), n_bins)
    max_entropy = np.log2(n_bins) if n_bins > 1 else 1.0
    alpha = float(np.clip(entropies.mean() / max_entropy, 0.0, 1.0))
    return alpha, entropies


def _cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between query (D,) and candidates (N, D).

    cos(q, c_i) = (q . c_i) / (||q|| * ||c_i||)

    Returns array of shape (N,). Vectors with zero norm yield score 0.0.
    """
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-12:
        return np.zeros(len(candidates), dtype=np.float64)

    cand_norms = np.linalg.norm(candidates, axis=1)
    dot_products = candidates @ query
    denom = cand_norms * query_norm
    safe_denom = np.where(denom < 1e-12, 1.0, denom)
    cosine_scores = np.where(denom < 1e-12, 0.0, dot_products / safe_denom)
    return cosine_scores.astype(np.float64)


def _nmi_per_candidate(
    query: np.ndarray, candidates: np.ndarray, n_bins: int
) -> np.ndarray:
    """
    Compute NMI between query and each candidate using sklearn's
    normalized_mutual_info_score on discretized values.

    NMI(X,Y) = 2*MI(X,Y) / (H(X) + H(Y))

    Discretization uses equal-width binning into n_bins bins per dimension,
    then the joint sequence (query_discretized, candidate_discretized) is
    treated as paired categorical observations for NMI computation.

    Returns array of shape (N,).
    """
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None)
    n_cands = len(candidates)

    all_vectors = np.vstack([query.reshape(1, -1), candidates])
    try:
        kbd.fit(all_vectors)
        discretized = kbd.transform(all_vectors).astype(int)
    except ValueError:
        return np.zeros(n_cands, dtype=np.float64)

    query_disc_flat = discretized[0].flatten()
    nmi_scores = np.zeros(n_cands, dtype=np.float64)

    for i in range(n_cands):
        cand_disc_flat = discretized[i + 1].flatten()
        try:
            score = normalized_mutual_info_score(
                query_disc_flat,
                cand_disc_flat,
                average_method="arithmetic",
            )
        except Exception:
            score = 0.0
        nmi_scores[i] = score

    return nmi_scores


def _compute_composite_scores(
    query: np.ndarray,
    candidates: np.ndarray,
    n_bins: int,
    alpha_override: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Orchestrates entropy-weighted composite score computation.

    composite_i = alpha * NMI(query, candidate_i)
                + (1 - alpha) * cosine(query, candidate_i)

    Where alpha is derived from marginal entropy of the query unless overridden.

    Returns (composite_scores, nmi_scores, cosine_scores, alpha_used, marginal_entropies).
    """
    if alpha_override is not None:
        alpha = float(np.clip(alpha_override, 0.0, 1.0))
        marginal_entropies = _marginal_entropy_per_dim(
            np.vstack([query.reshape(1, -1), candidates]), n_bins
        )
    else:
        alpha, marginal_entropies = _entropy_weighted_nmi_alpha(query, n_bins)

    cosine_scores = _cosine_similarity_batch(query, candidates)
    nmi_scores = _nmi_per_candidate(query, candidates, n_bins)

    composite = alpha * nmi_scores + (1.0 - alpha) * cosine_scores
    return composite, nmi_scores, cosine_scores, alpha, marginal_entropies


@app.post(
    "/v1/similarity/search",
    response_model=SearchResponse,
    summary="Entropy-weighted NMI+Cosine similarity search (stateless)",
    tags=["Similarity"],
)
def search_similar_vectors(
    payload: VectorPayload,
    _key: str = Security(_require_api_key),
) -> SearchResponse:
    """
    Find the top-k candidates most similar to query using the composite
    NMI+Cosine score weighted by the query's marginal entropy distribution.

    Use when: you have a query vector and a candidate set per call, with no
    persistent index. Suitable for datasets with mixed linear/nonlinear feature
    dependencies.

    Do NOT use when: you need approximate nearest-neighbor over millions of
    pre-indexed vectors -- use Pinecone/Weaviate instead.
    """
    t0 = time.perf_counter()

    query_arr = np.array(payload.query, dtype=np.float64)
    candidates_arr = np.array(payload.candidates, dtype=np.float64)

    if not np.all(np.isfinite(query_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query contains non-finite values (NaN or Inf).",
        )
    if not np.all(np.isfinite(candidates_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="candidates contains non-finite values (NaN or Inf).",
        )

    composite, nmi_scores, cosine_scores, alpha, _ = _compute_composite_scores(
        query_arr, candidates_arr, payload.nmi_bins, payload.alpha
    )

    top_k = min(payload.top_k, len(candidates_arr))
    top_indices = np.argpartition(composite, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(composite[top_indices])[::-1]]

    results = [
        SimilarityResult(
            index=int(idx),
            composite_score=round(float(composite[idx]), 8),
            nmi_score=round(float(nmi_scores[idx]), 8),
            cosine_score=round(float(cosine_scores[idx]), 8),
            nmi_weight=round(alpha, 8),
        )
        for idx in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResponse(
        results=results,
        query_dim=len(payload.query),
        candidate_count=len(payload.candidates),
        nmi_weight_used=round(alpha, 8),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/v1/similarity/batch",
    response_model=BatchSearchResponse,
    summary="Batch entropy-weighted NMI+Cosine search for multiple queries",
    tags=["Similarity"],
)
def batch_search_similar_vectors(
    payload: BatchPayload,
    _key: str = Security(_require_api_key),
) -> BatchSearchResponse:
    """
    Run similarity search for multiple queries against the same candidate set
    in a single HTTP call. Each query gets its own alpha derived independently
    from its marginal entropy.

    Use when: you have 2-256 queries to evaluate against the same candidate pool.

    Do NOT use when: queries have heterogeneous dimensions or candidate sets --
    use /v1/similarity/search per query instead.
    """
    t0 = time.perf_counter()

    queries_arr = np.array(payload.queries, dtype=np.float64)
    candidates_arr = np.array(payload.candidates, dtype=np.float64)

    if not np.all(np.isfinite(queries_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="queries contains non-finite values (NaN or Inf).",
        )
    if not np.all(np.isfinite(candidates_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="candidates contains non-finite values (NaN or Inf).",
        )

    all_results: list[list[SimilarityResult]] = []

    for q_idx, query_vec in enumerate(queries_arr):
        composite, nmi_scores, cosine_scores, alpha, _ = _compute_composite_scores(
            query_vec, candidates_arr, payload.nmi_bins, payload.alpha
        )
        top_k = min(payload.top_k, len(candidates_arr))
        top_indices = np.argpartition(composite, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(composite[top_indices])[::-1]]

        query_results = [
            SimilarityResult(
                index=int(idx),
                composite_score=round(float(composite[idx]), 8),
                nmi_score=round(float(nmi_scores[idx]), 8),
                cosine_score=round(float(cosine_scores[idx]), 8),
                nmi_weight=round(alpha, 8),
            )
            for idx in top_indices
        ]
        all_results.append(query_results)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return BatchSearchResponse(
        results=all_results,
        query_count=len(payload.queries),
        candidate_count=len(payload.candidates),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/v1/similarity/metrics",
    response_model=MetricsResponse,
    summary="Decomposed NMI, cosine, and composite score between two vectors",
    tags=["Similarity"],
)
def decompose_similarity_metrics(
    payload: MetricsPayload,
    _key: str = Security(_require_api_key),
) -> MetricsResponse:
    """
    Return the full score decomposition (NMI, cosine, composite, alpha, per-dim entropies)
    for a query-reference pair. Useful for debugging, calibration, and understanding
    why the composite score takes a specific value.

    Use when: you want to audit or explain a similarity score between exactly two vectors.

    Do NOT use when: you have a pool of candidates -- /v1/similarity/search is more efficient.
    """
    query_arr = np.array(payload.query, dtype=np.float64)
    reference_arr = np.array(payload.reference, dtype=np.float64)

    if not np.all(np.isfinite(query_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query contains non-finite values (NaN or Inf).",
        )
    if not np.all(np.isfinite(reference_arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="reference contains non-finite values (NaN or Inf).",
        )

    composite, nmi_scores, cosine_scores, alpha, marginal_entropies = _compute_composite_scores(
        query_arr,
        reference_arr.reshape(1, -1),
        payload.nmi_bins,
        payload.alpha,
    )

    return MetricsResponse(
        query=payload.query,
        reference=payload.reference,
        nmi_score=round(float(nmi_scores[0]), 8),
        cosine_score=round(float(cosine_scores[0]), 8),
        composite_score=round(float(composite[0]), 8),
        nmi_weight=round(alpha, 8),
        marginal_entropies=[round(float(e), 8) for e in marginal_entropies],
    )


@app.get(
    "/v1/health",
    summary="Liveness check — confirms API is up and math dependencies are importable",
    tags=["Ops"],
)
def health_check() -> dict:
    """
    Use when: load balancer or orchestrator needs a liveness probe.

    Do NOT use when: you need to verify auth or score correctness -- this endpoint
    does not require an API key and performs no score computation.
    """
    try:
        _test = normalized_mutual_info_score([0, 1, 0], [0, 1, 1])
        _test2 = float(scipy_entropy([0.5, 0.5], base=2))
        deps_ok = np.isfinite(_test) and np.isfinite(_test2)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Math dependency check failed: {exc}",
        )

    return {
        "status": "ok",
        "deps_ok": deps_ok,
        "numpy_version": np.__version__,
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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_vectors_by_nmi_cosine_fusion', description='Ranks a candidate set of vectors against a query vector using a composite score that fuses Normalized Mutual Information (for nonlinear feature dependencies) and cosine similarity (for directional alignment), weighted by the marginal entropy of each dimension computed from the candidate distribution. Use when you need stateless ad-hoc similarity ranking without a database, especially when the feature space contains categorical-encoded or mixed-type dimensions where cosine alone underperforms. Do NOT use when you have more than 50,000 candidates per call (latency will exceed SLA) or when all features are already known to be linearly independent and continuous (cosine alone is sufficient and cheaper).')
async def rank_vectors_by_nmi_cosine_fusion(query_vector: Annotated[list[float], Field(..., description='The reference vector to rank against. Must have the same dimensionality as every vector in candidate_matrix. Values must be finite floats; NaN or Inf will be rejected.', min_length=2, max_length=4096)], candidate_matrix: Annotated[list[list[float]], Field(..., description='2-D array of candidate vectors. Each row is one candidate; columns must match the dimensionality of query_vector. Min 1 row, max 50,000 rows.', min_length=1, max_length=50000)], top_k: Annotated[float, Field(10, description='Number of top-ranked candidates to return. Must be between 1 and the total number of candidates. Defaults to 10.', ge=1, le=50000)], nmi_bins: Annotated[float, Field(None, description='Number of histogram bins used to discretize each continuous dimension for NMI estimation. Higher values increase fidelity for smooth distributions but raise compute cost quadratically per dimension. Valid range: 2-64. If omitted, set automatically via Sturges rule applied to the candidate count.', ge=2, le=64)], fusion_mode: Annotated[str, Field('entropy_weighted', description="Controls how NMI and cosine scores are combined. 'entropy_weighted' (default): each dimension's contribution to NMI vs cosine is weighted by its marginal entropy, making the balance data-driven. 'equal': unweighted arithmetic mean of NMI and cosine scores. Use 'equal' only for ablation studies or when you want to decouple the adaptive weighting.", min_length=5, max_length=16)]) -> dict[str, Any]:
    """NMI+Cosine Fusion Ranking"""
    params = {"query_vector": query_vector, "candidate_matrix": candidate_matrix, "top_k": top_k, "nmi_bins": nmi_bins, "fusion_mode": fusion_mode}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/rank', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix', description='Computes the full N x N composite NMI+Cosine similarity matrix for a set of vectors. Each cell (i, j) is the fusion score between vector i and vector j, with entropy-based dimensional weighting derived from the full set distribution. Use when you need an all-pairs similarity structure for clustering, graph construction, or manifold analysis. Do NOT use for query-vs-candidates ranking (use rank_vectors_by_nmi_cosine_fusion instead) or for sets larger than 2,000 vectors (O(N^2 * D) cost becomes prohibitive).')
async def compute_pairwise_nmi_cosine_matrix(vector_matrix: Annotated[list[list[float]], Field(..., description='2-D array where each row is a vector. All rows must share the same dimensionality. Min 2 rows, max 2,000 rows.', min_length=2, max_length=2000)], nmi_bins: Annotated[float, Field(None, description='Histogram bins for NMI discretization. Valid range: 2-64. Auto-set via Sturges rule if omitted.', ge=2, le=64)], diagonal_value: Annotated[float, Field(1.0, description='Value to place on the matrix diagonal (self-similarity). Typically 1.0 for normalized scores. Set to 0.0 if the matrix will be used as a distance or graph-edge weight where self-loops should be ignored.', ge=0.0, le=1.0)]) -> dict[str, Any]:
    """Pairwise Fusion Score Matrix"""
    params = {"vector_matrix": vector_matrix, "nmi_bins": nmi_bins, "diagonal_value": diagonal_value}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pairwise-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_decompose_fusion_score_by_dimension', description='Returns the per-dimension breakdown of the NMI+Cosine fusion score between a query vector and a single target vector: the marginal entropy weight, raw NMI contribution, raw cosine contribution, and weighted fusion contribution for each dimension. Use when you need to explain why two vectors scored as they did, identify which dimensions drive similarity, or audit the entropy-weighting behavior on a specific pair. Do NOT use for bulk ranking or matrix computation — it is designed for single-pair interpretability, not throughput.')
async def decompose_fusion_score_by_dimension(query_vector: Annotated[list[float], Field(..., description='Reference vector. Must have the same length as target_vector. Min 2 elements, max 4096 elements.', min_length=2, max_length=4096)], target_vector: Annotated[list[float], Field(..., description='Target vector to compare against the query. Must match query_vector dimensionality exactly.', min_length=2, max_length=4096)], reference_distribution: Annotated[list[list[float]], Field(None, description='Optional background matrix used to estimate marginal entropy weights per dimension. If omitted, entropy is estimated from query_vector and target_vector alone (two-sample estimate, lower fidelity). Providing the same candidate_matrix used in the ranking call ensures entropy weights are consistent with the ranking context. Max 50,000 rows.', min_length=2, max_length=50000)], nmi_bins: Annotated[float, Field(None, description='Histogram bins for NMI discretization. Valid range: 2-64.', ge=2, le=64)]) -> dict[str, Any]:
    """Per-Dimension Score Decomposition"""
    params = {"query_vector": query_vector, "target_vector": target_vector, "reference_distribution": reference_distribution, "nmi_bins": nmi_bins}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/decompose', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_dimensional_entropy_profile', description='Estimates the marginal Shannon entropy of each dimension across a vector matrix and returns the normalized entropy weights that the fusion scorer will apply internally. Use this as a diagnostic before running bulk ranking or pairwise-matrix calls, to understand which dimensions dominate the fusion score and whether the input distribution has degenerate (zero-entropy) dimensions that should be dropped. Do NOT use as a substitute for the full fusion score — it returns weights only, not similarity scores.')
async def estimate_dimensional_entropy_profile(vector_matrix: Annotated[list[list[float]], Field(..., description='2-D array of vectors whose dimensional entropy profile you want to inspect. Min 2 rows, max 50,000 rows. Min 2 columns, max 4096 columns.', min_length=2, max_length=50000)], nmi_bins: Annotated[float, Field(None, description='Histogram bins for entropy discretization. Valid range: 2-64. Auto-set via Sturges rule if omitted.', ge=2, le=64)], flag_degenerate_threshold: Annotated[float, Field(0.01, description='Normalized entropy value below which a dimension is flagged as degenerate (near-constant, contributing near-zero information to the fusion score). Range: 0.0-0.5. Default 0.01 means dimensions with less than 1% of max possible entropy are flagged.', ge=0.0, le=0.5)]) -> dict[str, Any]:
    """Entropy Profile Estimator"""
    params = {"vector_matrix": vector_matrix, "nmi_bins": nmi_bins, "flag_degenerate_threshold": flag_degenerate_threshold}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/entropy-profile', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_threshold_filter_by_fusion_score', description='Returns only the candidates whose NMI+Cosine fusion score against the query vector meets or exceeds a specified threshold, without returning a ranked list. Use when the downstream system needs a binary in/out decision (e.g., deduplication, membership testing, anomaly gating) and the score threshold is already calibrated. Do NOT use when you need ranked ordering or when the threshold is unknown and you want to explore score distributions — use rank_vectors_by_nmi_cosine_fusion with top_k instead.')
async def threshold_filter_by_fusion_score(query_vector: Annotated[list[float], Field(..., description='Reference vector. Must match the dimensionality of every row in candidate_matrix. Min 2 elements, max 4096 elements.', min_length=2, max_length=4096)], candidate_matrix: Annotated[list[list[float]], Field(..., description='2-D array of candidate vectors to filter. Each row is one candidate. Min 1 row, max 50,000 rows.', min_length=1, max_length=50000)], min_fusion_score: Annotated[float, Field(..., description='Minimum composite NMI+Cosine fusion score (inclusive) for a candidate to be included in the response. Must be in [0.0, 1.0]. A value of 0.8 is a strong similarity threshold; 0.5 is permissive. Choosing a value without prior calibration via rank_vectors_by_nmi_cosine_fusion is discouraged.', ge=0.0, le=1.0)], nmi_bins: Annotated[float, Field(None, description='Histogram bins for NMI discretization. Valid range: 2-64. Should match the value used during threshold calibration to ensure score comparability.', ge=2, le=64)], fusion_mode: Annotated[str, Field('entropy_weighted', description="Score fusion mode: 'entropy_weighted' (default, data-driven weighting) or 'equal' (unweighted mean). Must match the fusion_mode used during threshold calibration.", min_length=5, max_length=16)]) -> dict[str, Any]:
    """Fusion Score Threshold Filter"""
    params = {"query_vector": query_vector, "candidate_matrix": candidate_matrix, "min_fusion_score": min_fusion_score, "nmi_bins": nmi_bins, "fusion_mode": fusion_mode}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/filter', params)


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
