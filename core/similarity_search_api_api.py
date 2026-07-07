from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cosine as cosine_distance
import os
import time
import hashlib

app = FastAPI(
    title="NMI-Cosine Similarity Search API",
    version="1.0.0",
    description="Stateless semantic similarity with adaptive NMI-cosine fusion calibrated by corpus entropy",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_API_KEYS = set(filter(None, os.environ.get("API_KEYS", "").split(",")))
_BASELINE_ENTROPY = float(os.environ.get("BASELINE_ENTROPY", "1.0"))


def _require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    if not _VALID_API_KEYS:
        return "open"
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key not in _VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class VectorSearchRequest(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=8192, description="Query vector")
    corpus: list[list[float]] = Field(..., min_length=1, max_length=2048, description="Corpus vectors to rank")
    corpus_ids: Optional[list[str]] = Field(None, description="Optional IDs aligned with corpus")
    top_k: int = Field(10, ge=1, le=500)
    nmi_bins: int = Field(20, ge=5, le=100, description="Histogram bins for NMI estimation")

    @field_validator("corpus")
    @classmethod
    def corpus_vectors_uniform_dimension(cls, v):
        if not v:
            raise ValueError("corpus must not be empty")
        dim0 = len(v[0])
        if dim0 < 2:
            raise ValueError("each corpus vector must have dimension >= 2")
        if dim0 > 8192:
            raise ValueError("vector dimension must be <= 8192")
        for i, vec in enumerate(v):
            if len(vec) != dim0:
                raise ValueError(f"corpus[{i}] has dimension {len(vec)}, expected {dim0}")
        return v

    @field_validator("query")
    @classmethod
    def query_not_zero(cls, v):
        arr = np.array(v, dtype=np.float64)
        if np.allclose(arr, 0.0):
            raise ValueError("query vector must not be the zero vector")
        return v

    @field_validator("corpus_ids")
    @classmethod
    def ids_match_corpus_length(cls, v, info):
        if v is not None and "corpus" in info.data:
            if len(v) != len(info.data["corpus"]):
                raise ValueError("corpus_ids length must match corpus length")
        return v


class SimilarityResult(BaseModel):
    id: str
    rank: int
    nmi_cosine_score: float
    cosine_similarity: float
    nmi_score: float
    adaptive_weight_nmi: float


class VectorSearchResponse(BaseModel):
    results: list[SimilarityResult]
    corpus_entropy: float
    adaptive_weight_nmi: float
    baseline_entropy: float
    latency_ms: float
    request_hash: str


class BatchFusionRequest(BaseModel):
    queries: list[list[float]] = Field(..., min_length=1, max_length=128)
    corpus: list[list[float]] = Field(..., min_length=1, max_length=2048)
    corpus_ids: Optional[list[str]] = None
    top_k: int = Field(10, ge=1, le=500)
    nmi_bins: int = Field(20, ge=5, le=100)

    @field_validator("queries")
    @classmethod
    def queries_have_valid_dim(cls, v):
        if not v:
            raise ValueError("queries must not be empty")
        dim0 = len(v[0])
        if dim0 < 2 or dim0 > 8192:
            raise ValueError("query dimension must be between 2 and 8192")
        for i, q in enumerate(v):
            if len(q) != dim0:
                raise ValueError(f"queries[{i}] dimension mismatch")
            if np.allclose(np.array(q, dtype=np.float64), 0.0):
                raise ValueError(f"queries[{i}] is a zero vector")
        return v

    @field_validator("corpus")
    @classmethod
    def corpus_valid(cls, v):
        if not v:
            raise ValueError("corpus must not be empty")
        dim0 = len(v[0])
        if dim0 < 2 or dim0 > 8192:
            raise ValueError("vector dimension must be between 2 and 8192")
        for i, vec in enumerate(v):
            if len(vec) != dim0:
                raise ValueError(f"corpus[{i}] dimension mismatch")
        return v


class BatchFusionResponse(BaseModel):
    query_results: list[list[SimilarityResult]]
    corpus_entropy: float
    adaptive_weight_nmi: float
    latency_ms: float


class EntropyDiagnosticsRequest(BaseModel):
    corpus: list[list[float]] = Field(..., min_length=2, max_length=2048)
    nmi_bins: int = Field(20, ge=5, le=100)

    @field_validator("corpus")
    @classmethod
    def corpus_has_min_size(cls, v):
        if len(v) < 2:
            raise ValueError("corpus must have at least 2 vectors for entropy estimation")
        dim0 = len(v[0])
        if dim0 < 2 or dim0 > 8192:
            raise ValueError("vector dimension must be between 2 and 8192")
        for i, vec in enumerate(v):
            if len(vec) != dim0:
                raise ValueError(f"corpus[{i}] dimension mismatch")
        return v


class EntropyDiagnosticsResponse(BaseModel):
    corpus_entropy: float
    per_dimension_entropy: list[float]
    effective_dimensions: int
    distribution_skew: float
    recommended_nmi_bins: int
    adaptive_weight_nmi: float
    baseline_entropy: float


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vec / norm


def _cosine_similarity_batch(query_norm: np.ndarray, corpus_norm: np.ndarray) -> np.ndarray:
    return np.clip(corpus_norm @ query_norm, -1.0, 1.0)


def _marginal_entropy_from_matrix(matrix: np.ndarray, n_bins: int) -> float:
    n_samples, n_dims = matrix.shape
    dim_sample = min(n_dims, 64)
    rng = np.random.default_rng(seed=42)
    sampled_dims = rng.choice(n_dims, size=dim_sample, replace=False) if n_dims > 64 else np.arange(n_dims)

    total_entropy = 0.0
    for d in sampled_dims:
        col = matrix[:, d]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            continue
        counts, _ = np.histogram(col, bins=n_bins, range=(col_min, col_max))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        total_entropy += float(-np.sum(probs * np.log2(probs + 1e-12)))

    avg_entropy = total_entropy / len(sampled_dims) if len(sampled_dims) > 0 else 0.0
    return avg_entropy


def _normalized_mutual_information_batch(
    query: np.ndarray,
    corpus: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    n_samples, n_dims = corpus.shape
    dim_sample = min(n_dims, 64)
    rng = np.random.default_rng(seed=0)
    sampled_dims = rng.choice(n_dims, size=dim_sample, replace=False) if n_dims > 64 else np.arange(n_dims)

    nmi_scores = np.zeros(n_samples, dtype=np.float64)

    for d in sampled_dims:
        q_val = float(query[d])
        c_col = corpus[:, d]

        combined_min = min(q_val, float(c_col.min()))
        combined_max = max(q_val, float(c_col.max()))
        if combined_max - combined_min < 1e-12:
            continue

        bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

        q_bin = int(np.digitize(q_val, bin_edges[1:-1]))
        q_bin = min(q_bin, n_bins - 1)

        c_bins = np.digitize(c_col, bin_edges[1:-1])
        c_bins = np.clip(c_bins, 0, n_bins - 1)

        c_counts = np.bincount(c_bins, minlength=n_bins).astype(np.float64)
        c_probs = c_counts / c_counts.sum()
        H_c = float(-np.sum(c_probs[c_probs > 0] * np.log2(c_probs[c_probs > 0] + 1e-12)))

        q_probs = np.zeros(n_bins)
        q_probs[q_bin] = 1.0
        H_q = 0.0

        joint_counts = np.zeros(n_bins, dtype=np.float64)
        joint_counts[q_bin] += 1.0
        for i in range(n_samples):
            if c_bins[i] == q_bin:
                joint_counts[q_bin] += 1.0

        for i in range(n_samples):
            cb = c_bins[i]
            joint_p = 1.0 / n_samples if cb == q_bin else 0.0
            p_q = 1.0
            p_c = c_probs[cb] if c_probs[cb] > 0 else 1e-12
            if joint_p > 0 and p_q > 0 and p_c > 0:
                nmi_scores[i] += joint_p * np.log2(joint_p / (p_q * p_c) + 1e-12)

    denom_per_dim = dim_sample if dim_sample > 0 else 1
    nmi_scores = nmi_scores / denom_per_dim

    denom_entropy = (H_q + H_c) / 2.0 if (H_q + H_c) > 1e-12 else 1.0
    nmi_normalized = np.clip(nmi_scores / denom_entropy, 0.0, 1.0)

    return nmi_normalized


def _compute_adaptive_weight_nmi(corpus_entropy: float, baseline_entropy: float) -> float:
    if baseline_entropy < 1e-12:
        return 0.5
    w_nmi = corpus_entropy / (corpus_entropy + baseline_entropy)
    return float(np.clip(w_nmi, 0.05, 0.95))


def _fused_nmi_cosine_ranking(
    query: np.ndarray,
    corpus: np.ndarray,
    corpus_ids: list[str],
    top_k: int,
    nmi_bins: int,
    baseline_entropy: float,
) -> tuple[list[SimilarityResult], float, float]:
    if query.shape[0] != corpus.shape[1]:
        raise ValueError(
            f"query dimension {query.shape[0]} does not match corpus dimension {corpus.shape[1]}"
        )

    query_norm = _l2_normalize(query)
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms < 1e-12, 1e-12, corpus_norms)
    corpus_normalized = corpus / corpus_norms

    cosine_scores = _cosine_similarity_batch(query_norm, corpus_normalized)
    corpus_entropy = _marginal_entropy_from_matrix(corpus_normalized, nmi_bins)
    w_nmi = _compute_adaptive_weight_nmi(corpus_entropy, baseline_entropy)
    w_cosine = 1.0 - w_nmi

    nmi_scores = _normalized_mutual_information_batch(query_norm, corpus_normalized, nmi_bins)

    fused_scores = w_cosine * ((cosine_scores + 1.0) / 2.0) + w_nmi * nmi_scores

    effective_top_k = min(top_k, len(corpus))
    top_indices = np.argpartition(fused_scores, -effective_top_k)[-effective_top_k:]
    top_indices = top_indices[np.argsort(fused_scores[top_indices])[::-1]]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(
            SimilarityResult(
                id=corpus_ids[int(idx)],
                rank=rank,
                nmi_cosine_score=float(round(fused_scores[int(idx)], 6)),
                cosine_similarity=float(round(cosine_scores[int(idx)], 6)),
                nmi_score=float(round(nmi_scores[int(idx)], 6)),
                adaptive_weight_nmi=float(round(w_nmi, 6)),
            )
        )

    return results, corpus_entropy, w_nmi


@app.post("/v1/search", response_model=VectorSearchResponse)
def nmi_cosine_vector_search(
    req: VectorSearchRequest,
    _key: str = Depends(_require_api_key),
) -> VectorSearchResponse:
    t0 = time.perf_counter()

    query_arr = np.array(req.query, dtype=np.float64)
    corpus_arr = np.array(req.corpus, dtype=np.float64)

    if query_arr.shape[0] != corpus_arr.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=f"query dimension {query_arr.shape[0]} != corpus dimension {corpus_arr.shape[1]}",
        )

    ids = req.corpus_ids if req.corpus_ids else [str(i) for i in range(len(req.corpus))]

    try:
        results, corpus_entropy, w_nmi = _fused_nmi_cosine_ranking(
            query=query_arr,
            corpus=corpus_arr,
            corpus_ids=ids,
            top_k=req.top_k,
            nmi_bins=req.nmi_bins,
            baseline_entropy=_BASELINE_ENTROPY,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    payload_bytes = (str(req.query) + str(len(req.corpus))).encode()
    request_hash = hashlib.sha256(payload_bytes).hexdigest()[:16]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return VectorSearchResponse(
        results=results,
        corpus_entropy=float(round(corpus_entropy, 6)),
        adaptive_weight_nmi=float(round(w_nmi, 6)),
        baseline_entropy=_BASELINE_ENTROPY,
        latency_ms=float(round(latency_ms, 3)),
        request_hash=request_hash,
    )


@app.post("/v1/search/batch", response_model=BatchFusionResponse)
def nmi_cosine_batch_search(
    req: BatchFusionRequest,
    _key: str = Depends(_require_api_key),
) -> BatchFusionResponse:
    t0 = time.perf_counter()

    corpus_arr = np.array(req.corpus, dtype=np.float64)
    ids = req.corpus_ids if req.corpus_ids else [str(i) for i in range(len(req.corpus))]

    if req.corpus_ids and len(req.corpus_ids) != len(req.corpus):
        raise HTTPException(status_code=422, detail="corpus_ids length must match corpus length")

    query_dim = len(req.queries[0])
    if query_dim != corpus_arr.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=f"query dimension {query_dim} != corpus dimension {corpus_arr.shape[1]}",
        )

    corpus_norms = np.linalg.norm(corpus_arr, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms < 1e-12, 1e-12, corpus_norms)
    corpus_normalized = corpus_arr / corpus_norms

    corpus_entropy = _marginal_entropy_from_matrix(corpus_normalized, req.nmi_bins)
    w_nmi = _compute_adaptive_weight_nmi(corpus_entropy, _BASELINE_ENTROPY)

    all_query_results: list[list[SimilarityResult]] = []

    for q_raw in req.queries:
        query_arr = np.array(q_raw, dtype=np.float64)
        try:
            results, _, _ = _fused_nmi_cosine_ranking(
                query=query_arr,
                corpus=corpus_arr,
                corpus_ids=ids,
                top_k=req.top_k,
                nmi_bins=req.nmi_bins,
                baseline_entropy=_BASELINE_ENTROPY,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        all_query_results.append(results)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return BatchFusionResponse(
        query_results=all_query_results,
        corpus_entropy=float(round(corpus_entropy, 6)),
        adaptive_weight_nmi=float(round(w_nmi, 6)),
        latency_ms=float(round(latency_ms, 3)),
    )


@app.post("/v1/corpus/entropy", response_model=EntropyDiagnosticsResponse)
def corpus_entropy_diagnostics(
    req: EntropyDiagnosticsRequest,
    _key: str = Depends(_require_api_key),
) -> EntropyDiagnosticsResponse:
    corpus_arr = np.array(req.corpus, dtype=np.float64)
    n_samples, n_dims = corpus_arr.shape

    corpus_norms = np.linalg.norm(corpus_arr, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms < 1e-12, 1e-12, corpus_norms)
    corpus_normalized = corpus_arr / corpus_norms

    per_dim_entropy: list[float] = []
    for d in range(n_dims):
        col = corpus_normalized[:, d]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            per_dim_entropy.append(0.0)
            continue
        counts, _ = np.histogram(col, bins=req.nmi_bins, range=(col_min, col_max))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log2(probs + 1e-12)))
        per_dim_entropy.append(h)

    dim_entropies = np.array(per_dim_entropy)
    corpus_entropy = float(np.mean(dim_entropies))

    max_entropy = np.log2(req.nmi_bins)
    effective_dims = int(np.sum(dim_entropies > 0.1 * max_entropy))

    flat = corpus_normalized.flatten()
    skewness = float(
        np.mean(((flat - flat.mean()) / (flat.std() + 1e-12)) ** 3)
    )

    recommended_bins = max(5, min(100, int(np.ceil(np.sqrt(n_samples)))))

    w_nmi = _compute_adaptive_weight_nmi(corpus_entropy, _BASELINE_ENTROPY)

    return EntropyDiagnosticsResponse(
        corpus_entropy=float(round(corpus_entropy, 6)),
        per_dimension_entropy=[float(round(h, 6)) for h in per_dim_entropy],
        effective_dimensions=effective_dims,
        distribution_skew=float(round(skewness, 6)),
        recommended_nmi_bins=recommended_bins,
        adaptive_weight_nmi=float(round(w_nmi, 6)),
        baseline_entropy=_BASELINE_ENTROPY,
    )


@app.get("/v1/health")
def readiness_probe() -> dict:
    try:
        _probe = np.dot(np.ones(4), np.ones(4))
        status = "ok"
    except Exception:
        status = "degraded"
    return {
        "status": status,
        "baseline_entropy": _BASELINE_ENTROPY,
        "api_keys_configured": len(_VALID_API_KEYS) > 0,
        "version": "1.0.0",
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


@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_vectors_by_nmi_cosine_fusion', description='Ranks a corpus of float vectors against a query vector using entropy-calibrated NMI-cosine weighted fusion. Use when the corpus distribution is unknown or potentially skewed and cosine similarity alone would produce unreliable rankings. Do NOT use when all vectors are already L2-normalized and the corpus is guaranteed uniform — plain cosine suffices and will be faster. Do NOT use with fewer than 3 corpus vectors, as entropy calibration is undefined on trivially small sets.')
async def rank_vectors_by_nmi_cosine_fusion(query_vector: Annotated[list[float], Field(..., description='Dense float vector representing the query. Must have the same dimensionality as every vector in corpus_vectors.', min_length=2, max_length=16384)], corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense float vectors to rank. Each inner array must match query_vector dimensionality. Order is preserved in the response for index mapping.', min_length=3, max_length=10000)], nmi_weight: Annotated[float, Field(None, description='Manual override for the NMI component weight in [0.0, 1.0]. When omitted, weight is derived automatically from corpus entropy. Set to 0.0 to degrade to pure cosine; set to 1.0 to use NMI only. Omit unless you have domain-specific reasons to bypass entropy calibration.', ge=0.0, le=1.0)], top_k: Annotated[float, Field(None, description='Number of top-ranked results to return. Must be between 1 and the length of corpus_vectors. Defaults to returning all corpus items ranked.', ge=1, le=10000)]) -> dict[str, Any]:
    """NMI-Cosine Fusion Ranking"""
    params = {"query_vector": query_vector, "corpus_vectors": corpus_vectors, "nmi_weight": nmi_weight, "top_k": top_k}
    return await _nexus_mcp_call_core('POST', '/v1/rank/vectors/nmi-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_rank_texts_by_nmi_cosine_fusion', description='Encodes a query string and a corpus of strings into embeddings internally, then ranks by entropy-calibrated NMI-cosine fusion. Use when the caller does not hold precomputed vectors and wants a fully stateless text-in / scores-out call. Do NOT use when you already have float vectors — use rank_vectors_by_nmi_cosine_fusion instead to avoid paying embedding latency twice. Do NOT use for corpora with highly heterogeneous languages in the same call; cross-lingual NMI calibration degrades without per-language entropy partitioning.')
async def rank_texts_by_nmi_cosine_fusion(query_text: Annotated[str, Field(..., description='Natural language query string to encode and match against the corpus.', min_length=1, max_length=2048)], corpus_texts: Annotated[list[str], Field(..., description='List of natural language strings to encode and rank. Minimum 3 required for valid entropy calibration.', min_length=3, max_length=2000)], nmi_weight: Annotated[float, Field(None, description='Manual override for the NMI component weight in [0.0, 1.0]. Omit to use entropy-derived adaptive weight.', ge=0.0, le=1.0)], top_k: Annotated[float, Field(None, description='Number of top results to return. Defaults to all corpus items ranked.', ge=1, le=2000)]) -> dict[str, Any]:
    """Text NMI-Cosine Fusion Ranking"""
    params = {"query_text": query_text, "corpus_texts": corpus_texts, "nmi_weight": nmi_weight, "top_k": top_k}
    return await _nexus_mcp_call_core('POST', '/v1/rank/texts/nmi-cosine', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix', description='Computes the full N x N entropy-calibrated NMI-cosine fusion similarity matrix for a set of vectors. Use for clustering preprocessing, graph construction, or any task requiring all pairwise similarities, not just query-to-corpus. Do NOT use as a substitute for rank_vectors_by_nmi_cosine_fusion when you only need one query ranked — this computes O(N^2) pairs and is substantially more expensive. Do NOT use with N > 1000 without confirming latency budget; complexity is O(N^2 * D).')
async def compute_pairwise_nmi_cosine_matrix(vectors: Annotated[list[list[float]], Field(..., description='List of dense float vectors. All must share the same dimensionality. Minimum 3 for valid entropy calibration; maximum 1000 to stay within per-call latency budget.', min_length=3, max_length=1000)], nmi_weight: Annotated[float, Field(None, description='Manual override for NMI weight in [0.0, 1.0]. Omit to derive from corpus entropy.', ge=0.0, le=1.0)]) -> dict[str, Any]:
    """Pairwise NMI-Cosine Matrix"""
    params = {"vectors": vectors, "nmi_weight": nmi_weight}
    return await _nexus_mcp_call_core('POST', '/v1/similarity/pairwise-matrix', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_explain_nmi_cosine_score', description='Returns a decomposed breakdown of a single NMI-cosine fusion score between one query vector and one target vector, including raw cosine similarity, raw NMI estimate, corpus entropy used for calibration, the derived adaptive weight, and the final fused score. Use for debugging unexpected rankings or auditing why two vectors scored the way they did in a previous rank call. Do NOT use in production ranking loops — it processes exactly one pair and is not vectorized; call rank_vectors_by_nmi_cosine_fusion for batch ranking.')
async def explain_nmi_cosine_score(query_vector: Annotated[list[float], Field(..., description='The query vector from the ranking call being explained.', min_length=2, max_length=16384)], target_vector: Annotated[list[float], Field(..., description='The specific corpus vector whose score is being explained.', min_length=2, max_length=16384)], corpus_entropy: Annotated[float, Field(None, description='The marginal entropy of the corpus used in the original ranking call, returned in every rank response under diagnostics.corpus_entropy. Providing it avoids recomputing entropy and ensures the explanation matches the original score exactly.', ge=0.0)]) -> dict[str, Any]:
    """Score Component Explainer"""
    params = {"query_vector": query_vector, "target_vector": target_vector, "corpus_entropy": corpus_entropy}
    return await _nexus_mcp_call_core('POST', '/v1/explain/score-decomposition', params)

@_nexus_mcp.tool(name='nexus_similarity_search_api_estimate_corpus_entropy_profile', description='Computes the marginal entropy, per-dimension variance profile, and recommended nmi_weight for a corpus of vectors without performing any ranking. Use as a preflight call to understand corpus distribution before committing to a batch ranking job, or to cache corpus_entropy for repeated queries against the same static corpus. Do NOT use if the corpus changes between the entropy call and the rank call — entropy is recalculated per rank request anyway when nmi_weight is omitted, making this redundant if the corpus is volatile.')
async def estimate_corpus_entropy_profile(corpus_vectors: Annotated[list[list[float]], Field(..., description='List of dense float vectors representing the corpus to profile. Minimum 3 for meaningful entropy estimation.', min_length=3, max_length=10000)], n_entropy_bins: Annotated[float, Field(32, description="Number of histogram bins used for marginal entropy estimation via Scott's rule approximation. Valid range 8-256. Lower values reduce sensitivity to high-dimensional sparsity; higher values increase resolution at the cost of variance in small corpora.", ge=8, le=256)]) -> dict[str, Any]:
    """Corpus Entropy Profiler"""
    params = {"corpus_vectors": corpus_vectors, "n_entropy_bins": n_entropy_bins}
    return await _nexus_mcp_call_core('POST', '/v1/corpus/entropy-profile', params)


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
