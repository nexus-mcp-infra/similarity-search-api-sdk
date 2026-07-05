from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import numpy as np
from scipy.stats import chi2_contingency
from scipy.special import entr
import os
import time
import hashlib
import hmac

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = set(filter(None, os.environ.get("SIMILARITY_API_KEYS", "dev-key-local").split(",")))

app = FastAPI(
    title="NMI-Cosine Similarity API",
    description="Hybrid NMI-cosine scoring with per-pair bootstrap p-values. Stateless — no vector persistence.",
    version="1.0.0",
)


class EmbeddingPair(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=4096, description="Query embedding vector.")
    candidates: list[list[float]] = Field(..., min_length=1, max_length=256, description="Candidate embedding vectors to rank.")
    bootstrap_samples: int = Field(default=499, ge=99, le=1999, description="Bootstrap iterations for p-value estimation. Higher = more precise, slower.")
    alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="Significance threshold for the returned confidence intervals.")
    top_k: Optional[int] = Field(default=None, ge=1, le=256, description="Return only top-k results. None returns all candidates.")

    @model_validator(mode="after")
    def validate_uniform_dimensionality(self) -> "EmbeddingPair":
        dim = len(self.query)
        for i, c in enumerate(self.candidates):
            if len(c) != dim:
                raise ValueError(f"Candidate {i} has dimension {len(c)}, expected {dim} (same as query).")
        return self


class SimilarityResult(BaseModel):
    rank: int
    candidate_index: int
    cosine_similarity: float
    nmi_score: float
    hybrid_score: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool


class SimilarityResponse(BaseModel):
    results: list[SimilarityResult]
    query_dim: int
    n_candidates_evaluated: int
    bootstrap_samples_used: int
    latency_ms: float


class BatchRequest(BaseModel):
    queries: list[EmbeddingPair] = Field(..., min_length=1, max_length=16)


class BatchResponse(BaseModel):
    responses: list[SimilarityResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


def _require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return api_key


def _freedman_diaconis_bins(data: np.ndarray) -> int:
    n = len(data)
    if n < 4:
        return 2
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr < 1e-12:
        return 2
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    data_range = data.max() - data.min()
    if bin_width < 1e-12 or data_range < 1e-12:
        return 2
    n_bins = int(np.ceil(data_range / bin_width))
    return max(2, min(n_bins, max(2, int(np.sqrt(n)))))


def _discretize_pair(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.stack([v1, v2], axis=0)
    d1_bins = np.zeros(len(v1), dtype=np.int32)
    d2_bins = np.zeros(len(v2), dtype=np.int32)
    n_bins_per_dim = []
    for dim_idx in range(len(v1)):
        col = combined[:, dim_idx]
        n_bins = _freedman_diaconis_bins(col)
        edges = np.linspace(col.min() - 1e-12, col.max() + 1e-12, n_bins + 1)
        d1_bins[dim_idx] = np.searchsorted(edges[1:], v1[dim_idx], side="left")
        d2_bins[dim_idx] = np.searchsorted(edges[1:], v2[dim_idx], side="left")
        n_bins_per_dim.append(n_bins)
    return d1_bins, d2_bins


def _joint_entropy_discrete(x_bins: np.ndarray, y_bins: np.ndarray, n_states_x: int, n_states_y: int) -> float:
    joint_counts = np.zeros((n_states_x, n_states_y), dtype=np.float64)
    for xi, yi in zip(x_bins, y_bins):
        xi = min(xi, n_states_x - 1)
        yi = min(yi, n_states_y - 1)
        joint_counts[xi, yi] += 1.0
    total = joint_counts.sum()
    if total < 1e-12:
        return 0.0
    p_joint = joint_counts / total
    return float(np.sum(entr(p_joint)))


def _marginal_entropy(bins: np.ndarray, n_states: int) -> float:
    counts = np.bincount(np.clip(bins, 0, n_states - 1), minlength=n_states).astype(np.float64)
    total = counts.sum()
    if total < 1e-12:
        return 0.0
    p = counts / total
    return float(np.sum(entr(p)))


def _compute_nmi(v1: np.ndarray, v2: np.ndarray) -> float:
    d1, d2 = _discretize_pair(v1, v2)
    n_states = max(d1.max(), d2.max()) + 1
    h_x = _marginal_entropy(d1, n_states)
    h_y = _marginal_entropy(d2, n_states)
    h_xy = _joint_entropy_discrete(d1, d2, n_states, n_states)
    denom = (h_x + h_y)
    if denom < 1e-12:
        return 0.0
    mi = max(0.0, h_x + h_y - h_xy)
    nmi = mi / (0.5 * denom)
    return float(np.clip(nmi, 0.0, 1.0))


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _bootstrap_nmi_pvalue(
    v1: np.ndarray,
    v2: np.ndarray,
    observed_nmi: float,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    null_nmis = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        v2_permuted = rng.permutation(v2)
        null_nmis[b] = _compute_nmi(v1, v2_permuted)
    p_value = float(np.mean(null_nmis >= observed_nmi))
    half_alpha = alpha / 2.0
    ci_lower = float(np.percentile(null_nmis, 100.0 * half_alpha))
    ci_upper = float(np.percentile(null_nmis, 100.0 * (1.0 - half_alpha)))
    return p_value, ci_lower, ci_upper


def _run_similarity_search(payload: EmbeddingPair) -> SimilarityResponse:
    t0 = time.perf_counter()
    query = np.array(payload.query, dtype=np.float64)
    candidates = [np.array(c, dtype=np.float64) for c in payload.candidates]
    rng = np.random.default_rng(seed=int(hashlib.md5(query.tobytes()).hexdigest()[:8], 16))

    results: list[SimilarityResult] = []
    for idx, cand in enumerate(candidates):
        cosine = _cosine_similarity(query, cand)
        nmi = _compute_nmi(query, cand)
        hybrid = 0.5 * cosine + 0.5 * nmi
        p_value, ci_lower, ci_upper = _bootstrap_nmi_pvalue(
            query, cand, nmi, payload.bootstrap_samples, payload.alpha, rng
        )
        results.append(
            SimilarityResult(
                rank=0,
                candidate_index=idx,
                cosine_similarity=round(cosine, 6),
                nmi_score=round(nmi, 6),
                hybrid_score=round(hybrid, 6),
                p_value=round(p_value, 6),
                ci_lower=round(ci_lower, 6),
                ci_upper=round(ci_upper, 6),
                significant=p_value < payload.alpha,
            )
        )

    results.sort(key=lambda r: r.hybrid_score, reverse=True)
    top_k = payload.top_k if payload.top_k is not None else len(results)
    results = results[:top_k]
    for rank, r in enumerate(results, start=1):
        r.rank = rank

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return SimilarityResponse(
        results=results,
        query_dim=len(query),
        n_candidates_evaluated=len(candidates),
        bootstrap_samples_used=payload.bootstrap_samples,
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/search", response_model=SimilarityResponse, tags=["core"])
def nmi_cosine_search(
    payload: EmbeddingPair,
    _api_key: str = Depends(_require_api_key),
) -> SimilarityResponse:
    if not payload.query:
        raise HTTPException(status_code=422, detail="query vector must not be empty.")
    if not payload.candidates:
        raise HTTPException(status_code=422, detail="candidates list must not be empty.")
    return _run_similarity_search(payload)


@app.post("/batch", response_model=BatchResponse, tags=["core"])
def nmi_cosine_batch_search(
    batch: BatchRequest,
    _api_key: str = Depends(_require_api_key),
) -> BatchResponse:
    t0 = time.perf_counter()
    responses = [_run_similarity_search(q) for q in batch.queries]
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return BatchResponse(responses=responses, total_latency_ms=total_ms)


@app.post("/score", response_model=SimilarityResult, tags=["core"])
def score_single_pair(
    payload: EmbeddingPair,
    _api_key: str = Depends(_require_api_key),
) -> SimilarityResult:
    if len(payload.candidates) != 1:
        raise HTTPException(
            status_code=422,
            detail="/score accepts exactly 1 candidate. Use /search for multiple candidates.",
        )
    response = _run_similarity_search(payload)
    return response.results[0]