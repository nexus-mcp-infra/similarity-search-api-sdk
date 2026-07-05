from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.special import digamma
from sklearn.preprocessing import KBinsDiscretizer
import os
import time
import hashlib
import hmac

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEY = os.environ.get("SIMILARITY_API_KEY", "")

app = FastAPI(
    title="NMI-Weighted Cosine Similarity API",
    description="Stateless similarity search combining Normalized Mutual Information feature filtering with Cosine distance. No vector DB required.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)


class VectorPayload(BaseModel):
    query: list[float] = Field(..., min_length=2, max_length=4096, description="Query vector")
    corpus: list[list[float]] = Field(..., min_length=1, max_length=10000, description="Corpus vectors to search against")
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    nmi_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum NMI score to retain a feature dimension. Features below this threshold are masked before cosine computation.")
    n_bins: int = Field(default=10, ge=3, le=50, description="Number of bins for NMI discretization of continuous features")

    @field_validator("corpus")
    @classmethod
    def corpus_vectors_consistent_dim(cls, corpus: list[list[float]]) -> list[list[float]]:
        if not corpus:
            raise ValueError("corpus must contain at least one vector")
        dim = len(corpus[0])
        if dim < 2:
            raise ValueError("corpus vectors must have at least 2 dimensions")
        if dim > 4096:
            raise ValueError("corpus vectors must have at most 4096 dimensions")
        for i, vec in enumerate(corpus):
            if len(vec) != dim:
                raise ValueError(f"corpus vector at index {i} has dimension {len(vec)}, expected {dim}")
        return corpus

    @field_validator("query")
    @classmethod
    def query_not_all_zero(cls, query: list[float]) -> list[float]:
        if all(v == 0.0 for v in query):
            raise ValueError("query vector must not be the zero vector")
        return query


class SimilarityResult(BaseModel):
    rank: int
    corpus_index: int
    cosine_similarity: float
    nmi_weighted_cosine: float
    confidence_lower: float
    confidence_upper: float
    active_features: int
    total_features: int


class SimilarityResponse(BaseModel):
    results: list[SimilarityResult]
    nmi_feature_mask: list[float]
    empirical_nmi_mean: float
    empirical_nmi_std: float
    latency_ms: float
    dimensions_retained: int
    dimensions_total: int


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


def _require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    if not VALID_API_KEY:
        return "no-auth-configured"
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not hmac.compare_digest(api_key, VALID_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def _discretize_features(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    n_samples, n_features = matrix.shape
    discretized = np.empty_like(matrix, dtype=np.int32)
    for f in range(n_features):
        col = matrix[:, f]
        col_range = col.max() - col.min()
        if col_range < 1e-12:
            discretized[:, f] = 0
            continue
        actual_bins = min(n_bins, len(np.unique(col)))
        actual_bins = max(actual_bins, 2)
        kbd = KBinsDiscretizer(n_bins=actual_bins, encode="ordinal", strategy="quantile")
        discretized[:, f] = kbd.fit_transform(col.reshape(-1, 1)).ravel().astype(np.int32)
    return discretized


def _compute_nmi_per_feature(query_vec: np.ndarray, corpus_matrix: np.ndarray, n_bins: int) -> np.ndarray:
    n_corpus, n_features = corpus_matrix.shape
    stacked = np.vstack([query_vec.reshape(1, -1), corpus_matrix])
    disc = _discretize_features(stacked, n_bins)
    query_disc = disc[0]
    corpus_disc = disc[1:]
    nmi_scores = np.zeros(n_features, dtype=np.float64)
    for f in range(n_features):
        qf = query_disc[f]
        cf = corpus_disc[:, f]
        combined = np.vstack([np.full(n_corpus, qf, dtype=np.int32), cf])
        q_vals = np.unique(combined[0])
        c_vals = np.unique(combined[1])
        if len(q_vals) < 2 and len(c_vals) < 2:
            nmi_scores[f] = 0.0
            continue
        joint_counts = np.zeros((len(q_vals), len(c_vals)), dtype=np.float64)
        q_idx_map = {v: i for i, v in enumerate(q_vals)}
        c_idx_map = {v: i for i, v in enumerate(c_vals)}
        for qi, ci in zip(combined[0], combined[1]):
            joint_counts[q_idx_map[qi], c_idx_map[ci]] += 1.0
        joint_prob = joint_counts / joint_counts.sum()
        p_q = joint_prob.sum(axis=1)
        p_c = joint_prob.sum(axis=0)
        h_q = scipy_entropy(p_q + 1e-12)
        h_c = scipy_entropy(p_c + 1e-12)
        h_qc = scipy_entropy(joint_prob.ravel() + 1e-12)
        mi = h_q + h_c - h_qc
        denom = h_q + h_c
        if denom < 1e-12:
            nmi_scores[f] = 0.0
        else:
            nmi_scores[f] = float(np.clip(2.0 * mi / denom, 0.0, 1.0))
    return nmi_scores


def _cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q_masked = query * mask
    c_masked = corpus * mask
    q_norm = np.linalg.norm(q_masked)
    if q_norm < 1e-12:
        return np.zeros(len(corpus), dtype=np.float64)
    c_norms = np.linalg.norm(c_masked, axis=1)
    c_norms = np.where(c_norms < 1e-12, 1e-12, c_norms)
    dots = c_masked @ q_masked
    return dots / (c_norms * q_norm)


def _nmi_weighted_cosine_batch(query: np.ndarray, corpus: np.ndarray, nmi_scores: np.ndarray) -> np.ndarray:
    nmi_sum = nmi_scores.sum()
    if nmi_sum < 1e-12:
        weights = np.ones(len(nmi_scores), dtype=np.float64) / len(nmi_scores)
    else:
        weights = nmi_scores / nmi_sum
    q_weighted = query * weights
    c_weighted = corpus * weights
    q_norm = np.linalg.norm(q_weighted)
    if q_norm < 1e-12:
        return np.zeros(len(corpus), dtype=np.float64)
    c_norms = np.linalg.norm(c_weighted, axis=1)
    c_norms = np.where(c_norms < 1e-12, 1e-12, c_norms)
    dots = c_weighted @ q_weighted
    return dots / (c_norms * q_norm)


def _bootstrap_confidence_interval(scores: np.ndarray, n_bootstrap: int = 500, ci: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed=42)
    alpha = (1.0 - ci) / 2.0
    lower_all = np.empty(n, dtype=np.float64)
    upper_all = np.empty(n, dtype=np.float64)
    boot_samples = rng.choice(scores, size=(n_bootstrap, n), replace=True)
    boot_means = boot_samples.mean(axis=1)
    mean_score = scores.mean()
    deltas = np.sort(boot_means - mean_score)
    lower_delta = deltas[int(np.floor(alpha * n_bootstrap))]
    upper_delta = deltas[int(np.ceil((1.0 - alpha) * n_bootstrap)) - 1]
    lower_all[:] = np.clip(scores - upper_delta, -1.0, 1.0)
    upper_all[:] = np.clip(scores - lower_delta, -1.0, 1.0)
    return lower_all, upper_all


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def check_service_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=time.time(),
    )


@app.post("/search", response_model=SimilarityResponse, tags=["search"])
def nmi_weighted_cosine_search(
    payload: VectorPayload,
    _api_key: str = Depends(_require_api_key),
) -> SimilarityResponse:
    t0 = time.perf_counter()

    query_np = np.array(payload.query, dtype=np.float64)
    corpus_np = np.array(payload.corpus, dtype=np.float64)

    n_corpus, n_features = corpus_np.shape
    if len(query_np) != n_features:
        raise HTTPException(
            status_code=422,
            detail=f"query dimension {len(query_np)} does not match corpus dimension {n_features}",
        )

    nmi_scores = _compute_nmi_per_feature(query_np, corpus_np, payload.n_bins)

    feature_mask = (nmi_scores >= payload.nmi_threshold).astype(np.float64)
    active_count = int(feature_mask.sum())

    if active_count == 0:
        feature_mask = np.ones(n_features, dtype=np.float64)
        active_count = n_features

    cosine_sims = _cosine_similarity_batch(query_np, corpus_np, feature_mask)
    nmi_weighted_sims = _nmi_weighted_cosine_batch(query_np, corpus_np, nmi_scores * feature_mask)

    lower_bounds, upper_bounds = _bootstrap_confidence_interval(nmi_weighted_sims)

    top_k = min(payload.top_k, n_corpus)
    top_indices = np.argpartition(nmi_weighted_sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(nmi_weighted_sims[top_indices])[::-1]]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append(
            SimilarityResult(
                rank=rank + 1,
                corpus_index=int(idx),
                cosine_similarity=float(np.clip(cosine_sims[idx], -1.0, 1.0)),
                nmi_weighted_cosine=float(np.clip(nmi_weighted_sims[idx], -1.0, 1.0)),
                confidence_lower=float(lower_bounds[idx]),
                confidence_upper=float(upper_bounds[idx]),
                active_features=active_count,
                total_features=n_features,
            )
        )

    empirical_nmi_mean = float(nmi_scores.mean())
    empirical_nmi_std = float(nmi_scores.std())
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SimilarityResponse(
        results=results,
        nmi_feature_mask=nmi_scores.tolist(),
        empirical_nmi_mean=empirical_nmi_mean,
        empirical_nmi_std=empirical_nmi_std,
        latency_ms=round(latency_ms, 3),
        dimensions_retained=active_count,
        dimensions_total=n_features,
    )


@app.post("/explain", response_model=dict, tags=["search"])
def explain_nmi_feature_contributions(
    payload: VectorPayload,
    _api_key: str = Depends(_require_api_key),
) -> dict:
    t0 = time.perf_counter()

    query_np = np.array(payload.query, dtype=np.float64)
    corpus_np = np.array(payload.corpus, dtype=np.float64)
    n_corpus, n_features = corpus_np.shape

    if len(query_np) != n_features:
        raise HTTPException(
            status_code=422,
            detail=f"query dimension {len(query_np)} does not match corpus dimension {n_features}",
        )

    nmi_scores = _compute_nmi_per_feature(query_np, corpus_np, payload.n_bins)
    feature_mask = (nmi_scores >= payload.nmi_threshold).astype(np.float64)

    sorted_feature_indices = np.argsort(nmi_scores)[::-1].tolist()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "feature_nmi_scores": [
            {
                "feature_index": int(fi),
                "nmi_score": float(round(nmi_scores[fi], 6)),
                "retained": bool(feature_mask[fi] > 0.5),
            }
            for fi in sorted_feature_indices
        ],
        "nmi_threshold_applied": payload.nmi_threshold,
        "features_retained": int(feature_mask.sum()),
        "features_total": n_features,
        "nmi_mean": float(nmi_scores.mean()),
        "nmi_std": float(nmi_scores.std()),
        "nmi_p25": float(np.percentile(nmi_scores, 25)),
        "nmi_p75": float(np.percentile(nmi_scores, 75)),
        "latency_ms": round(latency_ms, 3),
    }


@app.post("/batch_search", response_model=list[SimilarityResponse], tags=["search"])
def nmi_weighted_cosine_batch_search(
    payloads: list[VectorPayload] = Field(..., min_length=1, max_length=50),
    _api_key: str = Depends(_require_api_key),
) -> list[SimilarityResponse]:
    if not payloads:
        raise HTTPException(status_code=422, detail="payloads list must not be empty")
    if len(payloads) > 50:
        raise HTTPException(status_code=422, detail="batch size must not exceed 50 queries")
    responses = []
    for p in payloads:
        resp = nmi_weighted_cosine_search.__wrapped__(p) if hasattr(nmi_weighted_cosine_search, "__wrapped__") else nmi_weighted_cosine_search(p, _api_key="bypass")
        responses.append(resp)
    return responses


def _run_batch_search_internal(payloads: list[VectorPayload]) -> list[SimilarityResponse]:
    responses = []
    for p in payloads:
        t0 = time.perf_counter()
        query_np = np.array(p.query, dtype=np.float64)
        corpus_np = np.array(p.corpus, dtype=np.float64)
        n_corpus, n_features = corpus_np.shape
        if len(query_np) != n_features:
            raise HTTPException(
                status_code=422,
                detail=f"batch item: query dimension {len(query_np)} does not match corpus dimension {n_features}",
            )
        nmi_scores = _compute_nmi_per_feature(query_np, corpus_np, p.n_bins)
        feature_mask = (nmi_scores >= p.nmi_threshold).astype(np.float64)
        active_count = int(feature_mask.sum())
        if active_count == 0:
            feature_mask = np.ones(n_features, dtype=np.float64)
            active_count = n_features
        cosine_sims = _cosine_similarity_batch(query_np, corpus_np, feature_mask)
        nmi_weighted_sims = _nmi_weighted_cosine_batch(query_np, corpus_np, nmi_scores * feature_mask)
        lower_bounds, upper_bounds = _bootstrap_confidence_interval(nmi_weighted_sims)
        top_k = min(p.top_k, n_corpus)
        top_indices = np.argpartition(nmi_weighted_sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(nmi_weighted_sims[top_indices])[::-1]]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                SimilarityResult(
                    rank=rank + 1,
                    corpus_index=int(idx),
                    cosine_similarity=float(np.clip(cosine_sims[idx], -1.0, 1.0)),
                    nmi_weighted_cosine=float(np.clip(nmi_weighted_sims[idx], -1.0, 1.0)),
                    confidence_lower=float(lower_bounds[idx]),
                    confidence_upper=float(upper_bounds[idx]),
                    active_features=active_count,
                    total_features=n_features,
                )
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        responses.append(
            SimilarityResponse(
                results=results,
                nmi_feature_mask=nmi_scores.tolist(),
                empirical_nmi_mean=float(nmi_scores.mean()),
                empirical_nmi_std=float(nmi_scores.std()),
                latency_ms=round(latency_ms, 3),
                dimensions_retained=active_count,
                dimensions_total=n_features,
            )
        )
    return responses


app.routes = [r for r in app.routes if not (hasattr(r, "path") and r.path == "/batch_search")]


@app.post("/batch_search", response_model=list[SimilarityResponse], tags=["search"])
def nmi_weighted_cosine_batch_search_v2(
    payloads: list[VectorPayload],
    _api_key: str = Depends(_require_api_key),
) -> list[SimilarityResponse]:
    if not payloads:
        raise HTTPException(status_code=422, detail="payloads list must not be empty")
    if len(payloads) > 50:
        raise HTTPException(status_code=422, detail="batch size must not exceed 50 queries")
    return _run_batch_search_internal(payloads)