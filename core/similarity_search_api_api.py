import hashlib
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from typing import Literal
import os
import json
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("similarity_search_api")

app = FastAPI(
    title="Stateless Semantic Similarity API",
    description="NMI+Cosine composite scoring without vector persistence",
    version="1.0.0",
)

security = HTTPBearer()

DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "text":     {"alpha": 0.62, "n_bins": 12},
    "image":    {"alpha": 0.48, "n_bins": 16},
    "tabular":  {"alpha": 0.35, "n_bins": 20},
}

MAX_DIM = 4096
MIN_DIM = 2
MAX_BATCH = 128


def _validate_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    expected = os.environ.get("SIMILARITY_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-12 or norm_v < 1e-12:
        raise HTTPException(
            status_code=422,
            detail="One or both embeddings have near-zero norm; cosine similarity is undefined"
        )
    return float(np.dot(u, v) / (norm_u * norm_v))


def _normalized_mutual_information(u: np.ndarray, v: np.ndarray, n_bins: int) -> float:
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    u_disc = kbd.fit_transform(u.reshape(-1, 1)).ravel().astype(int)
    v_disc = kbd.fit_transform(v.reshape(-1, 1)).ravel().astype(int)
    mi = mutual_info_score(u_disc, v_disc)
    h_u = mutual_info_score(u_disc, u_disc)
    h_v = mutual_info_score(v_disc, v_disc)
    denom = 0.5 * (h_u + h_v)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))


def _composite_score(cosine: float, nmi: float, alpha: float) -> float:
    cosine_01 = (cosine + 1.0) / 2.0
    return float(alpha * cosine_01 + (1.0 - alpha) * nmi)


def _sha256_pair(vec_a: list[float], vec_b: list[float]) -> str:
    payload = json.dumps({"a": vec_a[:8], "b": vec_b[:8]}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _log_call(
    pair_hash: str,
    domain: str,
    composite: float,
    cosine: float,
    nmi: float,
    latency_ms: float,
) -> None:
    logger.info(
        json.dumps({
            "event": "similarity_call",
            "pair_hash": pair_hash,
            "domain": domain,
            "composite": round(composite, 6),
            "cosine": round(cosine, 6),
            "nmi": round(nmi, 6),
            "latency_ms": round(latency_ms, 3),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    )


class SimilarityRequest(BaseModel):
    embedding_a: list[float] = Field(..., min_length=MIN_DIM, max_length=MAX_DIM)
    embedding_b: list[float] = Field(..., min_length=MIN_DIM, max_length=MAX_DIM)
    domain: Literal["text", "image", "tabular"] = Field(default="text")

    @model_validator(mode="after")
    def check_equal_dimensions(self) -> "SimilarityRequest":
        if len(self.embedding_a) != len(self.embedding_b):
            raise ValueError(
                f"embedding_a has {len(self.embedding_a)} dims but embedding_b has "
                f"{len(self.embedding_b)} dims; they must match"
            )
        return self


class SimilarityResult(BaseModel):
    composite_score: float
    cosine_similarity: float
    nmi_score: float
    domain: str
    alpha: float
    pair_hash: str
    latency_ms: float


class BatchSimilarityRequest(BaseModel):
    pairs: list[SimilarityRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchSimilarityResult(BaseModel):
    results: list[SimilarityResult]
    total_latency_ms: float


class RankRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=MIN_DIM, max_length=MAX_DIM)
    candidate_embeddings: list[list[float]] = Field(..., min_length=1, max_length=MAX_BATCH)
    domain: Literal["text", "image", "tabular"] = Field(default="text")
    top_k: int = Field(default=10, ge=1, le=MAX_BATCH)

    @field_validator("candidate_embeddings")
    @classmethod
    def check_candidate_dims(cls, candidates: list[list[float]]) -> list[list[float]]:
        lengths = {len(c) for c in candidates}
        if len(lengths) > 1:
            raise ValueError("All candidate embeddings must share the same dimensionality")
        return candidates

    @model_validator(mode="after")
    def check_query_vs_candidates_dim(self) -> "RankRequest":
        if self.candidate_embeddings:
            cand_dim = len(self.candidate_embeddings[0])
            if len(self.query_embedding) != cand_dim:
                raise ValueError(
                    f"query_embedding has {len(self.query_embedding)} dims but candidates "
                    f"have {cand_dim} dims"
                )
        return self


class RankedCandidate(BaseModel):
    index: int
    composite_score: float
    cosine_similarity: float
    nmi_score: float


class RankResult(BaseModel):
    ranked: list[RankedCandidate]
    domain: str
    alpha: float
    total_latency_ms: float


def _compute_single(req: SimilarityRequest) -> tuple[SimilarityResult, float]:
    t0 = time.perf_counter()
    weights = DOMAIN_WEIGHTS[req.domain]
    alpha = weights["alpha"]
    n_bins = weights["n_bins"]
    u = np.array(req.embedding_a, dtype=np.float64)
    v = np.array(req.embedding_b, dtype=np.float64)
    cosine = _cosine_similarity(u, v)
    nmi = _normalized_mutual_information(u, v, n_bins)
    composite = _composite_score(cosine, nmi, alpha)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pair_hash = _sha256_pair(req.embedding_a, req.embedding_b)
    _log_call(pair_hash, req.domain, composite, cosine, nmi, latency_ms)
    result = SimilarityResult(
        composite_score=round(composite, 6),
        cosine_similarity=round(cosine, 6),
        nmi_score=round(nmi, 6),
        domain=req.domain,
        alpha=alpha,
        pair_hash=pair_hash,
        latency_ms=round(latency_ms, 3),
    )
    return result, latency_ms


@app.post("/v1/similarity", response_model=SimilarityResult, tags=["core"])
def compute_pairwise_similarity(
    req: SimilarityRequest,
    _: str = Depends(_validate_api_key),
) -> SimilarityResult:
    result, _ = _compute_single(req)
    return result


@app.post("/v1/similarity/batch", response_model=BatchSimilarityResult, tags=["core"])
def compute_pairwise_similarity_batch(
    req: BatchSimilarityRequest,
    _: str = Depends(_validate_api_key),
) -> BatchSimilarityResult:
    t_batch = time.perf_counter()
    results = []
    for pair in req.pairs:
        result, _ = _compute_single(pair)
        results.append(result)
    total_ms = (time.perf_counter() - t_batch) * 1000.0
    return BatchSimilarityResult(results=results, total_latency_ms=round(total_ms, 3))


@app.post("/v1/rank", response_model=RankResult, tags=["core"])
def rank_candidates_by_composite_score(
    req: RankRequest,
    _: str = Depends(_validate_api_key),
) -> RankResult:
    t0 = time.perf_counter()
    weights = DOMAIN_WEIGHTS[req.domain]
    alpha = weights["alpha"]
    n_bins = weights["n_bins"]
    q = np.array(req.query_embedding, dtype=np.float64)
    scored: list[tuple[int, float, float, float]] = []
    for idx, cand in enumerate(req.candidate_embeddings):
        c = np.array(cand, dtype=np.float64)
        cosine = _cosine_similarity(q, c)
        nmi = _normalized_mutual_information(q, c, n_bins)
        composite = _composite_score(cosine, nmi, alpha)
        scored.append((idx, composite, cosine, nmi))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: req.top_k]
    total_ms = (time.perf_counter() - t0) * 1000.0
    ranked = [
        RankedCandidate(
            index=i,
            composite_score=round(comp, 6),
            cosine_similarity=round(cos, 6),
            nmi_score=round(nmi, 6),
        )
        for i, comp, cos, nmi in top
    ]
    return RankResult(
        ranked=ranked,
        domain=req.domain,
        alpha=alpha,
        total_latency_ms=round(total_ms, 3),
    )


@app.get("/v1/domain-weights", tags=["meta"])
def get_domain_calibration_weights(
    _: str = Depends(_validate_api_key),
) -> dict:
    return {
        domain: {"alpha": cfg["alpha"], "beta": round(1.0 - cfg["alpha"], 4), "n_bins": cfg["n_bins"]}
        for domain, cfg in DOMAIN_WEIGHTS.items()
    }


@app.get("/healthz", include_in_schema=False)
def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0"}