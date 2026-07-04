from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
import hashlib
import time
import os

from src.math.information import NormalizedMutualInformation, TransferEntropy
from src.math.causal import DoCalculus, CausalDAG, build_nexus_dag
from src.math.game_theory import NashEquilibrium, MarketEntryGame
from src.math.statistics import Statistics

app = FastAPI(
    title="NMI Similarity Search API",
    version="1.0.0",
    description="Hybrid NMI+Cosine similarity over raw data without embedding pipeline",
)

security = HTTPBearer()

_corpus_registry: dict[str, dict] = {}
_api_keys: set[str] = set(os.environ.get("NEXUS_API_KEYS", "").split(","))

NMI_COMPUTE_LIMIT = 50_000
MAX_CORPUS_ITEMS = 100_000
MAX_QUERY_LENGTH = 4096
MAX_CORPUS_LABEL_LENGTH = 512


class CorpusIngestRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    documents: list[str] = Field(..., min_length=1, max_length=MAX_CORPUS_ITEMS)
    labels: Optional[list[str]] = None

    @field_validator("documents")
    @classmethod
    def documents_not_empty(cls, v):
        if any(len(d.strip()) == 0 for d in v):
            raise ValueError("Each document must be a non-empty string")
        if any(len(d) > MAX_QUERY_LENGTH for d in v):
            raise ValueError(f"Each document must be <= {MAX_QUERY_LENGTH} characters")
        return v

    @field_validator("labels")
    @classmethod
    def labels_length_matches(cls, v, info):
        if v is not None and "documents" in info.data:
            if len(v) != len(info.data["documents"]):
                raise ValueError("labels length must match documents length")
            if any(len(lbl) > MAX_CORPUS_LABEL_LENGTH for lbl in v):
                raise ValueError(f"Each label must be <= {MAX_CORPUS_LABEL_LENGTH} characters")
        return v


class SimilarityQueryRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=10, ge=1, le=100)
    alpha_override: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CorpusStatsRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")


class TransferEntropyRequest(BaseModel):
    source_sequence: list[float] = Field(..., min_length=2, max_length=10_000)
    target_sequence: list[float] = Field(..., min_length=2, max_length=10_000)
    lag: int = Field(default=1, ge=1, le=50)

    @field_validator("target_sequence")
    @classmethod
    def sequences_same_length(cls, v, info):
        if "source_sequence" in info.data and len(v) != len(info.data["source_sequence"]):
            raise ValueError("source_sequence and target_sequence must have equal length")
        return v


def _authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    if not token or token not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return token


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _bow_vector(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _build_vocab(all_tokens: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for tokens in all_tokens:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def _marginal_entropy_alpha(
    corpus_vectors: np.ndarray,
    vocab_size: int,
    stats: Statistics,
) -> float:
    if vocab_size <= 1:
        return 0.5
    marginal_freq = corpus_vectors.sum(axis=0)
    total = marginal_freq.sum()
    if total == 0:
        return 0.5
    marginal_prob = marginal_freq / total
    marginal_prob = marginal_prob[marginal_prob > 0]
    h_marginal = float(stats.entropy(marginal_prob.tolist()))
    log2_vocab = float(np.log2(vocab_size))
    if log2_vocab == 0:
        return 0.5
    alpha = float(np.clip(h_marginal / log2_vocab, 0.0, 1.0))
    return alpha


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hybrid_score(
    query_vec: np.ndarray,
    doc_vec: np.ndarray,
    query_tokens: list[str],
    doc_tokens: list[str],
    alpha: float,
    nmi_engine: NormalizedMutualInformation,
    alpha_override: Optional[float],
) -> float:
    effective_alpha = alpha_override if alpha_override is not None else alpha
    cos = _cosine_similarity(query_vec, doc_vec)
    nmi_score = 0.0
    if len(query_tokens) > 0 and len(doc_tokens) > 0:
        nmi_score = float(nmi_engine.compute(query_tokens, doc_tokens))
    return effective_alpha * cos + (1.0 - effective_alpha) * nmi_score


@app.post("/corpus/ingest")
def ingest_corpus(
    request: CorpusIngestRequest,
    _token: str = Depends(_authenticate),
) -> dict:
    stats = Statistics()
    nmi_engine = NormalizedMutualInformation()

    tokenized = [_tokenize(doc) for doc in request.documents]
    vocab = _build_vocab(tokenized)
    vocab_size = len(vocab)

    if vocab_size == 0:
        raise HTTPException(status_code=422, detail="Corpus produced an empty vocabulary after tokenization")

    corpus_vectors = np.array([_bow_vector(t, vocab) for t in tokenized], dtype=np.float32)
    alpha = _marginal_entropy_alpha(corpus_vectors, vocab_size, stats)

    corpus_fingerprint = hashlib.sha256(
        "".join(request.documents).encode("utf-8")
    ).hexdigest()[:16]

    corpus_nmi_matrix: Optional[list] = None
    n = len(request.documents)
    if n <= 500:
        dag = build_nexus_dag(tokenized)
        causal = DoCalculus(dag)
        corpus_nmi_matrix = causal.marginal_independence_summary()

    _corpus_registry[request.corpus_id] = {
        "documents": request.documents,
        "tokenized": tokenized,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "corpus_vectors": corpus_vectors,
        "alpha": alpha,
        "fingerprint": corpus_fingerprint,
        "ingested_at": time.time(),
        "causal_summary": corpus_nmi_matrix,
        "labels": request.labels,
    }

    return {
        "corpus_id": request.corpus_id,
        "document_count": n,
        "vocab_size": vocab_size,
        "alpha": round(alpha, 6),
        "fingerprint": corpus_fingerprint,
        "causal_summary_available": corpus_nmi_matrix is not None,
    }


@app.post("/similarity/query")
def query_similarity(
    request: SimilarityQueryRequest,
    _token: str = Depends(_authenticate),
) -> dict:
    if request.corpus_id not in _corpus_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{request.corpus_id}' not found. Call /corpus/ingest first.",
        )

    corpus = _corpus_registry[request.corpus_id]
    nmi_engine = NormalizedMutualInformation()

    query_tokens = _tokenize(request.query)
    if not query_tokens:
        raise HTTPException(status_code=422, detail="Query produced no tokens after tokenization")

    query_vec = _bow_vector(query_tokens, corpus["vocab"])
    alpha = corpus["alpha"]

    scores = []
    for idx, (doc_vec, doc_tokens) in enumerate(
        zip(corpus["corpus_vectors"], corpus["tokenized"])
    ):
        h = _hybrid_score(
            query_vec,
            doc_vec,
            query_tokens,
            doc_tokens,
            alpha,
            nmi_engine,
            request.alpha_override,
        )
        scores.append((idx, h))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[: request.top_k]

    results = []
    for rank, (idx, score) in enumerate(top):
        entry = {
            "rank": rank + 1,
            "document_index": idx,
            "hybrid_score": round(score, 6),
            "text_preview": corpus["documents"][idx][:120],
        }
        if corpus["labels"] is not None:
            entry["label"] = corpus["labels"][idx]
        results.append(entry)

    effective_alpha = request.alpha_override if request.alpha_override is not None else alpha

    return {
        "corpus_id": request.corpus_id,
        "query_length_tokens": len(query_tokens),
        "alpha_used": round(effective_alpha, 6),
        "results": results,
    }


@app.get("/corpus/stats/{corpus_id}")
def corpus_stats(
    corpus_id: str,
    _token: str = Depends(_authenticate),
) -> dict:
    if not corpus_id or len(corpus_id) > 64:
        raise HTTPException(status_code=422, detail="corpus_id must be 1-64 characters")

    if corpus_id not in _corpus_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{corpus_id}' not found.",
        )

    corpus = _corpus_registry[corpus_id]
    stats = Statistics()

    doc_lengths = [len(t) for t in corpus["tokenized"]]
    length_stats = stats.describe(doc_lengths)

    marginal_freq = corpus["corpus_vectors"].sum(axis=0)
    total = marginal_freq.sum()
    marginal_prob = (marginal_freq / total) if total > 0 else marginal_freq
    marginal_prob_nonzero = marginal_prob[marginal_prob > 0]
    h_marginal = float(stats.entropy(marginal_prob_nonzero.tolist()))

    return {
        "corpus_id": corpus_id,
        "document_count": len(corpus["documents"]),
        "vocab_size": corpus["vocab_size"],
        "alpha": round(corpus["alpha"], 6),
        "marginal_entropy_bits": round(h_marginal, 6),
        "log2_vocab": round(float(np.log2(max(corpus["vocab_size"], 2))), 6),
        "fingerprint": corpus["fingerprint"],
        "ingested_at": corpus["ingested_at"],
        "doc_length_tokens": {
            "mean": round(length_stats["mean"], 2),
            "std": round(length_stats["std"], 2),
            "min": length_stats["min"],
            "max": length_stats["max"],
        },
        "causal_summary_available": corpus.get("causal_summary") is not None,
    }


@app.post("/corpus/delete/{corpus_id}")
def delete_corpus(
    corpus_id: str,
    _token: str = Depends(_authenticate),
) -> dict:
    if not corpus_id or len(corpus_id) > 64:
        raise HTTPException(status_code=422, detail="corpus_id must be 1-64 characters")

    if corpus_id not in _corpus_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{corpus_id}' not found.",
        )

    doc_count = len(_corpus_registry[corpus_id]["documents"])
    del _corpus_registry[corpus_id]

    return {
        "corpus_id": corpus_id,
        "deleted": True,
        "documents_removed": doc_count,
    }


@app.post("/analysis/transfer-entropy")
def compute_transfer_entropy(
    request: TransferEntropyRequest,
    _token: str = Depends(_authenticate),
) -> dict:
    if len(request.source_sequence) < 2:
        raise HTTPException(status_code=422, detail="source_sequence requires at least 2 elements")

    te_engine = TransferEntropy()
    stats = Statistics()

    te_value = float(
        te_engine.compute(
            request.source_sequence,
            request.target_sequence,
            lag=request.lag,
        )
    )

    src_stats = stats.describe(request.source_sequence)
    tgt_stats = stats.describe(request.target_sequence)

    dag = build_nexus_dag([request.source_sequence, request.target_sequence])
    causal = DoCalculus(dag)
    causal_direction = causal.infer_direction(
        source=request.source_sequence,
        target=request.target_sequence,
        lag=request.lag,
    )

    ne = NashEquilibrium()
    market_game = MarketEntryGame(
        information_asymmetry=te_value,
        source_variance=src_stats["std"],
        target_variance=tgt_stats["std"],
    )
    equilibrium = ne.solve(market_game)

    return {
        "transfer_entropy_bits": round(te_value, 6),
        "lag": request.lag,
        "sequence_length": len(request.source_sequence),
        "causal_direction": causal_direction,
        "nash_equilibrium": {
            "strategy_source": round(equilibrium.strategy_a, 6),
            "strategy_target": round(equilibrium.strategy_b, 6),
            "payoff_source": round(equilibrium.payoff_a, 6),
            "payoff_target": round(equilibrium.payoff_b, 6),
        },
        "source_stats": {k: round(v, 4) for k, v in src_stats.items()},
        "target_stats": {k: round(v, 4) for k, v in tgt_stats.items()},
    }