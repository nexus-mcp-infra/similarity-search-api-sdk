import httpx
from typing import Any
import numpy as np


BASE_URL = "https://api.nexus-similarity.io/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_EMBEDDING_DIM = 4096
MAX_BATCH_PAIRS = 512


class SimilaritySearchAuthError(Exception):
    pass


class SimilaritySearchValidationError(Exception):
    pass


class SimilaritySearchAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class SimilaritySearchRateLimitError(SimilaritySearchAPIError):
    pass


def _validate_embedding(embedding: Any, label: str) -> list[float]:
    if embedding is None:
        raise SimilaritySearchValidationError(
            f"'{label}' must not be None — provide a non-empty list or numpy array of floats."
        )
    if isinstance(embedding, np.ndarray):
        if embedding.ndim != 1:
            raise SimilaritySearchValidationError(
                f"'{label}' must be a 1-D array, got shape {embedding.shape}."
            )
        embedding = embedding.tolist()
    if not isinstance(embedding, list):
        raise SimilaritySearchValidationError(
            f"'{label}' must be a list or numpy array, got {type(embedding).__name__}."
        )
    if len(embedding) == 0:
        raise SimilaritySearchValidationError(
            f"'{label}' must not be empty."
        )
    if len(embedding) > MAX_EMBEDDING_DIM:
        raise SimilaritySearchValidationError(
            f"'{label}' has {len(embedding)} dimensions; maximum supported is {MAX_EMBEDDING_DIM}."
        )
    if not all(isinstance(v, (int, float)) for v in embedding):
        raise SimilaritySearchValidationError(
            f"'{label}' must contain only numeric values (int or float)."
        )
    return [float(v) for v in embedding]


def _validate_embedding_batch(embeddings: Any, label: str) -> list[list[float]]:
    if embeddings is None:
        raise SimilaritySearchValidationError(
            f"'{label}' must not be None."
        )
    if not isinstance(embeddings, (list, np.ndarray)):
        raise SimilaritySearchValidationError(
            f"'{label}' must be a list of embeddings or a 2-D numpy array, got {type(embeddings).__name__}."
        )
    if isinstance(embeddings, np.ndarray):
        if embeddings.ndim != 2:
            raise SimilaritySearchValidationError(
                f"'{label}' must be a 2-D numpy array, got shape {embeddings.shape}."
            )
        embeddings = embeddings.tolist()
    if len(embeddings) == 0:
        raise SimilaritySearchValidationError(
            f"'{label}' must contain at least one embedding."
        )
    return [_validate_embedding(e, f"{label}[{i}]") for i, e in enumerate(embeddings)]


def _validate_domain(domain: str) -> str:
    allowed = {"text", "image", "tabular"}
    if not isinstance(domain, str):
        raise SimilaritySearchValidationError(
            f"'domain' must be a string, got {type(domain).__name__}."
        )
    if domain not in allowed:
        raise SimilaritySearchValidationError(
            f"'domain' must be one of {sorted(allowed)}, got '{domain}'."
        )
    return domain


def _validate_top_k(top_k: Any) -> int:
    if not isinstance(top_k, int):
        raise SimilaritySearchValidationError(
            f"'top_k' must be an integer, got {type(top_k).__name__}."
        )
    if top_k < 1 or top_k > MAX_BATCH_PAIRS:
        raise SimilaritySearchValidationError(
            f"'top_k' must be between 1 and {MAX_BATCH_PAIRS}, got {top_k}."
        )
    return top_k


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise SimilaritySearchAuthError(
            "Authentication failed: verify your api_key is correct and active."
        )
    if response.status_code == 422:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise SimilaritySearchValidationError(
            f"API rejected request with validation error: {detail}"
        )
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        raise SimilaritySearchRateLimitError(
            429,
            f"Rate limit exceeded. Retry after {retry_after} seconds."
        )
    if response.status_code >= 500:
        raise SimilaritySearchAPIError(
            response.status_code,
            f"Server error: {response.text}"
        )
    if response.status_code >= 400:
        raise SimilaritySearchAPIError(
            response.status_code,
            f"Client error: {response.text}"
        )


class Client:
    """
    Thin HTTP wrapper over the Similarity Search API.

    Provides stateless on-the-fly similarity scoring combining NMI and cosine
    with domain-calibrated weights, without requiring vector upsert or index setup.

    Usage:
        client = Client(api_key="sk-...")
        result = client.score_embedding_pair(query, candidate, domain="text")
        results = client.rank_candidates_by_composite_score(query, candidates, domain="text")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not api_key or not isinstance(api_key, str):
            raise SimilaritySearchAuthError(
                "api_key must be a non-empty string. "
                "Obtain yours at https://nexus-similarity.io/dashboard."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def main_method(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Generic dispatch entry-point for callers using the Client(api_key).main_method(data) pattern.

        'data' must be a dict containing:
          - 'operation' (str): one of 'score_pair', 'rank_candidates', 'batch_score_pairs'
          - operation-specific keys (see individual methods for full parameter docs)

        For production use, prefer the named methods directly:
          client.score_embedding_pair(...)
          client.rank_candidates_by_composite_score(...)
          client.batch_score_embedding_pairs(...)
        """
        if not isinstance(data, dict):
            raise SimilaritySearchValidationError(
                f"'data' must be a dict, got {type(data).__name__}. "
                "Expected keys: 'operation' and operation-specific parameters."
            )
        operation = data.get("operation")
        if operation is None:
            raise SimilaritySearchValidationError(
                "'data' must include an 'operation' key: 'score_pair', 'rank_candidates', or 'batch_score_pairs'."
            )
        if operation == "score_pair":
            return self.score_embedding_pair(
                query_embedding=data["query_embedding"],
                candidate_embedding=data["candidate_embedding"],
                domain=data.get("domain", "text"),
            )
        if operation == "rank_candidates":
            return self.rank_candidates_by_composite_score(
                query_embedding=data["query_embedding"],
                candidate_embeddings=data["candidate_embeddings"],
                domain=data.get("domain", "text"),
                top_k=data.get("top_k", 10),
            )
        if operation == "batch_score_pairs":
            return self.batch_score_embedding_pairs(
                pairs=data["pairs"],
                domain=data.get("domain", "text"),
            )
        raise SimilaritySearchValidationError(
            f"Unknown operation '{operation}'. "
            "Valid operations: 'score_pair', 'rank_candidates', 'batch_score_pairs'."
        )

    def score_embedding_pair(
        self,
        query_embedding: Any,
        candidate_embedding: Any,
        domain: str = "text",
    ) -> dict[str, Any]:
        """
        Compute the NMI+Cosine composite similarity score for a single embedding pair.

        Returns a dict with:
          - 'composite_score' (float): S = alpha*cosine + (1-alpha)*NMI, domain-calibrated alpha
          - 'cosine_score' (float): raw cosine similarity component
          - 'nmi_score' (float): normalized mutual information component
          - 'domain' (str): domain used for weight calibration
          - 'latency_ms' (float): server-side compute latency

        Use this when: you need a single on-the-fly similarity score without persisting vectors.
        Do NOT use this when: you need to rank more than one candidate — use rank_candidates_by_composite_score instead.

        Args:
            query_embedding: 1-D list or numpy array of floats, max {MAX_EMBEDDING_DIM} dims.
            candidate_embedding: same shape and dimension as query_embedding.
            domain: 'text', 'image', or 'tabular'. Selects calibrated alpha/beta weights.
        """
        query_vec = _validate_embedding(query_embedding, "query_embedding")
        candidate_vec = _validate_embedding(candidate_embedding, "candidate_embedding")
        _validate_domain(domain)
        if len(query_vec) != len(candidate_vec):
            raise SimilaritySearchValidationError(
                f"query_embedding and candidate_embedding must have the same dimension; "
                f"got {len(query_vec)} vs {len(candidate_vec)}."
            )
        payload = {
            "query_embedding": query_vec,
            "candidate_embedding": candidate_vec,
            "domain": domain,
        }
        response = self._client.post(f"{self._base_url}/similarity/score-pair", json=payload)
        _raise_for_status(response)
        return response.json()

    def rank_candidates_by_composite_score(
        self,
        query_embedding: Any,
        candidate_embeddings: Any,
        domain: str = "text",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Rank a list of candidate embeddings against a query using the NMI+Cosine composite score.

        Returns a dict with:
          - 'ranked_results' (list): each item has 'index', 'composite_score', 'cosine_score', 'nmi_score'
          - 'domain' (str): domain used for calibration
          - 'total_candidates' (int): number of candidates evaluated
          - 'latency_ms' (float): server-side compute latency

        Use this when: you need to find the top-K most similar items from a candidate set on-the-fly.
        Do NOT use this when: you have a persistent index to query — use a vector database instead.

        Args:
            query_embedding: 1-D list or numpy array, max {MAX_EMBEDDING_DIM} dims.
            candidate_embeddings: list of 1-D embeddings or 2-D numpy array; max {MAX_BATCH_PAIRS} candidates.
            domain: 'text', 'image', or 'tabular'.
            top_k: number of top results to return; must be between 1 and {MAX_BATCH_PAIRS}.
        """
        query_vec = _validate_embedding(query_embedding, "query_embedding")
        candidate_vecs = _validate_embedding_batch(candidate_embeddings, "candidate_embeddings")
        _validate_domain(domain)
        top_k = _validate_top_k(top_k)
        if len(candidate_vecs) > MAX_BATCH_PAIRS:
            raise SimilaritySearchValidationError(
                f"candidate_embeddings exceeds maximum batch size of {MAX_BATCH_PAIRS}; "
                f"got {len(candidate_vecs)}. Split into multiple calls."
            )
        for i, vec in enumerate(candidate_vecs):
            if len(vec) != len(query_vec):
                raise SimilaritySearchValidationError(
                    f"candidate_embeddings[{i}] has dimension {len(vec)}, "
                    f"but query_embedding has dimension {len(query_vec)}. All must match."
                )
        payload = {
            "query_embedding": query_vec,
            "candidate_embeddings": candidate_vecs,
            "domain": domain,
            "top_k": top_k,
        }
        response = self._client.post(
            f"{self._base_url}/similarity/rank-candidates", json=payload
        )
        _raise_for_status(response)
        return response.json()

    def batch_score_embedding_pairs(
        self,
        pairs: list[dict[str, Any]],
        domain: str = "text",
    ) -> dict[str, Any]:
        """
        Compute NMI+Cosine composite scores for multiple independent embedding pairs in a single call.

        Returns a dict with:
          - 'scores' (list): each item has 'pair_index', 'composite_score', 'cosine_score', 'nmi_score'
          - 'domain' (str): domain used for calibration
          - 'latency_ms' (float): server-side compute latency

        Use this when: you have N independent pairs and want to minimize round-trip overhead.
        Do NOT use this when: you need to rank candidates against a shared query — use rank_candidates_by_composite_score.

        Args:
            pairs: list of dicts, each with keys 'query_embedding' and 'candidate_embedding'.
                   Max {MAX_BATCH_PAIRS} pairs per call.
            domain: 'text', 'image', or 'tabular'.
        """
        if not isinstance(pairs, list):
            raise SimilaritySearchValidationError(
                f"'pairs' must be a list of dicts, got {type(pairs).__name__}."
            )
        if len(pairs) == 0:
            raise SimilaritySearchValidationError(
                "'pairs' must contain at least one pair."
            )
        if len(pairs) > MAX_BATCH_PAIRS:
            raise SimilaritySearchValidationError(
                f"'pairs' exceeds maximum batch size of {MAX_BATCH_PAIRS}; "
                f"got {len(pairs)}. Split into multiple calls."
            )
        _validate_domain(domain)
        validated_pairs = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise SimilaritySearchValidationError(
                    f"pairs[{i}] must be a dict with 'query_embedding' and 'candidate_embedding', "
                    f"got {type(pair).__name__}."
                )
            if "query_embedding" not in pair:
                raise SimilaritySearchValidationError(
                    f"pairs[{i}] is missing required key 'query_embedding'."
                )
            if "candidate_embedding" not in pair:
                raise SimilaritySearchValidationError(
                    f"pairs[{i}] is missing required key 'candidate_embedding'."
                )
            q = _validate_embedding(pair["query_embedding"], f"pairs[{i}].query_embedding")
            c = _validate_embedding(pair["candidate_embedding"], f"pairs[{i}].candidate_embedding")
            if len(q) != len(c):
                raise SimilaritySearchValidationError(
                    f"pairs[{i}]: query_embedding dim {len(q)} != candidate_embedding dim {len(c)}."
                )
            validated_pairs.append({"query_embedding": q, "candidate_embedding": c})
        payload = {"pairs": validated_pairs, "domain": domain}
        response = self._client.post(
            f"{self._base_url}/similarity/batch-score-pairs", json=payload
        )
        _raise_for_status(response)
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()