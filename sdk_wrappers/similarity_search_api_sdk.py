"""
similarity_search_sdk.py

Thin HTTP wrapper over the Similarity Search API.
Exposes composite-score similarity search (cosine + NMI) with calibrated p-values.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.similarity-search.nexus/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_ALPHA = 0.6


class SimilaritySearchError(Exception):
    """Raised when the API returns a non-2xx response or a network error occurs."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    """Raised when the API key is missing, empty, or rejected (401/403)."""


class RateLimitError(SimilaritySearchError):
    """Raised when the API returns 429. Inspect retry_after for back-off guidance."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(SimilaritySearchError):
    """Raised when the API rejects the request due to malformed input (422)."""


class CompositeScoreResult:
    """
    Represents one item returned by composite_score_search or composite_score_pair.

    Attributes
    ----------
    item_id : str
        Identifier of the candidate item.
    cosine_similarity : float
        Raw cosine similarity in [-1, 1].
    nmi_score : float
        Normalized Mutual Information over activation histograms in [0, 1].
    composite_score : float
        Weighted combination: alpha * cosine + (1 - alpha) * nmi_score.
    p_value : float
        Chi-squared p-value corrected for corpus size via Bonferroni.
        Values below 0.05 indicate statistically significant similarity.
    alpha : float
        The alpha used for this result (echoed from request).
    metadata : dict
        Any additional fields returned by the API for this item.
    """

    __slots__ = (
        "item_id",
        "cosine_similarity",
        "nmi_score",
        "composite_score",
        "p_value",
        "alpha",
        "metadata",
    )

    def __init__(
        self,
        item_id: str,
        cosine_similarity: float,
        nmi_score: float,
        composite_score: float,
        p_value: float,
        alpha: float,
        metadata: dict,
    ):
        self.item_id = item_id
        self.cosine_similarity = cosine_similarity
        self.nmi_score = nmi_score
        self.composite_score = composite_score
        self.p_value = p_value
        self.alpha = alpha
        self.metadata = metadata

    def is_statistically_significant(self, threshold: float = 0.05) -> bool:
        """Returns True if p_value < threshold (Bonferroni-corrected against corpus size)."""
        return self.p_value < threshold

    def __repr__(self) -> str:
        return (
            f"CompositeScoreResult(item_id={self.item_id!r}, "
            f"composite_score={self.composite_score:.4f}, "
            f"p_value={self.p_value:.4e}, "
            f"significant={self.is_statistically_significant()})"
        )


class IndexStats:
    """
    Corpus index statistics returned by corpus_index_stats.

    Attributes
    ----------
    corpus_id : str
    item_count : int
    embedding_dim : int
    index_created_at : str   ISO-8601 timestamp
    storage_bytes : int
    metadata : dict
    """

    __slots__ = (
        "corpus_id",
        "item_count",
        "embedding_dim",
        "index_created_at",
        "storage_bytes",
        "metadata",
    )

    def __init__(
        self,
        corpus_id: str,
        item_count: int,
        embedding_dim: int,
        index_created_at: str,
        storage_bytes: int,
        metadata: dict,
    ):
        self.corpus_id = corpus_id
        self.item_count = item_count
        self.embedding_dim = embedding_dim
        self.index_created_at = index_created_at
        self.storage_bytes = storage_bytes
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"IndexStats(corpus_id={self.corpus_id!r}, "
            f"item_count={self.item_count}, "
            f"embedding_dim={self.embedding_dim})"
        )


def _validate_embedding(embedding: Any, param_name: str) -> list[float]:
    if embedding is None:
        raise ValueError(f"'{param_name}' must not be None.")
    if not isinstance(embedding, (list, tuple)):
        raise TypeError(
            f"'{param_name}' must be a list or tuple of floats, got {type(embedding).__name__}."
        )
    if len(embedding) == 0:
        raise ValueError(f"'{param_name}' must not be empty.")
    result = []
    for i, v in enumerate(embedding):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"'{param_name}[{i}]' must be numeric, got {type(v).__name__}."
            )
        result.append(float(v))
    return result


def _validate_alpha(alpha: Any) -> float:
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"'alpha' must be a float in [0.0, 1.0], got {type(alpha).__name__}.")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"'alpha' must be in [0.0, 1.0], got {alpha}.")
    return alpha


def _validate_top_k(top_k: Any) -> int:
    if not isinstance(top_k, int):
        raise TypeError(f"'top_k' must be an int, got {type(top_k).__name__}.")
    if not (1 <= top_k <= 1000):
        raise ValueError(f"'top_k' must be between 1 and 1000, got {top_k}.")
    return top_k


def _validate_corpus_id(corpus_id: Any) -> str:
    if corpus_id is None:
        raise ValueError("'corpus_id' must not be None.")
    if not isinstance(corpus_id, str):
        raise TypeError(f"'corpus_id' must be a str, got {type(corpus_id).__name__}.")
    corpus_id = corpus_id.strip()
    if len(corpus_id) == 0:
        raise ValueError("'corpus_id' must not be empty or whitespace.")
    if len(corpus_id) > 128:
        raise ValueError("'corpus_id' must not exceed 128 characters.")
    return corpus_id


def _parse_composite_score_result(raw: dict) -> CompositeScoreResult:
    try:
        return CompositeScoreResult(
            item_id=str(raw["item_id"]),
            cosine_similarity=float(raw["cosine_similarity"]),
            nmi_score=float(raw["nmi_score"]),
            composite_score=float(raw["composite_score"]),
            p_value=float(raw["p_value"]),
            alpha=float(raw["alpha"]),
            metadata=raw.get("metadata") or {},
        )
    except KeyError as exc:
        raise SimilaritySearchError(
            f"API response missing expected field: {exc}. Raw item: {raw}"
        ) from exc


class Client:
    """
    HTTP client for the Similarity Search API.

    Provides composite-score similarity (cosine + NMI) with Bonferroni-corrected
    p-values — without requiring the caller to manage any vector database.

    Parameters
    ----------
    api_key : str
        Your Similarity Search API key. Must not be empty.
    base_url : str, optional
        Override the API base URL (useful for self-hosted or staging deployments).
    timeout : float, optional
        Per-request timeout in seconds. Default 30.0.
    max_retries : int, optional
        Number of automatic retries on transient 5xx errors. Default 2.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ):
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise AuthenticationError(
                "A non-empty 'api_key' string is required to initialize the Client."
            )
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"'timeout' must be a positive number, got {timeout}.")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"'max_retries' must be a non-negative int, got {max_retries}.")

        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def composite_score_search(
        self,
        query_embedding: list[float],
        corpus_id: str,
        top_k: int = 10,
        alpha: float = DEFAULT_ALPHA,
        min_composite_score: float | None = None,
        significance_threshold: float | None = None,
    ) -> list[CompositeScoreResult]:
        """
        Search a corpus for the top-k items most similar to query_embedding,
        ranked by composite score (alpha * cosine + (1-alpha) * NMI).

        Returns only results that pass optional filters for composite score
        and/or statistical significance.

        Parameters
        ----------
        query_embedding : list[float]
            Dense embedding vector of the query item. Must match the
            embedding dimension of the corpus (typically 384–3072 floats).
        corpus_id : str
            Identifier of the corpus to search. Max 128 characters.
        top_k : int, optional
            Number of candidates to return before filtering. Range [1, 1000].
        alpha : float, optional
            Weighting factor for cosine vs NMI. 0.0 = pure NMI, 1.0 = pure cosine.
            Default 0.6. Must be in [0.0, 1.0].
        min_composite_score : float, optional
            If provided, exclude results with composite_score below this threshold.
        significance_threshold : float, optional
            If provided, exclude results with p_value >= this threshold.
            Typical value: 0.05.

        Returns
        -------
        list[CompositeScoreResult]
            Ranked list of results, highest composite_score first.

        Raises
        ------
        AuthenticationError
            When the API key is rejected.
        RateLimitError
            When rate limit is exceeded.
        ValidationError
            When the API rejects the request payload.
        SimilaritySearchError
            On any other API or network failure.
        """
        query_embedding = _validate_embedding(query_embedding, "query_embedding")
        corpus_id = _validate_corpus_id(corpus_id)
        top_k = _validate_top_k(top_k)
        alpha = _validate_alpha(alpha)

        if min_composite_score is not None:
            if not isinstance(min_composite_score, (int, float)):
                raise TypeError("'min_composite_score' must be a float.")
            min_composite_score = float(min_composite_score)

        if significance_threshold is not None:
            if not isinstance(significance_threshold, (int, float)):
                raise TypeError("'significance_threshold' must be a float.")
            significance_threshold = float(significance_threshold)

        payload: dict[str, Any] = {
            "query_embedding": query_embedding,
            "corpus_id": corpus_id,
            "top_k": top_k,
            "alpha": alpha,
        }
        if min_composite_score is not None:
            payload["min_composite_score"] = min_composite_score
        if significance_threshold is not None:
            payload["significance_threshold"] = significance_threshold

        response = self._post("/search", payload)
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise SimilaritySearchError(
                f"Expected 'results' list in API response, got: {type(raw_results).__name__}."
            )
        return [_parse_composite_score_result(r) for r in raw_results]

    def composite_score_pair(
        self,
        embedding_a: list[float],
        embedding_b: list[float],
        corpus_size: int,
        alpha: float = DEFAULT_ALPHA,
    ) -> CompositeScoreResult:
        """
        Compute the composite similarity score between exactly two embeddings,
        with p-value calibrated against a user-specified corpus size
        (Bonferroni correction over corpus_size comparisons).

        Use this when you have two specific items to compare and want statistical
        significance without maintaining a persistent corpus index.
        Do NOT use this for batch search across many candidates — use
        composite_score_search instead.

        Parameters
        ----------
        embedding_a : list[float]
            First dense embedding vector.
        embedding_b : list[float]
            Second dense embedding vector. Must have the same dimension as embedding_a.
        corpus_size : int
            Number of items in the reference corpus for Bonferroni correction.
            Must be >= 1. Use 1 if comparing in isolation (no correction applied).
        alpha : float, optional
            Weighting factor for cosine vs NMI. Default 0.6.

        Returns
        -------
        CompositeScoreResult
            The composite score result. item_id will be 'pair' for pairwise calls.

        Raises
        ------
        ValueError
            If embedding dimensions differ.
        AuthenticationError, RateLimitError, ValidationError, SimilaritySearchError
            See composite_score_search for error semantics.
        """
        embedding_a = _validate_embedding(embedding_a, "embedding_a")
        embedding_b = _validate_embedding(embedding_b, "embedding_b")
        alpha = _validate_alpha(alpha)

        if len(embedding_a) != len(embedding_b):
            raise ValueError(
                f"'embedding_a' and 'embedding_b' must have the same dimension. "
                f"Got {len(embedding_a)} vs {len(embedding_b)}."
            )
        if not isinstance(corpus_size, int) or corpus_size < 1:
            raise ValueError(
                f"'corpus_size' must be a positive integer, got {corpus_size!r}."
            )

        payload = {
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
            "corpus_size": corpus_size,
            "alpha": alpha,
        }
        response = self._post("/pair", payload)
        result_raw = response.get("result")
        if not isinstance(result_raw, dict):
            raise SimilaritySearchError(
                f"Expected 'result' dict in API response, got: {type(result_raw).__name__}."
            )
        return _parse_composite_score_result(result_raw)

    def corpus_upsert(
        self,
        corpus_id: str,
        items: list[dict],
    ) -> dict:
        """
        Insert or update items in a corpus. Existing items with the same
        item_id are overwritten; new items are appended.

        Each item in 'items' must be a dict with at minimum:
            - 'item_id' (str): unique identifier within the corpus
            - 'embedding' (list[float]): dense embedding vector

        Optional per-item fields:
            - 'metadata' (dict): arbitrary key-value pairs returned in search results

        Parameters
        ----------
        corpus_id : str
            Target corpus. Created automatically if it does not exist.
        items : list[dict]
            List of item dicts. Maximum 5000 items per call.

        Returns
        -------
        dict
            API response with keys: 'corpus_id', 'upserted_count', 'skipped_count'.

        Raises
        ------
        ValueError
            If items is empty or exceeds 5000.
        AuthenticationError, RateLimitError, ValidationError, SimilaritySearchError
            Standard error semantics.
        """
        corpus_id = _validate_corpus_id(corpus_id)

        if not isinstance(items, list):
            raise TypeError(f"'items' must be a list, got {type(items).__name__}.")
        if len(items) == 0:
            raise ValueError("'items' must not be empty.")
        if len(items) > 5000:
            raise ValueError(
                f"'items' must not exceed 5000 per call, got {len(items)}. "
                "Split into multiple corpus_upsert calls."
            )

        validated_items = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise TypeError(f"items[{idx}] must be a dict, got {type(item).__name__}.")
            item_id = item.get("item_id")
            if not item_id or not isinstance(item_id, str):
                raise ValueError(f"items[{idx}]['item_id'] must be a non-empty string.")
            embedding = _validate_embedding(item.get("embedding"), f"items[{idx}]['embedding']")
            validated_item: dict[str, Any] = {"item_id": item_id, "embedding": embedding}
            if "metadata" in item:
                if not isinstance(item["metadata"], dict):
                    raise TypeError(f"items[{idx}]['metadata'] must be a dict.")
                validated_item["metadata"] = item["metadata"]
            validated_items.append(validated_item)

        payload = {"corpus_id": corpus_id, "items": validated_items}
        return self._post("/corpus/upsert", payload)

    def corpus_index_stats(self, corpus_id: str) -> IndexStats:
        """
        Retrieve statistics for a corpus index: item count, embedding dimension,
        creation timestamp, and storage usage.

        Use this to verify a corpus exists and is ready before issuing searches,
        or to determine corpus_size for Bonferroni correction in composite_score_pair.
        Do NOT use this as a health-check polling loop — it bills per call.

        Parameters
        ----------
        corpus_id : str
            Corpus to inspect. Max 128 characters.

        Returns
        -------
        IndexStats

        Raises
        ------
        SimilaritySearchError (status_code=404)
            If the corpus does not exist.
        AuthenticationError, RateLimitError, ValidationError, SimilaritySearchError
            Standard error semantics.
        """
        corpus_id = _validate_corpus_id(corpus_id)
        response = self._get(f"/corpus/{corpus_id}/stats")
        try:
            return IndexStats(
                corpus_id=str(response["corpus_id"]),
                item_count=int(response["item_count"]),
                embedding_dim=int(response["embedding_dim"]),
                index_created_at=str(response["index_created_at"]),
                storage_bytes=int(response["storage_bytes"]),
                metadata=response.get("metadata") or {},
            )
        except KeyError as exc:
            raise SimilaritySearchError(
                f"API response missing expected field: {exc}. Raw response: {response}"
            ) from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(url, json=payload)
                return self._handle_response(response)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise SimilaritySearchError(
                    f"Network error after {self._max_retries + 1} attempt(s) "
                    f"on POST {path}: {exc}"
                ) from exc
        raise SimilaritySearchError(
            f"Exhausted retries on POST {path}."
        ) from last_exc

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.get(url)
                return self._handle_response(response)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise SimilaritySearchError(
                    f"Network error after {self._max_retries + 1} attempt(s) "
                    f"on GET {path}: {exc}"
                ) from exc
        raise SimilaritySearchError(
            f"Exhausted retries on GET {path}."
        ) from last_exc

    def _handle_response(self, response: httpx.Response) -> dict:
        status = response.status_code
        if status == 200 or status == 201:
            try:
                return response.json()
            except Exception as exc:
                raise SimilaritySearchError(
                    f"API returned status {status} but response body is not valid JSON: {exc}. "
                    f"Body snippet: {response.text[:300]!r}"
                ) from exc

        body: Any = None
        try:
            body = response.json()
        except Exception:
            body = response.text[:500]

        detail = body.get("detail", body) if isinstance(body, dict) else body

        if status == 401 or status == 403:
            raise AuthenticationError(
                f"Authentication failed (HTTP {status}): {detail}",
                status_code=status,
                response_body=body,
            )
        if status == 422:
            raise ValidationError(
                f"Request validation error (HTTP 422): {detail}",
                status_code=422,
                response_body=body,
            )
        if status == 429:
            retry_after: float | None = None
            raw_retry = response.headers.get("Retry-After")
            if raw_retry is not None:
                try:
                    retry_after = float(raw_retry)
                except ValueError:
                    pass
            raise RateLimitError(
                f"Rate limit exceeded (HTTP 429): {detail}. "
                f"Retry after {retry_after}s." if retry_after else
                f"Rate limit exceeded (HTTP 429): {detail}.",
                retry_after=retry_after,
            )
        if status >= 500:
            raise SimilaritySearchError(
                f"API server error (HTTP {status}): {detail}",
                status_code=status,
                response_body=body,
            )
        raise SimilaritySearchError(
            f"Unexpected API response (HTTP {status}): {detail}",
            status_code=status,
            response_body=body,
        )

    def close(self):
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()