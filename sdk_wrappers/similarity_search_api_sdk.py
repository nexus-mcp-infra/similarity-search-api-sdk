from __future__ import annotations

import time
from typing import Any, Union

import httpx

__version__ = "0.1.0"
_DEFAULT_BASE_URL = "https://api.similarity-search.nexus/v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class SimilaritySearchError(Exception):
    """Base exception for all SDK errors."""
    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    """Raised when the API key is missing or invalid."""


class RateLimitError(SimilaritySearchError):
    """Raised when the server responds with 429 Too Many Requests."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(SimilaritySearchError):
    """Raised when input parameters fail server-side or client-side validation."""


class SimilaritySearchResult:
    """Parsed response from a ranking call."""

    def __init__(self, raw: dict[str, Any]):
        if not isinstance(raw, dict):
            raise ValidationError(
                f"Expected dict response body, got {type(raw).__name__}"
            )
        self._raw = raw
        self.query_id: str | None = raw.get("query_id")
        self.alpha: float = raw.get("alpha", 0.5)
        self.input_type: str = raw.get("input_type", "unknown")
        self.ranked_indices: list[int] = raw.get("ranked_indices", [])
        self.scores: list[float] = raw.get("scores", [])
        self.nmi_scores: list[float] = raw.get("nmi_scores", [])
        self.cosine_scores: list[float] = raw.get("cosine_scores", [])
        self.bias_correction_applied: bool = raw.get("bias_correction_applied", False)
        self.corpus_size: int = raw.get("corpus_size", 0)
        self.latency_ms: float = raw.get("latency_ms", 0.0)

    def __repr__(self) -> str:
        return (
            f"SimilaritySearchResult(corpus_size={self.corpus_size}, "
            f"input_type={self.input_type!r}, alpha={self.alpha}, "
            f"bias_correction_applied={self.bias_correction_applied})"
        )

    def to_dict(self) -> dict[str, Any]:
        return self._raw


class NMIRankingResponse:
    """Parsed response from a batch ranking call."""

    def __init__(self, raw: dict[str, Any]):
        if not isinstance(raw, dict):
            raise ValidationError(
                f"Expected dict response body, got {type(raw).__name__}"
            )
        self._raw = raw
        self.results: list[SimilaritySearchResult] = [
            SimilaritySearchResult(r) for r in raw.get("results", [])
        ]
        self.batch_id: str | None = raw.get("batch_id")
        self.total_latency_ms: float = raw.get("total_latency_ms", 0.0)

    def __repr__(self) -> str:
        return f"NMIRankingResponse(batch_size={len(self.results)}, batch_id={self.batch_id!r})"

    def to_dict(self) -> dict[str, Any]:
        return self._raw


Vector = list[float]
Distribution = list[float]
VectorOrDist = Union[Vector, Distribution]


def _validate_vector_or_distribution(
    value: Any, name: str, allow_empty: bool = False
) -> None:
    if value is None:
        raise ValidationError(f"'{name}' must not be None.")
    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"'{name}' must be a list or tuple of floats, got {type(value).__name__}."
        )
    if not allow_empty and len(value) == 0:
        raise ValidationError(f"'{name}' must not be empty.")
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)):
            raise ValidationError(
                f"'{name}[{i}]' must be a numeric value, got {type(v).__name__}."
            )


def _validate_corpus(corpus: Any) -> None:
    if corpus is None:
        raise ValidationError("'corpus' must not be None.")
    if not isinstance(corpus, (list, tuple)):
        raise ValidationError(
            f"'corpus' must be a list of vectors or distributions, got {type(corpus).__name__}."
        )
    if len(corpus) < 1:
        raise ValidationError("'corpus' must contain at least one item.")
    if len(corpus) > 50_000:
        raise ValidationError(
            f"'corpus' exceeds the maximum of 50,000 items (got {len(corpus)})."
        )
    reference_len = len(corpus[0])
    for i, item in enumerate(corpus):
        _validate_vector_or_distribution(item, f"corpus[{i}]")
        if len(item) != reference_len:
            raise ValidationError(
                f"All corpus items must have the same dimensionality. "
                f"corpus[0] has {reference_len} dimensions but corpus[{i}] has {len(item)}."
            )


def _validate_alpha(alpha: Any) -> None:
    if not isinstance(alpha, (int, float)):
        raise ValidationError(
            f"'alpha' must be a float between 0.0 and 1.0, got {type(alpha).__name__}."
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValidationError(
            f"'alpha' must be between 0.0 and 1.0, got {alpha}."
        )


def _validate_top_k(top_k: Any, corpus_size: int) -> None:
    if not isinstance(top_k, int):
        raise ValidationError(
            f"'top_k' must be a positive integer, got {type(top_k).__name__}."
        )
    if top_k < 1:
        raise ValidationError(f"'top_k' must be >= 1, got {top_k}.")
    if top_k > corpus_size:
        raise ValidationError(
            f"'top_k' ({top_k}) cannot exceed corpus size ({corpus_size})."
        )


class Client:
    """
    Thin HTTP wrapper for the Similarity Search API.

    Ranks a corpus of vectors or discrete probability distributions against a
    query using a hybrid NMI+Cosine scoring function. NMI is computed with
    Strehl-Ghosh bias correction for corpora smaller than 200 items.

    Parameters
    ----------
    api_key : str
        Your Similarity Search API key. Must not be empty.
    base_url : str, optional
        Override the default API base URL.
    timeout : float, optional
        Per-request timeout in seconds (default 30.0).
    max_retries : int, optional
        Maximum number of retries on retryable errors (default 3).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "'api_key' must be a non-empty string. "
                "Obtain your key at https://similarity-search.nexus/dashboard."
            )
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("'timeout' must be a positive number.")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValidationError("'max_retries' must be a non-negative integer.")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-SDK-Version": __version__,
            },
            timeout=self._timeout,
        )

    def rank_by_nmi_cosine(
        self,
        query: VectorOrDist,
        corpus: list[VectorOrDist],
        alpha: float = 0.5,
        top_k: int = 10,
        input_type: str = "dense_vector",
        n_bins: int | None = None,
    ) -> SimilaritySearchResult:
        """
        Rank corpus items against a query using a hybrid NMI+Cosine score.

        Score = alpha * NMI(query, item) + (1 - alpha) * Cosine(query, item)

        For dense vectors, the server performs adaptive binning + Strehl-Ghosh
        bias correction before computing NMI. For discrete distributions, NMI
        is computed directly without binning.

        Parameters
        ----------
        query : list[float]
            Query vector or discrete probability distribution. Must have the
            same dimensionality as all items in corpus.
        corpus : list[list[float]]
            Between 1 and 50,000 items to rank. All must share the same
            dimensionality as the query.
        alpha : float
            Weight of NMI in the hybrid score. 0.0 => pure Cosine;
            1.0 => pure NMI. Must be in [0.0, 1.0].
        top_k : int
            Number of top-ranked results to return. Must be >= 1 and
            <= len(corpus).
        input_type : str
            'dense_vector' — server will estimate joint distribution via
            adaptive binning with Strehl-Ghosh correction.
            'discrete_distribution' — server computes NMI directly; values
            must be non-negative and sum to 1.0 per item.
        n_bins : int or None
            Override the server's adaptive bin count for dense vector inputs.
            Ignored when input_type is 'discrete_distribution'.
            Must be between 2 and 512 if provided.

        Returns
        -------
        SimilaritySearchResult

        Raises
        ------
        ValidationError
            If any parameter is outside its valid range before the HTTP call.
        AuthenticationError
            If the API key is rejected by the server.
        RateLimitError
            If the server returns 429 after exhausting retries.
        SimilaritySearchError
            For any other non-2xx server response.
        """
        if input_type not in ("dense_vector", "discrete_distribution"):
            raise ValidationError(
                f"'input_type' must be 'dense_vector' or 'discrete_distribution', "
                f"got {input_type!r}."
            )
        _validate_vector_or_distribution(query, "query")
        _validate_corpus(corpus)
        _validate_alpha(alpha)
        _validate_top_k(top_k, len(corpus))

        if n_bins is not None:
            if not isinstance(n_bins, int) or not (2 <= n_bins <= 512):
                raise ValidationError(
                    f"'n_bins' must be an integer between 2 and 512, got {n_bins!r}."
                )

        payload: dict[str, Any] = {
            "query": list(query),
            "corpus": [list(item) for item in corpus],
            "alpha": float(alpha),
            "top_k": top_k,
            "input_type": input_type,
        }
        if n_bins is not None:
            payload["n_bins"] = n_bins

        raw = self._post("/rank", payload)
        return SimilaritySearchResult(raw)

    def rank_batch_by_nmi_cosine(
        self,
        queries: list[VectorOrDist],
        corpus: list[VectorOrDist],
        alpha: float = 0.5,
        top_k: int = 10,
        input_type: str = "dense_vector",
        n_bins: int | None = None,
    ) -> NMIRankingResponse:
        """
        Rank the same corpus against multiple queries in a single HTTP call.

        Each query produces an independent SimilaritySearchResult. The corpus
        is transmitted once and reused for all queries server-side.

        Parameters
        ----------
        queries : list[list[float]]
            Between 1 and 100 query vectors or distributions. All must share
            the same dimensionality as corpus items.
        corpus : list[list[float]]
            Between 1 and 50,000 items. See rank_by_nmi_cosine for details.
        alpha : float
            Shared hybrid weight for all queries in this batch.
        top_k : int
            Shared top-k for all queries.
        input_type : str
            'dense_vector' or 'discrete_distribution'. Applies to all items.
        n_bins : int or None
            Optional bin override. See rank_by_nmi_cosine.

        Returns
        -------
        NMIRankingResponse
            Contains one SimilaritySearchResult per query, in the same order.

        Raises
        ------
        ValidationError, AuthenticationError, RateLimitError, SimilaritySearchError
        """
        if queries is None:
            raise ValidationError("'queries' must not be None.")
        if not isinstance(queries, (list, tuple)) or len(queries) == 0:
            raise ValidationError("'queries' must be a non-empty list.")
        if len(queries) > 100:
            raise ValidationError(
                f"'queries' exceeds the batch maximum of 100 (got {len(queries)})."
            )
        if input_type not in ("dense_vector", "discrete_distribution"):
            raise ValidationError(
                f"'input_type' must be 'dense_vector' or 'discrete_distribution', "
                f"got {input_type!r}."
            )
        for i, q in enumerate(queries):
            _validate_vector_or_distribution(q, f"queries[{i}]")
        _validate_corpus(corpus)
        _validate_alpha(alpha)
        _validate_top_k(top_k, len(corpus))

        if n_bins is not None:
            if not isinstance(n_bins, int) or not (2 <= n_bins <= 512):
                raise ValidationError(
                    f"'n_bins' must be an integer between 2 and 512, got {n_bins!r}."
                )

        payload: dict[str, Any] = {
            "queries": [list(q) for q in queries],
            "corpus": [list(item) for item in corpus],
            "alpha": float(alpha),
            "top_k": top_k,
            "input_type": input_type,
        }
        if n_bins is not None:
            payload["n_bins"] = n_bins

        raw = self._post("/rank/batch", payload)
        return NMIRankingResponse(raw)

    def score_pair_nmi_cosine(
        self,
        vector_a: VectorOrDist,
        vector_b: VectorOrDist,
        alpha: float = 0.5,
        input_type: str = "dense_vector",
        n_bins: int | None = None,
    ) -> dict[str, float]:
        """
        Compute the hybrid NMI+Cosine score for a single pair.

        Use this when you need the decomposed NMI and Cosine components for
        a known pair rather than ranking a full corpus.

        Do NOT use this in a loop to simulate rank_by_nmi_cosine — each call
        incurs a full round-trip and lacks the server's vectorized NMI
        estimation across the corpus.

        Parameters
        ----------
        vector_a : list[float]
            First vector or distribution.
        vector_b : list[float]
            Second vector or distribution. Must match dimensionality of a.
        alpha : float
            NMI weight. See rank_by_nmi_cosine.
        input_type : str
            'dense_vector' or 'discrete_distribution'.
        n_bins : int or None
            Optional bin override for dense vectors.

        Returns
        -------
        dict with keys:
            'hybrid_score' (float) — alpha*NMI + (1-alpha)*Cosine
            'nmi_score' (float)
            'cosine_score' (float)
            'bias_correction_applied' (bool)
            'latency_ms' (float)

        Raises
        ------
        ValidationError, AuthenticationError, RateLimitError, SimilaritySearchError
        """
        if input_type not in ("dense_vector", "discrete_distribution"):
            raise ValidationError(
                f"'input_type' must be 'dense_vector' or 'discrete_distribution', "
                f"got {input_type!r}."
            )
        _validate_vector_or_distribution(vector_a, "vector_a")
        _validate_vector_or_distribution(vector_b, "vector_b")
        if len(vector_a) != len(vector_b):
            raise ValidationError(
                f"'vector_a' and 'vector_b' must have the same dimensionality "
                f"({len(vector_a)} vs {len(vector_b)})."
            )
        _validate_alpha(alpha)

        if n_bins is not None:
            if not isinstance(n_bins, int) or not (2 <= n_bins <= 512):
                raise ValidationError(
                    f"'n_bins' must be an integer between 2 and 512, got {n_bins!r}."
                )

        payload: dict[str, Any] = {
            "vector_a": list(vector_a),
            "vector_b": list(vector_b),
            "alpha": float(alpha),
            "input_type": input_type,
        }
        if n_bins is not None:
            payload["n_bins"] = n_bins

        return self._post("/score/pair", payload)

    def health(self) -> dict[str, Any]:
        """
        Return the API health status and version metadata.

        Use for liveness checks and to verify the API key resolves correctly
        before sending large corpora.

        Returns
        -------
        dict with keys 'status', 'version', 'region', 'latency_ms'.

        Raises
        ------
        AuthenticationError, SimilaritySearchError
        """
        return self._get("/health")

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_exc = SimilaritySearchError(
                    f"Request to {path} timed out after {self._timeout}s."
                )
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise last_exc from exc
            except httpx.RequestError as exc:
                raise SimilaritySearchError(
                    f"Network error on {path}: {exc}"
                ) from exc

            if response.status_code == 401:
                raise AuthenticationError(
                    "API key rejected by server. Verify your key at "
                    "https://similarity-search.nexus/dashboard.",
                    status_code=401,
                    response_body=self._safe_json(response),
                )
            if response.status_code == 422:
                body = self._safe_json(response)
                raise ValidationError(
                    f"Server rejected request parameters: {body}",
                    status_code=422,
                    response_body=body,
                )
            if response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                retry_after = float(retry_after_header) if retry_after_header else None
                if attempt < self._max_retries:
                    wait = retry_after if retry_after is not None else 2 ** attempt
                    time.sleep(wait)
                    last_exc = RateLimitError(
                        f"Rate limit exceeded on {path}. Retrying after {wait}s.",
                        retry_after=retry_after,
                    )
                    continue
                raise RateLimitError(
                    f"Rate limit exceeded on {path} after {self._max_retries} retries.",
                    retry_after=retry_after,
                )
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    last_exc = SimilaritySearchError(
                        f"Server error {response.status_code} on {path}.",
                        status_code=response.status_code,
                    )
                    continue
                raise SimilaritySearchError(
                    f"Server error {response.status_code} on {path} after "
                    f"{self._max_retries} retries.",
                    status_code=response.status_code,
                    response_body=self._safe_json(response),
                )
            if not response.is_success:
                body = self._safe_json(response)
                raise SimilaritySearchError(
                    f"Unexpected status {response.status_code} on {path}: {body}",
                    status_code=response.status_code,
                    response_body=body,
                )

            try:
                return response.json()
            except Exception as exc:
                raise SimilaritySearchError(
                    f"Server returned non-JSON body on {path}: "
                    f"{response.text[:200]!r}"
                ) from exc

        raise last_exc or SimilaritySearchError(
            f"All {self._max_retries} retries exhausted for {path}."
        )

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.get(url)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(
                f"Request to {path} timed out after {self._timeout}s."
            ) from exc
        except httpx.RequestError as exc:
            raise SimilaritySearchError(
                f"Network error on {path}: {exc}"
            ) from exc

        if response.status_code == 401:
            raise AuthenticationError(
                "API key rejected by server.",
                status_code=401,
                response_body=self._safe_json(response),
            )
        if not response.is_success:
            body = self._safe_json(response)
            raise SimilaritySearchError(
                f"Unexpected status {response.status_code} on {path}: {body}",
                status_code=response.status_code,
                response_body=body,
            )
        try:
            return response.json()
        except Exception as exc:
            raise SimilaritySearchError(
                f"Server returned non-JSON body on {path}: {response.text[:200]!r}"
            ) from exc

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text[:500]