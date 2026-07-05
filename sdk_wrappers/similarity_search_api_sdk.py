from __future__ import annotations

import os
import time
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://api.nexus-similarity.io/v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5


class SimilaritySearchError(Exception):
    """Raised when the Similarity Search API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AuthenticationError(SimilaritySearchError):
    """Raised when the API key is missing or rejected (401/403)."""


class RateLimitError(SimilaritySearchError):
    """Raised when the API signals rate limiting (429)."""


class SimilaritySearchResult:
    """Structured response from a NMI-weighted cosine similarity query."""

    __slots__ = (
        "matches",
        "nmi_weights",
        "confidence_interval",
        "effective_dimensions",
        "request_id",
    )

    def __init__(self, payload: dict[str, Any]) -> None:
        self.matches: list[dict[str, Any]] = payload.get("matches", [])
        self.nmi_weights: list[float] = payload.get("nmi_weights", [])
        self.confidence_interval: dict[str, float] = payload.get(
            "confidence_interval", {}
        )
        self.effective_dimensions: int = payload.get("effective_dimensions", 0)
        self.request_id: str = payload.get("request_id", "")

    def __repr__(self) -> str:
        return (
            f"SimilaritySearchResult("
            f"matches={len(self.matches)}, "
            f"effective_dimensions={self.effective_dimensions}, "
            f"confidence_interval={self.confidence_interval}, "
            f"request_id={self.request_id!r})"
        )


class Client:
    """
    HTTP wrapper for the Similarity Search API.

    Performs NMI-weighted cosine similarity over a query vector (or tokenised
    text / tabular features) against a local corpus in a single stateless call.

    Args:
        api_key: Secret key issued by NEXUS. Falls back to the environment
            variable SIMILARITY_SEARCH_API_KEY when omitted.
        base_url: Override the production endpoint (useful for staging).
        timeout: Per-request timeout in seconds.
        max_retries: Number of automatic retries on 5xx or network errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        resolved_key = api_key or os.environ.get("SIMILARITY_SEARCH_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                401,
                "No API key provided. Pass api_key= or set the "
                "SIMILARITY_SEARCH_API_KEY environment variable.",
            )
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise AuthenticationError(
                401,
                "api_key must be a non-empty string.",
            )

        self._api_key = resolved_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "nexus-similarity-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def nmi_cosine_search(
        self,
        query: list[float] | list[str] | list[list[float]],
        corpus: list[list[float]] | list[str] | list[list[float]],
        top_k: int = 10,
        nmi_threshold: float = 0.05,
        domain_hint: str | None = None,
        include_nmi_weights: bool = False,
    ) -> SimilaritySearchResult:
        """
        Run NMI-filtered cosine similarity between *query* and *corpus*.

        Use this method when:
        - You have a single query and an in-memory corpus (up to ~100k items).
        - You need the NMI confidence interval for auditability or downstream
          thresholding.
        - You are working with raw vectors, tokenised text, or tabular feature
          rows without a pre-built index.

        Do NOT use this method when:
        - Your corpus exceeds 500k items — use a dedicated vector DB instead.
        - You need real-time streaming results — this call is synchronous.
        - You want to persist or index the corpus on the server side — this
          API is stateless by design.

        Args:
            query: A single feature vector (list of floats), a list of tokens
                (list of strings), or a single tabular feature row
                (list of floats). Must not be empty.
            corpus: A list of items in the same representation as *query*.
                Must contain at least one item and use a consistent type with
                *query*.
            top_k: Number of nearest neighbours to return. Must be between 1
                and 1000 (inclusive).
            nmi_threshold: Minimum NMI score for a feature dimension to be
                included in the cosine distance calculation. Lower values
                retain more dimensions; 0.0 disables NMI filtering. Must be
                in [0.0, 1.0].
            domain_hint: Optional string label (e.g. "biomedical", "ecommerce")
                used to select the matching empirical NMI distribution in the
                percentile backend, yielding tighter confidence intervals.
                Ignored if the domain has fewer than 1000 observations.
            include_nmi_weights: When True, the response includes the per-
                dimension NMI weights used during the search.

        Returns:
            SimilaritySearchResult with ranked matches, confidence interval,
            and effective dimensionality after NMI filtering.

        Raises:
            ValueError: On invalid or empty inputs.
            AuthenticationError: On 401/403 responses.
            RateLimitError: On 429 responses.
            SimilaritySearchError: On all other non-2xx responses.
        """
        self._validate_nmi_cosine_inputs(query, corpus, top_k, nmi_threshold)

        body: dict[str, Any] = {
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
            "nmi_threshold": nmi_threshold,
            "include_nmi_weights": include_nmi_weights,
        }
        if domain_hint is not None:
            if not isinstance(domain_hint, str) or not domain_hint.strip():
                raise ValueError("domain_hint must be a non-empty string when provided.")
            body["domain_hint"] = domain_hint.strip()

        response_payload = self._post_with_retries(
            endpoint="/search/nmi-cosine", body=body
        )
        return SimilaritySearchResult(response_payload)

    def main_method(
        self,
        data: dict[str, Any],
    ) -> SimilaritySearchResult:
        """
        Convenience entry-point that delegates to nmi_cosine_search using a
        structured dict payload.

        Expected keys in *data*:
            query (required): same as nmi_cosine_search query parameter.
            corpus (required): same as nmi_cosine_search corpus parameter.
            top_k (optional, default 10): int in [1, 1000].
            nmi_threshold (optional, default 0.05): float in [0.0, 1.0].
            domain_hint (optional): str.
            include_nmi_weights (optional, default False): bool.

        Raises:
            TypeError: When *data* is not a dict.
            KeyError: When required keys are absent.
        """
        if data is None:
            raise TypeError(
                "data must be a non-None dict. Received None."
            )
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}."
            )
        missing = [k for k in ("query", "corpus") if k not in data]
        if missing:
            raise KeyError(
                f"data is missing required key(s): {missing}. "
                "Provide both 'query' and 'corpus'."
            )

        return self.nmi_cosine_search(
            query=data["query"],
            corpus=data["corpus"],
            top_k=data.get("top_k", 10),
            nmi_threshold=data.get("nmi_threshold", 0.05),
            domain_hint=data.get("domain_hint"),
            include_nmi_weights=data.get("include_nmi_weights", False),
        )

    def health(self) -> dict[str, Any]:
        """
        Ping the API health endpoint.

        Returns a dict with at least {"status": "ok", "version": str}.
        Raises SimilaritySearchError on any non-2xx response.

        Use this to verify connectivity and credentials before a batch run.
        Do NOT poll this endpoint in a tight loop — it counts against your
        rate limit quota.
        """
        return self._get_with_retries("/health")

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _post_with_retries(
        self, endpoint: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http.post(url, json=body)
            except httpx.TimeoutException as exc:
                last_exc = exc
                self._backoff(attempt)
                continue
            except httpx.RequestError as exc:
                last_exc = exc
                self._backoff(attempt)
                continue

            if response.status_code < 300:
                return self._parse_json_response(response)

            if response.status_code in (401, 403):
                detail = self._extract_detail(response)
                raise AuthenticationError(response.status_code, detail)

            if response.status_code == 422:
                detail = self._extract_detail(response)
                raise ValueError(
                    f"API rejected the request payload (HTTP 422): {detail}"
                )

            if response.status_code == 429:
                detail = self._extract_detail(response)
                raise RateLimitError(429, detail)

            if response.status_code >= 500:
                last_exc = SimilaritySearchError(
                    response.status_code, self._extract_detail(response)
                )
                self._backoff(attempt)
                continue

            detail = self._extract_detail(response)
            raise SimilaritySearchError(response.status_code, detail)

        if last_exc is not None:
            if isinstance(last_exc, SimilaritySearchError):
                raise last_exc
            raise SimilaritySearchError(
                503,
                f"Request failed after {self._max_retries} attempts: {last_exc}",
            )
        raise SimilaritySearchError(503, "Request failed for unknown reason.")

    def _get_with_retries(self, endpoint: str) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http.get(url)
            except httpx.TimeoutException as exc:
                last_exc = exc
                self._backoff(attempt)
                continue
            except httpx.RequestError as exc:
                last_exc = exc
                self._backoff(attempt)
                continue

            if response.status_code < 300:
                return self._parse_json_response(response)

            if response.status_code in (401, 403):
                raise AuthenticationError(
                    response.status_code, self._extract_detail(response)
                )

            if response.status_code == 429:
                raise RateLimitError(429, self._extract_detail(response))

            if response.status_code >= 500:
                last_exc = SimilaritySearchError(
                    response.status_code, self._extract_detail(response)
                )
                self._backoff(attempt)
                continue

            raise SimilaritySearchError(
                response.status_code, self._extract_detail(response)
            )

        if last_exc is not None:
            if isinstance(last_exc, SimilaritySearchError):
                raise last_exc
            raise SimilaritySearchError(
                503,
                f"Request failed after {self._max_retries} attempts: {last_exc}",
            )
        raise SimilaritySearchError(503, "Request failed for unknown reason.")

    @staticmethod
    def _parse_json_response(response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except Exception as exc:
            raise SimilaritySearchError(
                response.status_code,
                f"API returned non-JSON body: {exc}",
            )

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("detail") or body.get("message") or body)
        except Exception:
            return response.text[:500] if response.text else "(empty body)"

    @staticmethod
    def _backoff(attempt: int) -> None:
        delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
        time.sleep(delay)

    @staticmethod
    def _validate_nmi_cosine_inputs(
        query: Any,
        corpus: Any,
        top_k: Any,
        nmi_threshold: Any,
    ) -> None:
        if query is None:
            raise ValueError("query must not be None.")
        if not isinstance(query, list) or len(query) == 0:
            raise ValueError(
                "query must be a non-empty list of floats, strings, or lists."
            )

        if corpus is None:
            raise ValueError("corpus must not be None.")
        if not isinstance(corpus, list) or len(corpus) == 0:
            raise ValueError(
                "corpus must be a non-empty list of items matching the type "
                "of query."
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(
                f"top_k must be an int, got {type(top_k).__name__}."
            )
        if not (1 <= top_k <= 1000):
            raise ValueError(
                f"top_k must be between 1 and 1000 inclusive, got {top_k}."
            )

        if not isinstance(nmi_threshold, (int, float)) or isinstance(
            nmi_threshold, bool
        ):
            raise TypeError(
                f"nmi_threshold must be a float, got {type(nmi_threshold).__name__}."
            )
        if not (0.0 <= float(nmi_threshold) <= 1.0):
            raise ValueError(
                f"nmi_threshold must be in [0.0, 1.0], got {nmi_threshold}."
            )