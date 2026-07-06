import httpx
import time
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.similaritysearch.io/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_VECTOR_DIMENSION = 16384
MAX_BATCH_CANDIDATES = 10000


class SimilaritySearchAuthError(Exception):
    pass


class SimilaritySearchValidationError(Exception):
    pass


class SimilaritySearchAPIError(Exception):
    def __init__(self, message: str, status_code: int, response_body: dict):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class SimilaritySearchTimeoutError(Exception):
    pass


class NMICosineResult:
    def __init__(self, raw: dict):
        self._raw = raw

    @property
    def composite_score(self) -> float:
        return float(self._raw["composite_score"])

    @property
    def cosine_component(self) -> float:
        return float(self._raw["cosine_component"])

    @property
    def nmi_component(self) -> float:
        return float(self._raw["nmi_component"])

    @property
    def alpha(self) -> float:
        return float(self._raw["alpha"])

    @property
    def latency_ms(self) -> float:
        return float(self._raw.get("latency_ms", 0.0))

    @property
    def domain_tag(self) -> Optional[str]:
        return self._raw.get("domain_tag")

    def to_dict(self) -> dict:
        return dict(self._raw)

    def __repr__(self) -> str:
        return (
            f"NMICosineResult(composite={self.composite_score:.4f}, "
            f"cosine={self.cosine_component:.4f}, "
            f"nmi={self.nmi_component:.4f}, "
            f"alpha={self.alpha:.2f})"
        )


class NMICosineBatchResult:
    def __init__(self, raw: dict):
        self._raw = raw

    @property
    def rankings(self) -> list[dict]:
        return self._raw["rankings"]

    @property
    def query_latency_ms(self) -> float:
        return float(self._raw.get("query_latency_ms", 0.0))

    @property
    def candidate_count(self) -> int:
        return int(self._raw["candidate_count"])

    @property
    def top_k(self) -> int:
        return int(self._raw["top_k"])

    def to_dict(self) -> dict:
        return dict(self._raw)

    def __repr__(self) -> str:
        return (
            f"NMICosineBatchResult(candidates={self.candidate_count}, "
            f"top_k={self.top_k}, "
            f"latency_ms={self.query_latency_ms:.1f})"
        )


class AlphaRecommendationResult:
    def __init__(self, raw: dict):
        self._raw = raw

    @property
    def recommended_alpha(self) -> float:
        return float(self._raw["recommended_alpha"])

    @property
    def confidence(self) -> float:
        return float(self._raw["confidence"])

    @property
    def domain_tag(self) -> str:
        return self._raw["domain_tag"]

    @property
    def sample_count(self) -> int:
        return int(self._raw["sample_count"])

    def to_dict(self) -> dict:
        return dict(self._raw)

    def __repr__(self) -> str:
        return (
            f"AlphaRecommendationResult(domain='{self.domain_tag}', "
            f"alpha={self.recommended_alpha:.3f}, "
            f"confidence={self.confidence:.3f}, "
            f"n={self.sample_count})"
        )


def _validate_vector(vec: list, name: str) -> None:
    if vec is None:
        raise SimilaritySearchValidationError(
            f"'{name}' must not be None — provide a list of numeric values."
        )
    if not isinstance(vec, list):
        raise SimilaritySearchValidationError(
            f"'{name}' must be a list, got {type(vec).__name__}."
        )
    if len(vec) == 0:
        raise SimilaritySearchValidationError(
            f"'{name}' must not be empty — provide at least one dimension."
        )
    if len(vec) > MAX_VECTOR_DIMENSION:
        raise SimilaritySearchValidationError(
            f"'{name}' has {len(vec)} dimensions, which exceeds the maximum of "
            f"{MAX_VECTOR_DIMENSION}. Reduce dimensionality before calling the API."
        )
    for i, val in enumerate(vec):
        if not isinstance(val, (int, float)):
            raise SimilaritySearchValidationError(
                f"'{name}[{i}]' is {type(val).__name__}, expected int or float."
            )


def _validate_alpha(alpha: float) -> None:
    if not isinstance(alpha, (int, float)):
        raise SimilaritySearchValidationError(
            f"'alpha' must be a float between 0.0 and 1.0, got {type(alpha).__name__}."
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise SimilaritySearchValidationError(
            f"'alpha' must be between 0.0 and 1.0 inclusive, got {alpha}. "
            "alpha=1.0 is pure cosine; alpha=0.0 is pure NMI."
        )


def _validate_api_key(api_key: str) -> None:
    if not api_key or not isinstance(api_key, str):
        raise SimilaritySearchAuthError(
            "API key is missing or empty. Instantiate Client(api_key='your_key') "
            "or set the SIMILARITY_SEARCH_API_KEY environment variable."
        )
    if len(api_key.strip()) == 0:
        raise SimilaritySearchAuthError(
            "API key is blank (whitespace only). Provide a valid key."
        )


class Client:
    """
    Thin HTTP wrapper over the Similarity Search API.

    Exposes three operations:
      - score_nmi_cosine_pair: compute the NMI-Cosine composite for a single vector pair
      - rank_candidates_nmi_cosine: rank a list of candidate vectors against a query
      - get_alpha_recommendation: retrieve the data-flywheel-derived optimal alpha for a domain

    All calls are stateless — no index is maintained client-side.
    Pricing is per-call; see https://api.similaritysearch.io/pricing
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ):
        _validate_api_key(api_key)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise SimilaritySearchValidationError(
                f"'timeout' must be a positive number, got {timeout}."
            )
        if not isinstance(max_retries, int) or max_retries < 0:
            raise SimilaritySearchValidationError(
                f"'max_retries' must be a non-negative integer, got {max_retries}."
            )

        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=httpx.Timeout(timeout),
        )

    def score_nmi_cosine_pair(
        self,
        vector_u: list[Union[int, float]],
        vector_v: list[Union[int, float]],
        alpha: float = 0.5,
        domain_tag: Optional[str] = None,
    ) -> NMICosineResult:
        """
        Compute the NMI-Cosine composite score for a single pair of vectors.

        S = alpha * cosine(u, v) + (1 - alpha) * NMI(u, v)

        NMI is estimated via Freedman-Diaconis binning on the server side.
        Use this when you need a single pairwise score — for batch ranking
        use rank_candidates_nmi_cosine instead.

        Parameters
        ----------
        vector_u : list of float
            Query vector, 1 to 16384 dimensions.
        vector_v : list of float
            Candidate vector, same dimension as vector_u.
        alpha : float, default 0.5
            Weight of the cosine component in [0.0, 1.0].
            alpha=1.0 -> pure cosine similarity.
            alpha=0.0 -> pure NMI.
        domain_tag : str, optional
            Semantic label for the domain (e.g. 'recommendation', 'dedup').
            Used to feed the data flywheel for alpha suggestions.

        Returns
        -------
        NMICosineResult
            Composite score and both component scores.

        Raises
        ------
        SimilaritySearchValidationError
            If vectors differ in length, contain non-numeric values, or alpha is out of range.
        SimilaritySearchAuthError
            If the API key is invalid or expired.
        SimilaritySearchAPIError
            If the server returns a 4xx or 5xx response.
        SimilaritySearchTimeoutError
            If the request exceeds the configured timeout.
        """
        _validate_vector(vector_u, "vector_u")
        _validate_vector(vector_v, "vector_v")
        _validate_alpha(alpha)

        if len(vector_u) != len(vector_v):
            raise SimilaritySearchValidationError(
                f"'vector_u' has {len(vector_u)} dimensions but 'vector_v' has "
                f"{len(vector_v)}. Both vectors must share the same dimensionality."
            )

        payload: dict = {
            "vector_u": vector_u,
            "vector_v": vector_v,
            "alpha": float(alpha),
        }
        if domain_tag is not None:
            if not isinstance(domain_tag, str) or len(domain_tag.strip()) == 0:
                raise SimilaritySearchValidationError(
                    "'domain_tag' must be a non-empty string when provided."
                )
            payload["domain_tag"] = domain_tag.strip()

        raw = self._post("/score/pair", payload)
        return NMICosineResult(raw)

    def rank_candidates_nmi_cosine(
        self,
        query_vector: list[Union[int, float]],
        candidate_vectors: list[list[Union[int, float]]],
        alpha: float = 0.5,
        top_k: int = 10,
        domain_tag: Optional[str] = None,
    ) -> NMICosineBatchResult:
        """
        Rank a list of candidate vectors against a query vector using the NMI-Cosine score.

        Returns up to top_k candidates sorted by descending composite score.
        All computation is stateless per-request — no persistent index is used.

        Do NOT use this for datasets above 500k candidates. For larger corpora,
        use a dedicated ANN index and call score_nmi_cosine_pair for re-ranking.

        Parameters
        ----------
        query_vector : list of float
            The reference vector to rank against, 1 to 16384 dimensions.
        candidate_vectors : list of list of float
            Vectors to score and rank. Max 10000 candidates per call.
            All must share the same dimension as query_vector.
        alpha : float, default 0.5
            Weight of the cosine component in [0.0, 1.0].
        top_k : int, default 10
            Number of top results to return. Must be >= 1 and <= len(candidate_vectors).
        domain_tag : str, optional
            Semantic label for the domain, fed to the flywheel.

        Returns
        -------
        NMICosineBatchResult
            Ranked list of candidates with composite, cosine, and NMI scores per entry.

        Raises
        ------
        SimilaritySearchValidationError
            On dimension mismatch, empty inputs, or invalid parameter ranges.
        SimilaritySearchAuthError
            If the API key is invalid or expired.
        SimilaritySearchAPIError
            If the server returns a 4xx or 5xx response.
        SimilaritySearchTimeoutError
            If the request exceeds the configured timeout.
        """
        _validate_vector(query_vector, "query_vector")
        _validate_alpha(alpha)

        if candidate_vectors is None:
            raise SimilaritySearchValidationError(
                "'candidate_vectors' must not be None — provide a list of numeric lists."
            )
        if not isinstance(candidate_vectors, list):
            raise SimilaritySearchValidationError(
                f"'candidate_vectors' must be a list, got {type(candidate_vectors).__name__}."
            )
        if len(candidate_vectors) == 0:
            raise SimilaritySearchValidationError(
                "'candidate_vectors' is empty — provide at least one candidate vector."
            )
        if len(candidate_vectors) > MAX_BATCH_CANDIDATES:
            raise SimilaritySearchValidationError(
                f"'candidate_vectors' contains {len(candidate_vectors)} items, "
                f"exceeding the maximum of {MAX_BATCH_CANDIDATES} per call. "
                "Split your dataset into smaller batches."
            )

        query_dim = len(query_vector)
        for idx, cand in enumerate(candidate_vectors):
            _validate_vector(cand, f"candidate_vectors[{idx}]")
            if len(cand) != query_dim:
                raise SimilaritySearchValidationError(
                    f"'candidate_vectors[{idx}]' has {len(cand)} dimensions but "
                    f"'query_vector' has {query_dim}. All vectors must share the same dimension."
                )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise SimilaritySearchValidationError(
                f"'top_k' must be an integer, got {type(top_k).__name__}."
            )
        if top_k < 1:
            raise SimilaritySearchValidationError(
                f"'top_k' must be >= 1, got {top_k}."
            )
        if top_k > len(candidate_vectors):
            raise SimilaritySearchValidationError(
                f"'top_k' ({top_k}) exceeds the number of candidates ({len(candidate_vectors)}). "
                "Set top_k <= len(candidate_vectors)."
            )

        payload: dict = {
            "query_vector": query_vector,
            "candidate_vectors": candidate_vectors,
            "alpha": float(alpha),
            "top_k": top_k,
        }
        if domain_tag is not None:
            if not isinstance(domain_tag, str) or len(domain_tag.strip()) == 0:
                raise SimilaritySearchValidationError(
                    "'domain_tag' must be a non-empty string when provided."
                )
            payload["domain_tag"] = domain_tag.strip()

        raw = self._post("/score/rank", payload)
        return NMICosineBatchResult(raw)

    def get_alpha_recommendation(
        self,
        domain_tag: str,
        vector_dimension: Optional[int] = None,
    ) -> AlphaRecommendationResult:
        """
        Retrieve the data-flywheel-derived optimal alpha for a given domain.

        The recommendation is derived from the meta-model trained on historical
        (alpha, dim, domain_tag, composite_score, latency_ms) tuples recorded
        across all API calls for that domain. Requires at minimum ~50 historical
        calls for the domain to produce a statistically meaningful recommendation.

        Do NOT use this to override alpha programmatically per-request without
        validating on a holdout set — use it as a starting point for manual tuning.

        Parameters
        ----------
        domain_tag : str
            The domain label to query (e.g. 'recommendation', 'semantic_dedup').
            Must match the tag used in prior score calls.
        vector_dimension : int, optional
            If provided, the recommendation is conditioned on this dimensionality.
            Must be between 1 and 16384.

        Returns
        -------
        AlphaRecommendationResult
            Recommended alpha, model confidence [0, 1], and sample count used.

        Raises
        ------
        SimilaritySearchValidationError
            If domain_tag is empty or vector_dimension is out of range.
        SimilaritySearchAuthError
            If the API key is invalid or expired.
        SimilaritySearchAPIError
            If the server returns 4xx (including 404 when domain has no history) or 5xx.
        SimilaritySearchTimeoutError
            If the request exceeds the configured timeout.
        """
        if not domain_tag or not isinstance(domain_tag, str):
            raise SimilaritySearchValidationError(
                "'domain_tag' must be a non-empty string — "
                "provide the same tag used in your score calls."
            )
        if len(domain_tag.strip()) == 0:
            raise SimilaritySearchValidationError(
                "'domain_tag' is blank (whitespace only). Provide a meaningful domain label."
            )

        params: dict = {"domain_tag": domain_tag.strip()}

        if vector_dimension is not None:
            if not isinstance(vector_dimension, int) or isinstance(vector_dimension, bool):
                raise SimilaritySearchValidationError(
                    f"'vector_dimension' must be an integer, got {type(vector_dimension).__name__}."
                )
            if not (1 <= vector_dimension <= MAX_VECTOR_DIMENSION):
                raise SimilaritySearchValidationError(
                    f"'vector_dimension' must be between 1 and {MAX_VECTOR_DIMENSION}, "
                    f"got {vector_dimension}."
                )
            params["vector_dimension"] = vector_dimension

        raw = self._get("/alpha/recommendation", params)
        return AlphaRecommendationResult(raw)

    def main_method(
        self,
        data: dict,
    ) -> NMICosineResult:
        """
        Canonical entry point matching the SDK contract: client.main_method(data).

        Delegates to score_nmi_cosine_pair. 'data' must be a dict with keys:
          - 'vector_u'   : list of float (required)
          - 'vector_v'   : list of float (required)
          - 'alpha'      : float in [0, 1] (optional, default 0.5)
          - 'domain_tag' : str (optional)

        Parameters
        ----------
        data : dict
            Request payload. See above for required and optional keys.

        Returns
        -------
        NMICosineResult

        Raises
        ------
        SimilaritySearchValidationError
            If 'data' is None, not a dict, or is missing required keys.
        """
        if data is None:
            raise SimilaritySearchValidationError(
                "'data' must not be None — provide a dict with 'vector_u' and 'vector_v'."
            )
        if not isinstance(data, dict):
            raise SimilaritySearchValidationError(
                f"'data' must be a dict, got {type(data).__name__}. "
                "Expected keys: 'vector_u', 'vector_v', optionally 'alpha' and 'domain_tag'."
            )
        if "vector_u" not in data:
            raise SimilaritySearchValidationError(
                "'data' is missing required key 'vector_u'."
            )
        if "vector_v" not in data:
            raise SimilaritySearchValidationError(
                "'data' is missing required key 'vector_v'."
            )

        return self.score_nmi_cosine_pair(
            vector_u=data["vector_u"],
            vector_v=data["vector_v"],
            alpha=data.get("alpha", 0.5),
            domain_tag=data.get("domain_tag"),
        )

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._http.post(url, json=payload)
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "POST %s -> %d in %.1fms (attempt %d)",
                    path, response.status_code, elapsed_ms, attempt + 1,
                )
                return self._parse_response(response)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Request to %s timed out (attempt %d/%d), retrying...",
                        path, attempt + 1, self._max_retries + 1,
                    )
                    continue
                raise SimilaritySearchTimeoutError(
                    f"POST {path} timed out after {self._timeout}s "
                    f"({self._max_retries + 1} attempt(s)). "
                    "Increase Client(timeout=...) or reduce payload size."
                ) from exc
            except httpx.RequestError as exc:
                raise SimilaritySearchAPIError(
                    f"Network error on POST {path}: {exc}",
                    status_code=0,
                    response_body={},
                ) from exc

        raise SimilaritySearchTimeoutError(
            f"POST {path} failed after {self._max_retries + 1} attempts."
        ) from last_exc

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._http.get(url, params=params)
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "GET %s -> %d in %.1fms (attempt %d)",
                    path, response.status_code, elapsed_ms, attempt + 1,
                )
                return self._parse_response(response)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Request to %s timed out (attempt %d/%d), retrying...",
                        path, attempt + 1, self._max_retries + 1,
                    )
                    continue
                raise SimilaritySearchTimeoutError(
                    f"GET {path} timed out after {self._timeout}s "
                    f"({self._max_retries + 1} attempt(s)). "
                    "Increase Client(timeout=...) or reduce payload size."
                ) from exc
            except httpx.RequestError as exc:
                raise SimilaritySearchAPIError(
                    f"Network error on GET {path}: {exc}",
                    status_code=0,
                    response_body={},
                ) from exc

        raise SimilaritySearchTimeoutError(
            f"GET {path} failed after {self._max_retries + 1} attempts."
        ) from last_exc

    def _parse_response(self, response: httpx.Response) -> dict:
        if response.status_code == 401:
            raise SimilaritySearchAuthError(
                "Authentication failed (HTTP 401). "
                "Check that your API key is valid and has not been revoked. "
                "Obtain a new key at https://api.similaritysearch.io/keys"
            )
        if response.status_code == 403:
            raise SimilaritySearchAuthError(
                "Access denied (HTTP 403). "
                "Your API key does not have permission for this operation."
            )

        try:
            body = response.json()
        except Exception:
            body = {}

        if response.status_code == 422:
            detail = body.get("detail", str(body))
            raise SimilaritySearchValidationError(
                f"Unprocessable entity (HTTP 422) — the server rejected the payload: {detail}. "
                "Verify vector dimensions, alpha range, and that all values are finite floats."
            )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise SimilaritySearchAPIError(
                f"Rate limit exceeded (HTTP 429). Retry after {retry_after}s. "
                "Contact sales@similaritysearch.io to raise your quota.",
                status_code=429,
                response_body=body,
            )

        if not response.is_success:
            error_msg = body.get("error") or body.get("detail") or str(body)
            raise SimilaritySearchAPIError(
                f"API error (HTTP {response.status_code}): {error_msg}",
                status_code=response.status_code,
                response_body=body,
            )

        return body

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args) -> None:
        self.close()