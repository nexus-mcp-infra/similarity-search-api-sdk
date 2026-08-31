# --- PATCH sdk_route_grounding_manual_backfill ---
# similarity-search-api-sdk, sdk_wrappers/similarity_search_api_sdk.py
#
# Regrounded contra el codigo real deployado (core/similarity_search_api_api.py
# / https://similarity-search-api-production.up.railway.app/openapi.json),
# mismo principio que sdk_route_grounding en el repo NEXUS principal
# (CLAUDE.md SS9.43) -- esta build en particular es anterior a ese fix
# (deployada 2026-07-18) y nunca lo recibio (CLAUDE.md SS9.53).
#
# 3 bugs reales encontrados, los 3 corregidos aca:
#   1. SIMILARITY_SEARCH_BASE_URL apuntaba a un dominio que nunca existio
#      (NXDOMAIN, confirmado con nslookup antes de este patch).
#   2. Las 3 rutas de negocio eran inventadas ("/search", "/alpha",
#      "/score") -- las reales son "/similarity/search",
#      "/similarity/calibrate-alpha/v1", "/similarity/batch-score".
#   3. Header de auth incorrecto ("Authorization: Bearer") -- el server
#      real exige "X-API-Key" (confirmado contra
#      core/similarity_search_api_api.py: APIKeyHeader(name="X-API-Key"),
#      y contra components.securitySchemes del openapi.json real).
#
# Ademas de rutas/dominio/auth, los payloads de los 3 metodos de negocio
# nunca fueron groundeados contra los modelos Pydantic reales (mismo tipo
# de gap que sdk_field_mismatch/readme_field_mismatch shadow-mode en el
# repo principal, CLAUDE.md SS9.15/9.42 -- aca no hay gate que lo
# hubiera atrapado porque este repo no vive en output/cycle_archive, es
# un repo GitHub aparte). search()/score_pair() aceptaban 'query'/
# 'item_a'/'item_b' como str (busqueda semantica por texto) -- esa
# capacidad no existe en el servidor real, que SOLO acepta vectores ya
# calculados con un id de etiqueta (CorpusVector{id, vector}). Reescritos
# para reflejar exactamente lo que el servidor real puede hacer, en vez
# de solo cambiar las URLs y dejar cada llamada real fallando con 422 en
# lugar de 404/timeout.
#
# Sin esto, el 100% de los metodos de negocio del SDK fallaban con
# cualquier config default -- confirmado, no solo sospechado (CLAUDE.md
# SS9.53). Mitigante ya vigente: nunca se publico a PyPI (404 confirmado
# en pypi.org), asi que este bug nunca llego a un consumidor real.

import httpx
import time
from typing import Any

SIMILARITY_SEARCH_BASE_URL = "https://similarity-search-api-production.up.railway.app"
SIMILARITY_SEARCH_DEFAULT_TIMEOUT = 30.0
SIMILARITY_SEARCH_MAX_RETRIES = 3
SIMILARITY_SEARCH_RETRY_BACKOFF = 1.5


class SimilaritySearchAuthError(Exception):
    pass


class SimilaritySearchValidationError(Exception):
    pass


class SimilaritySearchRateLimitError(Exception):
    pass


class SimilaritySearchAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _validate_corpus_items(items: list[Any], param_name: str) -> None:
    if not isinstance(items, list):
        raise SimilaritySearchValidationError(
            f"'{param_name}' must be a list, got {type(items).__name__}"
        )
    if len(items) == 0:
        raise SimilaritySearchValidationError(
            f"'{param_name}' must contain at least one item"
        )
    if len(items) > 500_000:
        raise SimilaritySearchValidationError(
            f"'{param_name}' exceeds the 500,000-item corpus limit (got {len(items)})"
        )
    for i, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SimilaritySearchValidationError(
                f"'{param_name}[{i}]' must be an (id, vector) tuple, got {type(item).__name__}"
            )
        item_id, vector = item
        if not isinstance(item_id, str) or not item_id:
            raise SimilaritySearchValidationError(
                f"'{param_name}[{i}][0]' (id) must be a non-empty string"
            )
        if not isinstance(vector, list) or not vector:
            raise SimilaritySearchValidationError(
                f"'{param_name}[{i}][1]' (vector) must be a non-empty list of floats"
            )


def _validate_vector(vector: list[float], param_name: str) -> None:
    if not isinstance(vector, list) or not vector:
        raise SimilaritySearchValidationError(
            f"'{param_name}' must be a non-empty list of floats"
        )


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int):
        raise SimilaritySearchValidationError(
            f"'top_k' must be an int, got {type(top_k).__name__}"
        )
    if top_k < 1 or top_k > 1000:
        raise SimilaritySearchValidationError(
            f"'top_k' must be between 1 and 1000 (got {top_k})"
        )


def _validate_alpha(alpha: float | None) -> None:
    if alpha is None:
        return
    if not isinstance(alpha, (int, float)):
        raise SimilaritySearchValidationError(
            f"'alpha' must be a float between 0.0 and 1.0, got {type(alpha).__name__}"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise SimilaritySearchValidationError(
            f"'alpha' must be between 0.0 and 1.0 (got {alpha})"
        )


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = SIMILARITY_SEARCH_BASE_URL,
        timeout: float = SIMILARITY_SEARCH_DEFAULT_TIMEOUT,
        max_retries: int = SIMILARITY_SEARCH_MAX_RETRIES,
    ):
        # --- PATCH sdk_x402_only_auth_regrounding ---
        # api_key is optional as of repo commit 4e09c52 (2026-08-25): the
        # server dropped the X-API-Key gate on search()/compute_calibrated_alpha()/
        # score_pair(), x402 payment alone is sufficient now, and Stripe metered
        # billing explicitly excludes these 3 routes
        # (_NEXUS_BILLING_EXCLUDED_PATHS in core/similarity_search_api_api.py).
        # Still sent when the caller does pass one, for forward-compat with any
        # future re-gating (e.g. the deprecated /similarity/calibrate-alpha stub,
        # which kept its gate).
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "similarity-search-sdk-python/1.0.0",
        }
        if api_key:
            headers["X-API-Key"] = api_key
        self._http = httpx.Client(headers=headers, timeout=self._timeout)

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        last_exception: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_exception = SimilaritySearchAPIError(
                    f"Request to {endpoint} timed out after {self._timeout}s (attempt {attempt + 1}/{self._max_retries})"
                )
                time.sleep(SIMILARITY_SEARCH_RETRY_BACKOFF ** attempt)
                continue
            except httpx.RequestError as exc:
                raise SimilaritySearchAPIError(
                    f"Network error reaching {endpoint}: {exc}"
                ) from exc

            if response.status_code == 401:
                raise SimilaritySearchAuthError(
                    "Invalid or expired API key. Verify the key passed to Client(api_key=...)"
                )
            if response.status_code == 422:
                raise SimilaritySearchValidationError(
                    f"Server rejected the request payload: {response.text}"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise SimilaritySearchRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after}s. "
                    "Consider batching requests or upgrading your plan."
                )
            if response.status_code >= 500:
                last_exception = SimilaritySearchAPIError(
                    f"Server error {response.status_code} on {endpoint} (attempt {attempt + 1}/{self._max_retries}): {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
                time.sleep(SIMILARITY_SEARCH_RETRY_BACKOFF ** attempt)
                continue
            if not response.is_success:
                raise SimilaritySearchAPIError(
                    f"Unexpected status {response.status_code} from {endpoint}: {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            try:
                return response.json()
            except Exception as exc:
                raise SimilaritySearchAPIError(
                    f"Could not parse JSON response from {endpoint}: {response.text}"
                ) from exc

        raise last_exception or SimilaritySearchAPIError(
            f"All {self._max_retries} attempts to {endpoint} failed"
        )

    def main_method(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise SimilaritySearchValidationError(
                f"'data' must be a dict, got {type(data).__name__}. "
                "Use search(), compute_calibrated_alpha(), or score_pair() for typed calls."
            )
        return self._post_with_retry("/similarity/search", data)

    def search(
        self,
        query_vector: list[float],
        corpus: list[tuple[str, list[float]]],
        query_id: str = "query",
        top_k: int = 10,
        nmi_bins: int | None = None,
        alpha_override: float | None = None,
    ) -> dict[str, Any]:
        """
        query_vector: the vector to search with.
        corpus: list of (id, vector) tuples to search over -- the real
            server has no text-embedding step, every item (query
            included) must already be a numeric vector.
        """
        _validate_vector(query_vector, "query_vector")
        _validate_corpus_items(corpus, "corpus")
        _validate_top_k(top_k)
        _validate_alpha(alpha_override)

        payload: dict[str, Any] = {
            "query": {"id": query_id, "vector": query_vector},
            "corpus": [{"id": cid, "vector": vec} for cid, vec in corpus],
            "top_k": top_k,
        }
        if nmi_bins is not None:
            payload["nmi_bins"] = nmi_bins
        if alpha_override is not None:
            payload["alpha_override"] = float(alpha_override)

        return self._post_with_retry("/similarity/search", payload)

    def rank_corpus_by_nmi_cosine_fusion(
        self,
        query_vector: list[float],
        corpus: list[tuple[str, list[float]]],
        query_id: str = "query",
        top_k: int = 10,
        nmi_bins: int | None = None,
        alpha_override: float | None = None,
    ) -> list[dict[str, Any]]:
        response = self.search(
            query_vector=query_vector,
            corpus=corpus,
            query_id=query_id,
            top_k=top_k,
            nmi_bins=nmi_bins,
            alpha_override=alpha_override,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise SimilaritySearchAPIError(
                f"Unexpected response shape from /similarity/search: missing 'results' list. Got keys: {list(response.keys())}"
            )
        return results

    def compute_calibrated_alpha(
        self,
        corpus: list[tuple[str, list[float]]],
        nmi_bins: int | None = None,
    ) -> dict[str, Any]:
        """
        Calls /similarity/calibrate-alpha/v1 -- the unversioned
        /similarity/calibrate-alpha always returns 501 on the real
        server (see core/similarity_search_api_api.py), the /v1 path
        is the only usable one.
        """
        _validate_corpus_items(corpus, "corpus")
        payload: dict[str, Any] = {
            "corpus": [{"id": cid, "vector": vec} for cid, vec in corpus],
        }
        if nmi_bins is not None:
            payload["nmi_bins"] = nmi_bins
        return self._post_with_retry("/similarity/calibrate-alpha/v1", payload)

    def score_pair(
        self,
        vector_a: list[float],
        vector_b: list[float],
        alpha: float = 0.5,
        nmi_bins: int | None = None,
    ) -> dict[str, Any]:
        """
        The real server has no single-pair scoring endpoint -- this
        calls /similarity/batch-score with a single pair and unwraps
        the one score, which is the only real equivalent available.
        """
        _validate_vector(vector_a, "vector_a")
        _validate_vector(vector_b, "vector_b")
        _validate_alpha(alpha)

        payload: dict[str, Any] = {
            "pairs": [[vector_a, vector_b]],
            "alpha": float(alpha),
        }
        if nmi_bins is not None:
            payload["nmi_bins"] = nmi_bins

        response = self._post_with_retry("/similarity/batch-score", payload)
        scores = response.get("scores")
        if not isinstance(scores, list) or not scores:
            raise SimilaritySearchAPIError(
                f"Unexpected response shape from /similarity/batch-score: missing 'scores' list. Got keys: {list(response.keys())}"
            )
        return {
            "score": scores[0],
            "alpha_used": response.get("alpha_used"),
            "latency_ms": response.get("latency_ms"),
        }

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
