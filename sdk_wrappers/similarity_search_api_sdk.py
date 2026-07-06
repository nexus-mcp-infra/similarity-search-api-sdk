import httpx
from typing import Any
import numpy as np


BASE_URL = "https://api.similarity-search.nexus/v1"
DEFAULT_TIMEOUT = 30.0


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    pass


class ValidationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    pass


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError("api_key must be a non-empty string")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing API key",
                status_code=response.status_code,
                response_body=response.text,
            )
        if response.status_code == 422:
            raise ValidationError(
                f"Payload validation failed: {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s",
                status_code=response.status_code,
                response_body=response.text,
            )
        if response.status_code >= 500:
            raise SimilaritySearchError(
                f"Server error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        if response.status_code >= 400:
            raise SimilaritySearchError(
                f"Client error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )

    def _validate_vector_list(self, name: str, value: Any) -> list[list[float]]:
        if value is None:
            raise ValidationError(f"'{name}' must not be None")
        if not isinstance(value, (list, np.ndarray)):
            raise ValidationError(
                f"'{name}' must be a list of numeric vectors, got {type(value).__name__}"
            )
        result = []
        for i, row in enumerate(value):
            if not isinstance(row, (list, np.ndarray)):
                raise ValidationError(
                    f"'{name}[{i}]' must be a numeric vector (list or array), got {type(row).__name__}"
                )
            row_list = row.tolist() if isinstance(row, np.ndarray) else list(row)
            if len(row_list) == 0:
                raise ValidationError(f"'{name}[{i}]' must not be an empty vector")
            for j, val in enumerate(row_list):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValidationError(
                        f"'{name}[{i}][{j}]' must be numeric, got {type(val).__name__}"
                    )
            result.append(row_list)
        if len(result) == 0:
            raise ValidationError(f"'{name}' must contain at least one vector")
        return result

    def compute_nmi_cosine_similarity(
        self,
        query_vectors: list[list[float]] | np.ndarray,
        candidate_vectors: list[list[float]] | np.ndarray,
        top_k: int = 10,
        nmi_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Compute entropy-weighted NMI+Cosine composite similarity scores.

        Parameters
        ----------
        query_vectors : list of numeric vectors to use as queries.
        candidate_vectors : list of numeric vectors to search over.
        top_k : number of top matches to return per query (1-100).
        nmi_bins : histogram bin count used server-side for NMI estimation (2-50).

        Returns
        -------
        dict with keys:
            - 'results': list of per-query ranked matches, each with
              'candidate_index', 'composite_score', 'nmi_score', 'cosine_score',
              'entropy_weights' (one per dimension)
            - 'meta': request metadata including dimension count and vector counts
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError(f"'top_k' must be an integer, got {type(top_k).__name__}")
        if top_k < 1 or top_k > 100:
            raise ValidationError(f"'top_k' must be between 1 and 100, got {top_k}")
        if not isinstance(nmi_bins, int) or isinstance(nmi_bins, bool):
            raise ValidationError(f"'nmi_bins' must be an integer, got {type(nmi_bins).__name__}")
        if nmi_bins < 2 or nmi_bins > 50:
            raise ValidationError(f"'nmi_bins' must be between 2 and 50, got {nmi_bins}")

        query_list = self._validate_vector_list("query_vectors", query_vectors)
        candidate_list = self._validate_vector_list("candidate_vectors", candidate_vectors)

        dim_q = len(query_list[0])
        dim_c = len(candidate_list[0])
        if dim_q != dim_c:
            raise ValidationError(
                f"query_vectors and candidate_vectors must have the same dimensionality "
                f"(query dim={dim_q}, candidate dim={dim_c})"
            )
        for i, row in enumerate(query_list[1:], start=1):
            if len(row) != dim_q:
                raise ValidationError(
                    f"query_vectors[{i}] has dim {len(row)}, expected {dim_q}"
                )
        for i, row in enumerate(candidate_list[1:], start=1):
            if len(row) != dim_c:
                raise ValidationError(
                    f"candidate_vectors[{i}] has dim {len(row)}, expected {dim_c}"
                )

        payload = {
            "query_vectors": query_list,
            "candidate_vectors": candidate_list,
            "top_k": top_k,
            "nmi_bins": nmi_bins,
        }

        try:
            response = self._http.post(
                f"{self._base_url}/similarity/nmi-cosine",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(
                f"Request timed out after {self._http.timeout}s: {exc}"
            ) from exc
        except httpx.NetworkError as exc:
            raise SimilaritySearchError(f"Network error: {exc}") from exc

        self._raise_for_status(response)
        return response.json()

    def rank_by_composite_score(
        self,
        anchor_vector: list[float] | np.ndarray,
        candidate_vectors: list[list[float]] | np.ndarray,
        top_k: int = 10,
        nmi_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Rank a flat list of candidates against a single anchor using NMI+Cosine.
        Convenience wrapper over compute_nmi_cosine_similarity for the single-query case.

        Parameters
        ----------
        anchor_vector : single query vector.
        candidate_vectors : list of vectors to rank against the anchor.
        top_k : number of top-ranked candidates to return (1-100).
        nmi_bins : histogram bins for server-side NMI estimation (2-50).

        Returns
        -------
        dict with keys:
            - 'ranked': ordered list of dicts with 'candidate_index',
              'composite_score', 'nmi_score', 'cosine_score', 'entropy_weights'
            - 'meta': request metadata
        """
        if anchor_vector is None:
            raise ValidationError("'anchor_vector' must not be None")
        if isinstance(anchor_vector, np.ndarray):
            anchor_vector = anchor_vector.tolist()
        if not isinstance(anchor_vector, list) or len(anchor_vector) == 0:
            raise ValidationError(
                f"'anchor_vector' must be a non-empty list of floats, got {type(anchor_vector).__name__}"
            )
        for i, val in enumerate(anchor_vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(
                    f"'anchor_vector[{i}]' must be numeric, got {type(val).__name__}"
                )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError(f"'top_k' must be an integer, got {type(top_k).__name__}")
        if top_k < 1 or top_k > 100:
            raise ValidationError(f"'top_k' must be between 1 and 100, got {top_k}")
        if not isinstance(nmi_bins, int) or isinstance(nmi_bins, bool):
            raise ValidationError(f"'nmi_bins' must be an integer, got {type(nmi_bins).__name__}")
        if nmi_bins < 2 or nmi_bins > 50:
            raise ValidationError(f"'nmi_bins' must be between 2 and 50, got {nmi_bins}")

        candidate_list = self._validate_vector_list("candidate_vectors", candidate_vectors)

        dim_a = len(anchor_vector)
        dim_c = len(candidate_list[0])
        if dim_a != dim_c:
            raise ValidationError(
                f"anchor_vector dim={dim_a} does not match candidate_vectors dim={dim_c}"
            )

        payload = {
            "anchor_vector": anchor_vector,
            "candidate_vectors": candidate_list,
            "top_k": top_k,
            "nmi_bins": nmi_bins,
        }

        try:
            response = self._http.post(
                f"{self._base_url}/similarity/rank-by-composite",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(
                f"Request timed out after {self._http.timeout}s: {exc}"
            ) from exc
        except httpx.NetworkError as exc:
            raise SimilaritySearchError(f"Network error: {exc}") from exc

        self._raise_for_status(response)
        return response.json()

    def explain_composite_score(
        self,
        vector_a: list[float] | np.ndarray,
        vector_b: list[float] | np.ndarray,
        nmi_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Return a per-dimension decomposition of the NMI+Cosine composite score
        between two vectors, including entropy weights H(Xi) and dimension-level
        NMI and cosine contributions. Use this to audit why two vectors score
        as they do, or to identify which dimensions drive the composite score.

        Do NOT use this for bulk ranking — use rank_by_composite_score instead.

        Parameters
        ----------
        vector_a, vector_b : the two vectors to compare (must have equal length).
        nmi_bins : histogram bins for NMI estimation (2-50).

        Returns
        -------
        dict with keys:
            - 'composite_score': float, final entropy-weighted NMI+Cosine score
            - 'nmi_score': float, marginal NMI component
            - 'cosine_score': float, marginal cosine component
            - 'dimension_breakdown': list of per-dimension dicts with
              'dimension_index', 'entropy_weight', 'nmi_contribution',
              'cosine_contribution'
            - 'meta': total entropy, dimension count, nmi_bins used
        """
        if vector_a is None:
            raise ValidationError("'vector_a' must not be None")
        if vector_b is None:
            raise ValidationError("'vector_b' must not be None")
        if isinstance(vector_a, np.ndarray):
            vector_a = vector_a.tolist()
        if isinstance(vector_b, np.ndarray):
            vector_b = vector_b.tolist()
        if not isinstance(vector_a, list) or len(vector_a) == 0:
            raise ValidationError(
                f"'vector_a' must be a non-empty list of floats, got {type(vector_a).__name__}"
            )
        if not isinstance(vector_b, list) or len(vector_b) == 0:
            raise ValidationError(
                f"'vector_b' must be a non-empty list of floats, got {type(vector_b).__name__}"
            )
        if len(vector_a) != len(vector_b):
            raise ValidationError(
                f"vector_a (dim={len(vector_a)}) and vector_b (dim={len(vector_b)}) must have equal length"
            )
        for i, val in enumerate(vector_a):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"'vector_a[{i}]' must be numeric, got {type(val).__name__}")
        for i, val in enumerate(vector_b):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"'vector_b[{i}]' must be numeric, got {type(val).__name__}")
        if not isinstance(nmi_bins, int) or isinstance(nmi_bins, bool):
            raise ValidationError(f"'nmi_bins' must be an integer, got {type(nmi_bins).__name__}")
        if nmi_bins < 2 or nmi_bins > 50:
            raise ValidationError(f"'nmi_bins' must be between 2 and 50, got {nmi_bins}")

        payload = {
            "vector_a": vector_a,
            "vector_b": vector_b,
            "nmi_bins": nmi_bins,
        }

        try:
            response = self._http.post(
                f"{self._base_url}/similarity/explain",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(
                f"Request timed out after {self._http.timeout}s: {exc}"
            ) from exc
        except httpx.NetworkError as exc:
            raise SimilaritySearchError(f"Network error: {exc}") from exc

        self._raise_for_status(response)
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()