import httpx
import numpy as np
from typing import Union
import time


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int = None, response_body: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(SimilaritySearchError):
    pass


class SimilaritySearchClient:
    BASE_URL = "https://api.similaritysearch.io/v1"
    MAX_CORPUS_VECTORS = 10_000
    MAX_VECTOR_DIM = 4_096
    MAX_TEXT_LENGTH = 8_192
    MAX_QUERY_VECTORS = 100
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        timeout: float = None,
        max_retries: int = 3,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "api_key must be a non-empty string. Obtain one at https://similaritysearch.io/dashboard"
            )
        self._api_key = api_key
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
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

    def _request(self, method: str, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                response = self._http.request(method, url, json=payload)
            except httpx.TimeoutException as exc:
                last_exc = SimilaritySearchError(
                    f"Request timed out after {self._timeout}s (attempt {attempt + 1}/{self._max_retries})"
                )
                time.sleep(2 ** attempt)
                continue
            except httpx.RequestError as exc:
                last_exc = SimilaritySearchError(f"Network error: {exc}")
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    raise SimilaritySearchError(
                        "API returned non-JSON response",
                        status_code=response.status_code,
                        response_body={"raw": response.text[:500]},
                    )

            try:
                body = response.json()
            except Exception:
                body = {"detail": response.text[:500]}

            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid or expired API key. Check your credentials at https://similaritysearch.io/dashboard",
                    status_code=401,
                    response_body=body,
                )
            if response.status_code == 422:
                raise ValidationError(
                    f"Payload validation failed: {body.get('detail', body)}",
                    status_code=422,
                    response_body=body,
                )
            if response.status_code == 429:
                retry_after = None
                try:
                    retry_after = float(response.headers.get("Retry-After", 60))
                except (TypeError, ValueError):
                    retry_after = 60.0
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after}s.",
                    retry_after=retry_after,
                )
            if response.status_code >= 500:
                last_exc = SimilaritySearchError(
                    f"Server error {response.status_code}: {body.get('detail', 'unknown error')}",
                    status_code=response.status_code,
                    response_body=body,
                )
                time.sleep(2 ** attempt)
                continue

            raise SimilaritySearchError(
                f"Unexpected status {response.status_code}: {body.get('detail', body)}",
                status_code=response.status_code,
                response_body=body,
            )

        raise last_exc or SimilaritySearchError("Max retries exceeded with no successful response")

    def _validate_vector_list(self, vectors: list, label: str, max_count: int) -> list:
        if not isinstance(vectors, list) or len(vectors) == 0:
            raise ValidationError(f"'{label}' must be a non-empty list of numeric vectors")
        if len(vectors) > max_count:
            raise ValidationError(
                f"'{label}' exceeds maximum allowed count: {len(vectors)} > {max_count}"
            )
        result = []
        for i, vec in enumerate(vectors):
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            if not isinstance(vec, list) or len(vec) == 0:
                raise ValidationError(
                    f"'{label}[{i}]' must be a non-empty list of floats, got {type(vec).__name__}"
                )
            if len(vec) > self.MAX_VECTOR_DIM:
                raise ValidationError(
                    f"'{label}[{i}]' dimension {len(vec)} exceeds max allowed {self.MAX_VECTOR_DIM}"
                )
            try:
                vec = [float(v) for v in vec]
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"'{label}[{i}]' contains non-numeric values: {exc}"
                )
            result.append(vec)
        return result

    def _validate_text_list(self, texts: list, label: str, max_count: int) -> list:
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValidationError(f"'{label}' must be a non-empty list of strings")
        if len(texts) > max_count:
            raise ValidationError(
                f"'{label}' exceeds maximum allowed count: {len(texts)} > {max_count}"
            )
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValidationError(
                    f"'{label}[{i}]' must be a string, got {type(t).__name__}"
                )
            if len(t) == 0:
                raise ValidationError(f"'{label}[{i}]' must not be empty")
            if len(t) > self.MAX_TEXT_LENGTH:
                raise ValidationError(
                    f"'{label}[{i}]' length {len(t)} exceeds max {self.MAX_TEXT_LENGTH} chars"
                )
        return texts

    def ranked_vector_similarity(
        self,
        query_vectors: list,
        corpus_vectors: list,
        top_k: int = 10,
        nmi_weight_override: float = None,
    ) -> dict:
        """
        Rank corpus_vectors against query_vectors using NMI-cosine weighted fusion.
        The adaptive entropy weight is computed server-side from the full corpus batch.

        query_vectors: list of numeric lists or np.ndarray, shape (Q, D)
        corpus_vectors: list of numeric lists or np.ndarray, shape (C, D), max 10_000
        top_k: number of top results to return per query, between 1 and min(C, 500)
        nmi_weight_override: optional float in [0.0, 1.0] to pin w_nmi instead of entropy-derived value;
                             use only for ablation studies, not production ranking

        Returns dict with keys: results (list of ranked matches per query), metadata (entropy, w_nmi used)
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError("top_k must be a positive integer")
        if top_k > 500:
            raise ValidationError("top_k cannot exceed 500")
        if nmi_weight_override is not None:
            if not isinstance(nmi_weight_override, (int, float)):
                raise ValidationError("nmi_weight_override must be a float in [0.0, 1.0]")
            if not (0.0 <= float(nmi_weight_override) <= 1.0):
                raise ValidationError("nmi_weight_override must be in [0.0, 1.0]")

        query_vectors = self._validate_vector_list(query_vectors, "query_vectors", self.MAX_QUERY_VECTORS)
        corpus_vectors = self._validate_vector_list(corpus_vectors, "corpus_vectors", self.MAX_CORPUS_VECTORS)

        q_dim = len(query_vectors[0])
        c_dim = len(corpus_vectors[0])
        if q_dim != c_dim:
            raise ValidationError(
                f"Dimension mismatch: query_vectors have dim {q_dim}, corpus_vectors have dim {c_dim}"
            )
        if top_k > len(corpus_vectors):
            raise ValidationError(
                f"top_k ({top_k}) cannot exceed corpus size ({len(corpus_vectors)})"
            )

        payload = {
            "query_vectors": query_vectors,
            "corpus_vectors": corpus_vectors,
            "top_k": top_k,
        }
        if nmi_weight_override is not None:
            payload["nmi_weight_override"] = float(nmi_weight_override)

        return self._request("POST", "/ranked-vector-similarity", payload)

    def ranked_text_similarity(
        self,
        query_texts: list,
        corpus_texts: list,
        top_k: int = 10,
        embedding_model: str = "default",
        nmi_weight_override: float = None,
    ) -> dict:
        """
        Embed query_texts and corpus_texts server-side, then rank using NMI-cosine fusion.
        Use this when you have raw text and want the full stateless pipeline in one call.
        Do NOT use this if you already have precomputed embeddings — use ranked_vector_similarity instead.

        query_texts: list of strings, max 100 items, each up to 8192 chars
        corpus_texts: list of strings, max 10_000 items
        top_k: number of top results per query, between 1 and min(C, 500)
        embedding_model: 'default' uses the server-configured model; pass a specific slug to override
        nmi_weight_override: see ranked_vector_similarity

        Returns dict with keys: results, metadata (entropy, w_nmi used, embedding_model resolved)
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError("top_k must be a positive integer")
        if top_k > 500:
            raise ValidationError("top_k cannot exceed 500")
        if not isinstance(embedding_model, str) or not embedding_model:
            raise ValidationError("embedding_model must be a non-empty string")
        if nmi_weight_override is not None:
            if not isinstance(nmi_weight_override, (int, float)):
                raise ValidationError("nmi_weight_override must be a float in [0.0, 1.0]")
            if not (0.0 <= float(nmi_weight_override) <= 1.0):
                raise ValidationError("nmi_weight_override must be in [0.0, 1.0]")

        query_texts = self._validate_text_list(query_texts, "query_texts", self.MAX_QUERY_VECTORS)
        corpus_texts = self._validate_text_list(corpus_texts, "corpus_texts", self.MAX_CORPUS_VECTORS)

        if top_k > len(corpus_texts):
            raise ValidationError(
                f"top_k ({top_k}) cannot exceed corpus size ({len(corpus_texts)})"
            )

        payload = {
            "query_texts": query_texts,
            "corpus_texts": corpus_texts,
            "top_k": top_k,
            "embedding_model": embedding_model,
        }
        if nmi_weight_override is not None:
            payload["nmi_weight_override"] = float(nmi_weight_override)

        return self._request("POST", "/ranked-text-similarity", payload)

    def corpus_entropy_profile(self, corpus_vectors: list) -> dict:
        """
        Compute the marginal entropy and NMI weight that would be assigned to a given corpus
        without performing a full similarity search. Use this to understand how adaptive weighting
        will behave before committing to a ranked_vector_similarity call.
        Do NOT use this as a substitute for ranked_vector_similarity — it does not return ranked results.

        corpus_vectors: list of numeric lists or np.ndarray, shape (C, D), max 10_000

        Returns dict with keys: marginal_entropy, baseline_entropy, w_nmi, distribution_summary
        """
        corpus_vectors = self._validate_vector_list(corpus_vectors, "corpus_vectors", self.MAX_CORPUS_VECTORS)
        payload = {"corpus_vectors": corpus_vectors}
        return self._request("POST", "/corpus-entropy-profile", payload)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


Client = SimilaritySearchClient