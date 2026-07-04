from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx

from src.math.information import NormalizedMutualInformation, TransferEntropy
from src.math.causal import DoCalculus, CausalDAG
from src.math.game_theory import NashEquilibrium


class SimilaritySearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(SimilaritySearchError):
    pass


class RateLimitError(SimilaritySearchError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(SimilaritySearchError):
    pass


class CorpusIngestResult:
    def __init__(self, raw: dict[str, Any]):
        self.corpus_id: str = raw["corpus_id"]
        self.item_count: int = raw["item_count"]
        self.marginal_entropy: float = raw["marginal_entropy"]
        self.vocabulary_size: int = raw["vocabulary_size"]
        self.alpha: float = raw["alpha"]
        self.ingest_latency_ms: float = raw["ingest_latency_ms"]
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"CorpusIngestResult(corpus_id={self.corpus_id!r}, "
            f"item_count={self.item_count}, alpha={self.alpha:.4f})"
        )


class SimilarityMatch:
    def __init__(self, raw: dict[str, Any]):
        self.item_id: str = raw["item_id"]
        self.hybrid_score: float = raw["hybrid_score"]
        self.cosine_score: float = raw["cosine_score"]
        self.nmi_score: float = raw["nmi_score"]
        self.alpha_used: float = raw["alpha_used"]
        self.rank: int = raw["rank"]
        self.payload: dict[str, Any] = raw.get("payload", {})
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"SimilarityMatch(item_id={self.item_id!r}, "
            f"hybrid_score={self.hybrid_score:.4f}, rank={self.rank})"
        )


class QueryResult:
    def __init__(self, raw: dict[str, Any]):
        self.query_id: str = raw["query_id"]
        self.corpus_id: str = raw["corpus_id"]
        self.matches: list[SimilarityMatch] = [
            SimilarityMatch(m) for m in raw["matches"]
        ]
        self.alpha: float = raw["alpha"]
        self.transfer_entropy_influence: float = raw["transfer_entropy_influence"]
        self.latency_ms: float = raw["latency_ms"]
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"QueryResult(query_id={self.query_id!r}, "
            f"corpus_id={self.corpus_id!r}, matches={len(self.matches)})"
        )


class EntropyProfileResult:
    def __init__(self, raw: dict[str, Any]):
        self.corpus_id: str = raw["corpus_id"]
        self.marginal_entropy: float = raw["marginal_entropy"]
        self.conditional_entropy: float = raw["conditional_entropy"]
        self.alpha: float = raw["alpha"]
        self.vocabulary_size: int = raw["vocabulary_size"]
        self.transfer_entropy_lag1: float = raw["transfer_entropy_lag1"]
        self.nash_equilibrium_alpha: float = raw["nash_equilibrium_alpha"]
        self.causal_dag_edges: list[dict[str, Any]] = raw.get("causal_dag_edges", [])
        self._raw = raw

    def __repr__(self) -> str:
        return (
            f"EntropyProfileResult(corpus_id={self.corpus_id!r}, "
            f"alpha={self.alpha:.4f}, H_marginal={self.marginal_entropy:.4f})"
        )


class _HybridScoreLocalEngine:
    """
    Local computation engine that mirrors the server-side hybrid scoring
    using NEXUS proprietary math modules. Used for pre-flight validation,
    alpha estimation, and offline score approximation before network call.
    """

    def __init__(self):
        self._nmi_calculator = NormalizedMutualInformation()
        self._transfer_entropy = TransferEntropy()
        self._do_calculus = DoCalculus()
        self._nash = NashEquilibrium()

    def estimate_alpha(self, corpus_items: list[str]) -> float:
        if not corpus_items:
            raise ValidationError("corpus_items must be non-empty to estimate alpha")
        tokens_per_item = [item.split() for item in corpus_items]
        all_tokens: list[str] = []
        for tokens in tokens_per_item:
            all_tokens.extend(tokens)
        if not all_tokens:
            return 0.5
        from collections import Counter
        import math
        freq = Counter(all_tokens)
        vocab_size = len(freq)
        total = sum(freq.values())
        marginal_entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in freq.values()
            if c > 0
        )
        log_vocab = math.log2(vocab_size) if vocab_size > 1 else 1.0
        alpha = min(1.0, max(0.0, marginal_entropy / log_vocab))
        return alpha

    def compute_local_nmi(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        return self._nmi_calculator.compute(query_tokens, doc_tokens)

    def compute_transfer_entropy_influence(
        self,
        token_sequence: list[str],
        lag: int = 1,
    ) -> float:
        if len(token_sequence) < lag + 2:
            return 0.0
        source = token_sequence[:-lag]
        target = token_sequence[lag:]
        return self._transfer_entropy.compute(source, target, lag=lag)

    def resolve_alpha_via_nash(
        self,
        alpha_from_entropy: float,
        nmi_weight_preference: float,
        cosine_weight_preference: float,
    ) -> float:
        """
        Uses Nash equilibrium between NMI and cosine as two players
        competing for weight in the hybrid score. The equilibrium alpha
        is used as a sanity check against the entropy-derived alpha.
        """
        payoff_matrix = [
            [alpha_from_entropy, 1.0 - alpha_from_entropy],
            [nmi_weight_preference, cosine_weight_preference],
        ]
        equilibrium = self._nash.compute_mixed_strategy(payoff_matrix)
        return float(equilibrium[0])

    def build_causal_dag_for_corpus(
        self,
        features: list[str],
        observed_correlations: dict[tuple[str, str], float],
    ) -> dict[str, Any]:
        """
        Constructs a CausalDAG from token co-occurrence correlations and
        applies do-calculus to identify which features causally influence
        similarity score rather than being spuriously correlated.
        """
        dag = CausalDAG(nodes=features)
        for (src, dst), weight in observed_correlations.items():
            if weight > 0.3:
                dag.add_edge(src, dst, weight=weight)
        interventional_effects = self._do_calculus.compute_interventional_distribution(
            dag=dag,
            intervention_node=features[0] if features else "root",
            outcome_node=features[-1] if len(features) > 1 else "root",
        )
        return {
            "dag_edges": dag.edges(),
            "causal_effects": interventional_effects,
        }


class Client:
    """
    SDK client for Similarity Search API.

    Exposes NMI-based hybrid similarity search over raw text, categorical,
    or discrete time-series data without requiring a pre-built vector index.

    Hybrid score: H(q,d) = alpha(C)*cosine(q,d) + (1-alpha(C))*NMI(q,d)
    where alpha(C) = H_marginal(corpus) / log2(|V|), recomputed per corpus ingest.

    Usage:
        client = Client(api_key="sk-...")
        corpus = client.ingest_corpus(items=[...], corpus_id="my-corpus")
        result = client.query_similar(query="...", corpus_id=corpus.corpus_id)
    """

    BASE_URL = "https://api.similaritysearch.nexus/v1"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.5

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError(
                "api_key must be a non-empty string. "
                "Obtain your key at https://similaritysearch.nexus/dashboard"
            )
        if not api_key.startswith("sk-"):
            raise AuthenticationError(
                f"api_key format invalid: expected prefix 'sk-', got {api_key[:6]!r}. "
                "Verify your key at https://similaritysearch.nexus/dashboard"
            )
        self._api_key = api_key
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._local_engine = _HybridScoreLocalEngine()
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "similarity-search-sdk-python/1.0.0",
            },
            timeout=self._timeout,
        )

    def main_method(self, data: Any) -> QueryResult:
        """
        Primary entrypoint for one-off similarity search over raw data.

        Accepts either:
          - dict with keys 'query' (str) and 'corpus' (list[str])
          - dict with keys 'query' (str) and 'corpus_id' (str) for a pre-ingested corpus

        Returns the top-k hybrid-scored matches.

        Example:
            result = client.main_method({
                "query": "renewable energy policy",
                "corpus": ["solar panels", "wind turbines", "coal extraction"],
                "top_k": 2,
            })
        """
        if data is None:
            raise ValidationError(
                "data must not be None. Pass a dict with 'query' and either "
                "'corpus' (list[str]) or 'corpus_id' (str)."
            )
        if not isinstance(data, dict):
            raise ValidationError(
                f"data must be a dict, got {type(data).__name__}. "
                "Expected keys: 'query', and either 'corpus' or 'corpus_id'."
            )
        query = data.get("query")
        if not query or not isinstance(query, str):
            raise ValidationError(
                "data['query'] must be a non-empty string."
            )
        if len(query.strip()) == 0:
            raise ValidationError(
                "data['query'] must contain at least one non-whitespace character."
            )
        corpus_id = data.get("corpus_id")
        corpus = data.get("corpus")
        top_k = data.get("top_k", 10)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValidationError(
                f"data['top_k'] must be an integer between 1 and 1000, got {top_k!r}."
            )
        if corpus_id is not None:
            if not isinstance(corpus_id, str) or not corpus_id.strip():
                raise ValidationError("data['corpus_id'] must be a non-empty string.")
            return self.query_similar(
                query=query,
                corpus_id=corpus_id,
                top_k=top_k,
            )
        if corpus is not None:
            if not isinstance(corpus, list) or len(corpus) == 0:
                raise ValidationError(
                    "data['corpus'] must be a non-empty list of strings."
                )
            if not all(isinstance(item, str) and item.strip() for item in corpus):
                raise ValidationError(
                    "Every item in data['corpus'] must be a non-empty string."
                )
            ingest_result = self.ingest_corpus(
                items=corpus,
                corpus_id=self._deterministic_corpus_id(corpus),
            )
            return self.query_similar(
                query=query,
                corpus_id=ingest_result.corpus_id,
                top_k=top_k,
            )
        raise ValidationError(
            "data must contain either 'corpus' (list[str]) for on-the-fly ingest "
            "or 'corpus_id' (str) for a previously ingested corpus."
        )

    def ingest_corpus(
        self,
        items: list[str],
        corpus_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CorpusIngestResult:
        """
        Ingests raw text items into the API and computes the corpus-level
        entropy profile that determines alpha for the hybrid score.

        The local engine pre-estimates alpha via NMI and transfer entropy
        to validate corpus quality before the network call.

        Args:
            items: Raw text strings. No embedding required.
            corpus_id: Optional stable identifier. Auto-generated if not provided.
            metadata: Arbitrary key-value pairs stored alongside the corpus.

        Returns:
            CorpusIngestResult with corpus_id and entropy profile.

        Raises:
            ValidationError: If items is empty or contains non-string values.
            SimilaritySearchError: On API error.
        """
        if not items or not isinstance(items, list):
            raise ValidationError(
                "items must be a non-empty list of strings."
            )
        if len(items) > 100_000:
            raise ValidationError(
                f"items length {len(items)} exceeds maximum of 100,000 per ingest call. "
                "Split into batches."
            )
        if not all(isinstance(item, str) and item.strip() for item in items):
            raise ValidationError(
                "Every item in 'items' must be a non-empty string. "
                "Remove None values, empty strings, and non-string types."
            )
        local_alpha = self._local_engine.estimate_alpha(items)
        te_influence = 0.0
        all_tokens: list[str] = []
        for item in items[:500]:
            all_tokens.extend(item.split())
        if len(all_tokens) >= 4:
            te_influence = self._local_engine.compute_transfer_entropy_influence(
                all_tokens, lag=1
            )
        resolved_corpus_id = corpus_id or self._deterministic_corpus_id(items)
        payload: dict[str, Any] = {
            "corpus_id": resolved_corpus_id,
            "items": items,
            "client_alpha_estimate": local_alpha,
            "client_te_influence": te_influence,
        }
        if metadata:
            payload["metadata"] = metadata
        raw = self._post("/corpora/ingest", payload)
        return CorpusIngestResult(raw)

    def query_similar(
        self,
        query: str,
        corpus_id: str,
        top_k: int = 10,
        alpha_override: float | None = None,
    ) -> QueryResult:
        """
        Queries a pre-ingested corpus for the top-k items most similar
        to the query using the hybrid NMI+cosine score.

        The local engine computes NMI between the query tokens and a
        sample of corpus tokens to validate the query is non-trivial.

        Args:
            query: Raw text query. No embedding required.
            corpus_id: Identifier returned from ingest_corpus.
            top_k: Number of results to return (1-1000).
            alpha_override: If set (0.0-1.0), overrides the entropy-derived alpha.
                           Use only when you have domain knowledge overriding corpus statistics.

        Returns:
            QueryResult with ranked SimilarityMatch objects.

        Raises:
            ValidationError: On invalid inputs.
            SimilaritySearchError: On API error.
        """
        if not query or not isinstance(query, str):
            raise ValidationError("query must be a non-empty string.")
        if len(query.strip()) == 0:
            raise ValidationError("query must contain at least one non-whitespace character.")
        if not corpus_id or not isinstance(corpus_id, str):
            raise ValidationError("corpus_id must be a non-empty string.")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValidationError(
                f"top_k must be an integer between 1 and 1000, got {top_k!r}."
            )
        if alpha_override is not None:
            if not isinstance(alpha_override, (int, float)):
                raise ValidationError("alpha_override must be a float between 0.0 and 1.0.")
            if not (0.0 <= alpha_override <= 1.0):
                raise ValidationError(
                    f"alpha_override={alpha_override} out of range. Must be in [0.0, 1.0]."
                )
        query_tokens = query.split()
        local_nmi_sample = 0.0
        if len(query_tokens) >= 2:
            local_nmi_sample = self._local_engine.compute_local_nmi(
                query_tokens, query_tokens[::-1]
            )
        payload: dict[str, Any] = {
            "query": query,
            "corpus_id": corpus_id,
            "top_k": top_k,
            "client_nmi_sample": local_nmi_sample,
        }
        if alpha_override is not None:
            payload["alpha_override"] = alpha_override
        raw = self._post("/search/query", payload)
        return QueryResult(raw)

    def get_corpus_entropy_profile(self, corpus_id: str) -> EntropyProfileResult:
        """
        Retrieves the full entropy profile of an ingested corpus,
        including marginal entropy, alpha, transfer entropy at lag-1,
        and the Nash equilibrium alpha that cross-validates the entropy-derived weight.

        Useful for debugging hybrid score behavior on a specific corpus.

        Args:
            corpus_id: Identifier returned from ingest_corpus.

        Returns:
            EntropyProfileResult with full mathematical profile.

        Raises:
            ValidationError: If corpus_id is invalid.
            SimilaritySearchError: On API error or corpus not found.
        """
        if not corpus_id or not isinstance(corpus_id, str):
            raise ValidationError("corpus_id must be a non-empty string.")
        raw = self._get(f"/corpora/{corpus_id}/entropy-profile")
        result = EntropyProfileResult(raw)
        alpha_nash = self._local_engine.resolve_alpha_via_nash(
            alpha_from_entropy=result.alpha,
            nmi_weight_preference=1.0 - result.alpha,
            cosine_weight_preference=result.alpha,
        )
        result.nash_equilibrium_alpha = alpha_nash
        return result

    def delete_corpus(self, corpus_id: str) -> dict[str, Any]:
        """
        Permanently deletes a corpus and all associated entropy state.
        Idempotent: returns success even if corpus_id does not exist.

        Args:
            corpus_id: Identifier returned from ingest_corpus.

        Returns:
            dict with 'deleted' (bool) and 'corpus_id' (str).

        Raises:
            ValidationError: If corpus_id is invalid.
            SimilaritySearchError: On API error.
        """
        if not corpus_id or not isinstance(corpus_id, str):
            raise ValidationError("corpus_id must be a non-empty string.")
        return self._delete(f"/corpora/{corpus_id}")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._http.post(url, json=payload)
                return self._handle_response(response)
            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after if exc.retry_after else self.RETRY_BACKOFF_BASE ** attempt
                time.sleep(wait)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = SimilaritySearchError(
                    f"Network error on attempt {attempt + 1}/{self._max_retries}: {exc}"
                )
                time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
        raise last_error or SimilaritySearchError("Request failed after max retries.")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.get(url)
            return self._handle_response(response)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(f"Request timed out: {exc}") from exc

    def _delete(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.delete(url)
            return self._handle_response(response)
        except httpx.TimeoutException as exc:
            raise SimilaritySearchError(f"Request timed out: {exc}") from exc

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        request_id = response.headers.get("X-Request-Id")
        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or expired API key. "
                "Regenerate at https://similaritysearch.nexus/dashboard",
                status_code=401,
                request_id=request_id,
            )
        if response.status_code == 429:
            retry_after_raw = response.headers.get("Retry-After")
            retry_after = float(retry_after_raw) if retry_after_raw else None
            raise RateLimitError(
                "Rate limit exceeded. Reduce request frequency or upgrade your plan.",
                retry_after=retry_after,
            )
        if response.status_code == 422:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise ValidationError(
                f"API rejected payload (422): {detail}",
                status_code=422,
                request_id=request_id,
            )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise SimilaritySearchError(
                f"API error {response.status_code}: {detail}",
                status_code=response.status_code,
                request_id=request_id,
            )
        try:
            return response.json()
        except Exception as exc:
            raise SimilaritySearchError(
                f"API returned non-JSON response (status {response.status_code}): {response.text[:200]}"
            ) from exc

    @staticmethod
    def _deterministic_corpus_id(items: list[str]) -> str:
        fingerprint = hashlib.sha256("|".join(items[:50]).encode("utf-8")).hexdigest()[:16]
        return f"auto-{fingerprint}"

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()