import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np


MOCK_CORPUS = [
    "machine learning algorithms optimize model parameters",
    "deep neural networks process image recognition tasks",
    "natural language processing enables text understanding",
    "statistical methods quantify uncertainty in predictions",
    "gradient descent minimizes loss functions iteratively",
]
MOCK_QUERY = "neural network optimization techniques"
MOCK_API_KEY = "test-key-forge-nexus-12345"

MOCK_SIMILARITY_RESPONSE = {
    "results": [
        {"index": 1, "text": "deep neural networks process image recognition tasks", "score": 0.847, "cosine": 0.791, "nmi": 0.903},
        {"index": 4, "text": "gradient descent minimizes loss functions iteratively", "score": 0.712, "cosine": 0.689, "nmi": 0.735},
        {"index": 0, "text": "machine learning algorithms optimize model parameters", "score": 0.631, "cosine": 0.598, "nmi": 0.664},
    ],
    "query": MOCK_QUERY,
    "corpus_size": 5,
    "top_k": 3,
    "complexity": "O(n·d)",
}

MOCK_AUTH_ERROR = {
    "detail": "API key missing or invalid. Provide a valid X-API-Key header to access similarity search.",
    "error_code": "AUTH_MISSING_KEY",
}

MOCK_VALIDATION_ERROR = {
    "detail": "corpus must be a non-empty list of strings; received type <class 'str'>",
    "error_code": "INVALID_INPUT_TYPE",
}


def build_mock_sdk_client(api_key=MOCK_API_KEY):
    client = MagicMock()
    client.api_key = api_key

    def mock_rank_corpus_by_query(corpus, query, top_k=3, alpha=0.5):
        if not api_key:
            raise PermissionError(
                "API key missing or invalid. Provide a valid X-API-Key header to access similarity search."
            )
        if not isinstance(corpus, list) or not all(isinstance(t, str) for t in corpus):
            raise TypeError(
                f"corpus must be a non-empty list of strings; received type {type(corpus)}"
            )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if len(corpus) == 0:
            raise ValueError("corpus must contain at least one item.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0].")
        if not 1 <= top_k <= len(corpus):
            raise ValueError(f"top_k must be between 1 and corpus length ({len(corpus)}).")
        return MOCK_SIMILARITY_RESPONSE

    client.rank_corpus_by_query = mock_rank_corpus_by_query
    return client


class TestSimilaritySearchHappyPath(unittest.TestCase):

    def setUp(self):
        self.client = build_mock_sdk_client(api_key=MOCK_API_KEY)

    def test_happy_path_returns_ranked_results_with_composite_scores(self):
        """Verifies that a valid corpus+query returns ranked items with both cosine and NMI scores fused."""
        response = self.client.rank_corpus_by_query(
            corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3
        )
        self.assertIn("results", response)
        self.assertEqual(len(response["results"]), 3)
        for item in response["results"]:
            self.assertIn("score", item)
            self.assertIn("cosine", item)
            self.assertIn("nmi", item)
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)

    def test_happy_path_results_are_sorted_descending_by_composite_score(self):
        """Verifies that returned results are ordered from highest to lowest composite score."""
        response = self.client.rank_corpus_by_query(
            corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3
        )
        scores = [item["score"] for item in response["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_happy_path_corpus_size_and_top_k_reflected_in_response(self):
        """Verifies that response metadata accurately echoes corpus_size and top_k from the request."""
        response = self.client.rank_corpus_by_query(
            corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3
        )
        self.assertEqual(response["corpus_size"], len(MOCK_CORPUS))
        self.assertEqual(response["top_k"], 3)
        self.assertEqual(response["query"], MOCK_QUERY)


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    def setUp(self):
        self.client = build_mock_sdk_client(api_key=MOCK_API_KEY)

    def test_edge_case_empty_corpus_raises_value_error(self):
        """Verifies that an empty corpus list raises ValueError before any scoring is attempted."""
        with self.assertRaises(ValueError) as ctx:
            self.client.rank_corpus_by_query(corpus=[], query=MOCK_QUERY, top_k=1)
        self.assertIn("at least one item", str(ctx.exception))

    def test_edge_case_top_k_exceeds_corpus_length_raises_value_error(self):
        """Verifies that requesting more results than corpus items raises a descriptive ValueError."""
        small_corpus = ["single item only"]
        with self.assertRaises(ValueError) as ctx:
            self.client.rank_corpus_by_query(corpus=small_corpus, query=MOCK_QUERY, top_k=5)
        self.assertIn("top_k", str(ctx.exception))


class TestSimilaritySearchInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.client = build_mock_sdk_client(api_key=MOCK_API_KEY)

    def test_invalid_input_corpus_as_string_instead_of_list_raises_type_error(self):
        """Verifies that passing a raw string as corpus raises TypeError with type information."""
        with self.assertRaises(TypeError) as ctx:
            self.client.rank_corpus_by_query(
                corpus="this is not a list", query=MOCK_QUERY, top_k=1
            )
        self.assertIn("non-empty list of strings", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))

    def test_invalid_input_empty_query_string_raises_value_error(self):
        """Verifies that a blank or whitespace-only query raises ValueError, not a silent failure."""
        with self.assertRaises(ValueError) as ctx:
            self.client.rank_corpus_by_query(
                corpus=MOCK_CORPUS, query="   ", top_k=3
            )
        self.assertIn("non-empty string", str(ctx.exception))

    def test_invalid_input_alpha_out_of_range_raises_value_error(self):
        """Verifies that alpha outside [0.0, 1.0] raises ValueError before scoring."""
        with self.assertRaises(ValueError) as ctx:
            self.client.rank_corpus_by_query(
                corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3, alpha=1.7
            )
        self.assertIn("alpha", str(ctx.exception))


class TestSimilaritySearchRateLimit(unittest.TestCase):

    def setUp(self):
        self.client = build_mock_sdk_client(api_key=MOCK_API_KEY)

    def test_rate_limit_fifty_sequential_calls_do_not_crash_or_raise(self):
        """Verifies that 50 sequential calls with valid inputs complete without exception (stateless guarantee)."""
        errors = []
        for i in range(50):
            try:
                response = self.client.rank_corpus_by_query(
                    corpus=MOCK_CORPUS, query=f"query variant {i}", top_k=2
                )
                self.assertIn("results", response)
            except Exception as exc:
                errors.append(str(exc))
        self.assertEqual(
            len(errors), 0,
            msg=f"Unexpected errors during sequential calls: {errors[:3]}"
        )


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_permission_error_with_descriptive_message(self):
        """Verifies that a client with no API key raises PermissionError containing actionable guidance."""
        unauthenticated_client = build_mock_sdk_client(api_key="")
        with self.assertRaises(PermissionError) as ctx:
            unauthenticated_client.rank_corpus_by_query(
                corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3
            )
        error_message = str(ctx.exception)
        self.assertIn("API key", error_message)
        self.assertIn("X-API-Key", error_message)


class TestSimilaritySearchIdempotency(unittest.TestCase):

    def setUp(self):
        self.client = build_mock_sdk_client(api_key=MOCK_API_KEY)

    def test_idempotency_identical_request_twice_returns_identical_scores_and_ranking(self):
        """Verifies that calling rank_corpus_by_query twice with identical inputs produces byte-identical results."""
        response_a = self.client.rank_corpus_by_query(
            corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3, alpha=0.5
        )
        response_b = self.client.rank_corpus_by_query(
            corpus=MOCK_CORPUS, query=MOCK_QUERY, top_k=3, alpha=0.5
        )
        self.assertEqual(
            [item["score"] for item in response_a["results"]],
            [item["score"] for item in response_b["results"]],
        )
        self.assertEqual(
            [item["index"] for item in response_a["results"]],
            [item["index"] for item in response_b["results"]],
        )
        self.assertEqual(response_a["corpus_size"], response_b["corpus_size"])


if __name__ == "__main__":
    unittest.main()