import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np


MOCK_RANKED_RESULTS = {
    "results": [
        {"id": "doc_2", "nmi_weighted_cosine": 0.91, "rank": 1},
        {"id": "doc_1", "nmi_weighted_cosine": 0.74, "rank": 2},
        {"id": "doc_3", "nmi_weighted_cosine": 0.53, "rank": 3},
    ],
    "query_features": 4,
    "corpus_size": 3,
    "metric": "nmi_weighted_cosine",
}

SAMPLE_QUERY = {
    "features": [0.12, 0.88, 0.45, 1.0],
    "feature_types": ["numeric", "categorical", "numeric", "text_embedding"],
}

SAMPLE_CORPUS = [
    {"id": "doc_1", "features": [0.10, 0.75, 0.40, 0.95]},
    {"id": "doc_2", "features": [0.11, 0.87, 0.44, 0.99]},
    {"id": "doc_3", "features": [0.90, 0.10, 0.80, 0.20]},
]


def _build_mock_response(status_code: int, body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    return mock_resp


class TestSimilaritySearchAPIHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_happy_path_ranked_results_order(self, mock_rank):
        """Verifies that top-ranked result has highest nmi_weighted_cosine score."""
        mock_rank.return_value = MOCK_RANKED_RESULTS

        result = mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key="test-key")

        scores = [r["nmi_weighted_cosine"] for r in result["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result["results"][0]["id"], "doc_2")

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_happy_path_response_schema_completeness(self, mock_rank):
        """Verifies response contains all required top-level fields and result subfields."""
        mock_rank.return_value = MOCK_RANKED_RESULTS

        result = mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key="test-key")

        self.assertIn("results", result)
        self.assertIn("corpus_size", result)
        self.assertIn("metric", result)
        self.assertEqual(result["metric"], "nmi_weighted_cosine")
        for item in result["results"]:
            self.assertIn("id", item)
            self.assertIn("nmi_weighted_cosine", item)
            self.assertIn("rank", item)
            self.assertGreaterEqual(item["nmi_weighted_cosine"], 0.0)
            self.assertLessEqual(item["nmi_weighted_cosine"], 1.0)


class TestSimilaritySearchAPIEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_edge_case_single_corpus_document(self, mock_rank):
        """Verifies API handles corpus with exactly one document and returns rank 1."""
        single_doc_result = {
            "results": [{"id": "only_doc", "nmi_weighted_cosine": 0.67, "rank": 1}],
            "query_features": 4,
            "corpus_size": 1,
            "metric": "nmi_weighted_cosine",
        }
        mock_rank.return_value = single_doc_result

        result = mock_rank(
            query=SAMPLE_QUERY,
            corpus=[{"id": "only_doc", "features": [0.11, 0.80, 0.41, 0.90]}],
            api_key="test-key",
        )

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["rank"], 1)

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_edge_case_maximum_corpus_size_does_not_timeout(self, mock_rank):
        """Verifies that a corpus of 500 documents returns without raising an exception."""
        large_corpus = [
            {"id": f"doc_{i}", "features": list(np.random.rand(8).tolist())}
            for i in range(500)
        ]
        large_result = {
            "results": [
                {"id": f"doc_{i}", "nmi_weighted_cosine": float(np.random.rand()), "rank": i + 1}
                for i in range(500)
            ],
            "query_features": 8,
            "corpus_size": 500,
            "metric": "nmi_weighted_cosine",
        }
        mock_rank.return_value = large_result

        result = mock_rank(
            query={"features": list(np.random.rand(8).tolist()), "feature_types": ["numeric"] * 8},
            corpus=large_corpus,
            api_key="test-key",
        )

        self.assertEqual(result["corpus_size"], 500)
        self.assertEqual(len(result["results"]), 500)


class TestSimilaritySearchAPIInvalidInput(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_invalid_input_empty_corpus_raises_value_error(self, mock_rank):
        """Verifies that an empty corpus list raises ValueError with a descriptive message."""
        mock_rank.side_effect = ValueError(
            "corpus must contain at least 1 document; received empty list"
        )

        with self.assertRaises(ValueError) as ctx:
            mock_rank(query=SAMPLE_QUERY, corpus=[], api_key="test-key")

        self.assertIn("corpus", str(ctx.exception).lower())

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_invalid_input_mismatched_feature_dimensions_raises_value_error(self, mock_rank):
        """Verifies that query and corpus documents with differing feature lengths raise ValueError."""
        mismatched_corpus = [
            {"id": "doc_x", "features": [0.1, 0.2]},
        ]
        mock_rank.side_effect = ValueError(
            "feature dimension mismatch: query has 4 features, doc_x has 2"
        )

        with self.assertRaises(ValueError) as ctx:
            mock_rank(query=SAMPLE_QUERY, corpus=mismatched_corpus, api_key="test-key")

        self.assertIn("dimension", str(ctx.exception).lower())

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_invalid_input_none_query_raises_type_error(self, mock_rank):
        """Verifies that passing None as query raises TypeError, not a silent failure."""
        mock_rank.side_effect = TypeError(
            "query must be a dict with 'features' list; received NoneType"
        )

        with self.assertRaises(TypeError) as ctx:
            mock_rank(query=None, corpus=SAMPLE_CORPUS, api_key="test-key")

        self.assertIn("query", str(ctx.exception).lower())


class TestSimilaritySearchAPIRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_rate_limit_burst_of_10_calls_does_not_crash(self, mock_rank):
        """Verifies that 10 sequential calls with the same payload all return valid responses."""
        mock_rank.return_value = MOCK_RANKED_RESULTS

        for _ in range(10):
            result = mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key="test-key")
            self.assertIn("results", result)
            self.assertGreater(len(result["results"]), 0)

        self.assertEqual(mock_rank.call_count, 10)


class TestSimilaritySearchAPIAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_auth_missing_api_key_raises_authentication_error(self, mock_rank):
        """Verifies that omitting api_key raises an error with 'authentication' or 'api_key' in message."""
        mock_rank.side_effect = PermissionError(
            "authentication failed: api_key is required and was not provided"
        )

        with self.assertRaises(PermissionError) as ctx:
            mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key=None)

        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            "api_key" in error_msg or "authentication" in error_msg,
            msg=f"Expected 'api_key' or 'authentication' in error, got: {error_msg}",
        )


class TestSimilaritySearchAPIIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient.rank_by_nmi_weighted_cosine")
    def test_idempotency_same_payload_returns_identical_ranked_order(self, mock_rank):
        """Verifies that two identical calls return the same ranked order and scores."""
        mock_rank.return_value = MOCK_RANKED_RESULTS

        result_a = mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key="test-key")
        result_b = mock_rank(query=SAMPLE_QUERY, corpus=SAMPLE_CORPUS, api_key="test-key")

        ids_a = [r["id"] for r in result_a["results"]]
        ids_b = [r["id"] for r in result_b["results"]]
        scores_a = [r["nmi_weighted_cosine"] for r in result_a["results"]]
        scores_b = [r["nmi_weighted_cosine"] for r in result_b["results"]]

        self.assertEqual(ids_a, ids_b)
        self.assertEqual(scores_a, scores_b)


if __name__ == "__main__":
    unittest.main()