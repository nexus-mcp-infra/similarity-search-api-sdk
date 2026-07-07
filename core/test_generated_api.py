import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np


MOCK_API_KEY = "test-key-nexus-forge-001"
BASE_URL = "https://api.nexus.io/v1/similarity"

VECTOR_A = [0.1, 0.4, 0.9, 0.2, 0.7]
VECTOR_B = [0.2, 0.3, 0.8, 0.1, 0.6]
VECTOR_C = [0.9, 0.1, 0.2, 0.8, 0.3]

MOCK_RANKED_RESPONSE = {
    "results": [
        {"index": 0, "nmi_score": 0.87, "cosine_score": 0.992, "fused_score": 0.941},
        {"index": 1, "nmi_score": 0.21, "cosine_score": 0.143, "fused_score": 0.177},
    ],
    "query_dims": 5,
    "candidates_evaluated": 2,
    "fusion_weights": {"nmi": 0.4, "cosine": 0.6},
}


def _make_mock_response(status_code: int, body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
    return mock_resp


class TestSimilaritySearchHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_vector_ranked_results_order(self, mock_post):
        """Verifica que resultados vienen ordenados descendente por fused_score con vectores válidos."""
        mock_post.return_value = MOCK_RANKED_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)
        response = client.rank_by_fused_similarity(
            query_vector=VECTOR_A,
            candidate_vectors=[VECTOR_B, VECTOR_C],
        )

        scores = [r["fused_score"] for r in response["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(response["results"]), 2)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_fusion_weights_reflected_in_response(self, mock_post):
        """Verifica que los pesos NMI/cosine enviados se reflejan en la respuesta de fusión."""
        custom_response = {**MOCK_RANKED_RESPONSE, "fusion_weights": {"nmi": 0.3, "cosine": 0.7}}
        mock_post.return_value = custom_response

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)
        response = client.rank_by_fused_similarity(
            query_vector=VECTOR_A,
            candidate_vectors=[VECTOR_B, VECTOR_C],
            nmi_weight=0.3,
            cosine_weight=0.7,
        )

        self.assertAlmostEqual(response["fusion_weights"]["nmi"], 0.3)
        self.assertAlmostEqual(response["fusion_weights"]["cosine"], 0.7)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_fused_score_within_unit_interval(self, mock_post):
        """Verifica que todos los fused_scores retornados caen en [0.0, 1.0]."""
        mock_post.return_value = MOCK_RANKED_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)
        response = client.rank_by_fused_similarity(
            query_vector=VECTOR_A,
            candidate_vectors=[VECTOR_B, VECTOR_C],
        )

        for result in response["results"]:
            self.assertGreaterEqual(result["fused_score"], 0.0)
            self.assertLessEqual(result["fused_score"], 1.0)


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_single_candidate_returns_one_result(self, mock_post):
        """Verifica que un solo candidato produce exactamente un resultado rankeado."""
        single_result_response = {
            "results": [{"index": 0, "nmi_score": 0.75, "cosine_score": 0.80, "fused_score": 0.78}],
            "query_dims": 5,
            "candidates_evaluated": 1,
            "fusion_weights": {"nmi": 0.4, "cosine": 0.6},
        }
        mock_post.return_value = single_result_response

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)
        response = client.rank_by_fused_similarity(
            query_vector=VECTOR_A,
            candidate_vectors=[VECTOR_B],
        )

        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(response["candidates_evaluated"], 1)

    def test_edge_case_empty_candidates_raises_value_error(self):
        """Verifica que lista vacía de candidatos lanza ValueError antes de llamar a la API."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        with self.assertRaises(ValueError) as ctx:
            client.rank_by_fused_similarity(
                query_vector=VECTOR_A,
                candidate_vectors=[],
            )

        self.assertIn("candidate_vectors", str(ctx.exception).lower())

    def test_edge_case_none_query_vector_raises_value_error(self):
        """Verifica que query_vector=None lanza ValueError con mensaje descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        with self.assertRaises(ValueError) as ctx:
            client.rank_by_fused_similarity(
                query_vector=None,
                candidate_vectors=[VECTOR_B],
            )

        self.assertIn("query_vector", str(ctx.exception).lower())


class TestSimilaritySearchInvalidInput(unittest.TestCase):

    def test_invalid_input_string_vector_raises_type_error(self):
        """Verifica que pasar strings en lugar de floats como vector lanza TypeError."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        with self.assertRaises(TypeError) as ctx:
            client.rank_by_fused_similarity(
                query_vector=["a", "b", "c"],
                candidate_vectors=[VECTOR_B],
            )

        self.assertIn("float", str(ctx.exception).lower())

    def test_invalid_input_mismatched_dimensions_raises_value_error(self):
        """Verifica que query y candidatos con dimensiones distintas lanzan ValueError."""
        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        mismatched_candidate = [0.1, 0.2]

        with self.assertRaises(ValueError) as ctx:
            client.rank_by_fused_similarity(
                query_vector=VECTOR_A,
                candidate_vectors=[mismatched_candidate],
            )

        self.assertIn("dimension", str(ctx.exception).lower())


class TestSimilaritySearchRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_rate_limit_burst_calls_do_not_crash(self, mock_post):
        """Verifica que 50 llamadas consecutivas sin demora no lanzan excepción en el cliente."""
        mock_post.return_value = MOCK_RANKED_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        exceptions_raised = 0
        for _ in range(50):
            try:
                client.rank_by_fused_similarity(
                    query_vector=VECTOR_A,
                    candidate_vectors=[VECTOR_B, VECTOR_C],
                )
            except Exception:
                exceptions_raised += 1

        self.assertEqual(exceptions_raised, 0)
        self.assertEqual(mock_post.call_count, 50)


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_authentication_error(self):
        """Verifica que instanciar el cliente sin api_key lanza AuthenticationError descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError

        with self.assertRaises(AuthenticationError) as ctx:
            SimilaritySearchClient(api_key=None)

        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            "api_key" in error_msg or "authentication" in error_msg or "key" in error_msg,
            msg=f"Error message not descriptive enough: {ctx.exception}",
        )

    def test_auth_empty_string_api_key_raises_authentication_error(self):
        """Verifica que api_key='' (string vacío) lanza AuthenticationError antes de cualquier llamada HTTP."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError

        with self.assertRaises(AuthenticationError):
            SimilaritySearchClient(api_key="")


class TestSimilaritySearchIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_idempotency_same_vectors_same_fused_scores(self, mock_post):
        """Verifica que dos llamadas idénticas devuelven exactamente los mismos fused_scores."""
        mock_post.return_value = MOCK_RANKED_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient
        client = SimilaritySearchClient(api_key=MOCK_API_KEY)

        payload = dict(query_vector=VECTOR_A, candidate_vectors=[VECTOR_B, VECTOR_C])
        response_1 = client.rank_by_fused_similarity(**payload)
        response_2 = client.rank_by_fused_similarity(**payload)

        scores_1 = [r["fused_score"] for r in response_1["results"]]
        scores_2 = [r["fused_score"] for r in response_2["results"]]
        self.assertEqual(scores_1, scores_2)


if __name__ == "__main__":
    unittest.main()