import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_SIMILARITY_RESPONSE = {
    "query_id": "test-query-001",
    "results": [
        {
            "item_id": "doc-1",
            "fused_score": 0.847,
            "cosine_similarity": 0.912,
            "nmi_score": 0.783,
        },
        {
            "item_id": "doc-2",
            "fused_score": 0.621,
            "cosine_similarity": 0.701,
            "nmi_score": 0.541,
        },
    ],
    "top_k": 2,
    "fusion_weight_cosine": 0.5,
    "fusion_weight_nmi": 0.5,
}

MOCK_AUTH_ERROR_RESPONSE = {
    "detail": "Missing or invalid API key. Provide X-API-Key header with a valid credential."
}

MOCK_RATE_LIMIT_RESPONSE = {
    "detail": "Rate limit exceeded. Maximum 60 requests per minute per API key."
}


def build_mock_client(status_code=200, json_body=None):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_body if json_body is not None else MOCK_SIMILARITY_RESPONSE
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )
    return mock_response


class TestSimilaritySearchHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_valid_vectors_return_fused_scores(self, mock_post):
        """Verifica que vectores válidos devuelven resultados con fused_score, cosine_similarity y nmi_score."""
        mock_post.return_value = MOCK_SIMILARITY_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        query_vector = [0.1, 0.4, 0.9, 0.3]
        candidate_vectors = [
            {"item_id": "doc-1", "vector": [0.2, 0.35, 0.85, 0.25]},
            {"item_id": "doc-2", "vector": [0.9, 0.1, 0.2, 0.7]},
        ]

        result = client.search(query_vector=query_vector, candidates=candidate_vectors, top_k=2)

        self.assertEqual(len(result["results"]), 2)
        first = result["results"][0]
        self.assertIn("fused_score", first)
        self.assertIn("cosine_similarity", first)
        self.assertIn("nmi_score", first)
        self.assertGreaterEqual(first["fused_score"], 0.0)
        self.assertLessEqual(first["fused_score"], 1.0)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_top_k_limits_results(self, mock_post):
        """Verifica que el parámetro top_k restringe el número de resultados devueltos."""
        response_top1 = {**MOCK_SIMILARITY_RESPONSE, "results": [MOCK_SIMILARITY_RESPONSE["results"][0]], "top_k": 1}
        mock_post.return_value = response_top1

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        result = client.search(
            query_vector=[0.1, 0.4, 0.9, 0.3],
            candidates=[
                {"item_id": "doc-1", "vector": [0.2, 0.35, 0.85, 0.25]},
                {"item_id": "doc-2", "vector": [0.9, 0.1, 0.2, 0.7]},
            ],
            top_k=1,
        )

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["top_k"], 1)

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_happy_path_fusion_weights_reflected_in_response(self, mock_post):
        """Verifica que los pesos de fusión custom enviados se reflejan correctamente en la respuesta."""
        custom_response = {**MOCK_SIMILARITY_RESPONSE, "fusion_weight_cosine": 0.7, "fusion_weight_nmi": 0.3}
        mock_post.return_value = custom_response

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        result = client.search(
            query_vector=[0.5, 0.5, 0.5, 0.5],
            candidates=[{"item_id": "doc-1", "vector": [0.5, 0.5, 0.5, 0.5]}],
            fusion_weight_cosine=0.7,
            fusion_weight_nmi=0.3,
        )

        self.assertAlmostEqual(result["fusion_weight_cosine"], 0.7)
        self.assertAlmostEqual(result["fusion_weight_nmi"], 0.3)


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_edge_case_single_candidate_returns_one_result(self, mock_post):
        """Verifica que un único candidato devuelve exactamente un resultado sin error."""
        single_result_response = {
            **MOCK_SIMILARITY_RESPONSE,
            "results": [MOCK_SIMILARITY_RESPONSE["results"][0]],
            "top_k": 1,
        }
        mock_post.return_value = single_result_response

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        result = client.search(
            query_vector=[0.3, 0.6, 0.1, 0.8],
            candidates=[{"item_id": "only-doc", "vector": [0.3, 0.6, 0.1, 0.8]}],
            top_k=1,
        )

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["item_id"], "only-doc")

    def test_edge_case_empty_candidates_raises_value_error(self):
        """Verifica que una lista vacía de candidatos lanza ValueError antes de llamar a la API."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(ValueError) as ctx:
            client.search(
                query_vector=[0.1, 0.2, 0.3],
                candidates=[],
                top_k=1,
            )

        self.assertIn("candidates", str(ctx.exception).lower())

    def test_edge_case_none_query_vector_raises_value_error(self):
        """Verifica que query_vector=None lanza ValueError con mensaje descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(ValueError) as ctx:
            client.search(
                query_vector=None,
                candidates=[{"item_id": "doc-1", "vector": [0.1, 0.2]}],
                top_k=1,
            )

        self.assertIn("query_vector", str(ctx.exception).lower())


class TestSimilaritySearchInvalidInput(unittest.TestCase):

    def test_invalid_input_non_numeric_vector_raises_type_error(self):
        """Verifica que un vector con strings en lugar de floats lanza TypeError."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(TypeError) as ctx:
            client.search(
                query_vector=["a", "b", "c"],
                candidates=[{"item_id": "doc-1", "vector": [0.1, 0.2, 0.3]}],
                top_k=1,
            )

        self.assertIn("numeric", str(ctx.exception).lower())

    def test_invalid_input_top_k_zero_raises_value_error(self):
        """Verifica que top_k=0 lanza ValueError indicando el rango mínimo permitido."""
        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")

        with self.assertRaises(ValueError) as ctx:
            client.search(
                query_vector=[0.1, 0.2, 0.3],
                candidates=[{"item_id": "doc-1", "vector": [0.1, 0.2, 0.3]}],
                top_k=0,
            )

        self.assertIn("top_k", str(ctx.exception).lower())


class TestSimilaritySearchRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_rate_limit_burst_calls_do_not_crash_client(self, mock_post):
        """Verifica que 20 llamadas consecutivas no provocan excepción en el cliente."""
        mock_post.return_value = MOCK_SIMILARITY_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        query = [0.1, 0.4, 0.9, 0.3]
        candidates = [{"item_id": f"doc-{i}", "vector": [0.1 * i, 0.2, 0.3, 0.4]} for i in range(1, 4)]

        results = []
        for _ in range(20):
            r = client.search(query_vector=query, candidates=candidates, top_k=2)
            results.append(r)

        self.assertEqual(len(results), 20)
        self.assertEqual(mock_post.call_count, 20)


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_authentication_error(self):
        """Verifica que instanciar el cliente sin API key lanza AuthenticationError antes de cualquier llamada."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError

        with self.assertRaises(AuthenticationError) as ctx:
            SimilaritySearchClient(api_key=None)

        self.assertIn("api_key", str(ctx.exception).lower())

    def test_auth_empty_string_api_key_raises_authentication_error(self):
        """Verifica que api_key='' (cadena vacía) es rechazado con AuthenticationError descriptivo."""
        from similarity_search_api_sdk import SimilaritySearchClient, AuthenticationError

        with self.assertRaises(AuthenticationError) as ctx:
            SimilaritySearchClient(api_key="")

        self.assertIn("api_key", str(ctx.exception).lower())


class TestSimilaritySearchIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.SimilaritySearchClient._post")
    def test_idempotency_identical_inputs_return_identical_fused_scores(self, mock_post):
        """Verifica que dos llamadas con el mismo payload devuelven fused_scores idénticos."""
        mock_post.return_value = MOCK_SIMILARITY_RESPONSE

        from similarity_search_api_sdk import SimilaritySearchClient

        client = SimilaritySearchClient(api_key="test-key-mock")
        payload = dict(
            query_vector=[0.1, 0.4, 0.9, 0.3],
            candidates=[
                {"item_id": "doc-1", "vector": [0.2, 0.35, 0.85, 0.25]},
                {"item_id": "doc-2", "vector": [0.9, 0.1, 0.2, 0.7]},
            ],
            top_k=2,
        )

        result_a = client.search(**payload)
        result_b = client.search(**payload)

        scores_a = [r["fused_score"] for r in result_a["results"]]
        scores_b = [r["fused_score"] for r in result_b["results"]]
        self.assertEqual(scores_a, scores_b)


if __name__ == "__main__":
    unittest.main()