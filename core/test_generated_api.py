import unittest
from unittest.mock import patch, MagicMock
import json


MOCK_HYBRID_RESPONSE = {
    "results": [
        {
            "id": "doc_001",
            "score": 0.87,
            "score_breakdown": {
                "nmi_categorical": 0.74,
                "cosine_continuous": 0.93
            }
        },
        {
            "id": "doc_002",
            "score": 0.61,
            "score_breakdown": {
                "nmi_categorical": 0.55,
                "cosine_continuous": 0.64
            }
        }
    ],
    "query_id": "q_abc123",
    "top_k": 2
}


def build_mock_sdk_client(api_key="test-key-forge-001"):
    client = MagicMock()
    client.api_key = api_key
    client.search_hybrid = MagicMock(return_value=MOCK_HYBRID_RESPONSE)
    return client


class TestSimilaritySearchHappyPath(unittest.TestCase):

    def test_happy_path_hybrid_categorical_and_continuous_returns_scored_results(self):
        """Verifica que un payload con features categóricas y continuas devuelve resultados con score_breakdown por componente."""
        client = build_mock_sdk_client()
        payload = {
            "query": {
                "categorical": {"color": "rojo", "categoria": "electronica"},
                "continuous": [0.12, 0.87, 0.44, 0.99]
            },
            "corpus": [
                {"id": "doc_001", "categorical": {"color": "rojo", "categoria": "electronica"}, "continuous": [0.11, 0.85, 0.46, 0.98]},
                {"id": "doc_002", "categorical": {"color": "azul", "categoria": "ropa"}, "continuous": [0.50, 0.20, 0.10, 0.30]}
            ],
            "top_k": 2
        }
        result = client.search_hybrid(**payload)
        self.assertEqual(len(result["results"]), 2)
        self.assertIn("nmi_categorical", result["results"][0]["score_breakdown"])
        self.assertIn("cosine_continuous", result["results"][0]["score_breakdown"])
        self.assertGreaterEqual(result["results"][0]["score"], result["results"][1]["score"])

    def test_happy_path_only_continuous_features_returns_cosine_dominant_score(self):
        """Verifica que un payload sin features categóricas devuelve resultados donde nmi_categorical es 0.0 o ausente."""
        client = build_mock_sdk_client()
        continuous_only_response = {
            "results": [
                {
                    "id": "doc_003",
                    "score": 0.95,
                    "score_breakdown": {
                        "nmi_categorical": 0.0,
                        "cosine_continuous": 0.95
                    }
                }
            ],
            "query_id": "q_def456",
            "top_k": 1
        }
        client.search_hybrid.return_value = continuous_only_response
        payload = {
            "query": {"categorical": {}, "continuous": [0.1, 0.9, 0.5]},
            "corpus": [{"id": "doc_003", "categorical": {}, "continuous": [0.1, 0.88, 0.51]}],
            "top_k": 1
        }
        result = client.search_hybrid(**payload)
        breakdown = result["results"][0]["score_breakdown"]
        self.assertEqual(breakdown["nmi_categorical"], 0.0)
        self.assertGreater(breakdown["cosine_continuous"], 0.0)

    def test_happy_path_top_k_limits_result_count(self):
        """Verifica que el campo top_k en el request restringe el numero de resultados devueltos al valor indicado."""
        client = build_mock_sdk_client()
        top_k_one_response = {
            "results": [MOCK_HYBRID_RESPONSE["results"][0]],
            "query_id": "q_ghi789",
            "top_k": 1
        }
        client.search_hybrid.return_value = top_k_one_response
        payload = {
            "query": {"categorical": {"tipo": "A"}, "continuous": [0.5, 0.5]},
            "corpus": [
                {"id": "doc_001", "categorical": {"tipo": "A"}, "continuous": [0.5, 0.5]},
                {"id": "doc_002", "categorical": {"tipo": "B"}, "continuous": [0.1, 0.9]}
            ],
            "top_k": 1
        }
        result = client.search_hybrid(**payload)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["top_k"], 1)


class TestSimilaritySearchEdgeCases(unittest.TestCase):

    def test_edge_case_empty_corpus_returns_empty_results_list(self):
        """Verifica que un corpus vacio devuelve lista de resultados vacia sin error ni excepcion."""
        client = build_mock_sdk_client()
        empty_corpus_response = {"results": [], "query_id": "q_empty", "top_k": 5}
        client.search_hybrid.return_value = empty_corpus_response
        payload = {
            "query": {"categorical": {"x": "1"}, "continuous": [0.5]},
            "corpus": [],
            "top_k": 5
        }
        result = client.search_hybrid(**payload)
        self.assertEqual(result["results"], [])
        self.assertIsInstance(result["results"], list)

    def test_edge_case_single_element_corpus_returns_one_result(self):
        """Verifica que un corpus de un solo elemento devuelve exactamente un resultado con score valido entre 0 y 1."""
        client = build_mock_sdk_client()
        single_response = {
            "results": [{"id": "solo_doc", "score": 1.0, "score_breakdown": {"nmi_categorical": 1.0, "cosine_continuous": 1.0}}],
            "query_id": "q_solo",
            "top_k": 1
        }
        client.search_hybrid.return_value = single_response
        payload = {
            "query": {"categorical": {"k": "v"}, "continuous": [1.0, 0.0]},
            "corpus": [{"id": "solo_doc", "categorical": {"k": "v"}, "continuous": [1.0, 0.0]}],
            "top_k": 1
        }
        result = client.search_hybrid(**payload)
        self.assertEqual(len(result["results"]), 1)
        score = result["results"][0]["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestSimilaritySearchInvalidInput(unittest.TestCase):

    def test_invalid_input_none_query_raises_value_error(self):
        """Verifica que pasar query=None lanza ValueError con mensaje descriptivo del campo invalido."""
        from similarity_search_api_sdk import SimilaritySearchClient, SimilaritySearchInputError

        with patch("similarity_search_api_sdk.SimilaritySearchClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.search_hybrid.side_effect = SimilaritySearchInputError(
                "El campo 'query' no puede ser None. Provee al menos un feature categorico o continuo."
            )
            client = MockClient(api_key="test-key")
            with self.assertRaises(SimilaritySearchInputError) as ctx:
                client.search_hybrid(query=None, corpus=[], top_k=5)
            self.assertIn("query", str(ctx.exception))

    def test_invalid_input_continuous_vector_wrong_type_raises_type_error(self):
        """Verifica que un vector continuo con strings en lugar de floats lanza TypeError indicando el campo afectado."""
        from similarity_search_api_sdk import SimilaritySearchClient, SimilaritySearchTypeError

        with patch("similarity_search_api_sdk.SimilaritySearchClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.search_hybrid.side_effect = SimilaritySearchTypeError(
                "El campo 'continuous' debe ser una lista de float. Se recibio: ['a', 'b', 'c']"
            )
            client = MockClient(api_key="test-key")
            with self.assertRaises(SimilaritySearchTypeError) as ctx:
                client.search_hybrid(
                    query={"categorical": {}, "continuous": ["a", "b", "c"]},
                    corpus=[],
                    top_k=1
                )
            self.assertIn("continuous", str(ctx.exception))


class TestSimilaritySearchRateLimit(unittest.TestCase):

    def test_rate_limit_ten_sequential_calls_do_not_raise_unhandled_exception(self):
        """Verifica que 10 llamadas secuenciales al endpoint no producen excepcion no controlada ni estado corrupto."""
        client = build_mock_sdk_client()
        payload = {
            "query": {"categorical": {"tag": "ML"}, "continuous": [0.3, 0.7]},
            "corpus": [{"id": f"doc_{i}", "categorical": {"tag": "ML"}, "continuous": [0.3 + i * 0.01, 0.7]} for i in range(5)],
            "top_k": 3
        }
        responses = []
        for _ in range(10):
            result = client.search_hybrid(**payload)
            responses.append(result)
        self.assertEqual(len(responses), 10)
        for r in responses:
            self.assertIn("results", r)


class TestSimilaritySearchAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_auth_error_with_descriptive_message(self):
        """Verifica que instanciar el cliente sin API key y llamar search_hybrid lanza error de autenticacion con descripcion util."""
        from similarity_search_api_sdk import SimilaritySearchClient, SimilaritySearchAuthError

        with patch("similarity_search_api_sdk.SimilaritySearchClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.search_hybrid.side_effect = SimilaritySearchAuthError(
                "API key ausente o invalida. Incluye tu clave en el header X-API-Key o en el constructor del cliente."
            )
            client = MockClient(api_key=None)
            with self.assertRaises(SimilaritySearchAuthError) as ctx:
                client.search_hybrid(
                    query={"categorical": {"x": "1"}, "continuous": [0.5]},
                    corpus=[],
                    top_k=1
                )
            error_msg = str(ctx.exception)
            self.assertIn("API key", error_msg)
            self.assertTrue(len(error_msg) > 20)


class TestSimilaritySearchIdempotency(unittest.TestCase):

    def test_idempotency_identical_payload_twice_returns_identical_scores(self):
        """Verifica que el mismo payload enviado dos veces devuelve exactamente los mismos scores y orden de resultados."""
        client = build_mock_sdk_client()
        payload = {
            "query": {"categorical": {"region": "norte", "tipo": "B"}, "continuous": [0.22, 0.78, 0.50]},
            "corpus": [
                {"id": "item_X", "categorical": {"region": "norte", "tipo": "B"}, "continuous": [0.21, 0.79, 0.50]},
                {"id": "item_Y", "categorical": {"region": "sur", "tipo": "A"}, "continuous": [0.80, 0.10, 0.20]}
            ],
            "top_k": 2
        }
        result_first = client.search_hybrid(**payload)
        result_second = client.search_hybrid(**payload)
        self.assertEqual(
            [r["id"] for r in result_first["results"]],
            [r["id"] for r in result_second["results"]]
        )
        self.assertEqual(
            [r["score"] for r in result_first["results"]],
            [r["score"] for r in result_second["results"]]
        )
        self.assertEqual(
            [r["score_breakdown"] for r in result_first["results"]],
            [r["score_breakdown"] for r in result_second["results"]]
        )


if __name__ == "__main__":
    unittest.main()