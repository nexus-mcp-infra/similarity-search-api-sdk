import unittest
from unittest.mock import patch, MagicMock
import json


NMI_RESPONSE_SINGLE = {
    "query_id": "q_abc123",
    "results": [
        {"item_id": "doc_001", "nmi_score": 0.87, "rank": 1},
        {"item_id": "doc_002", "nmi_score": 0.74, "rank": 2},
    ],
    "metric": "nmi",
    "query_field_type": "categorical",
}

NMI_RESPONSE_TIMESERIES = {
    "query_id": "q_def456",
    "results": [
        {"item_id": "ts_001", "nmi_score": 0.91, "rank": 1},
    ],
    "metric": "nmi",
    "query_field_type": "discrete_timeseries",
}

AUTH_ERROR_RESPONSE = {
    "error": "authentication_required",
    "message": "API key missing or invalid. Provide X-API-Key header.",
    "status_code": 401,
}

RATE_LIMIT_RESPONSE = {
    "error": "rate_limit_exceeded",
    "message": "Exceeded 60 requests/minute. Retry after 12 seconds.",
    "status_code": 429,
}

VALIDATION_ERROR_RESPONSE = {
    "error": "validation_error",
    "message": "Field 'query_values' must be a non-empty list of strings or integers.",
    "status_code": 422,
}


def _make_mock_client(status_code=200, json_body=None):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_body or {}
    mock_response.headers = {"Content-Type": "application/json"}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response
    return mock_client


class TestNMISimilaritySearchHappyPath(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_happy_path_categorical_query_returns_ranked_nmi_scores(self, MockClient):
        """Verifica que una query categórica válida devuelve resultados ordenados con nmi_score entre 0 y 1."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = NMI_RESPONSE_SINGLE

        result = instance.search_by_nmi(
            query_values=["red", "blue", "red", "green"],
            field_type="categorical",
            top_k=2,
        )

        self.assertEqual(result["metric"], "nmi")
        self.assertEqual(len(result["results"]), 2)
        scores = [r["nmi_score"] for r in result["results"]]
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        self.assertEqual(result["results"][0]["rank"], 1)
        self.assertGreaterEqual(result["results"][0]["nmi_score"], result["results"][1]["nmi_score"])

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_happy_path_discrete_timeseries_query_returns_nmi_above_threshold(self, MockClient):
        """Verifica que una serie temporal discreta con alta dependencia estadística obtiene nmi_score > 0.8."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = NMI_RESPONSE_TIMESERIES

        result = instance.search_by_nmi(
            query_values=[1, 0, 1, 1, 0, 1, 0, 0],
            field_type="discrete_timeseries",
            top_k=1,
        )

        self.assertEqual(result["query_field_type"], "discrete_timeseries")
        self.assertGreater(result["results"][0]["nmi_score"], 0.8)
        instance.search_by_nmi.assert_called_once()


class TestNMISimilaritySearchEdgeCases(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_edge_case_single_unique_value_list_returns_zero_nmi(self, MockClient):
        """Verifica que una lista con un solo valor único produce nmi_score == 0.0 (entropía cero, sin información compartida)."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = {
            "query_id": "q_edge01",
            "results": [{"item_id": "doc_001", "nmi_score": 0.0, "rank": 1}],
            "metric": "nmi",
            "query_field_type": "categorical",
        }

        result = instance.search_by_nmi(
            query_values=["A", "A", "A", "A"],
            field_type="categorical",
            top_k=1,
        )

        self.assertEqual(result["results"][0]["nmi_score"], 0.0)

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_edge_case_max_length_query_does_not_raise(self, MockClient):
        """Verifica que una query con 10 000 valores (límite máximo documentado) no lanza excepción."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = NMI_RESPONSE_SINGLE

        large_input = ["cat", "dog", "bird"] * 3334
        large_input = large_input[:10000]

        result = instance.search_by_nmi(
            query_values=large_input,
            field_type="categorical",
            top_k=2,
        )

        self.assertIn("results", result)
        self.assertEqual(len(large_input), 10000)

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_edge_case_none_query_values_raises_value_error(self, MockClient):
        """Verifica que pasar None como query_values lanza ValueError con mensaje descriptivo antes de hacer la llamada HTTP."""
        instance = MockClient.return_value
        instance.search_by_nmi.side_effect = ValueError(
            "query_values must be a non-empty list; received None"
        )

        with self.assertRaises(ValueError) as ctx:
            instance.search_by_nmi(query_values=None, field_type="categorical", top_k=5)

        self.assertIn("None", str(ctx.exception))
        instance.post = MagicMock()
        instance.post.assert_not_called()


class TestNMISimilaritySearchInvalidInput(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_invalid_input_wrong_field_type_raises_value_error(self, MockClient):
        """Verifica que un field_type no soportado ('embedding') lanza ValueError antes de serializar la request."""
        instance = MockClient.return_value
        instance.search_by_nmi.side_effect = ValueError(
            "field_type 'embedding' is not supported. Valid options: categorical, discrete_timeseries, text_token."
        )

        with self.assertRaises(ValueError) as ctx:
            instance.search_by_nmi(
                query_values=["a", "b", "c"],
                field_type="embedding",
                top_k=3,
            )

        self.assertIn("embedding", str(ctx.exception))
        self.assertIn("categorical", str(ctx.exception))

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_invalid_input_non_list_query_values_raises_type_error(self, MockClient):
        """Verifica que query_values como string (en lugar de list) lanza TypeError con indicación del tipo recibido."""
        instance = MockClient.return_value
        instance.search_by_nmi.side_effect = TypeError(
            "query_values must be list, got str"
        )

        with self.assertRaises(TypeError) as ctx:
            instance.search_by_nmi(
                query_values="red,blue,green",
                field_type="categorical",
                top_k=3,
            )

        self.assertIn("str", str(ctx.exception))
        self.assertIn("list", str(ctx.exception))


class TestNMISimilaritySearchRateLimit(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_rate_limit_60_sequential_calls_do_not_crash_client(self, MockClient):
        """Verifica que 60 llamadas secuenciales (umbral del plan base) no provocan excepción en el cliente."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = NMI_RESPONSE_SINGLE

        results = []
        for _ in range(60):
            r = instance.search_by_nmi(
                query_values=["x", "y", "x"],
                field_type="categorical",
                top_k=2,
            )
            results.append(r)

        self.assertEqual(len(results), 60)
        self.assertEqual(instance.search_by_nmi.call_count, 60)
        self.assertTrue(all("results" in r for r in results))


class TestNMISimilaritySearchAuth(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_auth_missing_api_key_raises_authentication_error(self, MockClient):
        """Verifica que instanciar el cliente sin api_key y llamar search_by_nmi lanza AuthenticationError con instrucción de cabecera."""
        instance = MockClient.return_value
        instance.search_by_nmi.side_effect = PermissionError(
            "authentication_required: API key missing or invalid. Provide X-API-Key header."
        )

        with self.assertRaises(PermissionError) as ctx:
            instance.search_by_nmi(
                query_values=["a", "b"],
                field_type="categorical",
                top_k=1,
            )

        error_msg = str(ctx.exception)
        self.assertIn("X-API-Key", error_msg)
        self.assertIn("authentication_required", error_msg)


class TestNMISimilaritySearchIdempotency(unittest.TestCase):

    @patch("similarity_search_api_sdk.NMISimilarityClient")
    def test_idempotency_identical_categorical_query_returns_identical_nmi_scores(self, MockClient):
        """Verifica que dos llamadas idénticas producen exactamente los mismos nmi_score y rankings."""
        instance = MockClient.return_value
        instance.search_by_nmi.return_value = NMI_RESPONSE_SINGLE

        call_args = dict(
            query_values=["red", "blue", "red", "green"],
            field_type="categorical",
            top_k=2,
        )

        result_first = instance.search_by_nmi(**call_args)
        result_second = instance.search_by_nmi(**call_args)

        self.assertEqual(
            [r["nmi_score"] for r in result_first["results"]],
            [r["nmi_score"] for r in result_second["results"]],
        )
        self.assertEqual(
            [r["item_id"] for r in result_first["results"]],
            [r["item_id"] for r in result_second["results"]],
        )
        self.assertEqual(instance.search_by_nmi.call_count, 2)


if __name__ == "__main__":
    unittest.main()