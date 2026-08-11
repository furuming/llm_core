import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from presentation.app import create_app
from presentation.dependencies import get_transformers_runtime


class FakeRuntime:
    device = "cpu"

    def device_name(self):
        return None


class AppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.dependency_overrides[get_transformers_runtime] = FakeRuntime
        self.client = TestClient(self.app)

    def test_health_route_is_composed(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device"], "cpu")

    def test_models_route_uses_catalog(self) -> None:
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        self.assertIn("qwen-coder", [item["family"] for item in response.json()])

    def test_unknown_completion_model_returns_bad_request(self) -> None:
        response = self.client.post(
            "/v1/completions", json={"model": "unknown", "prompt": "x"}
        )
        self.assertEqual(response.status_code, 400)
