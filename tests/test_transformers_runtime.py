import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from infrastructure.gateways.transformers_runtime import TransformersRuntime


class TransformersRuntimeStatusTest(unittest.TestCase):
    def test_reports_loaded_models_without_gpu(self) -> None:
        runtime = TransformersRuntime(device="cpu")
        runtime._models = {"org/model-b": object(), "org/model-a": object()}

        with patch(
            "infrastructure.gateways.transformers_runtime.torch.cuda.is_available",
            return_value=False,
        ):
            status = runtime.runtime_status()

        self.assertEqual(status["loaded_models"], ["org/model-a", "org/model-b"])
        self.assertEqual(status["vram"], {"available": False, "devices": []})

    def test_reports_gpu_and_process_memory_usage(self) -> None:
        runtime = TransformersRuntime(device="cuda")

        with (
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.device_count",
                return_value=1,
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.mem_get_info",
                return_value=(400, 1000),
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.get_device_name",
                return_value="Fake GPU",
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.memory_allocated",
                return_value=500,
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.memory_reserved",
                return_value=550,
            ),
        ):
            status = runtime.runtime_status()

        device = status["vram"]["devices"][0]
        self.assertEqual(device["used_bytes"], 600)
        self.assertEqual(device["process_allocated_bytes"], 500)
        self.assertEqual(device["process_reserved_bytes"], 550)

    def test_unload_model_removes_model_and_tokenizer(self) -> None:
        runtime = TransformersRuntime(device="cpu")
        runtime._models = {"org/model": object()}
        runtime._tokenizers = {"org/model": object()}

        with patch("infrastructure.gateways.transformers_runtime.gc.collect") as collect:
            unloaded = runtime.unload_model("org/model")

        self.assertTrue(unloaded)
        self.assertNotIn("org/model", runtime._models)
        self.assertNotIn("org/model", runtime._tokenizers)
        collect.assert_called_once_with()

    def test_unload_missing_model_does_not_run_cleanup(self) -> None:
        runtime = TransformersRuntime(device="cpu")

        with patch("infrastructure.gateways.transformers_runtime.gc.collect") as collect:
            unloaded = runtime.unload_model("org/missing")

        self.assertFalse(unloaded)
        collect.assert_not_called()

    def test_unload_cuda_model_releases_cached_memory(self) -> None:
        runtime = TransformersRuntime(device="cuda")
        runtime._models = {"org/model": object()}

        with (
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "infrastructure.gateways.transformers_runtime.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            runtime.unload_model("org/model")

        empty_cache.assert_called_once_with()

    def test_unload_waits_for_active_inference(self) -> None:
        runtime = TransformersRuntime(device="cpu")
        model = object()
        runtime._models = {"org/model": model}
        runtime._tokenizers = {"org/model": object()}
        inference_started = threading.Event()
        finish_inference = threading.Event()
        unload_finished = threading.Event()

        def infer() -> None:
            with runtime._inference("org/model") as (_, active_model):
                self.assertIs(active_model, model)
                inference_started.set()
                finish_inference.wait(timeout=2)

        def unload() -> None:
            runtime.unload_model("org/model")
            unload_finished.set()

        inference_thread = threading.Thread(target=infer)
        unload_thread = threading.Thread(target=unload)
        inference_thread.start()
        self.assertTrue(inference_started.wait(timeout=2))
        unload_thread.start()

        self.assertFalse(unload_finished.wait(timeout=0.05))
        self.assertIn("org/model", runtime.runtime_status()["loaded_models"])
        finish_inference.set()
        inference_thread.join(timeout=2)
        unload_thread.join(timeout=2)

        self.assertFalse(inference_thread.is_alive())
        self.assertFalse(unload_thread.is_alive())
        self.assertTrue(unload_finished.is_set())
        self.assertNotIn("org/model", runtime.runtime_status()["loaded_models"])


if __name__ == "__main__":
    unittest.main()
