import sys
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


if __name__ == "__main__":
    unittest.main()
