import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from application.services.completion import build_fim_prompt, truncate_at_stop
from application.services.model_catalog import ModelCatalog, UnknownModelError
from domain.entities.model_preset import ModelPreset


class FakeTokenizer:
    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = vocabulary

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}


class AutocompleteHelpersTest(unittest.TestCase):
    def test_builds_qwen_fill_in_the_middle_prompt(self) -> None:
        tokenizer = FakeTokenizer(
            {"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"}
        )
        prompt = build_fim_prompt(tokenizer, "def greet():\n", "\nprint(greet())")
        self.assertEqual(
            prompt,
            "<|fim_prefix|>def greet():\n<|fim_suffix|>\nprint(greet())<|fim_middle|>",
        )

    def test_leaves_preformatted_prompt_unchanged(self) -> None:
        self.assertEqual(
            build_fim_prompt(FakeTokenizer(set()), "existing prompt", None),
            "existing prompt",
        )

    def test_truncates_at_earliest_stop_sequence(self) -> None:
        text, stopped = truncate_at_stop("alpha END beta STOP", ["STOP", "END"])
        self.assertEqual(text, "alpha ")
        self.assertTrue(stopped)


class ModelCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ModelCatalog((ModelPreset("short", "A model", "org/model"),))

    def test_resolves_family_and_full_name(self) -> None:
        self.assertEqual(self.catalog.resolve("short"), "org/model")
        self.assertEqual(self.catalog.resolve("org/model"), "org/model")

    def test_rejects_models_outside_allowlist(self) -> None:
        with self.assertRaises(UnknownModelError):
            self.catalog.resolve("unknown/model")


if __name__ == "__main__":
    unittest.main()
