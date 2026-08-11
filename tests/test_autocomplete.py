import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import main


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

        prompt = main.build_fim_prompt(tokenizer, "def greet():\n", "\nprint(greet())")

        self.assertEqual(
            prompt,
            "<|fim_prefix|>def greet():\n"
            "<|fim_suffix|>\nprint(greet())<|fim_middle|>",
        )

    def test_leaves_preformatted_prompt_unchanged(self) -> None:
        tokenizer = FakeTokenizer(set())

        self.assertEqual(main.build_fim_prompt(tokenizer, "existing prompt", None), "existing prompt")

    def test_truncates_at_earliest_stop_sequence(self) -> None:
        text, stopped = main.truncate_at_stop("alpha END beta STOP", ["STOP", "END"])

        self.assertEqual(text, "alpha ")
        self.assertTrue(stopped)


class CompletionEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_openai_compatible_completion(self) -> None:
        request = main.CompletionRequest(model="qwen-coder", prompt="pri", stop="\n")
        tokenizer = FakeTokenizer(set())

        with (
            patch.object(main, "get_tokenizer_by_model", return_value=tokenizer),
            patch.object(main, "generate_text", return_value=("print('ok')\nmore", 3, 5, "length")),
        ):
            response = await main.complete(request)

        self.assertEqual(response.object, "text_completion")
        self.assertEqual(response.model, "Qwen/Qwen2.5-Coder-1.5B")
        self.assertEqual(response.choices[0].text, "print('ok')")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 8)

    async def test_stream_ends_with_done_event(self) -> None:
        request = main.CompletionRequest(model="qwen-coder", prompt="pri", stream=True)

        with (
            patch.object(main, "get_tokenizer_by_model", return_value=FakeTokenizer(set())),
            patch.object(main, "generate_text", return_value=("nt()", 1, 2, "stop")),
        ):
            response = await main.complete(request)
            events = [chunk async for chunk in response.body_iterator]

        self.assertIn('"text": "nt()"', events[0])
        self.assertEqual(events[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
