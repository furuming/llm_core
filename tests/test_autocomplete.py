import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import main
import torch


class FakeTokenizer:
    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = vocabulary

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}


class StreamingTokenizer(FakeTokenizer):
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, token_ids, **kwargs) -> str:
        pieces = {1: "pri", 2: "one ", 3: "two "}
        return "".join(pieces[int(token)] for token in token_ids)


class StreamingModel:
    def __init__(self) -> None:
        self.worker_thread: threading.Thread | None = None

    def generate(self, input_ids, streamer, **kwargs):
        self.worker_thread = threading.current_thread()
        streamer.put(input_ids)
        streamer.put(torch.tensor([[2]]))
        streamer.put(torch.tensor([[3]]))
        streamer.end()
        return torch.tensor([[1, 2, 3]])


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
        events = iter(
            [
                'data: {"choices": [{"text": "nt()"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )

        with (
            patch.object(main, "get_tokenizer_by_model", return_value=FakeTokenizer(set())),
            patch.object(main, "stream_completion_events", return_value=events),
        ):
            response = await main.complete(request)
            events = [chunk async for chunk in response.body_iterator]

        self.assertIn('"text": "nt()"', events[0])
        self.assertEqual(events[-1], "data: [DONE]\n\n")

    def test_stream_does_not_load_or_generate_until_consumed(self) -> None:
        request = main.CompletionRequest(model="qwen-coder", prompt="pri", stream=True)

        with patch.object(main, "get_model_by_model_name") as get_model:
            events = main.stream_completion_events(
                completion_id="cmpl-test",
                created=0,
                model_name="Qwen/Qwen2.5-Coder-1.5B",
                prompt="pri",
                request=request,
            )

            get_model.assert_not_called()
            events.close()

    def test_streamer_emits_incremental_chunks_from_worker(self) -> None:
        request = main.CompletionRequest(model="qwen-coder", prompt="pri", stream=True)
        model = StreamingModel()

        with (
            patch.object(main, "get_tokenizer_by_model", return_value=StreamingTokenizer(set())),
            patch.object(main, "get_model_by_model_name", return_value=model),
        ):
            events = list(
                main.stream_completion_events(
                    completion_id="cmpl-test",
                    created=0,
                    model_name="Qwen/Qwen2.5-Coder-1.5B",
                    prompt="pri",
                    request=request,
                )
            )

        text_events = [event for event in events if '"text": "one "' in event or '"text": "two "' in event]
        self.assertEqual(len(text_events), 2)
        self.assertIsNot(model.worker_thread, threading.current_thread())
        self.assertEqual(events[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
