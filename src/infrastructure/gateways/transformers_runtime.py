import json
import os
import queue
import threading
from collections.abc import Iterator
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self, cancelled: threading.Event) -> None:
        self.cancelled = cancelled

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancelled.is_set()


class TransformersRuntime:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or detect_device()
        self._tokenizers: dict[str, object] = {}
        self._models: dict[str, object] = {}

    def device_name(self) -> str | None:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "Apple MPS" if self.device == "mps" else None

    def tokenizer(self, model_name: str):
        if model_name not in self._tokenizers:
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizers[model_name]

    def model(self, model_name: str):
        if model_name not in self._models:
            self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(self.device)
        return self._models[model_name]
    
    def generate(
        self,
        *,
        model_name: str,
        text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int, Literal["stop", "length"]]:
        tokenizer = self.tokenizer(model_name)
        model = self.model(model_name)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        print("=== GENERATE ===")
        print("text:", repr(text))
        print("input_ids:", inputs["input_ids"].shape)
        print("attention_mask:", inputs.get("attention_mask", None))
        print("================")

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        options = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if temperature > 0:
            options.update(
                temperature=temperature,
                top_p=top_p,
            )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **options,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]

        completion_tokens = len(generated_ids)
        reason = (
            "length"
            if completion_tokens >= max_tokens
            else "stop"
        )

        return (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ),
            input_length,
            completion_tokens,
            reason,
        )

    def stream(
        self,
        *,
        completion_id: str,
        created: int,
        model_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
        echo_prompt: str | None,
    ) -> Iterator[str]:
        tokenizer = self.tokenizer(model_name)
        model = self.model(model_name)
        inputs = {
            key: value.to(self.device)
            for key, value in tokenizer(prompt, return_tensors="pt").items()
        }
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.1
        )
        cancelled, errors, counts = threading.Event(), [], []
        options = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList(
                [CancellationStoppingCriteria(cancelled)]
            ),
        }
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)

        def worker_fn() -> None:
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **options)
                counts.append(outputs[0].shape[0] - inputs["input_ids"].shape[1])
            except Exception as error:  # noqa: BLE001 - re-raised by the iterator
                errors.append(error)
                streamer.on_finalized_text("", stream_end=True)

        worker = threading.Thread(target=worker_fn, daemon=True)
        worker.start()
        stops = (
            []
            if stop is None
            else [stop]
            if isinstance(stop, str)
            else [item for item in stop if item]
        )
        retained, stopped, first = "", False, True

        def event(text: str, reason: str | None = None) -> str:
            nonlocal first
            if first and echo_prompt is not None:
                text = echo_prompt + text
            first = False
            body = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": reason,
                    }
                ],
            }
            return f"data: {json.dumps(body)}\n\n"

        try:
            iterator = iter(streamer)
            while True:
                try:
                    retained += next(iterator)
                except queue.Empty:
                    if worker.is_alive():
                        continue
                    break
                except StopIteration:
                    break
                positions = [
                    retained.find(value) for value in stops if retained.find(value) >= 0
                ]
                if positions:
                    output, retained, stopped = retained[: min(positions)], "", True
                    if output or first:
                        yield event(output)
                    cancelled.set()
                    break
                keep = max((len(value) - 1 for value in stops), default=0)
                if len(retained) > keep:
                    output = retained[:-keep] if keep else retained
                    retained = retained[-keep:] if keep else ""
                    if output:
                        yield event(output)
            if errors:
                raise errors[0]
            if retained:
                yield event(retained)
            reason = (
                "length"
                if counts and counts[0] >= max_tokens and not stopped
                else "stop"
            )
            yield event("", reason)
            yield "data: [DONE]\n\n"
        finally:
            cancelled.set()
