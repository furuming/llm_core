import gc
import json
import os
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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
        self._model_lock = threading.RLock()
        self._model_condition = threading.Condition(self._model_lock)
        self._active_inferences: dict[str, int] = {}
        self._unloading_models: set[str] = set()

    def device_name(self) -> str | None:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "Apple MPS" if self.device == "mps" else None

    def tokenizer(self, model_name: str):
        with self._model_lock:
            if model_name not in self._tokenizers:
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            return self._tokenizers[model_name]

    def model(self, model_name: str):
        with self._model_lock:
            if model_name not in self._models:
                self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name
                ).to(self.device)
            return self._models[model_name]

    def runtime_status(self) -> dict:
        """Return the models resident in this process and current GPU memory use."""
        with self._model_lock:
            loaded_models = sorted(self._models)

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_bytes": total_bytes,
                        "free_bytes": free_bytes,
                        "used_bytes": total_bytes - free_bytes,
                        "process_allocated_bytes": torch.cuda.memory_allocated(index),
                        "process_reserved_bytes": torch.cuda.memory_reserved(index),
                    }
                )

        return {
            "loaded_models": loaded_models,
            "vram": {
                "available": bool(devices),
                "devices": devices,
            },
        }

    @contextmanager
    def _inference(self, model_name: str):
        """Keep a model resident until one inference has completely finished."""
        with self._model_condition:
            while model_name in self._unloading_models:
                self._model_condition.wait()
            self._active_inferences[model_name] = (
                self._active_inferences.get(model_name, 0) + 1
            )

        try:
            yield self.tokenizer(model_name), self.model(model_name)
        finally:
            with self._model_condition:
                remaining = self._active_inferences[model_name] - 1
                if remaining:
                    self._active_inferences[model_name] = remaining
                else:
                    del self._active_inferences[model_name]
                    self._model_condition.notify_all()

    def unload_model(self, model_name: str) -> bool:
        """Remove a loaded model and its tokenizer and release cached device memory."""
        with self._model_condition:
            while model_name in self._unloading_models:
                self._model_condition.wait()
            self._unloading_models.add(model_name)
            while self._active_inferences.get(model_name, 0):
                self._model_condition.wait()
            model = self._models.pop(model_name, None)
            tokenizer = self._tokenizers.pop(model_name, None)

        try:
            if model is None and tokenizer is None:
                return False

            del model, tokenizer
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return True
        finally:
            with self._model_condition:
                self._unloading_models.remove(model_name)
                self._model_condition.notify_all()

    def generate(
        self,
        *,
        model_name: str,
        text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int, Literal["stop", "length"]]:
        with self._inference(model_name) as (tokenizer, model):
            return self._generate_with_model(
                tokenizer=tokenizer,
                model=model,
                text=text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    def _generate_with_model(
        self, *, tokenizer, model, text, max_tokens, temperature, top_p
    ) -> tuple[str, int, int, Literal["stop", "length"]]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

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
        reason = "length" if completion_tokens >= max_tokens else "stop"

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
        with self._inference(model_name) as (tokenizer, model):
            yield from self._stream_with_model(
                tokenizer=tokenizer,
                model=model,
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo_prompt=echo_prompt,
            )

    def _stream_with_model(
        self,
        *,
        tokenizer,
        model,
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
                counts.append(outputs[0].shape[-1] - inputs["input_ids"].shape[-1])
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
            worker.join()
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
            worker.join()
