import time

from openai import AsyncOpenAI

from domain.entities.generation_config import GenerationConfig
from domain.entities.inference_result import InferenceResult
from domain.value_objects.message import Message


class VllmOpenAIGateway:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def generate(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig,
    ) -> InferenceResult:
        payload = [m.model_dump() for m in messages]

        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        elapsed = time.perf_counter() - started

        choice = response.choices[0]
        usage = getattr(response, "usage", None)

        return InferenceResult(
            engine="vllm",
            model=model,
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            elapsed_sec=elapsed,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )
