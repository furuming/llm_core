from typing import Protocol

from domain.entities.generation_config import GenerationConfig
from domain.entities.inference_result import InferenceResult
from domain.value_objects.message import Message


class ChatGateway(Protocol):
    async def generate(
        self,
        model: str,
        messages: list[Message],
        config: GenerationConfig,
    ) -> InferenceResult:
     ...
