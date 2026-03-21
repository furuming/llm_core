import asyncio
import uuid

from domain.entities.generation_config import GenerationConfig
from domain.entities.model_target import ModelTarget
from domain.entities.inference_result import InferenceResult
from domain.ports.chat_gateway import ChatGateway
from domain.ports.result_repository import ResultRepository
from domain.value_objects.message import Message


class LLMModelsUseCase:
    def __init__(
        self,
        chat_gateway: ChatGateway,
        result_repository: ResultRepository,
    ) -> None:
        self.chat_gateway = chat_gateway
        self.result_repository = result_repository

    async def execute(
        self,
        models: list[ModelTarget],
        messages: list[Message],
        config: GenerationConfig,
    ) -> dict:
        comparison_id = str(uuid.uuid4())

        tasks = [
            self.chat_gateway.generate(
                model=model.name,
                messages=messages,
                config=config,
            )
            for model in models
        ]

        results: list[InferenceResult] = await asyncio.gather(*tasks)

        await self.result_repository.save_comparison(
            comparison_id=comparison_id,
            results=results,
        )

        return {
            "comparison_id": comparison_id,
            "results": results,
        }