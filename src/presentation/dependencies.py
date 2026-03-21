from functools import lru_cache

from application.usecases.llm_models import LLMModelsUseCase
from infrastructure.gateways.vllm_openai_gateway import VllmOpenAIGateway
from infrastructure.repositories.in_memory_result_repository import (
    InMemoryResultRepository,
)
from infrastructure.settings.config import get_settings


@lru_cache
def get_usecase() -> LLMModelsUseCase:
    settings = get_settings()

    gateway = VllmOpenAIGateway(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=settings.request_timeout_sec,
    )
    repository = InMemoryResultRepository()

    return LLMModelsUseCase(
        chat_gateway=gateway,
        result_repository=repository,
    )