from functools import lru_cache

from application.services.model_catalog import ModelCatalog
from application.usecases.llm_models import LLMModelsUseCase
from domain.entities.model_preset import ModelPreset
from infrastructure.gateways.transformers_runtime import TransformersRuntime
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


@lru_cache
def get_model_catalog() -> ModelCatalog:
    settings = get_settings()
    return ModelCatalog(
        (
            ModelPreset("gemma", "Gemma 3 4B IT", "google/gemma-3-4b-it"),
            ModelPreset(
                "llama", "Llama 3.2 3B Instruct", "meta-llama/Llama-3.2-3B-Instruct"
            ),
            ModelPreset("qwen", "Qwen2.5 7B Instruct", "Qwen/Qwen2.5-7B-Instruct"),
            ModelPreset("qwen-coder", "Qwen2.5 Coder 1.5B", "Qwen/Qwen2.5-Coder-1.5B"),
            ModelPreset(
                "phi", "Phi-3.5 Mini Instruct", "microsoft/Phi-3.5-mini-instruct"
            ),
        )
    )


@lru_cache
def get_transformers_runtime() -> TransformersRuntime:
    return TransformersRuntime()
