from pydantic import BaseModel

from domain.entities.generation_config import GenerationConfig
from domain.entities.model_target import ModelTarget
from domain.value_objects.message import Message


class CompareRequest(BaseModel):
    models: list[ModelTarget]
    messages: list[Message]
    config: GenerationConfig = GenerationConfig()
