from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    model: str = Field(
        description="A model family key or model identifier returned by /models."
    )
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class GenerateResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class CompletionRequest(BaseModel):
    model: str = "qwen-coder"
    prompt: str
    suffix: str | None = None
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stop: str | list[str] | None = None
    stream: bool = False
    echo: bool = False


class TextCompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: None = None
    finish_reason: Literal["stop", "length"]


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextCompletionChoice]
    usage: CompletionUsage


class ModelPresetOption(BaseModel):
    family: str
    label: str
    model_name: str
