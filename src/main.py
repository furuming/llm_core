import json
import os
import time
import uuid
from functools import lru_cache
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import login
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from infrastructure.settings.config import get_settings


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SETTINGS = get_settings()
DEFAULT_MODEL_FAMILY = SETTINGS.default_model_family
DEFAULT_MODEL_NAME = SETTINGS.default_model_name
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "gemma": {
        "label": "Gemma 3 4B",
        "model_name": DEFAULT_MODEL_NAME,
    },
    "llama": {
        "label": "Llama 3.2 3B Instruct",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "qwen": {
        "label": "Qwen2.5 3B Instruct",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
    },
    "qwen-coder": {
        "label": "Qwen2.5 Coder 1.5B",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
    },
    "phi": {
        "label": "Phi-3.5 Mini Instruct",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
    },
}


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = detect_device()
print(DEVICE)


def get_torch_device_name() -> str | None:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if DEVICE == "mps":
        return "Apple MPS"
    return None


def login_to_huggingface_if_needed() -> None:
    if not SETTINGS.hf_token:
        return

    login(token=SETTINGS.hf_token, add_to_git_credential=False, skip_if_logged_in=True)


login_to_huggingface_if_needed()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    model: str = Field(
        examples=["gemma", "meta-llama/Llama-3.2-3B-Instruct"],
        description="A model family key or model identifier returned by /models.",
    )
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class CompletionRequest(BaseModel):
    """OpenAI-compatible legacy completion request used by Continue autocomplete."""

    model: str = Field(
        default="qwen-coder",
        description="A model family key or model identifier returned by /models.",
    )
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


@lru_cache
def get_tokenizer_by_model(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache
def get_model_by_model_name(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model.to(DEVICE)


def resolve_model_name(model: str) -> str:
    preset = MODEL_PRESETS.get(model)
    model_name = preset["model_name"] if preset is not None else model
    known_model_names = {item["model_name"] for item in MODEL_PRESETS.values()}
    if model_name not in known_model_names:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    return model_name


def build_fim_prompt(tokenizer, prefix: str, suffix: str | None) -> str:
    """Build a FIM prompt when the client uses OpenAI's ``suffix`` field.

    Continue can also send an already formatted FIM prompt; in that case suffix is
    omitted and the prompt is passed through unchanged.
    """
    if suffix is None:
        return prefix

    vocabulary = tokenizer.get_vocab()
    token_sets = (
        ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
        ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    )
    for prefix_token, suffix_token, middle_token in token_sets:
        if all(token in vocabulary for token in (prefix_token, suffix_token, middle_token)):
            return f"{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}"

    # A non-FIM tokenizer can still provide useful prefix completion. Appending
    # the suffix would cause the model to continue after it instead of filling it.
    return prefix


def generate_text(
    *,
    model_name: str,
    text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, int, Literal["stop", "length"]]:
    tokenizer = get_tokenizer_by_model(model_name)
    model = get_model_by_model_name(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    generation_options = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_options.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_options)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = len(generated_ids)
    finish_reason = "length" if completion_tokens >= max_tokens else "stop"
    return result_text, prompt_tokens, completion_tokens, finish_reason


def truncate_at_stop(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    if stop is None:
        return text, False
    stops = [stop] if isinstance(stop, str) else stop
    positions = [position for value in stops if value and (position := text.find(value)) >= 0]
    if not positions:
        return text, False
    return text[:min(positions)], True


app = FastAPI(title="LLM Compare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_name": get_torch_device_name(),
    }


@app.get("/models", response_model=list[ModelPresetOption])
async def models() -> list[ModelPresetOption]:
    return [
        ModelPresetOption(
            family=family,
            label=preset["label"],
            model_name=preset["model_name"],
        )
        for family, preset in MODEL_PRESETS.items()
    ]


@app.post("/v1/chat/completions", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    model_name = resolve_model_name(req.model)

    tokenizer = get_tokenizer_by_model(model_name)
    messages = [message.model_dump() for message in req.messages]

    text = "\n".join(message.content for message in req.messages)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                template=chat_template,
            )
        except ValueError:
            pass

    result_text, prompt_tokens, completion_tokens, finish_reason = generate_text(
        model_name=model_name,
        text=text,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    return GenerateResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result_text),
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def complete(req: CompletionRequest):
    """Serve Continue's OpenAI-provider autocomplete requests."""
    model_name = resolve_model_name(req.model)
    tokenizer = get_tokenizer_by_model(model_name)
    prompt = build_fim_prompt(tokenizer, req.prompt, req.suffix)
    result_text, prompt_tokens, completion_tokens, finish_reason = generate_text(
        model_name=model_name,
        text=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    result_text, stopped = truncate_at_stop(result_text, req.stop)
    if stopped:
        finish_reason = "stop"
    if req.echo:
        result_text = req.prompt + result_text

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    choice = TextCompletionChoice(text=result_text, finish_reason=finish_reason)
    response = CompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[choice],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if not req.stream:
        return response

    def event_stream():
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [choice.model_dump()],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.app_port)
