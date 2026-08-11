import os
import time
import uuid
from functools import lru_cache
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    preset = MODEL_PRESETS.get(req.model)
    model_name = preset["model_name"] if preset is not None else req.model
    known_model_names = {item["model_name"] for item in MODEL_PRESETS.values()}
    if model_name not in known_model_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {req.model}",
        )

    tokenizer = get_tokenizer_by_model(model_name)
    model = get_model_by_model_name(model_name)

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

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    generation_options = {
        "max_new_tokens": req.max_tokens,
        "do_sample": req.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if req.temperature > 0:
        generation_options.update(temperature=req.temperature, top_p=req.top_p)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_options,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = len(generated_ids)
    finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.app_port)
