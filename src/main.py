import os
import time
from functools import lru_cache

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


class CompareRequest(BaseModel):
    prompt: str
    model_family: str = Field(
        default=DEFAULT_MODEL_FAMILY,
        examples=["gemma", "llama", "qwen", "phi"],
        description="Model family key.",
    )
    max_new_tokens: int = 256
    temperature: float = 0.2


class CompareResponse(BaseModel):
    model: str
    device: str
    text: str
    elapsed_sec: float


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


@app.post("/generate", response_model=CompareResponse)
async def generate(req: CompareRequest) -> CompareResponse:
    preset = MODEL_PRESETS.get(req.model_family)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_family: {req.model_family}",
        )

    model_name = preset["model_name"]
    tokenizer = get_tokenizer_by_model(model_name)
    model = get_model_by_model_name(model_name)

    messages = [
        {"role": "user", "content": req.prompt},
    ]

    text = req.prompt
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
            text = req.prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return CompareResponse(
        model=model_name,
        device=DEVICE,
        text=result_text,
        elapsed_sec=elapsed,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.app_port)
