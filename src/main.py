# from fastapi import FastAPI

# from api.routers.router import router as compare_router

# app = FastAPI(title="LLM Compare API")
# app.include_router(compare_router)


# @app.get("/health")
# async def health() -> dict:
#     return {"status": "ok"}


import os
import time
from functools import lru_cache

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from infrastructure.settings.config import get_settings


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SETTINGS = get_settings()
DEFAULT_MODEL_NAME = SETTINGS.default_model_name


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = detect_device()
print(DEVICE)


def login_to_huggingface_if_needed() -> None:
    if not SETTINGS.hf_token:
        return

    login(token=SETTINGS.hf_token, add_to_git_credential=False, skip_if_logged_in=True)


login_to_huggingface_if_needed()


class CompareRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2


class CompareResponse(BaseModel):
    model: str
    device: str
    text: str
    elapsed_sec: float


@lru_cache
def get_tokenizer():
    return AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)


@lru_cache
def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        # torch_dtype=torch.float16 if DEVICE == "mps" else "auto",
    )
    return model.to(DEVICE)


app = FastAPI(title="LLM Compare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"]
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device": DEVICE}


@app.post("/generate", response_model=CompareResponse)
async def generate(req: CompareRequest) -> CompareResponse:
    tokenizer = get_tokenizer()
    model = get_model()

    messages = [
        {"role": "user", "content": req.prompt},
    ]

    # chat_template が設定されていない場合や、apply_chat_template が例外を投げる場合には
    # 元のプロンプトをそのまま使うフォールバックを用意する
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
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            # temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return CompareResponse(
        model=DEFAULT_MODEL_NAME,
        device=DEVICE,
        text=result_text,
        elapsed_sec=elapsed,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.app_port)
