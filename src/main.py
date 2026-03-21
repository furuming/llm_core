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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "google/gemma-3-4b-it"


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = detect_device()


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
    return AutoTokenizer.from_pretrained(MODEL_ID)


@lru_cache
def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
        model=MODEL_ID,
        device=DEVICE,
        text=result_text,
        elapsed_sec=elapsed,
    )
