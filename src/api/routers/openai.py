import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.openai import (
    ChatCompletionChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    GenerateRequest,
    GenerateResponse,
    TextCompletionChoice,
)
from application.services.completion import build_fim_prompt, truncate_at_stop
from application.services.model_catalog import ModelCatalog, UnknownModelError
from infrastructure.gateways.transformers_runtime import TransformersRuntime
from presentation.dependencies import get_model_catalog, get_transformers_runtime

router = APIRouter(prefix="/v1", tags=["OpenAI compatible"])
CatalogDependency = Annotated[ModelCatalog, Depends(get_model_catalog)]
RuntimeDependency = Annotated[TransformersRuntime, Depends(get_transformers_runtime)]


def resolve(catalog: ModelCatalog, model: str) -> str:
    try:
        return catalog.resolve(model)
    except UnknownModelError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/chat/completions", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    catalog: CatalogDependency,
    runtime: RuntimeDependency,
) -> GenerateResponse:
    model_name = resolve(catalog, request.model)
    tokenizer = runtime.tokenizer(model_name)
    messages = [message.model_dump() for message in request.messages]
    text = "\n".join(message.content for message in request.messages)
    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            pass
    result, prompt_tokens, completion_tokens, reason = runtime.generate(
        model_name=model_name,
        text=text,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return GenerateResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result),
                finish_reason=reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@router.post("/completions", response_model=CompletionResponse)
async def complete(
    request: CompletionRequest,
    catalog: CatalogDependency,
    runtime: RuntimeDependency,
):
    model_name = resolve(catalog, request.model)
    prompt = build_fim_prompt(
        runtime.tokenizer(model_name), request.prompt, request.suffix
    )
    completion_id, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
    if request.stream:
        return StreamingResponse(
            runtime.stream(
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                echo_prompt=request.prompt if request.echo else None,
            ),
            media_type="text/event-stream",
        )
    result, prompt_tokens, completion_tokens, reason = runtime.generate(
        model_name=model_name,
        text=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    result, stopped = truncate_at_stop(result, request.stop)
    if stopped:
        reason = "stop"
    if request.echo:
        result = request.prompt + result
    return CompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[TextCompletionChoice(text=result, finish_reason=reason)],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
