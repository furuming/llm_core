from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas.request import CompareRequest
from api.schemas.response import CompareResponse
from application.usecases.llm_models import LLMModelsUseCase
from presentation.dependencies import get_usecase

router = APIRouter(prefix="/compare", tags=["compare"])
UseCaseDependency = Annotated[LLMModelsUseCase, Depends(get_usecase)]


@router.post("", response_model=CompareResponse)
async def compare(
    request: CompareRequest,
    usecase: UseCaseDependency,
) -> CompareResponse:
    result = await usecase.execute(
        models=request.models,
        messages=request.messages,
        config=request.config,
    )
    return CompareResponse(**result)
