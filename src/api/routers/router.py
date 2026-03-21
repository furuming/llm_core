from fastapi import APIRouter, Depends

from api.schemas.request import CompareRequest
from api.schemas.response import CompareResponse
from presentation.dependencies import get_usecase
from application.usecases.llm_models import LLMModelsUseCase

router = APIRouter(prefix="/compare", tags=["compare"])

@router.post("", response_model=CompareResponse)
async def compare(
    request: CompareRequest,
    usecase: LLMModelsUseCase = Depends(get_usecase),
) -> CompareResponse:
    result = await usecase.execute(
        models=request.models,
        messages=request.messages,
        config=request.config,
    )
    return CompareResponse(**result)
