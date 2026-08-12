from typing import Annotated

import torch
from fastapi import APIRouter, Depends

from api.schemas.openai import ModelPresetOption
from api.schemas.system import RuntimeStatusResponse
from application.services.model_catalog import ModelCatalog
from infrastructure.gateways.transformers_runtime import TransformersRuntime
from presentation.dependencies import get_model_catalog, get_transformers_runtime

router = APIRouter(tags=["system"])
CatalogDependency = Annotated[ModelCatalog, Depends(get_model_catalog)]
RuntimeDependency = Annotated[TransformersRuntime, Depends(get_transformers_runtime)]


@router.get("/health")
async def health(
    runtime: RuntimeDependency,
) -> dict:
    return {
        "status": "ok",
        "device": runtime.device,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_name": runtime.device_name(),
    }


@router.get("/models", response_model=list[ModelPresetOption])
async def models(
    catalog: CatalogDependency,
) -> list[ModelPresetOption]:
    return [
        ModelPresetOption(
            family=item.family, label=item.label, model_name=item.model_name
        )
        for item in catalog.list()
    ]


@router.get("/runtime", response_model=RuntimeStatusResponse)
def runtime_status(runtime: RuntimeDependency) -> RuntimeStatusResponse:
    """Collect potentially blocking runtime metrics in FastAPI's thread pool."""
    return RuntimeStatusResponse.model_validate(runtime.runtime_status())
