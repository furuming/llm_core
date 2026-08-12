from typing import Annotated

import torch
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.openai import ModelPresetOption
from api.schemas.system import ModelUnloadResponse, RuntimeStatusResponse
from application.services.model_catalog import ModelCatalog, UnknownModelError
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


@router.delete("/runtime/models/{model:path}", response_model=ModelUnloadResponse)
def unload_model(
    model: str,
    catalog: CatalogDependency,
    runtime: RuntimeDependency,
) -> ModelUnloadResponse:
    """Unload a model in FastAPI's thread pool because memory cleanup can block."""
    try:
        model_name = catalog.resolve(model)
    except UnknownModelError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    unloaded = runtime.unload_model(model_name)
    loaded_models = runtime.runtime_status()["loaded_models"]
    return ModelUnloadResponse(
        model=model_name,
        unloaded=unloaded,
        loaded_models=loaded_models,
    )
