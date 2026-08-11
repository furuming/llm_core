from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

from api.routers.openai import router as openai_router
from api.routers.router import router as compare_router
from api.routers.system import router as system_router
from infrastructure.settings.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    if settings.hf_token:
        login(
            token=settings.hf_token, add_to_git_credential=False, skip_if_logged_in=True
        )

    app = FastAPI(title="LLM Compare API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_credentials=True,
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(openai_router)
    app.include_router(compare_router)
    return app
