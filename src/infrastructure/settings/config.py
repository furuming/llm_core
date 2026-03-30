from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "llm-compare"
    app_port: int = 9001
    env: str = "local"
    hf_token: Optional[str] = None
    default_model_name: str = "google/gemma-3-1b-it"

    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str = "dummy"
    request_timeout_sec: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
