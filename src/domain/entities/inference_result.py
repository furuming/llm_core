from pydantic import BaseModel


class InferenceResult(BaseModel):
    engine: str
    model: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_sec: float
    finish_reason: str | None = None
    raw_response: dict | None = None
