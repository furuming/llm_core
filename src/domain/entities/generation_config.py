from pydantic import BaseModel


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    seed: int | None = 42