from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., examples=["system", "user", "assistant"])
    content: str
