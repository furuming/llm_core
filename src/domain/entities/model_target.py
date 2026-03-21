from pydantic import BaseModel


class ModelTarget(BaseModel):
    name: str
    label: str | None = None
