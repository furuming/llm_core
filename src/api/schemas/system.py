from pydantic import BaseModel


class VramDevice(BaseModel):
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    process_allocated_bytes: int
    process_reserved_bytes: int


class VramStatus(BaseModel):
    available: bool
    devices: list[VramDevice]


class RuntimeStatusResponse(BaseModel):
    loaded_models: list[str]
    vram: VramStatus
