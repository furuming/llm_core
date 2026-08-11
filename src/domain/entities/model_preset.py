from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPreset:
    family: str
    label: str
    model_name: str
