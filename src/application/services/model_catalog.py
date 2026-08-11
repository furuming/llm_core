from domain.entities.model_preset import ModelPreset


class UnknownModelError(ValueError):
    pass


class ModelCatalog:
    def __init__(self, presets: tuple[ModelPreset, ...]) -> None:
        self._presets = presets
        self._by_family = {preset.family: preset for preset in presets}
        self._known_names = {preset.model_name for preset in presets}

    def list(self) -> tuple[ModelPreset, ...]:
        return self._presets

    def resolve(self, model: str) -> str:
        preset = self._by_family.get(model)
        model_name = preset.model_name if preset is not None else model
        if model_name not in self._known_names:
            raise UnknownModelError(f"Unknown model: {model}")
        return model_name
