from domain.entities.inference_result import InferenceResult


class InMemoryResultRepository:
    def __init__(self) -> None:
        self.store: dict[str, list[InferenceResult]] = {}

    async def save_comparison(
        self,
        comparison_id: str,
        results: list[InferenceResult],
    ) -> None:
        self.store[comparison_id] = results
