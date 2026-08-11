from typing import Protocol

from domain.entities.inference_result import InferenceResult


class ResultRepository(Protocol):
    async def save_comparison(
        self,
        comparison_id: str,
        results: list[InferenceResult],
    ) -> None: ...
