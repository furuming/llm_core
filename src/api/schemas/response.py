from pydantic import BaseModel

from domain.entities.inference_result import InferenceResult


class CompareResponse(BaseModel):
    comparison_id: str
    results: list[InferenceResult]
