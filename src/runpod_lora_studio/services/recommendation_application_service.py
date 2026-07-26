from __future__ import annotations

from runpod_lora_studio.domain.recommendation_models import TrainingRecommendation
from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
)
from runpod_lora_studio.services.training_recommendation_service import (
    recommendation_has_blocking_warning,
)
from runpod_lora_studio.services.training_service import TrainingService


class RecommendationApplicationService:
    """Apply a selected recommendation without creating or starting a job."""

    def __init__(self, training_service: TrainingService) -> None:
        self.training_service = training_service

    def apply(
        self,
        base: TrainingConfigInput,
        recommendation: TrainingRecommendation,
        *,
        current_fingerprint: str | None = None,
    ) -> TrainingConfig:
        if recommendation_has_blocking_warning(recommendation):
            raise ValueError("blocking recommendation warnings must be resolved")
        return self.training_service.apply_recommendation(
            base,
            recommendation,
            current_fingerprint=current_fingerprint,
        )
