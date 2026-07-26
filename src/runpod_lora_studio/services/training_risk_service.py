from __future__ import annotations

from collections.abc import Iterable

from runpod_lora_studio.domain.recommendation_models import (
    RecommendationWarning,
    TrainingRecommendation,
    WarningSeverity,
)


class TrainingRiskService:
    """Centralize the apply/start gate for recommendation warnings."""

    @staticmethod
    def warnings(
        recommendation: TrainingRecommendation,
    ) -> tuple[RecommendationWarning, ...]:
        return recommendation.warnings

    @staticmethod
    def can_apply(recommendation: TrainingRecommendation) -> bool:
        return not any(
            warning.severity is WarningSeverity.BLOCKING
            for warning in recommendation.warnings
        )

    @staticmethod
    def has_unresolved_security_or_model_warning(
        warnings: Iterable[RecommendationWarning],
    ) -> bool:
        protected = {"MODEL_HASH_UNVERIFIED", "ENVIRONMENT_CHANGED"}
        return any(warning.code in protected for warning in warnings)
