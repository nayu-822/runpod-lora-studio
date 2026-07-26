from __future__ import annotations

from dataclasses import replace

from runpod_lora_studio.domain.recommendation_models import (
    RecommendationInput,
    TrainingRecommendation,
)
from runpod_lora_studio.domain.training_models import TrainingConfigInput
from runpod_lora_studio.services.training_recommendation_engine import (
    RuleBasedRecommendationEngine,
)


class TrainingRecommendationService:
    """Facade for deterministic recommendations; it never starts a job."""

    def __init__(self, engine: RuleBasedRecommendationEngine | None = None) -> None:
        self.engine = engine or RuleBasedRecommendationEngine()

    def recommend(
        self, data: RecommendationInput
    ) -> tuple[TrainingRecommendation, ...]:
        return self.engine.recommend(data)

    @staticmethod
    def apply_to_config(
        base: TrainingConfigInput,
        recommendation: TrainingRecommendation,
        *,
        current_fingerprint: str | None = None,
    ) -> TrainingConfigInput:
        if current_fingerprint is not None and (
            current_fingerprint != recommendation.settings_fingerprint
        ):
            raise ValueError("recommendation is stale")
        extra_options = dict(base.extra_options)
        change_diff = {
            "batch_size": recommendation.batch_size,
            "epochs": recommendation.epochs,
            "learning_rate": recommendation.learning_rate,
            "optimizer": recommendation.optimizer,
            "scheduler": recommendation.scheduler,
            "network_module": recommendation.network_module,
            "network_dim": recommendation.network_dim,
            "network_alpha": recommendation.network_alpha,
            "mixed_precision": recommendation.mixed_precision,
            "cache_latents": recommendation.cache_latents,
            "gradient_checkpointing": recommendation.gradient_checkpointing,
        }
        return replace(
            base,
            batch_size=recommendation.batch_size,
            epochs=recommendation.epochs,
            learning_rate=recommendation.learning_rate,
            optimizer=recommendation.optimizer,
            scheduler=recommendation.scheduler,
            network_module=recommendation.network_module,
            network_dim=recommendation.network_dim,
            network_alpha=recommendation.network_alpha,
            mixed_precision=recommendation.mixed_precision,
            cache_latents=recommendation.cache_latents,
            gradient_checkpointing=recommendation.gradient_checkpointing,
            extra_options=extra_options,
            recommendation_id=recommendation.id,
            recommendation_engine_version=recommendation.engine_version,
            recommendation_change_diff=change_diff,
        )


def recommendation_has_blocking_warning(
    recommendation: TrainingRecommendation,
) -> bool:
    return any(
        warning.severity.value == "blocking" for warning in recommendation.warnings
    )
