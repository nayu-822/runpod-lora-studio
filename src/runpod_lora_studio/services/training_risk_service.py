from __future__ import annotations

from collections.abc import Iterable

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    RecommendationWarning,
    TrainingRecommendation,
    WarningSeverity,
)
from runpod_lora_studio.domain.training_models import TrainingConfigInput
from runpod_lora_studio.services.training_log_parser import TrainingStepEstimator
from runpod_lora_studio.services.training_memory_estimator import (
    TrainingMemoryEstimator,
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
    def evaluate(
        recommendation: TrainingRecommendation,
        config: TrainingConfigInput,
        *,
        environment: ComputeEnvironmentInfo,
        effective_image_count: int,
        repeats: tuple[int, ...],
        allowed_optimizers: frozenset[str],
        allowed_schedulers: frozenset[str],
    ) -> tuple[RecommendationWarning, ...]:
        warnings = list(recommendation.warnings)
        if config.optimizer not in allowed_optimizers:
            warnings.append(
                RecommendationWarning(
                    "OPTIMIZER_NOT_ALLOWED",
                    WarningSeverity.BLOCKING,
                    "optimizer is outside the trusted allowlist",
                    "optimizer",
                )
            )
        if config.scheduler not in allowed_schedulers:
            warnings.append(
                RecommendationWarning(
                    "SCHEDULER_NOT_ALLOWED",
                    WarningSeverity.BLOCKING,
                    "scheduler is outside the trusted allowlist",
                    "scheduler",
                )
            )
        if config.network_alpha > config.network_dim:
            warnings.append(
                RecommendationWarning(
                    "NETWORK_ALPHA_INVALID",
                    WarningSeverity.BLOCKING,
                    "network alpha must not exceed network dim",
                    "network_alpha",
                )
            )
        if config.mixed_precision == "bf16" and environment.bf16_supported is not True:
            warnings.append(
                RecommendationWarning(
                    "BF16_UNSUPPORTED",
                    WarningSeverity.BLOCKING,
                    "bf16 is not supported by the current environment",
                    "mixed_precision",
                )
            )
        if config.optimizer == "AdamW8bit" and not environment.bitsandbytes_available:
            warnings.append(
                RecommendationWarning(
                    "OPTIMIZER_DEPENDENCY_MISSING",
                    WarningSeverity.BLOCKING,
                    "bitsandbytes is unavailable for AdamW8bit",
                    "optimizer",
                )
            )
        gpu = environment.gpu_devices[0] if environment.gpu_devices else None
        estimate = TrainingMemoryEstimator().estimate(
            gpu=gpu,
            resolution=config.resolution,
            batch_size=config.batch_size,
            network_dim=config.network_dim,
            mixed_precision=config.mixed_precision,
            cache_latents=config.cache_latents,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        if not estimate.valid:
            warnings.append(
                RecommendationWarning(
                    "INSUFFICIENT_FREE_VRAM"
                    if gpu is not None
                    else "GPU_NOT_AVAILABLE",
                    WarningSeverity.BLOCKING,
                    "current training values do not fit the safe VRAM budget",
                    "batch_size",
                )
            )
        elif "VRAM_ESTIMATE_NEAR_LIMIT" in estimate.warnings:
            warnings.append(
                RecommendationWarning(
                    "VRAM_ESTIMATE_NEAR_LIMIT",
                    WarningSeverity.WARNING,
                    "current training values are close to the safe VRAM limit",
                    "batch_size",
                )
            )
        plan = TrainingStepEstimator.estimate(
            subset_image_counts=(effective_image_count,),
            num_repeats=(1,),
            batch_size=config.batch_size,
            epochs=config.epochs,
        )
        if plan.total_steps is not None and plan.total_steps > 100_000:
            warnings.append(
                RecommendationWarning(
                    "TOTAL_STEPS_TOO_HIGH",
                    WarningSeverity.WARNING,
                    "current training values produce excessive total steps",
                    "epochs",
                )
            )
        del repeats
        return tuple(warnings)

    @staticmethod
    def has_unresolved_security_or_model_warning(
        warnings: Iterable[RecommendationWarning],
    ) -> bool:
        protected = {"MODEL_HASH_UNVERIFIED", "ENVIRONMENT_CHANGED"}
        return any(warning.code in protected for warning in warnings)
