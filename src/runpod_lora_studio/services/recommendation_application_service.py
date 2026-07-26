from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import DatasetSnapshotStatus
from runpod_lora_studio.domain.recommendation_models import (
    RecommendationInput,
    RecommendationRequest,
    RecommendationStatus,
    RecommendationWarning,
    TrainingRecommendation,
    WarningSeverity,
)
from runpod_lora_studio.domain.storage_models import ManagedModelStatus
from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotRecord,
    ManagedModelRecord,
)
from runpod_lora_studio.services.dataset_statistics_service import (
    DatasetStatisticsService,
)
from runpod_lora_studio.services.environment_diagnostic_service import (
    ComputeEnvironmentService,
    TrainingEnvironmentService,
)
from runpod_lora_studio.services.gpu_memory_metrics import gpu_uuid_fingerprint
from runpod_lora_studio.services.recommendation_calibration_service import (
    TrainingCalibrationService,
)
from runpod_lora_studio.services.recommendation_fingerprint import input_fingerprint
from runpod_lora_studio.services.recommendation_persistence_service import (
    RecommendationPersistenceService,
)
from runpod_lora_studio.services.training_command import SdScriptsCommandBuilder
from runpod_lora_studio.services.training_risk_service import TrainingRiskService
from runpod_lora_studio.services.training_service import TrainingService


class RecommendationStaleError(ValueError):
    code = "RECOMMENDATION_STALE"


class RecommendationApplicationService:
    """Single trusted path from a persisted recommendation to a config."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        training_service: TrainingService | None = None,
        persistence: RecommendationPersistenceService | None = None,
        compute_service: ComputeEnvironmentService | None = None,
        training_environment_service: TrainingEnvironmentService | None = None,
        dataset_statistics: DatasetStatisticsService | None = None,
        calibration_service: TrainingCalibrationService | None = None,
    ) -> None:
        self.settings = settings
        self.training_service = training_service or TrainingService(settings)
        self.persistence = persistence or RecommendationPersistenceService(settings)
        self.compute_service = compute_service or ComputeEnvironmentService(settings)
        self.training_environment_service = (
            training_environment_service or TrainingEnvironmentService(settings)
        )
        self.dataset_statistics = dataset_statistics or DatasetStatisticsService(
            settings
        )
        self.calibration_service = calibration_service or TrainingCalibrationService(
            settings
        )

    def preview(
        self,
        recommendation_id: UUID,
        *,
        input_fingerprint_value: str | None = None,
    ) -> TrainingRecommendation:
        request, recommendation = self.persistence.get_recommendation(recommendation_id)
        if recommendation.request_id != request.id:
            raise ValueError("recommendation request relation is invalid")
        if request.status is not RecommendationStatus.COMPLETED:
            raise ValueError("recommendation request is not applicable")
        self._validate_warning_gate(recommendation)
        self._validate_calibration(request, recommendation)
        if (
            input_fingerprint_value is not None
            and input_fingerprint_value != request.input_fingerprint
        ):
            raise RecommendationStaleError("recommendation input fingerprint changed")
        return recommendation

    def apply(
        self,
        recommendation_id: UUID,
        base: TrainingConfigInput,
        *,
        input_fingerprint_value: str,
    ) -> TrainingConfig:
        request, recommendation = self.persistence.get_recommendation(recommendation_id)
        if recommendation.request_id != request.id:
            raise ValueError("recommendation request relation is invalid")
        if request.status is not RecommendationStatus.COMPLETED:
            raise ValueError("recommendation request is not applicable")
        self._validate_warning_gate(recommendation)
        self._validate_calibration(request, recommendation)
        if base.project_id != request.project_id:
            raise ValueError("recommendation project does not match current project")
        if base.dataset_snapshot_id != request.dataset_snapshot_id:
            raise RecommendationStaleError("dataset snapshot changed")
        if base.managed_model_id != request.model_id:
            raise RecommendationStaleError("model changed")
        if input_fingerprint_value != request.input_fingerprint:
            raise RecommendationStaleError("recommendation input fingerprint changed")
        current_data, snapshot_repeats = self._current_input(request, base)
        self._validate_calibration(
            request, recommendation, base=base, current_data=current_data
        )
        current_fingerprint = input_fingerprint(
            current_data, engine_version=request.engine_version
        )
        if current_fingerprint != request.input_fingerprint:
            raise RecommendationStaleError("current environment or dataset changed")
        warnings = TrainingRiskService.evaluate(
            recommendation,
            base,
            environment=current_data.environment,
            effective_image_count=current_data.dataset.effective_image_count,
            repeats=snapshot_repeats,
            allowed_optimizers=SdScriptsCommandBuilder.allowed_optimizers,
            allowed_schedulers=SdScriptsCommandBuilder.allowed_schedulers,
        )
        self._validate_risk(warnings)
        applied = replace(
            base,
            recommendation_id=recommendation.id,
            recommendation_engine_version=recommendation.engine_version,
            recommendation_change_diff=_change_diff(recommendation, base),
        )
        return self.training_service.create_config(applied)

    def _current_input(
        self, request: RecommendationRequest, base: TrainingConfigInput
    ) -> tuple[RecommendationInput, tuple[int, ...]]:
        with create_session_factory(self.settings)() as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.id == str(base.dataset_snapshot_id)
                )
            )
            model = session.scalar(
                select(ManagedModelRecord).where(
                    ManagedModelRecord.id == str(base.managed_model_id)
                )
            )
            if snapshot is None or snapshot.project_id != str(base.project_id):
                raise ValueError("dataset snapshot does not belong to the project")
            if snapshot.status != DatasetSnapshotStatus.COMPLETED.value:
                raise ValueError("dataset snapshot is not completed")
            if model is None or model.status != ManagedModelStatus.AVAILABLE.value:
                raise ValueError("model is not available")
            if not model.local_sha256:
                raise ValueError("model hash is unavailable")
            model_sha256 = model.local_sha256
        compute = self.compute_service.detect()
        training_environment = self.training_environment_service.detect()
        dataset = self.dataset_statistics.calculate(base.dataset_snapshot_id)
        constraints = dict(request.user_constraints)
        constraints["model_sha256"] = model_sha256
        constraints["model_hash_verified"] = True
        current_config = dict(request.current_config)
        current_config["resolution"] = base.resolution
        return (
            RecommendationInput(
                project_id=base.project_id,
                dataset_snapshot_id=base.dataset_snapshot_id,
                model_id=base.managed_model_id,
                environment_snapshot_id=request.environment_snapshot_id,
                environment=compute,
                training_environment=training_environment,
                dataset=dataset,
                concept_type=request.concept_type,
                quality_profile=request.quality_profile,
                speed_profile=request.speed_profile,
                user_constraints=constraints,
                current_config=current_config,
            ),
            dataset.repeats,
        )

    @staticmethod
    def _validate_warning_gate(recommendation: TrainingRecommendation) -> None:
        blocking = [
            warning.code
            for warning in recommendation.warnings
            if warning.severity.value == "blocking"
        ]
        if blocking:
            raise ValueError("blocking recommendation warnings: " + ", ".join(blocking))

    def _validate_calibration(
        self,
        request: RecommendationRequest,
        recommendation: TrainingRecommendation,
        *,
        base: TrainingConfigInput | None = None,
        current_data: RecommendationInput | None = None,
    ) -> None:
        if recommendation.calibration_snapshot_id is None:
            return
        if self.calibration_service.is_stale(
            recommendation.calibration_snapshot_id, request.project_id
        ):
            raise RecommendationStaleError("calibration snapshot changed or is stale")
        snapshot = self.calibration_service.get(recommendation.calibration_snapshot_id)
        if snapshot is None:
            raise RecommendationStaleError("calibration snapshot is unavailable")
        pairs = (
            (snapshot.resolution, recommendation.resolution),
            (snapshot.batch_size, recommendation.batch_size),
            (snapshot.network_module, recommendation.network_module),
            (snapshot.network_dim, recommendation.network_dim),
            (snapshot.network_alpha, recommendation.network_alpha),
            (snapshot.optimizer, recommendation.optimizer),
            (snapshot.mixed_precision, recommendation.mixed_precision),
            (snapshot.cache_latents, recommendation.cache_latents),
            (snapshot.gradient_checkpointing, recommendation.gradient_checkpointing),
        )
        if any(
            expected is not None and expected != actual for expected, actual in pairs
        ):
            raise RecommendationStaleError(
                "calibration snapshot is incompatible with recommendation"
            )
        if base is not None:
            base_pairs = (
                (snapshot.resolution, base.resolution),
                (snapshot.batch_size, base.batch_size),
                (snapshot.network_module, base.network_module),
                (snapshot.network_dim, base.network_dim),
                (snapshot.network_alpha, base.network_alpha),
                (snapshot.optimizer, base.optimizer),
                (snapshot.mixed_precision, base.mixed_precision),
                (snapshot.cache_latents, base.cache_latents),
                (snapshot.gradient_checkpointing, base.gradient_checkpointing),
            )
            if any(
                expected is not None and expected != actual
                for expected, actual in base_pairs
            ):
                raise RecommendationStaleError(
                    "calibration snapshot is incompatible with applied config"
                )
        if current_data is not None:
            gpu = (
                current_data.environment.gpu_devices[0]
                if current_data.environment.gpu_devices
                else None
            )
            current_gpu = gpu_uuid_fingerprint(gpu.uuid) if gpu and gpu.uuid else None
            if current_gpu != snapshot.gpu_identity_fingerprint:
                raise RecommendationStaleError(
                    "calibration snapshot does not match current GPU"
                )
            if (
                snapshot.gpu_total_vram_class is not None
                and gpu is not None
                and _vram_class(gpu.total_vram_bytes) != snapshot.gpu_total_vram_class
            ):
                raise RecommendationStaleError(
                    "calibration snapshot does not match current VRAM class"
                )

    @staticmethod
    def _validate_risk(warnings: tuple[RecommendationWarning, ...]) -> None:
        blocking = [
            warning.code
            for warning in warnings
            if warning.severity is WarningSeverity.BLOCKING
        ]
        if blocking:
            raise ValueError("blocking training risk: " + ", ".join(blocking))


def _change_diff(
    recommendation: TrainingRecommendation, applied: TrainingConfigInput
) -> dict[str, dict[str, object]]:
    pairs = {
        "resolution": (recommendation.resolution, applied.resolution),
        "batch_size": (recommendation.batch_size, applied.batch_size),
        "epochs": (recommendation.epochs, applied.epochs),
        "learning_rate": (recommendation.learning_rate, applied.learning_rate),
        "optimizer": (recommendation.optimizer, applied.optimizer),
        "scheduler": (recommendation.scheduler, applied.scheduler),
        "network_module": (recommendation.network_module, applied.network_module),
        "network_dim": (recommendation.network_dim, applied.network_dim),
        "network_alpha": (recommendation.network_alpha, applied.network_alpha),
        "mixed_precision": (recommendation.mixed_precision, applied.mixed_precision),
        "cache_latents": (recommendation.cache_latents, applied.cache_latents),
        "gradient_checkpointing": (
            recommendation.gradient_checkpointing,
            applied.gradient_checkpointing,
        ),
        "save_every_n_epochs": (
            recommendation.save_every_n_epochs,
            applied.save_every_n_epochs,
        ),
        "seed": (recommendation.seed, applied.seed),
    }
    return {
        name: {"recommended": recommended, "applied": actual}
        for name, (recommended, actual) in pairs.items()
    }


def _vram_class(value: int | None) -> str | None:
    return f"{round(value / 1024**3)}GiB" if value and value > 0 else None
