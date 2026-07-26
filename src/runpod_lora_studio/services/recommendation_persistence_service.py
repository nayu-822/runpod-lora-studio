from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import (
    QualityProfile,
    RecommendationInput,
    RecommendationRequest,
    RecommendationStatus,
    RecommendationWarning,
    SpeedProfile,
    TrainingRecommendation,
    WarningSeverity,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    ManagedModelRecord,
    TrainingRecommendationRecord,
    TrainingRecommendationRequestRecord,
)
from runpod_lora_studio.services.recommendation_fingerprint import input_fingerprint


class RecommendationPersistenceService:
    """Persist recommendation provenance independently of job execution."""

    def __init__(self, settings: AppSettings) -> None:
        self.session_factory = create_session_factory(settings)

    def save(
        self,
        data: RecommendationInput,
        recommendations: tuple[TrainingRecommendation, ...],
    ) -> RecommendationRequest:
        if not recommendations:
            raise ValueError("at least one recommendation is required")
        request_id = recommendations[0].request_id
        now = datetime.now(UTC)
        warnings = sum(len(item.warnings) for item in recommendations)
        status = (
            RecommendationStatus.INVALID
            if any(
                warning.severity is WarningSeverity.BLOCKING
                for item in recommendations
                for warning in item.warnings
            )
            else RecommendationStatus.COMPLETED
        )
        fingerprint = _input_fingerprint(data)
        request = RecommendationRequest(
            id=request_id,
            project_id=data.project_id,
            dataset_snapshot_id=data.dataset_snapshot_id,
            model_id=data.model_id,
            environment_snapshot_id=data.environment_snapshot_id,
            concept_type=data.concept_type,
            quality_profile=data.quality_profile,
            speed_profile=data.speed_profile,
            user_constraints=dict(data.user_constraints),
            input_fingerprint=fingerprint,
            engine_version=recommendations[0].engine_version,
            status=status,
            warning_count=warnings,
            created_at=now,
            updated_at=now,
            current_config=dict(data.current_config),
        )
        with self.session_factory() as session:
            session.add(
                TrainingRecommendationRequestRecord(
                    id=str(request.id),
                    project_id=str(request.project_id),
                    dataset_snapshot_id=str(request.dataset_snapshot_id),
                    managed_model_id=str(request.model_id),
                    environment_snapshot_id=str(request.environment_snapshot_id),
                    concept_type=request.concept_type,
                    quality_profile=request.quality_profile.value,
                    speed_profile=request.speed_profile.value,
                    user_constraints_json=json.dumps(
                        request.user_constraints, ensure_ascii=False, sort_keys=True
                    ),
                    current_config_json=json.dumps(
                        request.current_config, ensure_ascii=False, sort_keys=True
                    ),
                    input_fingerprint=request.input_fingerprint,
                    engine_version=request.engine_version,
                    status=request.status.value,
                    warning_count=request.warning_count,
                    created_at=now,
                    updated_at=now,
                )
            )
            for recommendation in recommendations:
                settings = asdict(recommendation)
                settings.pop("warnings", None)
                settings.pop("reasons", None)
                session.add(
                    TrainingRecommendationRecord(
                        id=str(recommendation.id),
                        request_id=str(request.id),
                        rank=recommendation.rank,
                        profile_name=recommendation.profile_name,
                        settings_json=json.dumps(
                            settings, default=str, ensure_ascii=False, sort_keys=True
                        ),
                        reasons_json=json.dumps(
                            recommendation.reasons, ensure_ascii=False
                        ),
                        warnings_json=json.dumps(
                            [asdict(item) for item in recommendation.warnings],
                            default=str,
                            ensure_ascii=False,
                        ),
                        settings_fingerprint=recommendation.settings_fingerprint,
                        engine_version=recommendation.engine_version,
                        created_at=recommendation.created_at,
                    )
                )
            session.commit()
        return request

    def get_request(self, request_id: UUID) -> RecommendationRequest:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingRecommendationRequestRecord).where(
                    TrainingRecommendationRequestRecord.id == str(request_id)
                )
            )
            if record is None:
                raise ValueError("recommendation request not found")
            return _request_from_record(record)

    def model_context(self, model_id: UUID) -> tuple[str, str | None, bool]:
        with self.session_factory() as session:
            record = session.scalar(
                select(ManagedModelRecord).where(ManagedModelRecord.id == str(model_id))
            )
            if record is None:
                raise ValueError("model not found")
            verified = record.status == "available" and bool(record.local_sha256)
            return record.status, record.local_sha256, verified

    def get_recommendation(
        self, recommendation_id: UUID
    ) -> tuple[RecommendationRequest, TrainingRecommendation]:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingRecommendationRecord).where(
                    TrainingRecommendationRecord.id == str(recommendation_id)
                )
            )
            if record is None:
                raise ValueError("recommendation not found")
            request_record = session.scalar(
                select(TrainingRecommendationRequestRecord).where(
                    TrainingRecommendationRequestRecord.id == record.request_id
                )
            )
            if request_record is None:
                raise ValueError("recommendation request not found")
            return _request_from_record(request_record), _recommendation_from_record(
                record
            )


def _input_fingerprint(data: RecommendationInput) -> str:
    return input_fingerprint(data, engine_version="phase7a-rules-v1")


def _request_from_record(
    record: TrainingRecommendationRequestRecord,
) -> RecommendationRequest:
    return RecommendationRequest(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        dataset_snapshot_id=UUID(record.dataset_snapshot_id),
        model_id=UUID(record.managed_model_id),
        environment_snapshot_id=UUID(record.environment_snapshot_id),
        concept_type=record.concept_type,
        quality_profile=QualityProfile(record.quality_profile),
        speed_profile=SpeedProfile(record.speed_profile),
        user_constraints=json.loads(record.user_constraints_json),
        input_fingerprint=record.input_fingerprint,
        engine_version=record.engine_version,
        status=RecommendationStatus(record.status),
        warning_count=record.warning_count,
        created_at=record.created_at,
        updated_at=record.updated_at,
        current_config=json.loads(record.current_config_json or "{}"),
    )


def _recommendation_from_record(
    record: TrainingRecommendationRecord,
) -> TrainingRecommendation:
    settings = json.loads(record.settings_json)
    warning_values = json.loads(record.warnings_json)
    return TrainingRecommendation(
        id=UUID(record.id),
        request_id=UUID(record.request_id),
        rank=record.rank,
        profile_name=record.profile_name,
        batch_size=int(settings["batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        network_module=str(settings["network_module"]),
        network_dim=int(settings["network_dim"]),
        network_alpha=int(settings["network_alpha"]),
        epochs=int(settings["epochs"]),
        repeats=tuple(int(value) for value in settings["repeats"]),
        learning_rate=float(settings["learning_rate"]),
        optimizer=str(settings["optimizer"]),
        scheduler=str(settings["scheduler"]),
        mixed_precision=str(settings["mixed_precision"]),
        cache_latents=bool(settings["cache_latents"]),
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        estimated_images_per_epoch=settings.get("estimated_images_per_epoch"),
        estimated_steps_per_epoch=settings.get("estimated_steps_per_epoch"),
        estimated_total_steps=settings.get("estimated_total_steps"),
        estimated_vram_bytes=settings.get("estimated_vram_bytes"),
        estimated_duration_seconds=settings.get("estimated_duration_seconds"),
        confidence=str(settings["confidence"]),
        reasons=tuple(json.loads(record.reasons_json)),
        warnings=tuple(
            RecommendationWarning(
                code=str(value["code"]),
                severity=WarningSeverity(value["severity"]),
                message=str(value["message"]),
                parameter=value.get("parameter"),
                measured=value.get("measured"),
            )
            for value in warning_values
        ),
        settings_fingerprint=record.settings_fingerprint,
        engine_version=record.engine_version,
        created_at=record.created_at,
        resolution=int(settings.get("resolution", 1024)),
        save_every_n_epochs=int(settings.get("save_every_n_epochs", 1)),
        seed=int(settings.get("seed", 42)),
    )
