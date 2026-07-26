from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import (
    RecommendationInput,
    RecommendationRequest,
    RecommendationStatus,
    TrainingRecommendation,
    WarningSeverity,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    TrainingRecommendationRecord,
    TrainingRecommendationRequestRecord,
)


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


def _input_fingerprint(data: RecommendationInput) -> str:
    payload = {
        "project_id": str(data.project_id),
        "dataset_snapshot_id": str(data.dataset_snapshot_id),
        "model_id": str(data.model_id),
        "environment_snapshot_id": str(data.environment_snapshot_id),
        "concept_type": data.concept_type,
        "quality_profile": data.quality_profile.value,
        "speed_profile": data.speed_profile.value,
        "constraints": data.user_constraints,
        "dataset": asdict(data.dataset),
        "environment": asdict(data.environment),
        "training_environment": asdict(data.training_environment),
    }
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
