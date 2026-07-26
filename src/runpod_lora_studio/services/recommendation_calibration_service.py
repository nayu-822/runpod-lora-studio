from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    TrainingCalibrationSnapshot,
    TrainingExecutionSummary,
    TrainingFailureCategory,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    RecommendationCalibrationSnapshotRecord,
    RecommendationCalibrationSourceRecord,
    TrainingExecutionSummaryRecord,
)
from runpod_lora_studio.services.training_calibration_service import (
    RecommendationCalibrationService,
)
from runpod_lora_studio.services.training_performance_service import (
    _summary_from_record,
)


class TrainingCalibrationService:
    """Persist and rebuild deterministic calibration snapshots."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        builder: RecommendationCalibrationService | None = None,
    ) -> None:
        self.session_factory = create_session_factory(settings)
        self.builder = builder or RecommendationCalibrationService()

    def list_summaries(
        self, project_id: UUID | None = None
    ) -> list[TrainingExecutionSummary]:
        with self.session_factory() as session:
            query = select(TrainingExecutionSummaryRecord)
            if project_id is not None:
                query = query.where(
                    TrainingExecutionSummaryRecord.project_id == str(project_id)
                )
            rows = session.scalars(
                query.order_by(TrainingExecutionSummaryRecord.created_at.desc())
            ).all()
            return [_summary_from_record(row) for row in rows]

    def build_for_project(
        self,
        project_id: UUID,
        *,
        gpu_identity_fingerprint: str,
        gpu_total_vram_bytes: int | None = None,
        resolution: int | None = None,
        optimizer: str | None = None,
        mixed_precision: str | None = None,
        cache_latents: bool | None = None,
        gradient_checkpointing: bool | None = None,
    ) -> TrainingCalibrationSnapshot:
        summaries = self.list_summaries(project_id)
        snapshot = self.builder.build(
            summaries,
            gpu_identity_fingerprint=gpu_identity_fingerprint,
            gpu_total_vram_bytes=gpu_total_vram_bytes,
            resolution=resolution,
            optimizer=optimizer,
            mixed_precision=mixed_precision,
            cache_latents=cache_latents,
            gradient_checkpointing=gradient_checkpointing,
            scope_project_id=project_id,
        )
        with self.session_factory() as session:
            existing = session.scalar(
                select(RecommendationCalibrationSnapshotRecord).where(
                    RecommendationCalibrationSnapshotRecord.calibration_fingerprint
                    == snapshot.calibration_fingerprint
                )
            )
            if existing is not None:
                return _snapshot_from_record(existing, session)
            session.add(_snapshot_record(snapshot))
            for summary_id in snapshot.source_summary_ids:
                session.add(
                    RecommendationCalibrationSourceRecord(
                        calibration_id=str(snapshot.id), summary_id=str(summary_id)
                    )
                )
            session.commit()
        return snapshot

    def get(self, snapshot_id: UUID) -> TrainingCalibrationSnapshot | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(RecommendationCalibrationSnapshotRecord).where(
                    RecommendationCalibrationSnapshotRecord.id == str(snapshot_id)
                )
            )
            return _snapshot_from_record(record, session) if record else None

    def mark_stale(self, snapshot_id: UUID) -> None:
        with self.session_factory() as session:
            record = session.scalar(
                select(RecommendationCalibrationSnapshotRecord).where(
                    RecommendationCalibrationSnapshotRecord.id == str(snapshot_id)
                )
            )
            if record is not None:
                record.stale = True
                session.commit()

    def rebuild(self, project_id: UUID, **match: object) -> TrainingCalibrationSnapshot:
        return self.build_for_project(project_id, **match)  # type: ignore[arg-type]

    def list_calibrations(
        self, project_id: UUID | None = None
    ) -> list[TrainingCalibrationSnapshot]:
        with self.session_factory() as session:
            query = select(RecommendationCalibrationSnapshotRecord)
            if project_id is not None:
                query = query.where(
                    RecommendationCalibrationSnapshotRecord.scope_project_id
                    == str(project_id)
                )
            rows = session.scalars(
                query.order_by(
                    RecommendationCalibrationSnapshotRecord.generated_at.desc()
                )
            ).all()
            return [_snapshot_from_record(row, session) for row in rows]

    def is_stale(self, snapshot_id: UUID, project_id: UUID | None = None) -> bool:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            return True
        summaries = self.list_summaries(project_id or snapshot.scope_project_id)
        return self.builder.is_stale(snapshot, summaries)


class RecommendationHistoryService:
    """Read and explicitly include/exclude empirical outcomes."""

    def __init__(self, settings: AppSettings) -> None:
        self.session_factory = create_session_factory(settings)

    def list(self, project_id: UUID | None = None) -> list[TrainingExecutionSummary]:
        with self.session_factory() as session:
            query = select(TrainingExecutionSummaryRecord)
            if project_id is not None:
                query = query.where(
                    TrainingExecutionSummaryRecord.project_id == str(project_id)
                )
            rows = session.scalars(
                query.order_by(TrainingExecutionSummaryRecord.created_at.desc())
            ).all()
            return [_summary_from_record(row) for row in rows]

    def set_included(
        self, summary_id: UUID, included: bool, reason: str | None = None
    ) -> None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingExecutionSummaryRecord).where(
                    TrainingExecutionSummaryRecord.id == str(summary_id)
                )
            )
            if record is None:
                raise ValueError("training execution summary not found")
            record.calibration_included = included
            record.manual_exclusion_reason = (
                None if included else (reason or "user_excluded")
            )
            record.updated_at = datetime.now(UTC)
            session.commit()

    def include(self, summary_id: UUID) -> None:
        self.set_included(summary_id, True)

    def exclude(self, summary_id: UUID, reason: str = "user_excluded") -> None:
        self.set_included(summary_id, False, reason)

    def reclassify(
        self,
        summary_id: UUID,
        *,
        category: TrainingFailureCategory,
        evidence_code: str = "manual_reclassification",
    ) -> None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingExecutionSummaryRecord).where(
                    TrainingExecutionSummaryRecord.id == str(summary_id)
                )
            )
            if record is None:
                raise ValueError("training execution summary not found")
            record.failure_category = category.value
            record.oom_detected = category in {
                TrainingFailureCategory.CUDA_OUT_OF_MEMORY,
                TrainingFailureCategory.SYSTEM_OUT_OF_MEMORY,
            }
            record.failure_evidence_codes_json = json.dumps([evidence_code])
            record.usable_for_speed_calibration = (
                False if record.oom_detected else record.usable_for_speed_calibration
            )
            record.updated_at = datetime.now(UTC)
            session.commit()


def _snapshot_record(
    snapshot: TrainingCalibrationSnapshot,
) -> RecommendationCalibrationSnapshotRecord:
    return RecommendationCalibrationSnapshotRecord(
        id=str(snapshot.id),
        scope_project_id=str(snapshot.scope_project_id)
        if snapshot.scope_project_id
        else None,
        gpu_identity_fingerprint=snapshot.gpu_identity_fingerprint,
        gpu_total_vram_class=snapshot.gpu_total_vram_class,
        resolution=snapshot.resolution,
        optimizer=snapshot.optimizer,
        mixed_precision=snapshot.mixed_precision,
        cache_latents=snapshot.cache_latents,
        gradient_checkpointing=snapshot.gradient_checkpointing,
        sample_count=snapshot.sample_count,
        successful_sample_count=snapshot.successful_sample_count,
        oom_sample_count=snapshot.oom_sample_count,
        median_steps_per_second=snapshot.median_steps_per_second,
        lower_percentile_steps_per_second=snapshot.lower_percentile_steps_per_second,
        median_peak_vram_bytes=snapshot.median_peak_vram_bytes,
        upper_percentile_peak_vram_bytes=snapshot.upper_percentile_peak_vram_bytes,
        confidence=snapshot.confidence.value,
        calibration_fingerprint=snapshot.calibration_fingerprint,
        calibration_version=snapshot.calibration_version,
        source_summary_fingerprint=snapshot.source_summary_fingerprint,
        reason_codes_json=json.dumps(snapshot.reason_codes, sort_keys=True),
        generated_at=snapshot.generated_at or datetime.now(UTC),
        expires_at=snapshot.expires_at,
        stale=snapshot.stale,
    )


def _snapshot_from_record(
    record: RecommendationCalibrationSnapshotRecord, session: Session
) -> TrainingCalibrationSnapshot:
    # SQLAlchemy Session is intentionally kept as a structural dependency here;
    # callers only receive immutable domain data.
    source_rows = session.scalars(
        select(RecommendationCalibrationSourceRecord).where(
            RecommendationCalibrationSourceRecord.calibration_id == record.id
        )
    ).all()
    return TrainingCalibrationSnapshot(
        id=UUID(record.id),
        scope_project_id=UUID(record.scope_project_id)
        if record.scope_project_id
        else None,
        gpu_identity_fingerprint=record.gpu_identity_fingerprint,
        gpu_total_vram_class=record.gpu_total_vram_class,
        resolution=record.resolution,
        optimizer=record.optimizer,
        mixed_precision=record.mixed_precision,
        cache_latents=bool(record.cache_latents)
        if record.cache_latents is not None
        else None,
        gradient_checkpointing=bool(record.gradient_checkpointing)
        if record.gradient_checkpointing is not None
        else None,
        sample_count=record.sample_count,
        successful_sample_count=record.successful_sample_count,
        oom_sample_count=record.oom_sample_count,
        median_steps_per_second=record.median_steps_per_second,
        lower_percentile_steps_per_second=record.lower_percentile_steps_per_second,
        median_peak_vram_bytes=record.median_peak_vram_bytes,
        upper_percentile_peak_vram_bytes=record.upper_percentile_peak_vram_bytes,
        confidence=CalibrationConfidence(record.confidence),
        calibration_fingerprint=record.calibration_fingerprint,
        calibration_version=record.calibration_version,
        generated_at=record.generated_at,
        expires_at=record.expires_at,
        stale=bool(record.stale),
        reason_codes=tuple(json.loads(record.reason_codes_json or "[]")),
        source_summary_ids=tuple(UUID(row.summary_id) for row in source_rows),
        source_summary_fingerprint=record.source_summary_fingerprint,
    )
