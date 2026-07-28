from __future__ import annotations

import hashlib
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
        gpu_architecture: str | None = None,
        compute_capability: str | None = None,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        effective_batch_size: int | None = None,
        network_module: str | None = None,
        network_dim: int | None = None,
        network_alpha: int | None = None,
        world_size: int | None = None,
        sd_scripts_version: str | None = None,
        xformers_available: bool | None = None,
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
            gpu_architecture=gpu_architecture,
            compute_capability=compute_capability,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            effective_batch_size=effective_batch_size,
            network_module=network_module,
            network_dim=network_dim,
            network_alpha=network_alpha,
            world_size=world_size,
            sd_scripts_version=sd_scripts_version,
            xformers_available=xformers_available,
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
                if existing.stale:
                    existing.stale = False
                    session.commit()
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
            _refresh_summary_fingerprints(record)
            _mark_calibrations_stale(session, record.id)
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
            _refresh_summary_fingerprints(record)
            _mark_calibrations_stale(session, record.id)
            session.commit()

    def set_usability(
        self,
        summary_id: UUID,
        *,
        usable_for_speed: bool | None = None,
        usable_for_memory: bool | None = None,
    ) -> None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingExecutionSummaryRecord).where(
                    TrainingExecutionSummaryRecord.id == str(summary_id)
                )
            )
            if record is None:
                raise ValueError("training execution summary not found")
            if usable_for_speed is not None:
                record.usable_for_speed_calibration = usable_for_speed
            if usable_for_memory is not None:
                record.usable_for_memory_calibration = usable_for_memory
            record.updated_at = datetime.now(UTC)
            _refresh_summary_fingerprints(record)
            _mark_calibrations_stale(session, record.id)
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
        gpu_architecture=snapshot.gpu_architecture,
        compute_capability=snapshot.compute_capability,
        resolution=snapshot.resolution,
        batch_size=snapshot.batch_size,
        gradient_accumulation_steps=snapshot.gradient_accumulation_steps,
        effective_batch_size=snapshot.effective_batch_size,
        network_module=snapshot.network_module,
        network_dim=snapshot.network_dim,
        network_alpha=snapshot.network_alpha,
        world_size=snapshot.world_size,
        sd_scripts_version=snapshot.sd_scripts_version,
        xformers_available=snapshot.xformers_available,
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
        gpu_architecture=record.gpu_architecture,
        compute_capability=record.compute_capability,
        resolution=record.resolution,
        batch_size=record.batch_size,
        gradient_accumulation_steps=record.gradient_accumulation_steps,
        effective_batch_size=record.effective_batch_size,
        network_module=record.network_module,
        network_dim=record.network_dim,
        network_alpha=record.network_alpha,
        world_size=record.world_size,
        sd_scripts_version=record.sd_scripts_version,
        xformers_available=bool(record.xformers_available)
        if record.xformers_available is not None
        else None,
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


def _refresh_summary_fingerprints(record: TrainingExecutionSummaryRecord) -> None:
    content = record.summary_content_fingerprint or _hash_record_content(record)
    record.summary_content_fingerprint = content
    state = _hash(
        {
            "summary": content,
            "included": bool(record.calibration_included),
            "manual_exclusion_reason": record.manual_exclusion_reason,
            "failure_category": record.failure_category,
            "failure_evidence_codes_json": record.failure_evidence_codes_json,
            "oom_detected": bool(record.oom_detected),
            "usable_for_speed": bool(record.usable_for_speed_calibration),
            "usable_for_memory": bool(record.usable_for_memory_calibration),
            "collector_version": record.collector_version,
            "classifier_version": record.classifier_version,
        }
    )
    record.calibration_state_fingerprint = state
    record.summary_fingerprint = _hash({"content": content, "state": state})


def _hash_record_content(record: TrainingExecutionSummaryRecord) -> str:
    fields = (
        "training_job_id",
        "project_id",
        "training_config_id",
        "dataset_snapshot_id",
        "managed_model_id",
        "training_job_environment_snapshot_id",
        "job_result_status",
        "gpu_identity_fingerprint",
        "gpu_architecture",
        "gpu_index",
        "gpu_total_vram_bytes",
        "settings_fingerprint",
        "resolution",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "network_module",
        "network_dim",
        "network_alpha",
        "optimizer",
        "scheduler",
        "mixed_precision",
        "cache_latents",
        "gradient_checkpointing",
        "world_size",
        "sd_scripts_version",
        "xformers_available",
        "total_epochs",
        "planned_total_steps",
        "completed_steps",
        "elapsed_seconds",
        "measured_steps_per_second",
        "measured_images_per_second",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "free_vram_before_bytes",
        "free_vram_after_bytes",
        "minimum_free_vram_bytes",
        "whole_gpu_peak_used_vram_bytes",
        "other_process_peak_vram_bytes",
        "memory_sample_count",
        "memory_failed_sample_count",
        "memory_confidence",
        "memory_first_sampled_at",
        "memory_last_sampled_at",
        "memory_coverage_seconds",
        "process_identity_verified",
        "gpu_identity_verified",
        "measurement_version",
        "exit_code",
    )
    return _hash({name: getattr(record, name) for name in fields})


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True).encode()
    ).hexdigest()


def _mark_calibrations_stale(session: Session, summary_id: str) -> None:
    calibration_ids = session.scalars(
        select(RecommendationCalibrationSourceRecord.calibration_id).where(
            RecommendationCalibrationSourceRecord.summary_id == summary_id
        )
    ).all()
    if calibration_ids:
        session.query(RecommendationCalibrationSnapshotRecord).filter(
            RecommendationCalibrationSnapshotRecord.id.in_(calibration_ids)
        ).update({"stale": True}, synchronize_session=False)
