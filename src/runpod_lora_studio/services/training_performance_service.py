from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    GpuMemorySummary,
    TrainingExecutionSummary,
    TrainingFailureCategory,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    ComputeEnvironmentSnapshotRecord,
    TrainingConfigRecord,
    TrainingExecutionSummaryRecord,
    TrainingJobRecord,
    TrainingMetricPointRecord,
    TrainingProgressRecord,
    TrainingRecommendationRecord,
    TrainingRecommendationRequestRecord,
)
from runpod_lora_studio.persistence.training_repository import utc_now
from runpod_lora_studio.services.gpu_memory_metrics import (
    GpuMemoryMetricsAdapter,
    summarize_gpu_memory,
)
from runpod_lora_studio.services.training_failure_classifier import (
    TrainingFailureClassifier,
)

logger = logging.getLogger("runpod_lora_studio.training_performance")


class TrainingPerformanceCollector:
    """Collect a bounded, immutable view of a terminal training job."""

    version = "phase7b-collector-v1"

    def __init__(
        self,
        settings: AppSettings,
        *,
        memory_adapter: GpuMemoryMetricsAdapter | None = None,
        failure_classifier: TrainingFailureClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.memory_adapter = memory_adapter
        self.failure_classifier = failure_classifier or TrainingFailureClassifier()

    def collect(self, job_id: UUID, *, force: bool = False) -> TrainingExecutionSummary:
        with self.session_factory() as session:
            existing = session.scalar(
                select(TrainingExecutionSummaryRecord).where(
                    TrainingExecutionSummaryRecord.training_job_id == str(job_id)
                )
            )
            if existing is not None and not force:
                return _summary_from_record(existing)
            job = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            if job is None:
                raise ValueError("training job not found")
            config = session.scalar(
                select(TrainingConfigRecord).where(
                    TrainingConfigRecord.id == job.training_config_id
                )
            )
            progress = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == str(job_id)
                )
            )
            if config is None:
                raise ValueError("training config not found")
            recommendation = (
                session.scalar(
                    select(TrainingRecommendationRecord).where(
                        TrainingRecommendationRecord.id == config.recommendation_id
                    )
                )
                if config.recommendation_id
                else None
            )
            request = (
                session.scalar(
                    select(TrainingRecommendationRequestRecord).where(
                        TrainingRecommendationRequestRecord.id
                        == recommendation.request_id
                    )
                )
                if recommendation is not None
                else None
            )
            environment_snapshot_id = (
                UUID(request.environment_snapshot_id) if request else None
            )
            recommendation_id = (
                UUID(config.recommendation_id) if config.recommendation_id else None
            )
            environment_payload: dict[str, object] = {}
            if environment_snapshot_id is not None:
                environment_record = session.scalar(
                    select(ComputeEnvironmentSnapshotRecord).where(
                        ComputeEnvironmentSnapshotRecord.id
                        == str(environment_snapshot_id)
                    )
                )
                if environment_record is not None:
                    try:
                        parsed_payload = json.loads(environment_record.payload_json)
                    except ValueError:
                        parsed_payload = {}
                    if isinstance(parsed_payload, dict):
                        environment_payload = parsed_payload
            metadata = _read_metadata(job.runtime_directory)
            gpu_fingerprint, gpu_total = _gpu_identity(metadata, environment_payload)
            memory = self._memory_summary(job.pid)
            stdout = _read_tail(
                job.stdout_log_path, self.settings.training_log_tail_bytes
            )
            stderr = _read_tail(
                job.stderr_log_path, self.settings.training_log_tail_bytes
            )
            classification = self.failure_classifier.classify(
                status=job.status,
                exit_code=job.exit_code,
                stdout=stdout,
                stderr=stderr,
                cancel_requested=bool(job.cancel_requested),
                stale=job.status == "stale",
            )
            metric_rows = session.scalars(
                select(TrainingMetricPointRecord)
                .where(TrainingMetricPointRecord.training_job_id == str(job_id))
                .order_by(
                    TrainingMetricPointRecord.step.asc(),
                    TrainingMetricPointRecord.internal_id.asc(),
                )
            ).all()
            speed_values = [
                row.value
                for row in metric_rows
                if row.metric_name == "steps_per_second" and row.value > 0
            ]
            completed_steps = progress.current_step if progress else None
            planned_steps = progress.total_steps if progress else None
            elapsed = progress.elapsed_seconds if progress else None
            if elapsed is None and job.started_at and job.finished_at:
                elapsed = max(0.0, (job.finished_at - job.started_at).total_seconds())
            settings_fingerprint = _settings_fingerprint(config)
            exclusion_reasons = _exclusion_reasons(
                status=job.status,
                completed_steps=completed_steps,
                planned_steps=planned_steps,
                elapsed_seconds=elapsed,
                speed_values=speed_values,
                resume_mode=job.resume_mode,
                initial_step=job.initial_step,
                parse_warning=progress.parse_warning if progress else None,
            )
            failure_category = classification.category
            usable_speed = (
                not exclusion_reasons
                and failure_category is TrainingFailureCategory.NONE
            )
            usable_memory = (
                gpu_fingerprint is not None
                and memory.sample_count >= 2
                and memory.confidence is not CalibrationConfidence.NONE
                and (
                    memory.other_process_ratio is None
                    or memory.other_process_ratio <= 0.5
                )
            )
            summary = _make_summary(
                job=job,
                config=config,
                progress=progress,
                metadata=metadata,
                gpu_fingerprint=gpu_fingerprint,
                gpu_total=gpu_total,
                memory=memory,
                completed_steps=completed_steps,
                planned_steps=planned_steps,
                elapsed=elapsed,
                speed_values=speed_values,
                failure_category=failure_category,
                evidence_codes=classification.evidence_codes,
                usable_speed=usable_speed,
                usable_memory=usable_memory,
                exclusion_reasons=exclusion_reasons,
                settings_fingerprint=settings_fingerprint,
                recommendation_id=recommendation_id,
                environment_snapshot_id=environment_snapshot_id,
                now=utc_now(),
            )
            if existing is None:
                session.add(_record_from_summary(summary))
            else:
                summary = TrainingExecutionSummary(
                    **{**asdict(summary), "id": UUID(existing.id)}
                )
                for key, value in _record_values(summary).items():
                    setattr(existing, key, value)
            session.commit()
            return summary

    def _memory_summary(self, pid: int | None) -> GpuMemorySummary:
        if self.memory_adapter is None:
            return GpuMemorySummary()
        try:
            return summarize_gpu_memory(self.memory_adapter.collect(pid=pid))
        except Exception:
            logger.exception("training_memory_collection_failed")
            return GpuMemorySummary()


# The longer name is part of the Phase 7B service contract.
TrainingExecutionCollector = TrainingPerformanceCollector


def _make_summary(
    *,
    job: TrainingJobRecord,
    config: TrainingConfigRecord,
    progress: TrainingProgressRecord | None,
    metadata: dict[str, object],
    gpu_fingerprint: str | None,
    gpu_total: int | None,
    memory: GpuMemorySummary,
    completed_steps: int | None,
    planned_steps: int | None,
    elapsed: float | None,
    speed_values: list[float],
    failure_category: TrainingFailureCategory,
    evidence_codes: tuple[str, ...],
    usable_speed: bool,
    usable_memory: bool,
    exclusion_reasons: tuple[str, ...],
    settings_fingerprint: str,
    recommendation_id: UUID | None,
    environment_snapshot_id: UUID | None,
    now: datetime,
) -> TrainingExecutionSummary:
    effective_batch = _int_option(
        config.extra_options, "gradient_accumulation_steps", 1
    )
    dataset_fingerprint = _text_option(metadata, "source_dataset_content_sha256")
    speed = median(speed_values) if speed_values else None
    images_per_second = speed * effective_batch if speed is not None else None
    values: dict[str, Any] = {
        "training_job_id": UUID(job.id),
        "project_id": UUID(job.project_id),
        "training_config_id": UUID(job.training_config_id),
        "dataset_snapshot_id": UUID(job.dataset_snapshot_id),
        "managed_model_id": UUID(job.managed_model_id),
        "job_result_status": job.status,
        "gpu_identity_fingerprint": gpu_fingerprint,
        "settings_fingerprint": settings_fingerprint,
        "recommendation_id": recommendation_id,
        "environment_snapshot_id": environment_snapshot_id,
        "dataset_scale_fingerprint": dataset_fingerprint,
        "gpu_total_vram_bytes": gpu_total,
        "resolution": config.resolution,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": effective_batch,
        "effective_batch_size": config.batch_size * effective_batch,
        "network_module": config.network_module,
        "network_dim": config.network_dim,
        "network_alpha": config.network_alpha,
        "optimizer": config.optimizer,
        "scheduler": config.scheduler,
        "mixed_precision": config.mixed_precision,
        "cache_latents": bool(config.cache_latents),
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "total_epochs": progress.total_epochs if progress else config.epochs,
        "planned_total_steps": planned_steps,
        "completed_steps": completed_steps,
        "resume_initial_step": job.initial_step,
        "resume_step_mode": job.resume_mode,
        "elapsed_seconds": elapsed,
        "measured_steps_per_second": speed,
        "measured_images_per_second": images_per_second,
        "peak_allocated_vram_bytes": memory.target_peak_allocated_bytes,
        "peak_reserved_vram_bytes": memory.target_peak_reserved_bytes,
        "free_vram_before_bytes": memory.free_before_bytes,
        "free_vram_after_bytes": memory.free_after_bytes,
        "memory_sample_count": memory.sample_count,
        "memory_confidence": memory.confidence,
        "exit_code": job.exit_code,
        "oom_detected": failure_category
        in {
            TrainingFailureCategory.CUDA_OUT_OF_MEMORY,
            TrainingFailureCategory.SYSTEM_OUT_OF_MEMORY,
        },
        "failure_category": failure_category,
        "failure_evidence_codes": evidence_codes,
        "usable_for_speed_calibration": usable_speed,
        "usable_for_memory_calibration": usable_memory,
        "exclusion_reasons": exclusion_reasons,
        "collector_version": TrainingPerformanceCollector.version,
        "created_at": now,
        "updated_at": now,
    }
    fingerprint = _fingerprint(values)
    return TrainingExecutionSummary(
        id=uuid4(), summary_fingerprint=fingerprint, **values
    )


def _summary_from_record(
    record: TrainingExecutionSummaryRecord,
) -> TrainingExecutionSummary:
    return TrainingExecutionSummary(
        id=UUID(record.id),
        training_job_id=UUID(record.training_job_id),
        project_id=UUID(record.project_id),
        training_config_id=UUID(record.training_config_id),
        dataset_snapshot_id=UUID(record.dataset_snapshot_id),
        managed_model_id=UUID(record.managed_model_id),
        job_result_status=record.job_result_status,
        gpu_identity_fingerprint=record.gpu_identity_fingerprint,
        settings_fingerprint=record.settings_fingerprint,
        recommendation_id=UUID(record.recommendation_id)
        if record.recommendation_id
        else None,
        environment_snapshot_id=UUID(record.environment_snapshot_id)
        if record.environment_snapshot_id
        else None,
        dataset_scale_fingerprint=record.dataset_scale_fingerprint,
        **{name: getattr(record, name) for name in _SUMMARY_FIELDS},
        failure_evidence_codes=tuple(
            json.loads(record.failure_evidence_codes_json or "[]")
        ),
        exclusion_reasons=tuple(json.loads(record.exclusion_reasons_json or "[]")),
        memory_confidence=CalibrationConfidence(record.memory_confidence),
        failure_category=TrainingFailureCategory(record.failure_category),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


_SUMMARY_FIELDS = (
    "gpu_total_vram_bytes",
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
    "total_epochs",
    "planned_total_steps",
    "completed_steps",
    "resume_initial_step",
    "resume_step_mode",
    "elapsed_seconds",
    "measured_steps_per_second",
    "measured_images_per_second",
    "peak_allocated_vram_bytes",
    "peak_reserved_vram_bytes",
    "free_vram_before_bytes",
    "free_vram_after_bytes",
    "memory_sample_count",
    "exit_code",
    "oom_detected",
    "usable_for_speed_calibration",
    "usable_for_memory_calibration",
    "collector_version",
    "summary_fingerprint",
    "calibration_included",
    "manual_exclusion_reason",
)


def _record_from_summary(
    summary: TrainingExecutionSummary,
) -> TrainingExecutionSummaryRecord:
    return TrainingExecutionSummaryRecord(**_record_values(summary))


def _record_values(summary: TrainingExecutionSummary) -> dict[str, object]:
    values = asdict(summary)
    values.pop("id")
    values["id"] = str(summary.id)
    for key in (
        "training_job_id",
        "project_id",
        "training_config_id",
        "dataset_snapshot_id",
        "managed_model_id",
    ):
        values[key] = str(values[key])
    for key in ("recommendation_id", "environment_snapshot_id"):
        if values[key] is not None:
            values[key] = str(values[key])
    values["failure_category"] = summary.failure_category.value
    values["memory_confidence"] = summary.memory_confidence.value
    values["failure_evidence_codes_json"] = json.dumps(
        summary.failure_evidence_codes, sort_keys=True
    )
    values["exclusion_reasons_json"] = json.dumps(
        summary.exclusion_reasons, sort_keys=True
    )
    values.pop("failure_evidence_codes")
    values.pop("exclusion_reasons")
    return values


def _read_metadata(runtime_directory: str | None) -> dict[str, object]:
    if not runtime_directory:
        return {}
    path = Path(runtime_directory) / "runtime" / "metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_tail(path_value: str | None, limit: int) -> str:
    if not path_value:
        return ""
    try:
        with Path(path_value).open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _gpu_identity(
    metadata: dict[str, object], environment: dict[str, object]
) -> tuple[str | None, int | None]:
    raw = metadata.get("gpu_identity")
    if not isinstance(raw, dict):
        devices = environment.get("gpu_devices")
        raw = devices[0] if isinstance(devices, list) and devices else None
    if isinstance(raw, dict):
        identity = raw.get("uuid") or raw.get("architecture") or raw.get("name")
        total = raw.get("total_vram_bytes")
        if identity:
            fingerprint = hashlib.sha256(f"gpu:{identity}".encode()).hexdigest()[:32]
            return fingerprint, int(total) if isinstance(total, int) else None
    return None, None


def _settings_fingerprint(config: TrainingConfigRecord) -> str:
    values = {
        "resolution": config.resolution,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "optimizer": config.optimizer,
        "scheduler": config.scheduler,
        "network_module": config.network_module,
        "network_dim": config.network_dim,
        "network_alpha": config.network_alpha,
        "mixed_precision": config.mixed_precision,
        "cache_latents": bool(config.cache_latents),
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "extra_options": config.extra_options,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _fingerprint(values: dict[str, object]) -> str:
    stable = {
        key: str(value)
        for key, value in values.items()
        if key not in {"created_at", "updated_at"}
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _text_option(values: dict[str, object], key: str) -> str | None:
    value = values.get(key)
    return str(value) if value else None


def _int_option(value: str, key: str, default: int) -> int:
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return default
    raw = parsed.get(key) if isinstance(parsed, dict) else None
    return int(raw) if isinstance(raw, int) and raw > 0 else default


def _exclusion_reasons(
    *,
    status: str,
    completed_steps: int | None,
    planned_steps: int | None,
    elapsed_seconds: float | None,
    speed_values: list[float],
    resume_mode: str | None,
    initial_step: int | None,
    parse_warning: str | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if status != "succeeded":
        reasons.append("job_not_succeeded")
    if completed_steps is None or planned_steps is None:
        reasons.append("steps_missing")
    elif completed_steps <= 0 or completed_steps > planned_steps * 1.1:
        reasons.append("steps_invalid")
    if elapsed_seconds is None or elapsed_seconds <= 0:
        reasons.append("elapsed_missing")
    if not speed_values:
        reasons.append("speed_measurement_missing")
    if initial_step and resume_mode not in {"local", "cumulative"}:
        reasons.append("resume_offset_ambiguous")
    if parse_warning and "failed" in parse_warning.lower():
        reasons.append("progress_parser_warning")
    return tuple(sorted(set(reasons)))
