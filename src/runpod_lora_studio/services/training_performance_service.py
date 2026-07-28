from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, replace
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
    RecommendationCalibrationSnapshotRecord,
    RecommendationCalibrationSourceRecord,
    TrainingConfigRecord,
    TrainingExecutionSummaryRecord,
    TrainingJobEnvironmentSnapshotRecord,
    TrainingJobRecord,
    TrainingJobSelectedGpuRecord,
    TrainingMemoryAggregateRecord,
    TrainingMetricPointRecord,
    TrainingProgressRecord,
    TrainingRecommendationRecord,
    TrainingRecommendationRequestRecord,
)
from runpod_lora_studio.persistence.training_repository import utc_now
from runpod_lora_studio.services.gpu_memory_metrics import (
    GpuMemoryMetricsAdapter,
    gpu_uuid_fingerprint,
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
            job_environment = session.scalar(
                select(TrainingJobEnvironmentSnapshotRecord).where(
                    TrainingJobEnvironmentSnapshotRecord.training_job_id == str(job_id)
                )
            )
            selected_gpu = session.scalar(
                select(TrainingJobSelectedGpuRecord).where(
                    TrainingJobSelectedGpuRecord.training_job_id == str(job_id)
                )
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
            if job_environment is not None:
                metadata = {
                    **metadata,
                    "sd_scripts_version": job_environment.sd_scripts_version
                    or metadata.get("sd_scripts_version"),
                    "xformers_available": (
                        job_environment.xformers_available
                        if job_environment.xformers_available is not None
                        else metadata.get("xformers_available")
                    ),
                }
            memory_record = session.scalar(
                select(TrainingMemoryAggregateRecord).where(
                    TrainingMemoryAggregateRecord.training_job_id == str(job_id)
                )
            )
            memory = _memory_summary_from_record(memory_record)
            measured_gpu_verified = (
                memory.gpu_identity_verified and memory.gpu_uuid_fingerprint is not None
            )
            gpu_fingerprint = (
                selected_gpu.gpu_uuid_fingerprint
                if selected_gpu is not None
                else job_environment.gpu_uuid_fingerprint
                if job_environment and job_environment.gpu_uuid_fingerprint is not None
                else memory.gpu_uuid_fingerprint
                if measured_gpu_verified
                else None
            )
            selected_gpu_changed = selected_gpu is not None and (
                selected_gpu.status != "ok"
                or "GPU_CHANGED_DURING_JOB"
                in _json_tuple(selected_gpu.warning_codes_json)
            )
            memory_gpu_changed = (
                "GPU_CHANGED_DURING_JOB" in memory.failure_codes
                or "GPU_CHANGED_DURING_JOB" in memory.warning_codes
            )
            expected_gpu_fingerprint = (
                selected_gpu.gpu_uuid_fingerprint
                if selected_gpu is not None
                else job_environment.gpu_uuid_fingerprint
                if job_environment is not None
                else None
            )
            memory_identity_mismatch = (
                memory.gpu_uuid_fingerprint is not None
                and expected_gpu_fingerprint is not None
                and memory.gpu_uuid_fingerprint != expected_gpu_fingerprint
            )
            summary_memory = (
                _invalidate_memory_summary_identity(memory)
                if memory_identity_mismatch
                else memory
            )
            if selected_gpu is not None:
                gpu_total = (
                    selected_gpu.total_vram_bytes
                    if selected_gpu.total_vram_bytes is not None
                    else memory.total_bytes
                    if measured_gpu_verified
                    and memory.gpu_uuid_fingerprint == gpu_fingerprint
                    else None
                )
            else:
                gpu_total = (
                    job_environment.total_vram_bytes
                    if job_environment and job_environment.total_vram_bytes is not None
                    else memory.total_bytes
                    if measured_gpu_verified
                    and (
                        gpu_fingerprint is None
                        or memory.gpu_uuid_fingerprint == gpu_fingerprint
                    )
                    else None
                )
            gpu_architecture = (
                selected_gpu.gpu_architecture
                if selected_gpu is not None
                else job_environment.gpu_architecture
                if job_environment
                else None
            )
            gpu_index = (
                selected_gpu.logical_gpu_index
                if selected_gpu is not None
                else job_environment.logical_gpu_index
                if job_environment and job_environment.logical_gpu_index is not None
                else None
            )
            physical_gpu_index = (
                selected_gpu.physical_gpu_index
                if selected_gpu is not None
                else job_environment.physical_gpu_index
                if job_environment is not None
                else memory.gpu_index
                if measured_gpu_verified
                else None
            )
            compute_capability = (
                selected_gpu.compute_capability
                if selected_gpu is not None
                else job_environment.compute_capability
                if job_environment is not None
                else None
            )
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
            recommendation_environment = selected_gpu or job_environment
            if _runtime_gpu_differs_from_recommendation(
                recommendation_environment, environment_payload
            ):
                exclusion_reasons = tuple(
                    sorted(
                        set(exclusion_reasons)
                        | {"gpu_environment_changed_since_recommendation"}
                    )
                )
            if selected_gpu_changed or memory_gpu_changed:
                exclusion_reasons = tuple(
                    sorted(set(exclusion_reasons) | {"gpu_changed_during_job"})
                )
            if memory_identity_mismatch:
                exclusion_reasons = tuple(
                    sorted(set(exclusion_reasons) | {"gpu_identity_mismatch"})
                )
            failure_category = classification.category
            usable_speed = (
                not exclusion_reasons
                and failure_category is TrainingFailureCategory.NONE
            )
            usable_memory = (
                selected_gpu is not None
                and selected_gpu.status == "ok"
                and not selected_gpu_changed
                and not memory_gpu_changed
                and not memory_identity_mismatch
                and gpu_fingerprint is not None
                and memory.target_peak_allocated_bytes is not None
                and memory.process_identity_verified
                and memory.gpu_identity_verified
                and memory.sample_count >= 2
                and memory.confidence
                in {CalibrationConfidence.MEDIUM, CalibrationConfidence.HIGH}
                and memory.coverage_seconds is not None
                and memory.coverage_seconds > 0
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
                gpu_architecture=gpu_architecture,
                gpu_index=gpu_index,
                physical_gpu_index=physical_gpu_index,
                compute_capability=compute_capability,
                memory=summary_memory,
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
                job_environment_snapshot_id=(
                    UUID(job_environment.id) if job_environment else None
                ),
                classifier_version=classification.classifier_version,
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
                _mark_related_calibrations_stale(session, existing.id)
            session.commit()
            return summary


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
    gpu_architecture: str | None,
    gpu_index: int | None,
    physical_gpu_index: int | None,
    compute_capability: str | None,
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
    job_environment_snapshot_id: UUID | None,
    classifier_version: str,
    now: datetime,
) -> TrainingExecutionSummary:
    effective_batch = _int_option(
        config.extra_options, "gradient_accumulation_steps", 1
    )
    world_size = _int_option(config.extra_options, "world_size", 1)
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
        "gpu_architecture": gpu_architecture,
        "gpu_index": gpu_index,
        "physical_gpu_index": physical_gpu_index,
        "compute_capability": compute_capability,
        "settings_fingerprint": settings_fingerprint,
        "recommendation_id": recommendation_id,
        "environment_snapshot_id": environment_snapshot_id,
        "training_job_environment_snapshot_id": job_environment_snapshot_id,
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
        "world_size": world_size,
        "sd_scripts_version": _text_option(metadata, "sd_scripts_version"),
        "xformers_available": _bool_option(metadata, "xformers_available"),
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
        "memory_failed_sample_count": memory.failed_sample_count,
        "memory_confidence": memory.confidence,
        "minimum_free_vram_bytes": memory.whole_gpu_min_free_bytes,
        "whole_gpu_peak_used_vram_bytes": memory.whole_gpu_peak_used_bytes,
        "other_process_peak_vram_bytes": memory.other_process_peak_used_bytes,
        "memory_first_sampled_at": memory.first_sampled_at,
        "memory_last_sampled_at": memory.last_sampled_at,
        "memory_coverage_seconds": memory.coverage_seconds,
        "process_identity_verified": memory.process_identity_verified,
        "gpu_identity_verified": memory.gpu_identity_verified,
        "measurement_version": memory.measurement_version,
        "memory_warning_codes": memory.warning_codes,
        "memory_failure_codes": memory.failure_codes,
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
        "classifier_version": classifier_version,
        "summary_content_fingerprint": "",
        "calibration_state_fingerprint": "",
        "calibration_included": True,
        "manual_exclusion_reason": None,
        "created_at": now,
        "updated_at": now,
    }
    content_values = {
        key: value
        for key, value in values.items()
        if key
        not in {
            "created_at",
            "updated_at",
            "summary_content_fingerprint",
            "calibration_state_fingerprint",
            "calibration_included",
            "manual_exclusion_reason",
            "failure_category",
            "failure_evidence_codes",
            "oom_detected",
            "usable_for_speed_calibration",
            "usable_for_memory_calibration",
            "collector_version",
            "classifier_version",
        }
    }
    content_fingerprint = _fingerprint(content_values)
    state_fingerprint = _fingerprint(
        {
            "summary": content_fingerprint,
            "included": True,
            "manual_exclusion_reason": None,
            "failure_category": failure_category.value,
            "failure_evidence_codes": evidence_codes,
            "oom_detected": values["oom_detected"],
            "usable_for_speed": usable_speed,
            "usable_for_memory": usable_memory,
            "exclusion_reasons": exclusion_reasons,
            "collector_version": TrainingPerformanceCollector.version,
            "classifier_version": classifier_version,
        }
    )
    values["summary_content_fingerprint"] = content_fingerprint
    values["calibration_state_fingerprint"] = state_fingerprint
    fingerprint = _fingerprint(
        {"content": content_fingerprint, "state": state_fingerprint}
    )
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
        training_job_environment_snapshot_id=(
            UUID(record.training_job_environment_snapshot_id)
            if record.training_job_environment_snapshot_id
            else None
        ),
        dataset_scale_fingerprint=record.dataset_scale_fingerprint,
        **{name: getattr(record, name) for name in _SUMMARY_FIELDS},
        failure_evidence_codes=tuple(
            json.loads(record.failure_evidence_codes_json or "[]")
        ),
        exclusion_reasons=tuple(json.loads(record.exclusion_reasons_json or "[]")),
        memory_confidence=CalibrationConfidence(record.memory_confidence),
        memory_warning_codes=_json_tuple(record.memory_warning_codes_json),
        memory_failure_codes=_json_tuple(record.memory_failure_codes_json),
        failure_category=TrainingFailureCategory(record.failure_category),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


_SUMMARY_FIELDS = (
    "gpu_total_vram_bytes",
    "gpu_architecture",
    "gpu_index",
    "physical_gpu_index",
    "compute_capability",
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
    "resume_initial_step",
    "resume_step_mode",
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
    "memory_first_sampled_at",
    "memory_last_sampled_at",
    "memory_coverage_seconds",
    "process_identity_verified",
    "gpu_identity_verified",
    "measurement_version",
    "exit_code",
    "oom_detected",
    "usable_for_speed_calibration",
    "usable_for_memory_calibration",
    "collector_version",
    "classifier_version",
    "summary_fingerprint",
    "summary_content_fingerprint",
    "calibration_state_fingerprint",
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
    for key in (
        "recommendation_id",
        "environment_snapshot_id",
        "training_job_environment_snapshot_id",
    ):
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
    values["memory_warning_codes_json"] = json.dumps(
        summary.memory_warning_codes, sort_keys=True
    )
    values["memory_failure_codes_json"] = json.dumps(
        summary.memory_failure_codes, sort_keys=True
    )
    values.pop("failure_evidence_codes")
    values.pop("exclusion_reasons")
    values.pop("memory_warning_codes")
    values.pop("memory_failure_codes")
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
) -> tuple[str | None, int | None, str | None, int | None]:
    raw = metadata.get("gpu_identity")
    if not isinstance(raw, dict):
        devices = environment.get("gpu_devices")
        raw = devices[0] if isinstance(devices, list) and devices else None
    if isinstance(raw, dict):
        identity = raw.get("uuid") or raw.get("architecture") or raw.get("name")
        total = raw.get("total_vram_bytes")
        if identity:
            fingerprint = (
                gpu_uuid_fingerprint(str(identity))
                if raw.get("uuid")
                else hashlib.sha256(f"gpu:{identity}".encode()).hexdigest()[:32]
            )
            architecture = _string_option(raw, "architecture")
            index = raw.get("index")
            return (
                fingerprint,
                int(total) if isinstance(total, int) else None,
                architecture,
                int(index) if isinstance(index, int) else None,
            )
    return None, None, None, None


def _memory_summary_from_record(
    record: TrainingMemoryAggregateRecord | None,
) -> GpuMemorySummary:
    if record is None:
        return GpuMemorySummary()
    coverage = None
    if record.first_sampled_at and record.last_sampled_at:
        coverage = max(
            0.0, (record.last_sampled_at - record.first_sampled_at).total_seconds()
        )
    total = record.gpu_total_vram_bytes
    other_ratio = None
    if total and record.other_process_peak_used_bytes is not None:
        other_ratio = record.other_process_peak_used_bytes / total
    return GpuMemorySummary(
        gpu_index=record.gpu_index,
        gpu_uuid_fingerprint=record.gpu_uuid_fingerprint,
        total_bytes=total,
        free_before_bytes=record.free_vram_before_bytes,
        free_after_bytes=record.free_vram_after_bytes,
        target_peak_allocated_bytes=record.target_process_peak_used_bytes,
        target_peak_reserved_bytes=record.target_process_peak_used_bytes,
        whole_gpu_min_free_bytes=record.minimum_free_vram_bytes,
        whole_gpu_peak_used_bytes=record.whole_gpu_peak_used_bytes,
        other_process_peak_used_bytes=record.other_process_peak_used_bytes,
        sample_count=record.sample_count,
        failed_sample_count=record.failed_sample_count,
        first_sampled_at=record.first_sampled_at,
        last_sampled_at=record.last_sampled_at,
        process_identity_verified=bool(record.process_identity_verified),
        gpu_identity_verified=bool(record.gpu_identity_verified),
        coverage_seconds=coverage,
        measurement_version=record.measurement_version,
        warning_codes=_json_tuple(record.warning_codes_json),
        failure_codes=_json_tuple(record.failure_codes_json),
        other_process_ratio=other_ratio,
        confidence=CalibrationConfidence(record.confidence),
    )


def _invalidate_memory_summary_identity(
    memory: GpuMemorySummary,
) -> GpuMemorySummary:
    """Prevent memory values from a different GPU entering the summary."""

    return replace(
        memory,
        gpu_index=None,
        gpu_uuid_fingerprint=None,
        total_bytes=None,
        free_before_bytes=None,
        free_after_bytes=None,
        target_peak_allocated_bytes=None,
        target_peak_reserved_bytes=None,
        whole_gpu_min_free_bytes=None,
        whole_gpu_peak_used_bytes=None,
        other_process_peak_used_bytes=None,
        sample_count=0,
        process_identity_verified=False,
        gpu_identity_verified=False,
        coverage_seconds=None,
        other_process_ratio=None,
        confidence=CalibrationConfidence.NONE,
        failure_codes=tuple(
            sorted(set(memory.failure_codes) | {"GPU_IDENTITY_MISMATCH"})
        ),
    )


def _runtime_gpu_differs_from_recommendation(
    job_environment: (
        TrainingJobEnvironmentSnapshotRecord | TrainingJobSelectedGpuRecord | None
    ),
    recommendation_payload: dict[str, object],
) -> bool:
    if job_environment is None:
        return False
    devices = recommendation_payload.get("gpu_devices")
    if not isinstance(devices, list):
        return False
    expected = {
        gpu_uuid_fingerprint(str(device["uuid"]))
        for device in devices
        if isinstance(device, dict) and device.get("uuid")
    }
    if not expected:
        return False
    if job_environment.gpu_uuid_fingerprint is not None:
        return job_environment.gpu_uuid_fingerprint not in expected
    try:
        visible = set(
            json.loads(getattr(job_environment, "visible_gpu_uuids_json", "[]") or "[]")
        )
    except (TypeError, ValueError):
        visible = set()
    return bool(visible) and not (visible & expected)


def _mark_related_calibrations_stale(session: Any, summary_id: str) -> None:
    calibration_ids = session.scalars(
        select(RecommendationCalibrationSourceRecord.calibration_id).where(
            RecommendationCalibrationSourceRecord.summary_id == summary_id
        )
    ).all()
    if calibration_ids:
        session.query(RecommendationCalibrationSnapshotRecord).filter(
            RecommendationCalibrationSnapshotRecord.id.in_(calibration_ids)
        ).update({"stale": True}, synchronize_session=False)


def _string_option(values: dict[str, object], key: str) -> str | None:
    value = values.get(key)
    return str(value) if value is not None else None


def _json_tuple(value: str | None) -> tuple[str, ...]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(sorted(str(item) for item in parsed if item))


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


def _bool_option(values: dict[str, object], key: str) -> bool | None:
    value = values.get(key)
    return value if isinstance(value, bool) else None


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
