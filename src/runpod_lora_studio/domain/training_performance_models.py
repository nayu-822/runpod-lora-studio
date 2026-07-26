from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class TrainingFailureCategory(StrEnum):
    NONE = "none"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    SYSTEM_OUT_OF_MEMORY = "system_out_of_memory"
    DISK_FULL = "disk_full"
    MODEL_LOAD_FAILURE = "model_load_failure"
    DATASET_FAILURE = "dataset_failure"
    INVALID_CONFIGURATION = "invalid_configuration"
    DEPENDENCY_FAILURE = "dependency_failure"
    PROCESS_KILLED = "process_killed"
    USER_CANCELED = "user_canceled"
    STALE_PROCESS = "stale_process"
    UNKNOWN_FAILURE = "unknown_failure"


class CalibrationConfidence(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class TrainingFailureClassification:
    category: TrainingFailureCategory
    evidence_codes: tuple[str, ...] = ()
    classifier_version: str = "phase7b-failure-v1"


@dataclass(frozen=True, slots=True)
class GpuMemorySample:
    timestamp: datetime
    gpu_index: int
    total_bytes: int | None
    free_bytes: int | None
    process_used_bytes: int | None = None
    process_identity: str | None = None
    other_process_used_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class GpuMemorySummary:
    total_bytes: int | None = None
    free_before_bytes: int | None = None
    free_after_bytes: int | None = None
    target_peak_allocated_bytes: int | None = None
    target_peak_reserved_bytes: int | None = None
    whole_gpu_min_free_bytes: int | None = None
    sample_count: int = 0
    missing_sample_count: int = 0
    other_process_ratio: float | None = None
    confidence: CalibrationConfidence = CalibrationConfidence.NONE


@dataclass(frozen=True, slots=True)
class TrainingExecutionSummary:
    id: UUID
    training_job_id: UUID
    project_id: UUID
    training_config_id: UUID
    dataset_snapshot_id: UUID
    managed_model_id: UUID
    job_result_status: str
    gpu_identity_fingerprint: str | None
    settings_fingerprint: str | None
    dataset_scale_fingerprint: str | None = None
    environment_snapshot_id: UUID | None = None
    recommendation_id: UUID | None = None
    gpu_total_vram_bytes: int | None = None
    resolution: int | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    effective_batch_size: int | None = None
    network_module: str | None = None
    network_dim: int | None = None
    network_alpha: int | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    mixed_precision: str | None = None
    cache_latents: bool | None = None
    gradient_checkpointing: bool | None = None
    total_epochs: int | None = None
    planned_total_steps: int | None = None
    completed_steps: int | None = None
    resume_initial_step: int | None = None
    resume_step_mode: str | None = None
    elapsed_seconds: float | None = None
    measured_steps_per_second: float | None = None
    measured_images_per_second: float | None = None
    peak_allocated_vram_bytes: int | None = None
    peak_reserved_vram_bytes: int | None = None
    free_vram_before_bytes: int | None = None
    free_vram_after_bytes: int | None = None
    memory_sample_count: int = 0
    memory_confidence: CalibrationConfidence = CalibrationConfidence.NONE
    exit_code: int | None = None
    oom_detected: bool = False
    failure_category: TrainingFailureCategory = TrainingFailureCategory.NONE
    failure_evidence_codes: tuple[str, ...] = ()
    usable_for_speed_calibration: bool = False
    usable_for_memory_calibration: bool = False
    exclusion_reasons: tuple[str, ...] = ()
    collector_version: str = "phase7b-collector-v1"
    summary_fingerprint: str = ""
    calibration_included: bool = True
    manual_exclusion_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TrainingCalibrationSnapshot:
    id: UUID
    scope_project_id: UUID | None
    gpu_identity_fingerprint: str
    gpu_total_vram_class: str | None
    resolution: int | None
    optimizer: str | None
    mixed_precision: str | None
    cache_latents: bool | None
    gradient_checkpointing: bool | None
    sample_count: int
    successful_sample_count: int
    oom_sample_count: int
    median_steps_per_second: float | None
    lower_percentile_steps_per_second: float | None
    median_peak_vram_bytes: int | None
    upper_percentile_peak_vram_bytes: int | None
    confidence: CalibrationConfidence
    calibration_fingerprint: str
    calibration_version: str = "phase7b-calibration-v1"
    generated_at: datetime | None = None
    expires_at: datetime | None = None
    stale: bool = False
    reason_codes: tuple[str, ...] = ()
    source_summary_ids: tuple[UUID, ...] = ()
    source_summary_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class CalibrationRecommendationResult:
    baseline_duration_seconds: float | None
    calibrated_duration_seconds: float | None
    baseline_vram_bytes: int | None
    calibrated_vram_bytes: int | None
    baseline_batch_size: int | None
    suggested_batch_size: int | None
    confidence: CalibrationConfidence
    sample_count: int
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    calibration_fingerprint: str | None = None


# Names used by the design document and by older service integrations.
TrainingPerformanceSummary = TrainingExecutionSummary
RecommendationCalibrationSnapshot = TrainingCalibrationSnapshot
