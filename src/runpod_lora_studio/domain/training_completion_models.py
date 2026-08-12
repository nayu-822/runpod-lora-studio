from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class TrainingCompletionStatus(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    UPLOADING = "uploading"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"
    CANCELED = "canceled"


class TrainingCompletionErrorCode(StrEnum):
    ELIGIBILITY_FAILED = "COMPLETION_ELIGIBILITY_FAILED"
    TRAINING_NOT_SUCCEEDED = "TRAINING_NOT_SUCCEEDED"
    TRAINING_PROCESS_ACTIVE = "TRAINING_PROCESS_ACTIVE"
    DATASET_NOT_COMPLETED = "DATASET_NOT_COMPLETED"
    DATASET_REMOTE_NOT_VERIFIED = "DATASET_REMOTE_NOT_VERIFIED"
    MODEL_NOT_VERIFIED = "MODEL_NOT_VERIFIED"
    FINAL_LORA_MISSING = "FINAL_LORA_MISSING"
    FINAL_LORA_INVALID = "FINAL_LORA_INVALID"
    FINAL_LORA_CHANGING = "FINAL_LORA_CHANGING"
    PREVIEW_STALE = "PREVIEW_STALE"
    LOCAL_EXPORT_CONFLICT = "LOCAL_EXPORT_CONFLICT"
    REMOTE_COMPLETION_CONFLICT = "REMOTE_COMPLETION_CONFLICT"
    REMOTE_VERIFICATION_FAILED = "REMOTE_VERIFICATION_FAILED"
    CANCELED = "CANCELED"
    WORKER_CLAIM_LOST = "WORKER_CLAIM_LOST"
    UNKNOWN = "COMPLETION_FAILED"


@dataclass(frozen=True, slots=True)
class TrainingCompletionFile:
    relative_path: str
    source_path: Path
    size_bytes: int
    sha256: str
    category: str


@dataclass(frozen=True, slots=True)
class TrainingCompletionPreview:
    token: str
    source_fingerprint: str
    training_job_id: UUID
    project_id: UUID
    training_config_id: UUID
    dataset_snapshot_id: UUID
    managed_model_id: UUID
    final_lora_filename: str
    final_lora_size_bytes: int
    final_lora_sha256: str
    local_export_relative_path: str
    remote_relative_path: str
    files: tuple[TrainingCompletionFile, ...]
    dataset_remote_transfer_job_id: UUID
    dataset_remote_manifest_sha256: str
    overwrite_policy: str
    verification_policy: str


@dataclass(frozen=True, slots=True)
class TrainingCompletionExport:
    id: UUID
    training_job_id: UUID
    project_id: UUID
    storage_transfer_job_id: UUID | None
    status: TrainingCompletionStatus
    current_stage: str
    worker_id: str | None
    claim_token: str | None
    worker_generation: int
    heartbeat_at: datetime | None
    cancel_requested: bool
    attempt_count: int
    source_fingerprint: str | None
    preview_fingerprint: str | None
    export_relative_path: str | None
    export_manifest_relative_path: str | None
    export_manifest_sha256: str | None
    export_fingerprint: str | None
    remote_relative_path: str | None
    remote_completion_manifest_relative_path: str | None
    completion_manifest_sha256: str | None
    error_code: str | None
    error_summary: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
