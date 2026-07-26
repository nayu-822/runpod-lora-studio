from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class ResumeValidationStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    STATE_CHANGED = "state_changed"
    SOURCE_JOB_ACTIVE = "source_job_active"
    MISSING_METADATA = "missing_metadata"
    UNSUPPORTED_STATE = "unsupported_state"
    VALIDATION_FAILED = "validation_failed"


class ResumeMode(StrEnum):
    COPY = "copy"


@dataclass(frozen=True, slots=True)
class ResumeStateFile:
    relative_path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedResumeState:
    source_job_id: UUID
    source_artifact_id: UUID
    source_relative_path: Path
    source_path: Path
    fingerprint: str
    files: tuple[ResumeStateFile, ...]
    total_size: int
    validator_version: str
    generated_at: datetime


@dataclass(frozen=True, slots=True)
class ResumeCompatibility:
    status: ResumeValidationStatus
    issues: tuple[str, ...]
    source_config_fingerprint: str | None
    target_config_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class TrainingResumePreview:
    source_job_id: UUID
    source_artifact_id: UUID
    target_config_id: UUID
    source_status: str
    source_state_name: str
    state_fingerprint: str
    current_epoch: int | None
    current_step: int | None
    source_total_epochs: int | None
    target_total_epochs: int
    output_name: str
    compatibility: ResumeCompatibility
    signature: str
    command_summary: str
