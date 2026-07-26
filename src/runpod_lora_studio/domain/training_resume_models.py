from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


def parse_non_negative_integer(value: object) -> int | None:
    """Parse only an ASCII decimal, non-negative integer value."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        try:
            return int(value)
        except ValueError:
            return None
    return None


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
class ValidatedStatePosition:
    epoch: int
    step: int
    epoch_source: str
    step_source: str


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
    state_epoch: int | None = None
    state_step: int | None = None
    state_position: ValidatedStatePosition | None = None


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
    state_epoch: int | None = None
    state_step: int | None = None
    initial_epoch: int | None = None
    initial_step: int | None = None
    progress_epoch_offset: int | None = None
    progress_step_offset: int | None = None
    position_warning: str | None = None
    state_epoch_source: str | None = None
    state_step_source: str | None = None
