from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID


class TrainingParseStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class TrainingProgressSource(StrEnum):
    LOG = "log"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class TrainingArtifactType(StrEnum):
    LORA_CHECKPOINT = "lora_checkpoint"
    TRAINING_STATE = "training_state"
    OPTIMIZER_STATE = "optimizer_state"
    METADATA = "metadata"
    CONFIG_SNAPSHOT = "config_snapshot"
    LOG = "log"
    OTHER_SUPPORTED = "other_supported"


class TrainingArtifactValidationStatus(StrEnum):
    DISCOVERED = "discovered"
    VALID = "valid"
    INVALID = "invalid"
    CHANGING = "changing"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class TrainingMetricEvent:
    name: str
    value: float
    epoch: int | None = None
    step: int | None = None
    logged_at: datetime | None = None
    source: str = "log"


@dataclass(frozen=True, slots=True)
class TrainingLogParserState:
    remainder: str = ""
    current_epoch: int | None = None
    total_epochs: int | None = None
    current_step: int | None = None
    total_steps: int | None = None
    latest_loss: float | None = None
    learning_rate: float | None = None
    speed: float | None = None
    elapsed_seconds: float | None = None
    remaining_seconds: float | None = None
    total_steps_source: TrainingProgressSource = TrainingProgressSource.UNKNOWN
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedTrainingProgress:
    current_epoch: int | None
    total_epochs: int | None
    current_step: int | None
    total_steps: int | None
    latest_loss: float | None
    learning_rate: float | None
    speed: float | None
    elapsed_seconds: float | None
    estimated_remaining_seconds: float | None
    progress_ratio: float | None
    progress_source: TrainingProgressSource
    metric_events: tuple[TrainingMetricEvent, ...] = ()
    warnings: tuple[str, ...] = ()
    state: TrainingLogParserState = field(default_factory=TrainingLogParserState)


@dataclass(frozen=True, slots=True)
class TrainingLogParseResult:
    progress: ParsedTrainingProgress
    state: TrainingLogParserState


@dataclass(frozen=True, slots=True)
class EstimatedTrainingPlan:
    steps_per_epoch: int | None
    total_steps: int | None
    formula: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingProgressSnapshot:
    job_id: UUID
    current_epoch: int | None
    total_epochs: int | None
    current_step: int | None
    total_steps: int | None
    progress_ratio: float | None
    latest_loss: float | None
    smoothed_loss: float | None
    learning_rate: float | None
    steps_per_second: float | None
    samples_per_second: float | None
    elapsed_seconds: float | None
    estimated_remaining_seconds: float | None
    latest_log_at: datetime | None
    parser_version: str
    parse_status: TrainingParseStatus
    parse_warning: str | None
    progress_source: TrainingProgressSource
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    id: UUID
    job_id: UUID
    artifact_type: TrainingArtifactType
    relative_path: Path
    filename: str
    epoch: int | None
    step: int | None
    file_size: int
    sha256: str | None
    modified_at: datetime | None
    validation_status: TrainingArtifactValidationStatus
    validation_code: str | None
    validation_message: str | None
    discovered_at: datetime
    last_verified_at: datetime | None
    metadata: dict[str, Any] | None = None
