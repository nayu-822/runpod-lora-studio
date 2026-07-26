from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID


class TrainingJobStatus(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STALE = "stale"


class TrainingJobTransitionError(ValueError):
    """Raised when a training job state transition is not allowed."""


class TrainingJobStateMachine:
    _transitions: dict[TrainingJobStatus, frozenset[TrainingJobStatus]] = {
        TrainingJobStatus.QUEUED: frozenset({TrainingJobStatus.STARTING}),
        TrainingJobStatus.STARTING: frozenset(
            {
                TrainingJobStatus.RUNNING,
                TrainingJobStatus.CANCEL_REQUESTED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.STALE,
            }
        ),
        TrainingJobStatus.RUNNING: frozenset(
            {
                TrainingJobStatus.CANCEL_REQUESTED,
                TrainingJobStatus.SUCCEEDED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.STALE,
            }
        ),
        TrainingJobStatus.CANCEL_REQUESTED: frozenset(
            {
                TrainingJobStatus.CANCELED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.STALE,
            }
        ),
        TrainingJobStatus.STALE: frozenset({TrainingJobStatus.FAILED}),
        TrainingJobStatus.CANCELED: frozenset(),
        TrainingJobStatus.SUCCEEDED: frozenset(),
        TrainingJobStatus.FAILED: frozenset(),
    }

    @classmethod
    def can_transition(
        cls, current: TrainingJobStatus, target: TrainingJobStatus
    ) -> bool:
        return target in cls._transitions.get(current, frozenset())

    @classmethod
    def transition(
        cls, current: TrainingJobStatus, target: TrainingJobStatus
    ) -> TrainingJobStatus:
        if not cls.can_transition(current, target):
            raise TrainingJobTransitionError(
                f"invalid training job transition: {current.value} -> {target.value}"
            )
        return target


@dataclass(frozen=True, slots=True)
class TrainingConfigInput:
    project_id: UUID
    dataset_snapshot_id: UUID
    managed_model_id: UUID
    name: str
    output_name: str
    output_directory: Path
    sd_scripts_root: Path
    trainer_script: str = "sdxl_train_network.py"
    resolution: int = 1024
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "AdamW8bit"
    scheduler: str = "cosine"
    network_module: str = "networks.lora"
    network_dim: int = 16
    network_alpha: int = 16
    mixed_precision: str = "fp16"
    save_every_n_epochs: int = 1
    cache_latents: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    id: UUID
    project_id: UUID
    dataset_snapshot_id: UUID
    managed_model_id: UUID
    name: str
    output_name: str
    output_directory: Path
    sd_scripts_root: Path
    trainer_script: str
    resolution: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    scheduler: str
    network_module: str
    network_dim: int
    network_alpha: int
    mixed_precision: str
    save_every_n_epochs: int
    cache_latents: bool
    gradient_checkpointing: bool
    seed: int
    extra_options: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "dataset_snapshot_id": str(self.dataset_snapshot_id),
            "managed_model_id": str(self.managed_model_id),
            "name": self.name,
            "output_name": self.output_name,
            "output_directory": str(self.output_directory),
            "sd_scripts_root": str(self.sd_scripts_root),
            "trainer_script": self.trainer_script,
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "network_module": self.network_module,
            "network_dim": self.network_dim,
            "network_alpha": self.network_alpha,
            "mixed_precision": self.mixed_precision,
            "save_every_n_epochs": self.save_every_n_epochs,
            "cache_latents": self.cache_latents,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "extra_options": self.extra_options,
        }

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class TrainingJob:
    id: UUID
    project_id: UUID
    training_config_id: UUID
    dataset_snapshot_id: UUID
    managed_model_id: UUID
    status: TrainingJobStatus
    cancel_requested: bool
    pid: int | None
    worker_id: str | None
    worker_heartbeat: datetime | None
    command_summary: str | None
    stdout_log_path: Path | None
    stderr_log_path: Path | None
    runtime_directory: Path | None
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    failure_code: str | None
    failure_message: str | None
    created_at: datetime
    updated_at: datetime
    parent_job_id: UUID | None = None
    resume_artifact_id: UUID | None = None
    resume_mode: str | None = None
    resume_validation_status: str | None = None
    resume_validation_code: str | None = None
    resume_validation_message: str | None = None
    initial_epoch: int | None = None
    initial_step: int | None = None
    progress_step_offset: int | None = None
    progress_epoch_offset: int | None = None
