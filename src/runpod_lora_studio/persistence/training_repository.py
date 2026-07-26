from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
    TrainingJob,
    TrainingJobStatus,
)
from runpod_lora_studio.persistence.models import (
    TrainingConfigRecord,
    TrainingJobRecord,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _config_from_record(record: TrainingConfigRecord) -> TrainingConfig:
    extra_options = json.loads(record.extra_options)
    if not isinstance(extra_options, dict):
        extra_options = {}
    return TrainingConfig(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        dataset_snapshot_id=UUID(record.dataset_snapshot_id),
        managed_model_id=UUID(record.managed_model_id),
        name=record.name,
        output_name=record.output_name,
        output_directory=Path(record.output_directory),
        sd_scripts_root=Path(record.sd_scripts_root),
        trainer_script=record.trainer_script,
        resolution=record.resolution,
        batch_size=record.batch_size,
        epochs=record.epochs,
        learning_rate=record.learning_rate,
        optimizer=record.optimizer,
        scheduler=record.scheduler,
        network_module=record.network_module,
        network_dim=record.network_dim,
        network_alpha=record.network_alpha,
        mixed_precision=record.mixed_precision,
        save_every_n_epochs=record.save_every_n_epochs,
        cache_latents=bool(record.cache_latents),
        gradient_checkpointing=bool(record.gradient_checkpointing),
        seed=record.seed,
        extra_options=cast(dict[str, Any], extra_options),
        recommendation_id=UUID(record.recommendation_id)
        if record.recommendation_id
        else None,
        recommendation_engine_version=record.recommendation_engine_version,
        recommendation_change_diff=cast(
            dict[str, Any], json.loads(record.recommendation_change_diff or "{}")
        ),
        created_at=_utc(record.created_at) or utc_now(),
        updated_at=_utc(record.updated_at) or utc_now(),
    )


def _job_from_record(record: TrainingJobRecord) -> TrainingJob:
    return TrainingJob(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        training_config_id=UUID(record.training_config_id),
        dataset_snapshot_id=UUID(record.dataset_snapshot_id),
        managed_model_id=UUID(record.managed_model_id),
        status=TrainingJobStatus(record.status),
        cancel_requested=bool(record.cancel_requested),
        pid=record.pid,
        worker_id=record.worker_id,
        worker_heartbeat=_utc(record.worker_heartbeat),
        command_summary=record.command_summary,
        stdout_log_path=Path(record.stdout_log_path)
        if record.stdout_log_path
        else None,
        stderr_log_path=Path(record.stderr_log_path)
        if record.stderr_log_path
        else None,
        runtime_directory=Path(record.runtime_directory)
        if record.runtime_directory
        else None,
        started_at=_utc(record.started_at),
        finished_at=_utc(record.finished_at),
        exit_code=record.exit_code,
        failure_code=record.failure_code,
        failure_message=record.failure_message,
        created_at=_utc(record.created_at) or utc_now(),
        updated_at=_utc(record.updated_at) or utc_now(),
        parent_job_id=UUID(record.parent_job_id) if record.parent_job_id else None,
        resume_artifact_id=(
            UUID(record.resume_artifact_id) if record.resume_artifact_id else None
        ),
        resume_mode=record.resume_mode,
        resume_validation_status=record.resume_validation_status,
        resume_validation_code=record.resume_validation_code,
        resume_validation_message=record.resume_validation_message,
        initial_epoch=record.initial_epoch,
        initial_step=record.initial_step,
        initial_epoch_source=record.initial_epoch_source,
        initial_step_source=record.initial_step_source,
        progress_step_offset=record.progress_step_offset,
        progress_epoch_offset=record.progress_epoch_offset,
        resume_request_fingerprint=record.resume_request_fingerprint,
    )


class TrainingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_config(self, data: TrainingConfigInput) -> TrainingConfigRecord:
        now = utc_now()
        record = TrainingConfigRecord(
            id=str(uuid4()),
            project_id=str(data.project_id),
            dataset_snapshot_id=str(data.dataset_snapshot_id),
            managed_model_id=str(data.managed_model_id),
            name=data.name,
            output_name=data.output_name,
            output_directory=str(data.output_directory),
            sd_scripts_root=str(data.sd_scripts_root),
            trainer_script=data.trainer_script,
            resolution=data.resolution,
            batch_size=data.batch_size,
            epochs=data.epochs,
            learning_rate=data.learning_rate,
            optimizer=data.optimizer,
            scheduler=data.scheduler,
            network_module=data.network_module,
            network_dim=data.network_dim,
            network_alpha=data.network_alpha,
            mixed_precision=data.mixed_precision,
            save_every_n_epochs=data.save_every_n_epochs,
            cache_latents=data.cache_latents,
            gradient_checkpointing=data.gradient_checkpointing,
            seed=data.seed,
            extra_options=json.dumps(
                data.extra_options, ensure_ascii=False, sort_keys=True
            ),
            recommendation_id=(
                str(data.recommendation_id) if data.recommendation_id else None
            ),
            recommendation_engine_version=data.recommendation_engine_version,
            recommendation_change_diff=json.dumps(
                data.recommendation_change_diff, ensure_ascii=False, sort_keys=True
            ),
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_config_record(self, config_id: UUID) -> TrainingConfigRecord | None:
        return cast(
            TrainingConfigRecord | None,
            self.session.scalar(
                select(TrainingConfigRecord).where(
                    TrainingConfigRecord.id == str(config_id)
                )
            ),
        )

    def get_config(self, config_id: UUID) -> TrainingConfig | None:
        record = self.get_config_record(config_id)
        return _config_from_record(record) if record else None

    def list_configs(self, project_id: UUID) -> list[TrainingConfig]:
        records = self.session.scalars(
            select(TrainingConfigRecord)
            .where(TrainingConfigRecord.project_id == str(project_id))
            .order_by(TrainingConfigRecord.updated_at.desc())
        ).all()
        return [_config_from_record(record) for record in records]

    def update_config(
        self, record: TrainingConfigRecord, data: TrainingConfigInput
    ) -> None:
        for name, value in {
            "project_id": str(data.project_id),
            "dataset_snapshot_id": str(data.dataset_snapshot_id),
            "managed_model_id": str(data.managed_model_id),
            "name": data.name,
            "output_name": data.output_name,
            "output_directory": str(data.output_directory),
            "sd_scripts_root": str(data.sd_scripts_root),
            "trainer_script": data.trainer_script,
            "resolution": data.resolution,
            "batch_size": data.batch_size,
            "epochs": data.epochs,
            "learning_rate": data.learning_rate,
            "optimizer": data.optimizer,
            "scheduler": data.scheduler,
            "network_module": data.network_module,
            "network_dim": data.network_dim,
            "network_alpha": data.network_alpha,
            "mixed_precision": data.mixed_precision,
            "save_every_n_epochs": data.save_every_n_epochs,
            "cache_latents": data.cache_latents,
            "gradient_checkpointing": data.gradient_checkpointing,
            "seed": data.seed,
            "extra_options": json.dumps(
                data.extra_options, ensure_ascii=False, sort_keys=True
            ),
            "recommendation_id": (
                str(data.recommendation_id) if data.recommendation_id else None
            ),
            "recommendation_engine_version": data.recommendation_engine_version,
            "recommendation_change_diff": json.dumps(
                data.recommendation_change_diff, ensure_ascii=False, sort_keys=True
            ),
        }.items():
            setattr(record, name, value)
        record.updated_at = utc_now()

    def create_job(
        self,
        config: TrainingConfig,
        status: TrainingJobStatus = TrainingJobStatus.QUEUED,
    ) -> TrainingJobRecord:
        now = utc_now()
        record = TrainingJobRecord(
            id=str(uuid4()),
            project_id=str(config.project_id),
            training_config_id=str(config.id),
            dataset_snapshot_id=str(config.dataset_snapshot_id),
            managed_model_id=str(config.managed_model_id),
            status=status.value,
            cancel_requested=False,
            pid=None,
            worker_id=None,
            worker_heartbeat=None,
            command_summary=None,
            stdout_log_path=None,
            stderr_log_path=None,
            runtime_directory=None,
            config_snapshot=None,
            parent_job_id=None,
            resume_artifact_id=None,
            resume_mode=None,
            resume_requested_at=None,
            resume_validation_status=None,
            resume_validation_code=None,
            resume_validation_message=None,
            initial_epoch=None,
            initial_step=None,
            initial_epoch_source=None,
            initial_step_source=None,
            progress_step_offset=None,
            progress_epoch_offset=None,
            process_start_time=None,
            process_group_id=None,
            process_identity=None,
            started_at=None,
            finished_at=None,
            exit_code=None,
            failure_code=None,
            failure_message=None,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_job_record(self, job_id: UUID) -> TrainingJobRecord | None:
        return cast(
            TrainingJobRecord | None,
            self.session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            ),
        )

    def get_job(self, job_id: UUID) -> TrainingJob | None:
        record = self.get_job_record(job_id)
        return _job_from_record(record) if record else None

    def list_jobs(self, project_id: UUID | None = None) -> list[TrainingJob]:
        query = select(TrainingJobRecord)
        if project_id is not None:
            query = query.where(TrainingJobRecord.project_id == str(project_id))
        records = self.session.scalars(
            query.order_by(TrainingJobRecord.created_at.desc())
        ).all()
        return [_job_from_record(record) for record in records]

    def list_active_records(self) -> list[TrainingJobRecord]:
        return list(
            self.session.scalars(
                select(TrainingJobRecord).where(
                    TrainingJobRecord.status.in_(
                        [
                            TrainingJobStatus.STARTING.value,
                            TrainingJobStatus.RUNNING.value,
                            TrainingJobStatus.CANCEL_REQUESTED.value,
                        ]
                    )
                )
            ).all()
        )

    def update_job(self, record: TrainingJobRecord, **values: Any) -> None:
        for name, value in values.items():
            if not hasattr(record, name):
                raise ValueError(f"unknown training job field: {name}")
            setattr(record, name, value)
        record.updated_at = utc_now()


__all__ = ["TrainingRepository", "_config_from_record", "_job_from_record", "utc_now"]
