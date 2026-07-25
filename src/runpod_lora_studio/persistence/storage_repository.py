from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.storage_models import (
    ManagedModel,
    ManagedModelStatus,
    ModelType,
    OverwritePolicy,
    ProjectStorageSettings,
    StorageKind,
    StorageTransferJob,
    StorageTransferType,
    TransferDirection,
    TransferStatus,
    VerificationPolicy,
)
from runpod_lora_studio.persistence.models import (
    ManagedModelRecord,
    ModelTransferRecord,
    ProjectStorageSettingsRecord,
    StorageTransferJobRecord,
    TransferItemRecord,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def managed_model_from_record(record: ManagedModelRecord) -> ManagedModel:
    return ManagedModel(
        id=UUID(record.id),
        display_name=record.display_name,
        model_type=ModelType(record.model_type),
        remote_name=record.remote_name,
        remote_relative_path=record.remote_relative_path,
        remote_file_name=record.remote_file_name,
        remote_size_bytes=record.remote_size_bytes,
        remote_modified_at=_utc(record.remote_modified_at),
        remote_hash_type=record.remote_hash_type,
        remote_hash_value=record.remote_hash_value,
        local_path=Path(record.local_path) if record.local_path else None,
        local_size_bytes=record.local_size_bytes,
        local_sha256=record.local_sha256,
        status=ManagedModelStatus(record.status),
        source=record.source,
        rclone_version=record.rclone_version,
        first_seen_at=_utc(record.first_seen_at) or utc_now(),
        last_seen_at=_utc(record.last_seen_at) or utc_now(),
        downloaded_at=_utc(record.downloaded_at),
        verified_at=_utc(record.verified_at),
        error_summary=record.error_summary,
    )


def storage_job_from_record(record: StorageTransferJobRecord) -> StorageTransferJob:
    return StorageTransferJob(
        id=UUID(record.id),
        project_id=UUID(record.project_id) if record.project_id else None,
        snapshot_id=UUID(record.snapshot_id) if record.snapshot_id else None,
        training_run_id=UUID(record.training_run_id)
        if record.training_run_id
        else None,
        transfer_type=StorageTransferType(record.transfer_type),
        source_kind=StorageKind(record.source_kind),
        destination_kind=StorageKind(record.destination_kind),
        status=TransferStatus(record.status),
        current_step=record.current_step,
        item_count=record.item_count,
        processed_item_count=record.processed_item_count,
        succeeded_item_count=record.succeeded_item_count,
        failed_item_count=record.failed_item_count,
        skipped_item_count=record.skipped_item_count,
        total_bytes=record.total_bytes,
        transferred_bytes=record.transferred_bytes,
        completed_transferred_bytes=record.completed_transferred_bytes,
        current_file_transferred_bytes=record.current_file_transferred_bytes,
        cancel_requested=bool(record.cancel_requested),
        pid=record.pid,
        worker_id=record.worker_id,
        heartbeat_at=_utc(record.heartbeat_at),
        started_at=_utc(record.started_at),
        completed_at=_utc(record.completed_at),
        error_summary=record.error_summary,
    )


class StorageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def managed_model_from_record(record: ManagedModelRecord) -> ManagedModel:
        return managed_model_from_record(record)

    @staticmethod
    def storage_job_from_record(record: StorageTransferJobRecord) -> StorageTransferJob:
        return storage_job_from_record(record)

    def get_model(self, model_id: UUID) -> ManagedModelRecord | None:
        return cast(
            ManagedModelRecord | None,
            self.session.scalar(
                select(ManagedModelRecord).where(ManagedModelRecord.id == str(model_id))
            ),
        )

    def get_model_by_remote(
        self, remote_name: str, relative_path: str
    ) -> ManagedModelRecord | None:
        return cast(
            ManagedModelRecord | None,
            self.session.scalar(
                select(ManagedModelRecord).where(
                    ManagedModelRecord.remote_name == remote_name,
                    ManagedModelRecord.remote_relative_path == relative_path,
                )
            ),
        )

    def list_models(self) -> list[ManagedModel]:
        records = self.session.scalars(
            select(ManagedModelRecord).order_by(
                ManagedModelRecord.display_name, ManagedModelRecord.remote_relative_path
            )
        ).all()
        return [managed_model_from_record(record) for record in records]

    def upsert_model(
        self,
        *,
        display_name: str,
        model_type: ModelType,
        remote_name: str,
        remote_relative_path: str,
        remote_file_name: str,
        remote_size_bytes: int,
        remote_modified_at: datetime | None,
        remote_hash_type: str | None,
        remote_hash_value: str | None,
        source: str,
        rclone_version: str | None,
    ) -> ManagedModelRecord:
        now = utc_now()
        record = self.get_model_by_remote(remote_name, remote_relative_path)
        if record is None:
            record = ManagedModelRecord(
                id=str(uuid4()),
                display_name=display_name,
                model_type=model_type.value,
                remote_name=remote_name,
                remote_relative_path=remote_relative_path,
                remote_file_name=remote_file_name,
                remote_size_bytes=remote_size_bytes,
                remote_modified_at=remote_modified_at,
                remote_hash_type=remote_hash_type,
                remote_hash_value=remote_hash_value,
                local_path=None,
                local_size_bytes=None,
                local_sha256=None,
                status=ManagedModelStatus.REMOTE_ONLY.value,
                source=source,
                rclone_version=rclone_version,
                first_seen_at=now,
                last_seen_at=now,
                downloaded_at=None,
                verified_at=None,
                error_summary=None,
                created_at=now,
                updated_at=now,
            )
            self.session.add(record)
        else:
            record.display_name = display_name
            record.model_type = model_type.value
            record.remote_file_name = remote_file_name
            record.remote_size_bytes = remote_size_bytes
            record.remote_modified_at = remote_modified_at
            record.remote_hash_type = remote_hash_type
            record.remote_hash_value = remote_hash_value
            record.source = source
            record.rclone_version = rclone_version
            record.last_seen_at = now
            record.updated_at = now
            if record.local_path and not Path(record.local_path).is_file():
                record.status = ManagedModelStatus.MISSING_LOCAL.value
        self.session.flush()
        return record

    def update_model(self, record: ManagedModelRecord, **values: Any) -> None:
        for name, value in values.items():
            if not hasattr(record, name):
                raise ValueError(f"unknown managed model field: {name}")
            setattr(record, name, value)
        record.updated_at = utc_now()

    def create_transfer(
        self,
        *,
        model_id: UUID,
        direction: TransferDirection,
        source_path: str,
        destination_path: str,
        expected_size_bytes: int,
        expected_hash: str | None,
        settings_snapshot: dict[str, Any],
    ) -> ModelTransferRecord:
        now = utc_now()
        record = ModelTransferRecord(
            id=str(uuid4()),
            managed_model_id=str(model_id),
            direction=direction.value,
            status=TransferStatus.PENDING.value,
            source_path=source_path,
            destination_path=destination_path,
            expected_size_bytes=expected_size_bytes,
            transferred_size_bytes=0,
            expected_hash=expected_hash,
            actual_hash=None,
            attempt_count=0,
            retry_count=0,
            started_at=None,
            completed_at=None,
            error_summary=None,
            rclone_exit_code=None,
            rclone_version=None,
            settings_snapshot=json.dumps(settings_snapshot, sort_keys=True),
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def create_job(
        self,
        *,
        project_id: UUID | None,
        snapshot_id: UUID | None,
        transfer_type: StorageTransferType,
        source_kind: StorageKind,
        destination_kind: StorageKind,
        item_count: int,
        total_bytes: int,
        status: TransferStatus = TransferStatus.PENDING,
    ) -> StorageTransferJobRecord:
        now = utc_now()
        record = StorageTransferJobRecord(
            id=str(uuid4()),
            project_id=str(project_id) if project_id else None,
            snapshot_id=str(snapshot_id) if snapshot_id else None,
            training_run_id=None,
            transfer_type=transfer_type.value,
            source_kind=source_kind.value,
            destination_kind=destination_kind.value,
            status=status.value,
            current_step="pending",
            item_count=item_count,
            processed_item_count=0,
            succeeded_item_count=0,
            failed_item_count=0,
            skipped_item_count=0,
            total_bytes=total_bytes,
            transferred_bytes=0,
            completed_transferred_bytes=0,
            current_file_transferred_bytes=0,
            cancel_requested=False,
            pid=None,
            worker_id=None,
            heartbeat_at=None,
            started_at=None,
            completed_at=None,
            error_summary=None,
            manifest_path=None,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_job(self, job_id: UUID) -> StorageTransferJobRecord | None:
        return cast(
            StorageTransferJobRecord | None,
            self.session.scalar(
                select(StorageTransferJobRecord).where(
                    StorageTransferJobRecord.id == str(job_id)
                )
            ),
        )

    def list_jobs(self, project_id: UUID | None = None) -> list[StorageTransferJob]:
        query = select(StorageTransferJobRecord)
        if project_id:
            query = query.where(StorageTransferJobRecord.project_id == str(project_id))
        records = self.session.scalars(
            query.order_by(StorageTransferJobRecord.created_at.desc())
        ).all()
        return [storage_job_from_record(record) for record in records]

    def update_job(self, record: StorageTransferJobRecord, **values: Any) -> None:
        for name, value in values.items():
            if not hasattr(record, name):
                raise ValueError(f"unknown transfer job field: {name}")
            setattr(record, name, value)
        record.updated_at = utc_now()

    def request_cancel(self, job_id: UUID) -> None:
        record = self.get_job(job_id)
        if record is not None:
            record.cancel_requested = True
            record.updated_at = utc_now()

    def add_item_if_missing(
        self,
        *,
        job_id: UUID,
        relative_path: str,
        item_type: str,
        direction: TransferDirection,
        expected_size: int,
        source_sha256: str | None,
    ) -> TransferItemRecord:
        record = cast(
            TransferItemRecord | None,
            self.session.scalar(
                select(TransferItemRecord).where(
                    TransferItemRecord.transfer_job_id == str(job_id),
                    TransferItemRecord.relative_path == relative_path,
                )
            ),
        )
        if record is not None:
            return record
        record = TransferItemRecord(
            id=str(uuid4()),
            transfer_job_id=str(job_id),
            relative_path=relative_path,
            item_type=item_type,
            direction=direction.value,
            expected_size=expected_size,
            transferred_size=0,
            source_sha256=source_sha256,
            destination_hash_type=None,
            destination_hash_value=None,
            verification_status="pending",
            status=TransferStatus.PENDING.value,
            retry_count=0,
            error_summary=None,
            created_at=utc_now(),
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_project_settings(self, project_id: UUID) -> ProjectStorageSettings | None:
        record = self.session.scalar(
            select(ProjectStorageSettingsRecord).where(
                ProjectStorageSettingsRecord.project_id == str(project_id)
            )
        )
        if record is None:
            return None
        return ProjectStorageSettings(
            project_id=project_id,
            project_remote_root=record.project_remote_root,
            snapshot_remote_root=record.snapshot_remote_root,
            training_remote_root=record.training_remote_root,
            artifact_remote_root=record.artifact_remote_root,
            selected_managed_model_id=(
                UUID(record.selected_managed_model_id)
                if record.selected_managed_model_id
                else None
            ),
            overwrite_policy=OverwritePolicy(record.overwrite_policy),
            verification_policy=VerificationPolicy(record.verification_policy),
        )

    def save_project_settings(self, settings: ProjectStorageSettings) -> None:
        record = self.session.scalar(
            select(ProjectStorageSettingsRecord).where(
                ProjectStorageSettingsRecord.project_id == str(settings.project_id)
            )
        )
        now = utc_now()
        values = {
            "project_remote_root": settings.project_remote_root,
            "snapshot_remote_root": settings.snapshot_remote_root,
            "training_remote_root": settings.training_remote_root,
            "artifact_remote_root": settings.artifact_remote_root,
            "selected_managed_model_id": (
                str(settings.selected_managed_model_id)
                if settings.selected_managed_model_id
                else None
            ),
            "overwrite_policy": settings.overwrite_policy.value,
            "verification_policy": settings.verification_policy.value,
            "updated_at": now,
        }
        if record is None:
            self.session.add(
                ProjectStorageSettingsRecord(
                    project_id=str(settings.project_id), created_at=now, **values
                )
            )
        else:
            for name, value in values.items():
                setattr(record, name, value)
