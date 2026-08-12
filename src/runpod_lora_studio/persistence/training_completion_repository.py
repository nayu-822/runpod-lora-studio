from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.training_completion_models import (
    TrainingCompletionExport,
    TrainingCompletionStatus,
)
from runpod_lora_studio.persistence.models import TrainingCompletionExportRecord


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def export_from_record(
    record: TrainingCompletionExportRecord,
) -> TrainingCompletionExport:
    return TrainingCompletionExport(
        id=UUID(record.id),
        training_job_id=UUID(record.training_job_id),
        project_id=UUID(record.project_id),
        storage_transfer_job_id=(
            UUID(record.storage_transfer_job_id)
            if record.storage_transfer_job_id
            else None
        ),
        status=TrainingCompletionStatus(record.status),
        current_stage=record.current_stage,
        worker_id=record.worker_id,
        claim_token=record.claim_token,
        worker_generation=record.worker_generation,
        heartbeat_at=_utc(record.heartbeat_at),
        cancel_requested=bool(record.cancel_requested),
        attempt_count=record.attempt_count,
        source_fingerprint=record.source_fingerprint,
        preview_fingerprint=record.preview_fingerprint,
        export_relative_path=record.export_relative_path,
        export_manifest_relative_path=record.export_manifest_relative_path,
        export_manifest_sha256=record.export_manifest_sha256,
        export_fingerprint=record.export_fingerprint,
        remote_relative_path=record.remote_relative_path,
        remote_completion_manifest_relative_path=(
            record.remote_completion_manifest_relative_path
        ),
        completion_manifest_sha256=record.completion_manifest_sha256,
        error_code=record.error_code,
        error_summary=record.error_summary,
        started_at=_utc(record.started_at),
        completed_at=_utc(record.completed_at),
        created_at=_utc(record.created_at) or utc_now(),
        updated_at=_utc(record.updated_at) or utc_now(),
    )


class TrainingCompletionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, export_id: UUID) -> TrainingCompletionExportRecord | None:
        return cast(
            TrainingCompletionExportRecord | None,
            self.session.scalar(
                select(TrainingCompletionExportRecord).where(
                    TrainingCompletionExportRecord.id == str(export_id)
                )
            ),
        )

    def get_by_training_job(
        self, training_job_id: UUID
    ) -> TrainingCompletionExportRecord | None:
        return cast(
            TrainingCompletionExportRecord | None,
            self.session.scalar(
                select(TrainingCompletionExportRecord).where(
                    TrainingCompletionExportRecord.training_job_id
                    == str(training_job_id)
                )
            ),
        )

    def list_exports(
        self, project_id: UUID | None = None
    ) -> list[TrainingCompletionExport]:
        query = select(TrainingCompletionExportRecord)
        if project_id is not None:
            query = query.where(
                TrainingCompletionExportRecord.project_id == str(project_id)
            )
        records = self.session.scalars(
            query.order_by(TrainingCompletionExportRecord.created_at.desc())
        ).all()
        return [export_from_record(record) for record in records]

    def list_recovery_records(self) -> list[TrainingCompletionExportRecord]:
        return list(
            self.session.scalars(
                select(TrainingCompletionExportRecord).where(
                    TrainingCompletionExportRecord.status.in_(
                        [
                            TrainingCompletionStatus.PREPARING.value,
                            TrainingCompletionStatus.READY.value,
                            TrainingCompletionStatus.UPLOADING.value,
                            TrainingCompletionStatus.VERIFYING.value,
                            TrainingCompletionStatus.STALE.value,
                        ]
                    )
                )
            ).all()
        )

    def create(
        self,
        training_job_id: UUID,
        project_id: UUID,
        source_fingerprint: str,
        preview_fingerprint: str | None = None,
    ) -> TrainingCompletionExportRecord:
        now = utc_now()
        record = TrainingCompletionExportRecord(
            id=str(uuid4()),
            training_job_id=str(training_job_id),
            project_id=str(project_id),
            storage_transfer_job_id=None,
            status=TrainingCompletionStatus.PENDING.value,
            current_stage=TrainingCompletionStatus.PENDING.value,
            worker_id=None,
            claim_token=None,
            worker_generation=0,
            heartbeat_at=None,
            cancel_requested=False,
            attempt_count=0,
            source_fingerprint=source_fingerprint,
            preview_fingerprint=preview_fingerprint,
            export_relative_path=None,
            export_manifest_relative_path=None,
            export_manifest_sha256=None,
            export_fingerprint=None,
            remote_relative_path=None,
            remote_completion_manifest_relative_path=None,
            completion_manifest_sha256=None,
            error_code=None,
            error_summary=None,
            started_at=None,
            completed_at=None,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def claim(
        self,
        export_id: UUID,
        *,
        worker_id: str,
        claim_token: str,
        now: datetime | None = None,
        stale_after_seconds: float = 120.0,
    ) -> bool:
        current_time = now or utc_now()
        cutoff = current_time - timedelta(seconds=stale_after_seconds)
        statuses = [
            TrainingCompletionStatus.PENDING.value,
            TrainingCompletionStatus.READY.value,
            TrainingCompletionStatus.STALE.value,
            TrainingCompletionStatus.PREPARING.value,
            TrainingCompletionStatus.UPLOADING.value,
            TrainingCompletionStatus.VERIFYING.value,
        ]
        record = self.get(export_id)
        if record is None:
            return False
        owner_available = record.worker_id is None or record.heartbeat_at is None
        heartbeat = _utc(record.heartbeat_at)
        owner_stale = heartbeat is None or heartbeat < cutoff
        result = self.session.execute(
            update(TrainingCompletionExportRecord)
            .execution_options(synchronize_session=False)
            .where(
                TrainingCompletionExportRecord.id == str(export_id),
                TrainingCompletionExportRecord.status.in_(statuses),
                (
                    (TrainingCompletionExportRecord.worker_id.is_(None))
                    | (TrainingCompletionExportRecord.heartbeat_at.is_(None))
                    | (TrainingCompletionExportRecord.heartbeat_at < cutoff)
                    | (
                        TrainingCompletionExportRecord.status
                        == TrainingCompletionStatus.STALE.value
                    )
                ),
            )
            .values(
                status=TrainingCompletionStatus.PREPARING.value,
                current_stage=TrainingCompletionStatus.PREPARING.value,
                worker_id=worker_id,
                claim_token=claim_token,
                worker_generation=TrainingCompletionExportRecord.worker_generation + 1,
                heartbeat_at=current_time,
                started_at=TrainingCompletionExportRecord.started_at
                if record.started_at is not None
                else current_time,
                attempt_count=TrainingCompletionExportRecord.attempt_count + 1,
                updated_at=current_time,
            )
        )
        del owner_available, owner_stale
        return bool(result.rowcount == 1)

    def update_claimed(
        self,
        export_id: UUID,
        *,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
        values: dict[str, Any],
    ) -> bool:
        allowed = {
            "status",
            "current_stage",
            "heartbeat_at",
            "storage_transfer_job_id",
            "source_fingerprint",
            "preview_fingerprint",
            "export_relative_path",
            "export_manifest_relative_path",
            "export_manifest_sha256",
            "export_fingerprint",
            "remote_relative_path",
            "remote_completion_manifest_relative_path",
            "completion_manifest_sha256",
            "error_code",
            "error_summary",
            "completed_at",
            "cancel_requested",
            "worker_id",
            "claim_token",
        }
        if any(key not in allowed for key in values):
            raise ValueError("unknown completion export field")
        values = {**values, "updated_at": utc_now()}
        result = self.session.execute(
            update(TrainingCompletionExportRecord)
            .execution_options(synchronize_session=False)
            .where(
                TrainingCompletionExportRecord.id == str(export_id),
                TrainingCompletionExportRecord.worker_id == worker_id,
                TrainingCompletionExportRecord.claim_token == claim_token,
                TrainingCompletionExportRecord.worker_generation == worker_generation,
            )
            .values(**values)
        )
        return bool(result.rowcount == 1)

    def request_cancel(self, export_id: UUID) -> bool:
        result = self.session.execute(
            update(TrainingCompletionExportRecord)
            .execution_options(synchronize_session=False)
            .where(
                TrainingCompletionExportRecord.id == str(export_id),
                TrainingCompletionExportRecord.status
                != TrainingCompletionStatus.COMPLETED.value,
            )
            .values(cancel_requested=True, updated_at=utc_now())
        )
        return bool(result.rowcount == 1)

    def reset_for_retry(self, export_id: UUID) -> bool:
        result = self.session.execute(
            update(TrainingCompletionExportRecord)
            .execution_options(synchronize_session=False)
            .where(
                TrainingCompletionExportRecord.id == str(export_id),
                TrainingCompletionExportRecord.status.in_(
                    [
                        TrainingCompletionStatus.FAILED.value,
                        TrainingCompletionStatus.STALE.value,
                        TrainingCompletionStatus.CANCELED.value,
                    ]
                ),
            )
            .values(
                status=TrainingCompletionStatus.PENDING.value,
                current_stage=TrainingCompletionStatus.PENDING.value,
                worker_id=None,
                claim_token=None,
                heartbeat_at=None,
                cancel_requested=False,
                storage_transfer_job_id=None,
                error_code=None,
                error_summary=None,
                completed_at=None,
                updated_at=utc_now(),
            )
        )
        return bool(result.rowcount == 1)


__all__ = ["TrainingCompletionRepository", "export_from_record", "utc_now"]
