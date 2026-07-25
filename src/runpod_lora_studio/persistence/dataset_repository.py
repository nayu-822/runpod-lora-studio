from __future__ import annotations

# ruff: noqa: E501
import json
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.models import (
    DatasetIssueCategory,
    DatasetIssueSeverity,
    DatasetSnapshotItem,
    DatasetSnapshotStatus,
    DatasetSnapshotSummary,
    DatasetValidationIssue,
)
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotItemRecord,
    DatasetSnapshotRecord,
    DatasetValidationIssueRecord,
    SnapshotCreationJobRecord,
)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _issue_json(issue: DatasetValidationIssue) -> dict[str, object]:
    return {
        "issue_code": issue.issue_code,
        "severity": issue.severity.value,
        "category": issue.category.value,
        "message": issue.message,
        "image_id": str(issue.image_id) if issue.image_id else None,
        "measured_value": issue.measured_value,
        "threshold_value": issue.threshold_value,
        "details": issue.details,
    }


def issue_from_record(record: DatasetValidationIssueRecord) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        issue_code=record.issue_code,
        severity=DatasetIssueSeverity(record.severity),
        category=DatasetIssueCategory(record.category),
        message=record.message,
        image_id=UUID(record.image_id) if record.image_id else None,
        measured_value=record.measured_value,
        threshold_value=record.threshold_value,
        details=record.details,
    )


class DatasetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_snapshot(
        self,
        *,
        project_id: UUID,
        snapshot_id: UUID,
        name: str,
        description: str,
        snapshot_version: str,
        generator_version: str,
        source_project_version: str,
        source_tagger_run_id: UUID | None,
        target_image_count: int,
        warning_count: int,
        total_size_bytes: int,
        snapshot_root: str,
        dataset_toml_path: str,
        manifest_path: str,
        report_path: str,
        settings_snapshot: str,
        validation_summary: str,
    ) -> DatasetSnapshotRecord:
        now = datetime.now(UTC)
        record = DatasetSnapshotRecord(
            id=str(snapshot_id),
            project_id=str(project_id),
            name=name,
            description=description,
            status=DatasetSnapshotStatus.CREATING.value,
            snapshot_version=snapshot_version,
            generator_version=generator_version,
            source_project_version=source_project_version,
            source_tagger_run_id=(
                str(source_tagger_run_id) if source_tagger_run_id else None
            ),
            source_created_at=now,
            target_image_count=target_image_count,
            copied_image_count=0,
            failed_image_count=0,
            warning_count=warning_count,
            total_size_bytes=total_size_bytes,
            snapshot_root=snapshot_root,
            dataset_toml_path=dataset_toml_path,
            manifest_path=manifest_path,
            report_path=report_path,
            manifest_sha256=None,
            dataset_toml_sha256=None,
            content_sha256=None,
            settings_snapshot=settings_snapshot,
            validation_summary=validation_summary,
            error_summary=None,
            started_at=now,
            completed_at=None,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.add(
            SnapshotCreationJobRecord(
                id=str(uuid4()),
                snapshot_id=str(snapshot_id),
                status=DatasetSnapshotStatus.CREATING.value,
                cancel_requested=False,
                current_step="initializing",
                processed_count=0,
                total_count=target_image_count,
                current_image_id=None,
                error_summary=None,
                started_at=now,
                completed_at=None,
                created_at=now,
            )
        )
        self.session.flush()
        return record

    def active_snapshot(self, project_id: UUID) -> DatasetSnapshotSummary | None:
        record = self.session.scalar(
            select(DatasetSnapshotRecord)
            .where(
                DatasetSnapshotRecord.project_id == str(project_id),
                DatasetSnapshotRecord.status.in_(
                    [
                        DatasetSnapshotStatus.DRAFT.value,
                        DatasetSnapshotStatus.VALIDATING.value,
                        DatasetSnapshotStatus.CREATING.value,
                        DatasetSnapshotStatus.DB_FINALIZATION_PENDING.value,
                    ]
                ),
            )
            .order_by(DatasetSnapshotRecord.created_at.desc())
        )
        return self._summary(record) if record else None

    def get(self, snapshot_id: UUID) -> DatasetSnapshotRecord | None:
        return cast(
            DatasetSnapshotRecord | None,
            self.session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.id == str(snapshot_id)
                )
            ),
        )

    def list_snapshots(self, project_id: UUID) -> list[DatasetSnapshotSummary]:
        records = self.session.scalars(
            select(DatasetSnapshotRecord)
            .where(DatasetSnapshotRecord.project_id == str(project_id))
            .order_by(
                DatasetSnapshotRecord.created_at.desc(), DatasetSnapshotRecord.id.desc()
            )
        ).all()
        return [self._summary(record) for record in records]

    def update_progress(
        self,
        snapshot_id: UUID,
        *,
        processed_count: int,
        current_step: str,
        current_image_id: UUID | None = None,
    ) -> None:
        record = self._required(snapshot_id)
        job = self.session.scalar(
            select(SnapshotCreationJobRecord).where(
                SnapshotCreationJobRecord.snapshot_id == str(snapshot_id)
            )
        )
        if job is not None:
            job.processed_count = processed_count
            job.current_step = current_step
            job.current_image_id = str(current_image_id) if current_image_id else None
        record.updated_at = datetime.now(UTC)

    def request_cancel(self, snapshot_id: UUID) -> None:
        job = self._job(snapshot_id)
        job.cancel_requested = True

    def cancel_requested(self, snapshot_id: UUID) -> bool:
        return bool(self._job(snapshot_id).cancel_requested)

    def finish(
        self,
        snapshot_id: UUID,
        status: DatasetSnapshotStatus,
        *,
        copied_image_count: int = 0,
        failed_image_count: int = 0,
        manifest_sha256: str | None = None,
        dataset_toml_sha256: str | None = None,
        content_sha256: str | None = None,
        error_summary: str | None = None,
    ) -> None:
        record = self._required(snapshot_id)
        now = datetime.now(UTC)
        record.status = status.value
        record.copied_image_count = copied_image_count
        record.failed_image_count = failed_image_count
        record.manifest_sha256 = manifest_sha256
        record.dataset_toml_sha256 = dataset_toml_sha256
        record.content_sha256 = content_sha256
        record.error_summary = error_summary
        record.completed_at = now
        record.updated_at = now
        job = self._job(snapshot_id)
        job.status = status.value
        job.completed_at = now
        job.current_step = status.value
        job.error_summary = error_summary

    def mark_db_finalization_pending(
        self, snapshot_id: UUID, error_summary: str
    ) -> None:
        record = self._required(snapshot_id)
        now = datetime.now(UTC)
        record.status = DatasetSnapshotStatus.DB_FINALIZATION_PENDING.value
        record.error_summary = error_summary
        record.completed_at = None
        record.updated_at = now
        job = self._job(snapshot_id)
        job.status = DatasetSnapshotStatus.DB_FINALIZATION_PENDING.value
        job.current_step = DatasetSnapshotStatus.DB_FINALIZATION_PENDING.value
        job.error_summary = error_summary

    def list_records_for_recovery(
        self, project_id: UUID | None = None
    ) -> list[DatasetSnapshotRecord]:
        statuses = [
            DatasetSnapshotStatus.DB_FINALIZATION_PENDING.value,
            DatasetSnapshotStatus.FAILED.value,
            DatasetSnapshotStatus.CREATING.value,
        ]
        query = select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.status.in_(statuses)
        )
        if project_id is not None:
            query = query.where(DatasetSnapshotRecord.project_id == str(project_id))
        return list(
            self.session.scalars(
                query.order_by(
                    DatasetSnapshotRecord.created_at, DatasetSnapshotRecord.id
                )
            ).all()
        )

    def add_item_if_missing(self, item: DatasetSnapshotItem) -> bool:
        existing = self.session.scalar(
            select(DatasetSnapshotItemRecord).where(
                DatasetSnapshotItemRecord.snapshot_id == str(item.snapshot_id),
                DatasetSnapshotItemRecord.image_id == str(item.image_id),
            )
        )
        if existing is not None:
            return False
        self.add_item(item)
        return True

    def add_issue_if_missing(
        self, snapshot_id: UUID, issue: DatasetValidationIssue
    ) -> bool:
        query = select(DatasetValidationIssueRecord).where(
            DatasetValidationIssueRecord.snapshot_id == str(snapshot_id),
            DatasetValidationIssueRecord.issue_code == issue.issue_code,
            DatasetValidationIssueRecord.image_id
            == (str(issue.image_id) if issue.image_id else None),
            DatasetValidationIssueRecord.message == issue.message,
        )
        if self.session.scalar(query) is not None:
            return False
        self.add_issue(snapshot_id, issue)
        return True

    def add_item(self, item: DatasetSnapshotItem) -> None:
        self.session.add(
            DatasetSnapshotItemRecord(
                id=str(uuid4()),
                snapshot_id=str(item.snapshot_id),
                image_id=str(item.image_id),
                source_image_path=str(item.source_image_path),
                snapshot_image_relative_path=item.snapshot_image_relative_path,
                caption_relative_path=item.caption_relative_path,
                sequence_number=item.sequence_number,
                source_image_sha256=item.source_image_sha256,
                snapshot_image_sha256=item.snapshot_image_sha256,
                source_file_size=item.source_file_size,
                snapshot_file_size=item.snapshot_file_size,
                width=item.width,
                height=item.height,
                aspect_ratio=item.aspect_ratio,
                mime_type=item.mime_type,
                caption_id=str(item.caption_id),
                caption_revision=item.caption_revision,
                caption_sha256=item.caption_sha256,
                caption_text=item.caption_text,
                tag_count=item.tag_count,
                trigger_word_count=item.trigger_word_count,
                quality_status=item.quality_status,
                exact_duplicate_status=item.exact_duplicate_status,
                similarity_group_id=(
                    str(item.similarity_group_id) if item.similarity_group_id else None
                ),
                is_similarity_representative=item.is_similarity_representative,
                warnings_snapshot=json.dumps(
                    [_issue_json(issue) for issue in item.warnings],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                created_at=datetime.now(UTC),
            )
        )

    def add_issue(self, snapshot_id: UUID, issue: DatasetValidationIssue) -> None:
        self.session.add(
            DatasetValidationIssueRecord(
                id=str(uuid4()),
                snapshot_id=str(snapshot_id),
                image_id=str(issue.image_id) if issue.image_id else None,
                issue_code=issue.issue_code,
                severity=issue.severity.value,
                category=issue.category.value,
                message=issue.message,
                measured_value=issue.measured_value,
                threshold_value=issue.threshold_value,
                details=issue.details,
                created_at=datetime.now(UTC),
            )
        )

    def list_items(self, snapshot_id: UUID) -> list[DatasetSnapshotItemRecord]:
        return list(
            self.session.scalars(
                select(DatasetSnapshotItemRecord)
                .where(DatasetSnapshotItemRecord.snapshot_id == str(snapshot_id))
                .order_by(DatasetSnapshotItemRecord.sequence_number)
            ).all()
        )

    def list_issues(self, snapshot_id: UUID) -> list[DatasetValidationIssue]:
        records = self.session.scalars(
            select(DatasetValidationIssueRecord)
            .where(DatasetValidationIssueRecord.snapshot_id == str(snapshot_id))
            .order_by(DatasetValidationIssueRecord.internal_id)
        ).all()
        return [issue_from_record(record) for record in records]

    def recover_stale(self, project_id: UUID | None = None) -> int:
        query = select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.status.in_(
                [
                    DatasetSnapshotStatus.VALIDATING.value,
                    DatasetSnapshotStatus.CREATING.value,
                ]
            )
        )
        if project_id is not None:
            query = query.where(DatasetSnapshotRecord.project_id == str(project_id))
        records = self.session.scalars(query).all()
        for record in records:
            record.status = DatasetSnapshotStatus.FAILED.value
            record.error_summary = (
                "アプリ再起動後に作成中状態を復元できないため失敗扱いにしました。"
            )
            record.completed_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)
            job = self.session.scalar(
                select(SnapshotCreationJobRecord).where(
                    SnapshotCreationJobRecord.snapshot_id == record.id
                )
            )
            if job is not None:
                job.status = DatasetSnapshotStatus.FAILED.value
                job.current_step = DatasetSnapshotStatus.FAILED.value
                job.error_summary = record.error_summary
                job.completed_at = record.completed_at
        return len(records)

    def _required(self, snapshot_id: UUID) -> DatasetSnapshotRecord:
        record = self.get(snapshot_id)
        if record is None:
            raise ValueError("dataset snapshot not found")
        return record

    def _job(self, snapshot_id: UUID) -> SnapshotCreationJobRecord:
        record = cast(
            SnapshotCreationJobRecord | None,
            self.session.scalar(
                select(SnapshotCreationJobRecord).where(
                    SnapshotCreationJobRecord.snapshot_id == str(snapshot_id)
                )
            ),
        )
        if record is None:
            raise ValueError("snapshot creation job not found")
        return record

    @staticmethod
    def _summary(record: DatasetSnapshotRecord) -> DatasetSnapshotSummary:
        return DatasetSnapshotSummary(
            id=UUID(record.id),
            project_id=UUID(record.project_id),
            name=record.name,
            description=record.description,
            status=DatasetSnapshotStatus(record.status),
            target_image_count=record.target_image_count,
            copied_image_count=record.copied_image_count,
            failed_image_count=record.failed_image_count,
            warning_count=record.warning_count,
            total_size_bytes=record.total_size_bytes,
            snapshot_root=__import__("pathlib").Path(record.snapshot_root),
            manifest_sha256=record.manifest_sha256,
            content_sha256=record.content_sha256,
            source_tagger_run_id=(
                UUID(record.source_tagger_run_id)
                if record.source_tagger_run_id
                else None
            ),
            created_at=_utc(record.created_at),  # type: ignore[arg-type]
            completed_at=_utc(record.completed_at),
            error_summary=record.error_summary,
        )
