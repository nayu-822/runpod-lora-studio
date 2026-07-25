from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.models import (
    CaptionEditSource,
    CaptionTagValue,
    ProjectTagRule,
    StoredCaption,
    StoredTaggingResult,
    TagCategory,
    TaggerInferenceSettings,
    TaggerModelIdentity,
    TaggerRunStatus,
    TaggerRunSummary,
    TaggingResultStatus,
    TagPrediction,
    TagSource,
)
from runpod_lora_studio.persistence.models import (
    CaptionEditHistoryRecord,
    CaptionTagRecord,
    DetectedTagRecord,
    ImageCaptionRecord,
    ImageTaggingResultRecord,
    ProjectTagRuleRecord,
    TaggerRunRecord,
)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _run_view(record: TaggerRunRecord) -> TaggerRunSummary:
    return TaggerRunSummary(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        adapter_name=record.adapter_name,
        model_identifier=record.model_identifier,
        model_revision=record.model_revision,
        model_path=record.model_path,
        device=record.device,
        status=TaggerRunStatus(record.status),
        target_image_count=record.target_image_count,
        processed_image_count=record.processed_image_count,
        succeeded_image_count=record.succeeded_image_count,
        failed_image_count=record.failed_image_count,
        skipped_image_count=record.skipped_image_count,
        current_image_id=(
            UUID(record.current_image_id) if record.current_image_id else None
        ),
        cancel_requested=bool(record.cancel_requested),
        started_at=_utc(record.started_at),
        completed_at=_utc(record.completed_at),
        error_summary=record.error_summary,
        created_at=_utc(record.created_at),  # type: ignore[arg-type]
    )


def _tag_view(record: DetectedTagRecord) -> TagPrediction:
    return TagPrediction(
        tag_name_raw=record.tag_name_raw,
        tag_name_normalized=record.tag_name_normalized,
        category=TagCategory(record.category),
        confidence=record.confidence,
        original_order=record.original_order,
        source=TagSource(record.source),
    )


class TaggingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        project_id: UUID,
        identity: TaggerModelIdentity,
        settings: TaggerInferenceSettings,
        settings_snapshot: str,
        target_image_count: int,
    ) -> TaggerRunRecord:
        now = datetime.now(UTC)
        record = TaggerRunRecord(
            id=str(uuid4()),
            project_id=str(project_id),
            adapter_name=identity.adapter_name,
            model_identifier=identity.model_identifier,
            model_revision=identity.model_revision,
            model_path=identity.model_path,
            device=settings.device,
            general_threshold=settings.general_threshold,
            character_threshold=settings.character_threshold,
            save_rating=settings.save_rating,
            save_character=settings.save_character,
            save_general=settings.save_general,
            underscore_to_space=settings.underscore_to_space,
            escape_mode=settings.escape_mode,
            batch_size=settings.batch_size,
            max_workers=settings.max_workers,
            status=TaggerRunStatus.PENDING.value,
            target_image_count=target_image_count,
            processed_image_count=0,
            succeeded_image_count=0,
            failed_image_count=0,
            skipped_image_count=0,
            current_image_id=None,
            cancel_requested=False,
            started_at=None,
            completed_at=None,
            error_summary=None,
            settings_snapshot=settings_snapshot,
            implementation_version=identity.implementation_version,
            created_at=now,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def active_run(self, project_id: UUID) -> TaggerRunSummary | None:
        record = self.session.scalar(
            select(TaggerRunRecord)
            .where(
                TaggerRunRecord.project_id == str(project_id),
                TaggerRunRecord.status.in_(
                    [TaggerRunStatus.PENDING.value, TaggerRunStatus.RUNNING.value]
                ),
            )
            .order_by(TaggerRunRecord.created_at.desc())
        )
        return _run_view(record) if record else None

    def get_run(self, run_id: UUID) -> TaggerRunSummary | None:
        record = self.session.scalar(
            select(TaggerRunRecord).where(TaggerRunRecord.id == str(run_id))
        )
        return _run_view(record) if record else None

    def get_run_record(self, run_id: UUID) -> TaggerRunRecord | None:
        return cast(
            TaggerRunRecord | None,
            self.session.scalar(
                select(TaggerRunRecord).where(TaggerRunRecord.id == str(run_id))
            ),
        )

    def list_runs(self, project_id: UUID) -> list[TaggerRunSummary]:
        records = self.session.scalars(
            select(TaggerRunRecord)
            .where(TaggerRunRecord.project_id == str(project_id))
            .order_by(TaggerRunRecord.created_at.desc(), TaggerRunRecord.id.desc())
        ).all()
        return [_run_view(record) for record in records]

    def mark_running(self, run_id: UUID, device: str) -> None:
        record = self._required_run(run_id)
        record.status = TaggerRunStatus.RUNNING.value
        record.device = device
        record.started_at = datetime.now(UTC)

    def update_progress(
        self,
        run_id: UUID,
        *,
        processed: int,
        succeeded: int,
        failed: int,
        skipped: int | None = None,
        current_image_id: UUID | None,
    ) -> None:
        record = self._required_run(run_id)
        record.processed_image_count = processed
        record.succeeded_image_count = succeeded
        record.failed_image_count = failed
        if skipped is not None:
            record.skipped_image_count = skipped
        record.current_image_id = str(current_image_id) if current_image_id else None

    def finish_run(
        self,
        run_id: UUID,
        status: TaggerRunStatus,
        error_summary: str | None = None,
    ) -> None:
        record = self._required_run(run_id)
        record.status = status.value
        record.completed_at = datetime.now(UTC)
        record.current_image_id = None
        record.error_summary = error_summary

    def request_cancel(self, run_id: UUID) -> None:
        self._required_run(run_id).cancel_requested = True

    def cancel_requested(self, run_id: UUID) -> bool:
        return bool(self._required_run(run_id).cancel_requested)

    def recover_stale_runs(self, project_id: UUID | None = None) -> int:
        query = select(TaggerRunRecord).where(
            TaggerRunRecord.status.in_(
                [TaggerRunStatus.PENDING.value, TaggerRunStatus.RUNNING.value]
            )
        )
        if project_id is not None:
            query = query.where(TaggerRunRecord.project_id == str(project_id))
        records = self.session.scalars(query).all()
        now = datetime.now(UTC)
        for record in records:
            record.status = TaggerRunStatus.STALE.value
            record.completed_at = now
            record.error_summary = (
                "アプリ再起動後に実行中状態を復元できなかったためstaleにしました。"
            )
            record.current_image_id = None
        return len(records)

    def save_result(
        self,
        run_id: UUID,
        image_id: UUID,
        status: TaggingResultStatus,
        result: Iterable[TagPrediction] = (),
        raw_output: str | None = None,
        error_summary: str | None = None,
        identity: TaggerModelIdentity | None = None,
    ) -> None:
        run = self._required_run(run_id)
        existing = self.session.scalar(
            select(ImageTaggingResultRecord).where(
                ImageTaggingResultRecord.image_id == str(image_id),
                ImageTaggingResultRecord.tagger_run_id == str(run_id),
            )
        )
        if existing is not None:
            raise ValueError("image tagging result already exists for this run")
        now = datetime.now(UTC)
        result_record = ImageTaggingResultRecord(
            image_id=str(image_id),
            tagger_run_id=str(run_id),
            status=status.value,
            error_summary=error_summary,
            tagged_at=now if status is TaggingResultStatus.COMPLETED else None,
            raw_output=raw_output,
        )
        run.results.append(result_record)
        self.session.flush()
        for tag in result:
            if identity is None:
                raise ValueError("tagger identity is required for detected tags")
            result_record.tags.append(
                DetectedTagRecord(
                    image_tagging_result_id=result_record.internal_id,
                    tag_name_raw=tag.tag_name_raw,
                    tag_name_normalized=tag.tag_name_normalized,
                    category=tag.category.value,
                    confidence=tag.confidence,
                    original_order=tag.original_order,
                    source=tag.source.value,
                    model_identifier=identity.model_identifier,
                    model_revision=identity.model_revision,
                    tagger_run_id=str(run_id),
                    created_at=now,
                )
            )

    def result_for_image(
        self, image_id: UUID, run_id: UUID
    ) -> StoredTaggingResult | None:
        record = self.session.scalar(
            select(ImageTaggingResultRecord).where(
                ImageTaggingResultRecord.image_id == str(image_id),
                ImageTaggingResultRecord.tagger_run_id == str(run_id),
            )
        )
        if record is None:
            return None
        return StoredTaggingResult(
            image_id=UUID(record.image_id),
            tagger_run_id=UUID(record.tagger_run_id),
            status=TaggingResultStatus(record.status),
            error_summary=record.error_summary,
            tagged_at=_utc(record.tagged_at),
            tags=tuple(
                _tag_view(tag)
                for tag in sorted(record.tags, key=lambda item: item.original_order)
            ),
        )

    def completed_result_for_image(
        self, image_id: UUID, project_id: UUID, settings_snapshot: str
    ) -> StoredTaggingResult | None:
        record = self.session.scalar(
            select(ImageTaggingResultRecord)
            .join(TaggerRunRecord)
            .where(
                ImageTaggingResultRecord.image_id == str(image_id),
                TaggerRunRecord.project_id == str(project_id),
                TaggerRunRecord.settings_snapshot == settings_snapshot,
                ImageTaggingResultRecord.status == TaggingResultStatus.COMPLETED.value,
            )
            .order_by(TaggerRunRecord.created_at.desc())
        )
        return self._result_view(record) if record else None

    def latest_completed_result_for_project_image(
        self, image_id: UUID, project_id: UUID
    ) -> StoredTaggingResult | None:
        record = self.session.scalar(
            select(ImageTaggingResultRecord)
            .join(TaggerRunRecord)
            .where(
                ImageTaggingResultRecord.image_id == str(image_id),
                TaggerRunRecord.project_id == str(project_id),
                ImageTaggingResultRecord.status == TaggingResultStatus.COMPLETED.value,
            )
            .order_by(TaggerRunRecord.created_at.desc())
        )
        return self._result_view(record) if record else None

    def latest_failed_result(
        self, image_id: UUID, project_id: UUID
    ) -> StoredTaggingResult | None:
        record = self.session.scalar(
            select(ImageTaggingResultRecord)
            .join(TaggerRunRecord)
            .where(
                ImageTaggingResultRecord.image_id == str(image_id),
                TaggerRunRecord.project_id == str(project_id),
                ImageTaggingResultRecord.status == TaggingResultStatus.FAILED.value,
            )
            .order_by(TaggerRunRecord.created_at.desc())
        )
        return self._result_view(record) if record else None

    def results_for_run(
        self, project_id: UUID, run_id: UUID, accepted_only: bool = True
    ) -> list[StoredTaggingResult]:
        from runpod_lora_studio.persistence.models import ImageAssetRecord

        query = (
            select(ImageTaggingResultRecord)
            .join(TaggerRunRecord)
            .join(ImageAssetRecord)
            .where(
                TaggerRunRecord.project_id == str(project_id),
                TaggerRunRecord.id == str(run_id),
                ImageTaggingResultRecord.status == TaggingResultStatus.COMPLETED.value,
            )
            .order_by(ImageAssetRecord.internal_id)
        )
        if accepted_only:
            query = query.where(ImageAssetRecord.selection_state == "accepted")
        return [
            self._result_view(record) for record in self.session.scalars(query).all()
        ]

    def get_rules(self, project_id: UUID) -> list[ProjectTagRule]:
        records = self.session.scalars(
            select(ProjectTagRuleRecord)
            .where(ProjectTagRuleRecord.project_id == str(project_id))
            .order_by(ProjectTagRuleRecord.normalized_tag_name)
        ).all()
        return [
            ProjectTagRule(
                normalized_tag_name=record.normalized_tag_name,
                action=record.action,
                category=TagCategory(record.category),
                updated_by=record.updated_by,
            )
            for record in records
        ]

    def upsert_rules(
        self,
        project_id: UUID,
        rules: Iterable[ProjectTagRule],
        updated_by: str = "user",
    ) -> None:
        now = datetime.now(UTC)
        for rule in rules:
            record = self.session.scalar(
                select(ProjectTagRuleRecord).where(
                    ProjectTagRuleRecord.project_id == str(project_id),
                    ProjectTagRuleRecord.normalized_tag_name
                    == rule.normalized_tag_name,
                )
            )
            if record is None:
                self.session.add(
                    ProjectTagRuleRecord(
                        project_id=str(project_id),
                        normalized_tag_name=rule.normalized_tag_name,
                        action=rule.action,
                        category=rule.category.value,
                        updated_at=now,
                        updated_by=updated_by,
                    )
                )
            else:
                record.action = rule.action
                record.category = rule.category.value
                record.updated_at = now
                record.updated_by = updated_by

    def get_current_caption(self, image_id: UUID) -> StoredCaption | None:
        record = self.session.scalar(
            select(ImageCaptionRecord).where(
                ImageCaptionRecord.image_id == str(image_id),
                ImageCaptionRecord.is_current == 1,
            )
        )
        return self._caption_view(record) if record else None

    def save_caption(
        self,
        image_id: UUID,
        caption_text: str,
        tags: Iterable[CaptionTagValue],
        source: CaptionEditSource,
        source_tagger_run_id: UUID | None,
        diff_snapshot: str,
    ) -> StoredCaption:
        current = self.session.scalar(
            select(ImageCaptionRecord).where(
                ImageCaptionRecord.image_id == str(image_id),
                ImageCaptionRecord.is_current == 1,
            )
        )
        tag_values = tuple(tags)
        if current is not None:
            before = current.caption_text
            previous_revision = current.revision
            if (
                before == caption_text
                and self._caption_tag_values(current) == tag_values
            ):
                return self._caption_view(current)
            current.is_current = False
            self.session.flush()
        else:
            before = ""
            previous_revision = None
        now = datetime.now(UTC)
        record = ImageCaptionRecord(
            id=str(uuid4()),
            image_id=str(image_id),
            source_tagger_run_id=(
                str(source_tagger_run_id) if source_tagger_run_id else None
            ),
            revision=(previous_revision or 0) + 1,
            caption_text=caption_text,
            caption_format_version="phase3-v1",
            is_current=True,
            edit_source=source.value,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        self.session.flush()
        for position, tag in enumerate(tag_values):
            record.tags.append(
                CaptionTagRecord(
                    image_caption_id=record.internal_id,
                    tag_name=tag.tag_name,
                    normalized_name=tag.normalized_name,
                    category=tag.category.value,
                    source=tag.source.value,
                    position=position,
                    confidence=tag.confidence,
                    manually_added=tag.manually_added,
                    manually_removed=tag.manually_removed,
                )
            )
        self.session.add(
            CaptionEditHistoryRecord(
                id=str(uuid4()),
                image_id=str(image_id),
                image_caption_id=record.internal_id,
                previous_revision=previous_revision,
                new_revision=record.revision,
                before_text=before,
                after_text=caption_text,
                diff_snapshot=diff_snapshot,
                edit_source=source.value,
                created_at=now,
            )
        )
        self.session.flush()
        return self._caption_view(record)

    def caption_history(self, image_id: UUID) -> list[CaptionEditHistoryRecord]:
        return list(
            self.session.scalars(
                select(CaptionEditHistoryRecord)
                .where(CaptionEditHistoryRecord.image_id == str(image_id))
                .order_by(CaptionEditHistoryRecord.created_at.desc())
            ).all()
        )

    def _required_run(self, run_id: UUID) -> TaggerRunRecord:
        record = self.get_run_record(run_id)
        if record is None:
            raise ValueError("tagger run not found")
        return record

    def _result_view(self, record: ImageTaggingResultRecord) -> StoredTaggingResult:
        return StoredTaggingResult(
            image_id=UUID(record.image_id),
            tagger_run_id=UUID(record.tagger_run_id),
            status=TaggingResultStatus(record.status),
            error_summary=record.error_summary,
            tagged_at=_utc(record.tagged_at),
            tags=tuple(
                _tag_view(tag)
                for tag in sorted(record.tags, key=lambda item: item.original_order)
            ),
        )

    def _caption_view(self, record: ImageCaptionRecord) -> StoredCaption:
        return StoredCaption(
            image_id=UUID(record.image_id),
            id=UUID(record.id),
            revision=record.revision,
            caption_text=record.caption_text,
            source_tagger_run_id=(
                UUID(record.source_tagger_run_id)
                if record.source_tagger_run_id
                else None
            ),
            edit_source=CaptionEditSource(record.edit_source),
            tags=tuple(
                CaptionTagValue(
                    tag_name=tag.tag_name,
                    normalized_name=tag.normalized_name,
                    category=TagCategory(tag.category),
                    source=TagSource(tag.source),
                    position=tag.position,
                    confidence=tag.confidence,
                    manually_added=bool(tag.manually_added),
                    manually_removed=bool(tag.manually_removed),
                )
                for tag in sorted(record.tags, key=lambda item: item.position)
            ),
            updated_at=_utc(record.updated_at),  # type: ignore[arg-type]
        )

    def _caption_tag_values(
        self, record: ImageCaptionRecord
    ) -> tuple[CaptionTagValue, ...]:
        return self._caption_view(record).tags
