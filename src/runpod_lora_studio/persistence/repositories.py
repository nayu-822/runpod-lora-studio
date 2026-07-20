from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.models import (
    ConceptType,
    ImageAsset,
    Project,
    ProjectStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.models import ImageAssetRecord, ProjectRecord


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def project_from_record(
    record: ProjectRecord, counts: Counter[str] | None = None
) -> Project:
    count_map = Counter(counts or {})
    return Project(
        id=UUID(record.id),
        name=record.name,
        description=record.description,
        concept_type=ConceptType(record.concept_type),
        trigger_words=tuple(json.loads(record.trigger_words)),
        status=ProjectStatus(record.status),
        schema_version=record.schema_version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        image_counts={state: count_map[state.value] for state in SelectionState},
    )


def image_from_record(record: ImageAssetRecord) -> ImageAsset:
    return ImageAsset(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        original_filename=record.original_filename,
        stored_filename=record.stored_filename,
        original_path=Path(record.original_path),
        thumbnail_path=Path(record.thumbnail_path),
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        file_size=record.file_size,
        mime_type=record.mime_type,
        selection_state=SelectionState(record.selection_state),
        exclusion_reasons=tuple(json.loads(record.exclusion_reasons)),
        source_type=record.source_type,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        selection_source=record.selection_source,
    )


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, project: Project) -> ProjectRecord:
        record = ProjectRecord(
            id=str(project.id),
            name=project.name,
            description=project.description,
            concept_type=project.concept_type.value,
            trigger_words=json.dumps(project.trigger_words, ensure_ascii=False),
            status=project.status.value,
            schema_version=project.schema_version,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
        self.session.add(record)
        return record

    def get(self, project_id: UUID) -> ProjectRecord | None:
        return cast(
            ProjectRecord | None,
            self.session.scalar(
                select(ProjectRecord).where(ProjectRecord.id == str(project_id))
            ),
        )

    def list_for_record(self, record: ProjectRecord) -> Project:
        rows = self.session.execute(
            select(ImageAssetRecord.selection_state, func.count())
            .where(ImageAssetRecord.project_id == record.id)
            .group_by(ImageAssetRecord.selection_state)
        ).all()
        counts = Counter({state: count for state, count in rows})
        return project_from_record(record, counts)

    def list(self) -> list[Project]:
        records = self.session.scalars(
            select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())
        ).all()
        if not records:
            return []
        project_ids = [record.id for record in records]
        rows = self.session.execute(
            select(
                ImageAssetRecord.project_id,
                ImageAssetRecord.selection_state,
                func.count(),
            )
            .where(ImageAssetRecord.project_id.in_(project_ids))
            .group_by(ImageAssetRecord.project_id, ImageAssetRecord.selection_state)
        ).all()
        counts: dict[str, Counter[str]] = {}
        for project_id, state, count in rows:
            counts.setdefault(project_id, Counter())[state] = count
        return [
            project_from_record(record, counts.get(record.id)) for record in records
        ]

    def update(
        self,
        record: ProjectRecord,
        *,
        name: str,
        description: str,
        concept_type: ConceptType,
        trigger_words: tuple[str, ...],
    ) -> ProjectRecord:
        record.name = name
        record.description = description
        record.concept_type = concept_type.value
        record.trigger_words = json.dumps(trigger_words, ensure_ascii=False)
        record.updated_at = utc_now()
        return record

    def touch(self, project_id: UUID) -> None:
        record = self.get(project_id)
        if record is not None:
            record.updated_at = utc_now()


class ImageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, image: ImageAsset) -> ImageAssetRecord:
        record = ImageAssetRecord(
            id=str(image.id),
            project_id=str(image.project_id),
            original_filename=image.original_filename,
            stored_filename=image.stored_filename,
            original_path=str(image.original_path),
            thumbnail_path=str(image.thumbnail_path),
            sha256=image.sha256,
            width=image.width,
            height=image.height,
            file_size=image.file_size,
            mime_type=image.mime_type,
            selection_state=image.selection_state.value,
            exclusion_reasons=json.dumps(image.exclusion_reasons, ensure_ascii=False),
            source_type=image.source_type,
            selection_source=image.selection_source,
            created_at=image.created_at,
            updated_at=image.updated_at,
        )
        self.session.add(record)
        return record

    def get(self, image_id: UUID) -> ImageAssetRecord | None:
        return cast(
            ImageAssetRecord | None,
            self.session.scalar(
                select(ImageAssetRecord).where(ImageAssetRecord.id == str(image_id))
            ),
        )

    def sha_exists(self, project_id: UUID, sha256: str) -> bool:
        return (
            self.session.scalar(
                select(ImageAssetRecord.internal_id).where(
                    ImageAssetRecord.project_id == str(project_id),
                    ImageAssetRecord.sha256 == sha256,
                )
            )
            is not None
        )

    def list_for_project(
        self,
        project_id: UUID,
        *,
        state: SelectionState | None = None,
        search: str = "",
        page: int = 1,
        page_size: int = 30,
    ) -> tuple[list[ImageAsset], int]:
        page = max(page, 1)
        page_size = max(page_size, 1)
        query = select(ImageAssetRecord).where(
            ImageAssetRecord.project_id == str(project_id)
        )
        count_query = (
            select(func.count())
            .select_from(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
        )
        if state is not None:
            query = query.where(ImageAssetRecord.selection_state == state.value)
            count_query = count_query.where(
                ImageAssetRecord.selection_state == state.value
            )
        if search.strip():
            pattern = f"%{search.strip()}%"
            query = query.where(ImageAssetRecord.original_filename.ilike(pattern))
            count_query = count_query.where(
                ImageAssetRecord.original_filename.ilike(pattern)
            )
        total = int(self.session.scalar(count_query) or 0)
        records = self.session.scalars(
            query.order_by(ImageAssetRecord.created_at.desc())
            .offset(max(page - 1, 0) * page_size)
            .limit(page_size)
        ).all()
        return [image_from_record(record) for record in records], total

    def update_state(
        self,
        project_id: UUID,
        image_ids: Iterable[UUID],
        state: SelectionState,
    ) -> int:
        ids = [str(image_id) for image_id in image_ids]
        if not ids:
            return 0
        records = self.session.scalars(
            select(ImageAssetRecord).where(
                ImageAssetRecord.project_id == str(project_id),
                ImageAssetRecord.id.in_(ids),
            )
        ).all()
        now = utc_now()
        for record in records:
            record.selection_state = state.value
            record.selection_source = "manual"
            record.updated_at = now
        return len(records)
