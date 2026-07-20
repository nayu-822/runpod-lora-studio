from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    ConceptType,
    Project,
    ProjectStatus,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import ProjectRepository

logger = logging.getLogger("runpod_lora_studio.projects")


class UserFacingError(ValueError):
    """An expected validation or selection error safe to show in the UI."""


def normalize_trigger_words(words: Iterable[str] | str | None) -> tuple[str, ...]:
    values = [words] if isinstance(words, str) else list(words or [])
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProjectInput:
    name: str
    description: str = ""
    concept_type: ConceptType = ConceptType.OTHER
    trigger_words: tuple[str, ...] = ()


class ProjectService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)

    def create(self, data: ProjectInput) -> Project:
        name = data.name.strip()
        if not name:
            raise UserFacingError("プロジェクト名を入力してください。")
        if len(name) > 200:
            raise UserFacingError("プロジェクト名は200文字以内で入力してください。")
        description = data.description.strip()
        project_id = uuid4()
        project_root = self.project_root(project_id)
        self.settings.projects_dir.mkdir(parents=True, exist_ok=True)
        project_root.mkdir(parents=True, exist_ok=False)
        try:
            for directory in ("originals", "thumbnails", "metadata", "logs"):
                (project_root / directory).mkdir()
            now = datetime.now(UTC)
            project = Project(
                id=project_id,
                name=name,
                description=description,
                concept_type=data.concept_type,
                trigger_words=normalize_trigger_words(data.trigger_words),
                status=ProjectStatus.DRAFT,
                schema_version=1,
                created_at=now,
                updated_at=now,
            )
            with self.session_factory() as session:
                ProjectRepository(session).add(project)
                session.commit()
            logger.info("project_created project_id=%s", project_id)
            return project
        except Exception:
            shutil.rmtree(project_root, ignore_errors=True)
            raise

    def project_root(self, project_id: UUID) -> Path:
        return self.settings.projects_dir / str(project_id)

    def list_projects(self) -> list[Project]:
        with self.session_factory() as session:
            return ProjectRepository(session).list()

    def get(self, project_id: UUID) -> Project:
        with self.session_factory() as session:
            record = ProjectRepository(session).get(project_id)
            if record is None:
                raise UserFacingError("指定されたプロジェクトが見つかりません。")
            return ProjectRepository(session).list_for_record(record)

    def update(self, project_id: UUID, data: ProjectInput) -> Project:
        name = data.name.strip()
        if not name:
            raise UserFacingError("プロジェクト名を入力してください。")
        if len(name) > 200:
            raise UserFacingError("プロジェクト名は200文字以内で入力してください。")
        with self.session_factory() as session:
            repository = ProjectRepository(session)
            record = repository.get(project_id)
            if record is None:
                raise UserFacingError("指定されたプロジェクトが見つかりません。")
            repository.update(
                record,
                name=name,
                description=data.description.strip(),
                concept_type=data.concept_type,
                trigger_words=normalize_trigger_words(data.trigger_words),
            )
            session.commit()
            return repository.list_for_record(record)
