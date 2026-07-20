from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class ConceptType(StrEnum):
    CHARACTER = "character"
    STYLE = "style"
    COSTUME = "costume"
    OBJECT = "object"
    OTHER = "other"


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    PREPARING = "preparing"
    READY = "ready"
    ARCHIVED = "archived"


class SelectionState(StrEnum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class Project:
    id: UUID
    name: str
    description: str
    concept_type: ConceptType
    trigger_words: tuple[str, ...]
    status: ProjectStatus
    schema_version: int
    created_at: datetime
    updated_at: datetime
    image_counts: dict[SelectionState, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageAsset:
    id: UUID
    project_id: UUID
    original_filename: str
    stored_filename: str
    original_path: Path
    thumbnail_path: Path
    sha256: str
    width: int
    height: int
    file_size: int
    mime_type: str
    selection_state: SelectionState
    exclusion_reasons: tuple[str, ...]
    source_type: str
    created_at: datetime
    updated_at: datetime
    selection_source: str = "manual"
