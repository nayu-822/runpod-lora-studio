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


class InspectionRule(StrEnum):
    EXACT_DUPLICATE = "exact_duplicate"
    RESOLUTION_TOO_SMALL = "resolution_too_small"
    ASPECT_RATIO_EXTREME = "aspect_ratio_extreme"
    LOW_INFORMATION = "low_information"
    BLUR_SCORE = "blur_score"


class InspectionStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAILED = "failed"


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


@dataclass(frozen=True, slots=True)
class ImageInspectionResult:
    image_id: UUID
    rule: InspectionRule
    status: InspectionStatus
    score: float | None
    threshold: float | None
    reason: str
    detector_version: str
    inspected_at: datetime


@dataclass(frozen=True, slots=True)
class InspectionSummary:
    project_id: UUID
    total_images: int
    inspected_images: int
    pass_count: int
    warning_count: int
    failed_count: int
    exact_duplicate_count: int
    resolution_too_small_count: int
    aspect_ratio_extreme_count: int
    low_information_count: int
    blur_score_count: int


@dataclass(frozen=True, slots=True)
class InspectionRunResult:
    summary: InspectionSummary
    inspected_image_count: int
    failed_image_count: int
