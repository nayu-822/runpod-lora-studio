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


class PerceptualHashStatus(StrEnum):
    CALCULATED = "calculated"
    FAILED = "failed"


class SimilarityReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CONFIRMED_SIMILAR = "confirmed_similar"
    REJECTED_SIMILARITY = "rejected_similarity"


class RepresentativeSource(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class TagCategory(StrEnum):
    GENERAL = "general"
    CHARACTER = "character"
    RATING = "rating"
    META = "meta"
    UNKNOWN = "unknown"
    MANUAL = "manual"
    TRIGGER = "trigger"


class TagSource(StrEnum):
    WD_TAGGER = "wd_tagger"
    SOURCE_METADATA = "source_metadata"
    MANUAL = "manual"
    TRIGGER_WORD = "trigger_word"


class TaggerRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELED = "canceled"
    STALE = "stale"


class TaggingResultStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CaptionEditSource(StrEnum):
    GENERATED = "generated"
    BULK_FILTER = "bulk_filter"
    MANUAL = "manual"
    RESTORED = "restored"


class TaggerRunMode(StrEnum):
    UNTAGGED_ONLY = "untagged_only"
    FAILED_ONLY = "failed_only"
    ALL_ACCEPTED = "all_accepted"


class ManualCaptionPolicy(StrEnum):
    KEEP_MANUAL = "keep_manual"
    REBUILD_FROM_SOURCE = "rebuild_from_source"
    EXCLUDE_MANUAL = "exclude_manual"


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


@dataclass(frozen=True, slots=True)
class PerceptualHash:
    image_id: UUID
    algorithm: str
    hash_value: str
    hash_size: int
    detector_version: str
    status: PerceptualHashStatus
    calculated_at: datetime
    error_summary: str | None = None


@dataclass(frozen=True, slots=True)
class RepresentativeCandidate:
    image_id: UUID
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class SimilarityGroupMember:
    group_id: UUID
    image_id: UUID
    representative_candidate_score: float
    is_representative: bool
    representative_distance: int | None
    minimum_distance: int | None
    review_status: SimilarityReviewStatus
    image: ImageAsset | None = None


@dataclass(frozen=True, slots=True)
class SimilarityGroup:
    id: UUID
    project_id: UUID
    group_type: str
    detector_version: str
    distance_threshold: int
    representative_image_id: UUID | None
    representative_source: RepresentativeSource
    created_at: datetime
    updated_at: datetime
    members: tuple[SimilarityGroupMember, ...] = ()
    rejected_pairs: tuple[tuple[UUID, UUID], ...] = ()


@dataclass(frozen=True, slots=True)
class SimilaritySummary:
    project_id: UUID
    calculated_count: int
    uncalculated_count: int
    failed_count: int
    group_count: int
    candidate_image_count: int
    exact_only_group_count: int
    unreviewed_group_count: int


@dataclass(frozen=True, slots=True)
class SimilarityRunResult:
    summary: SimilaritySummary
    calculated_image_count: int
    failed_image_count: int
    skipped_image_count: int
    group_count: int


@dataclass(frozen=True, slots=True)
class TaggerModelIdentity:
    adapter_name: str
    model_identifier: str
    model_revision: str
    model_path: str
    implementation_version: str


@dataclass(frozen=True, slots=True)
class TaggerInferenceSettings:
    device: str
    batch_size: int
    general_threshold: float
    character_threshold: float
    save_rating: bool
    save_character: bool
    save_general: bool
    underscore_to_space: bool
    escape_mode: str
    max_workers: int
    allow_model_download: bool


@dataclass(frozen=True, slots=True)
class TagPrediction:
    tag_name_raw: str
    tag_name_normalized: str
    category: TagCategory
    confidence: float | None
    original_order: int
    source: TagSource = TagSource.WD_TAGGER


@dataclass(frozen=True, slots=True)
class TaggingResult:
    tags: tuple[TagPrediction, ...]
    raw_output: str | None = None


@dataclass(frozen=True, slots=True)
class TagFrequency:
    tag_name_normalized: str
    display_name: str
    category: TagCategory
    image_count: int
    target_image_count: int
    occurrence_rate: float
    average_confidence: float | None
    minimum_confidence: float | None
    maximum_confidence: float | None
    keep: bool
    rule_origin: str


@dataclass(frozen=True, slots=True)
class CaptionTagValue:
    tag_name: str
    normalized_name: str
    category: TagCategory
    source: TagSource
    position: int
    confidence: float | None = None
    manually_added: bool = False
    manually_removed: bool = False


@dataclass(frozen=True, slots=True)
class CaptionChange:
    image_id: UUID
    filename: str
    before: str
    after: str
    added_tags: tuple[str, ...]
    removed_tags: tuple[str, ...]
    trigger_words: tuple[str, ...]
    manual_policy: ManualCaptionPolicy
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionPreview:
    token: str
    project_id: UUID
    tagger_run_id: UUID
    changes: tuple[CaptionChange, ...]
    target_image_count: int
    keep_tag_count: int
    remove_tag_count: int
    changed_image_count: int
    empty_caption_count: int
    trigger_image_count: int
    manual_image_count: int
    rules_snapshot: tuple[tuple[str, str, str], ...]
    trigger_words: tuple[str, ...]
    policy: ManualCaptionPolicy
    run_target_image_count: int
    run_succeeded_image_count: int
    run_failed_image_count: int
    run_skipped_image_count: int
    used_image_count: int


@dataclass(frozen=True, slots=True)
class TaggerRunSummary:
    id: UUID
    project_id: UUID
    adapter_name: str
    model_identifier: str
    model_revision: str
    model_path: str
    device: str
    status: TaggerRunStatus
    target_image_count: int
    processed_image_count: int
    succeeded_image_count: int
    failed_image_count: int
    skipped_image_count: int
    current_image_id: UUID | None
    cancel_requested: bool
    started_at: datetime | None
    completed_at: datetime | None
    error_summary: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredTaggingResult:
    image_id: UUID
    tagger_run_id: UUID
    status: TaggingResultStatus
    error_summary: str | None
    tagged_at: datetime | None
    tags: tuple[TagPrediction, ...]


@dataclass(frozen=True, slots=True)
class ProjectTagRule:
    normalized_tag_name: str
    action: str
    category: TagCategory
    updated_by: str


@dataclass(frozen=True, slots=True)
class StoredCaption:
    image_id: UUID
    id: UUID
    revision: int
    caption_text: str
    source_tagger_run_id: UUID | None
    edit_source: CaptionEditSource
    tags: tuple[CaptionTagValue, ...]
    updated_at: datetime


class DatasetSnapshotStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    CORRUPTED = "corrupted"


class DatasetIssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DatasetIssueCategory(StrEnum):
    FILE = "file"
    CAPTION = "caption"
    DUPLICATE = "duplicate"
    QUALITY = "quality"
    DISTRIBUTION = "distribution"
    TRIGGER = "trigger"
    CONFIGURATION = "configuration"
    STORAGE = "storage"
    INTEGRITY = "integrity"


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    resolution: int = 1024
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = True
    caption_extension: str = ".txt"
    shuffle_caption: bool = True
    keep_tokens: int = 0
    caption_separator: str = ", "
    flip_aug: bool = False
    color_aug: bool = False
    random_crop: bool = False
    face_crop_aug_range: tuple[float, float] | None = None
    debug_dataset: bool = False
    class_tokens: str = ""
    is_reg: bool = False
    num_repeats: int = 1
    allow_empty_caption: bool = False
    warning_confirmation_required: bool = True
    schema_version: str = "phase4-dataset-v1"


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    issue_code: str
    severity: DatasetIssueSeverity
    category: DatasetIssueCategory
    message: str
    image_id: UUID | None = None
    measured_value: str | None = None
    threshold_value: str | None = None
    details: str = ""


@dataclass(frozen=True, slots=True)
class DatasetPreviewImage:
    image_id: UUID
    original_filename: str
    source_image_path: Path
    width: int
    height: int
    aspect_ratio: float
    file_size: int
    source_sha256: str
    mime_type: str
    selection_state: SelectionState
    caption_id: UUID | None
    caption_revision: int | None
    caption_text: str
    caption_sha256: str
    tag_count: int
    trigger_word_count: int
    quality_status: str
    exact_duplicate_status: str
    similarity_group_id: UUID | None
    is_similarity_representative: bool | None
    warnings: tuple[DatasetValidationIssue, ...]
    errors: tuple[DatasetValidationIssue, ...]

    @property
    def can_include(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class DatasetPreviewSummary:
    target_image_count: int
    caption_present_count: int
    caption_missing_count: int
    missing_file_count: int
    corrupt_file_count: int
    quality_warning_image_count: int
    quality_failed_image_count: int
    exact_duplicate_count: int
    exact_duplicate_nonrepresentative_count: int
    approximate_duplicate_count: int
    approximate_duplicate_nonrepresentative_count: int
    unreviewed_group_count: int
    empty_caption_count: int
    trigger_missing_count: int
    warning_count: int
    error_count: int
    estimated_size_bytes: int
    available_disk_bytes: int
    estimated_free_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetPreview:
    token: str
    project_id: UUID
    project_updated_at: datetime
    project_name: str
    trigger_words: tuple[str, ...]
    settings: DatasetSettings
    images: tuple[DatasetPreviewImage, ...]
    issues: tuple[DatasetValidationIssue, ...]
    summary: DatasetPreviewSummary
    source_tagger_run_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class DatasetSnapshotSummary:
    id: UUID
    project_id: UUID
    name: str
    description: str
    status: DatasetSnapshotStatus
    target_image_count: int
    copied_image_count: int
    failed_image_count: int
    warning_count: int
    total_size_bytes: int
    snapshot_root: Path
    manifest_sha256: str | None
    content_sha256: str | None
    source_tagger_run_id: UUID | None
    created_at: datetime
    completed_at: datetime | None
    error_summary: str | None


@dataclass(frozen=True, slots=True)
class DatasetSnapshotItem:
    snapshot_id: UUID
    image_id: UUID
    source_image_path: Path
    snapshot_image_relative_path: str
    caption_relative_path: str
    sequence_number: int
    source_image_sha256: str
    snapshot_image_sha256: str
    source_file_size: int
    snapshot_file_size: int
    width: int
    height: int
    aspect_ratio: float
    mime_type: str
    caption_id: UUID
    caption_revision: int
    caption_sha256: str
    caption_text: str
    tag_count: int
    trigger_word_count: int
    quality_status: str
    exact_duplicate_status: str
    similarity_group_id: UUID | None
    is_similarity_representative: bool | None
    warnings: tuple[DatasetValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class DatasetReport:
    report_json: dict[str, object]
    report_markdown: str
    tag_frequency_csv: str
    resolution_csv: str
    aspect_ratio_csv: str
    warnings_json: str
