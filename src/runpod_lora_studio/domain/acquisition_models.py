from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class ImageSourceType(StrEnum):
    DANBOORU = "danbooru"


class ImageRating(StrEnum):
    GENERAL = "g"
    SENSITIVE = "s"
    QUESTIONABLE = "q"
    EXPLICIT = "e"


class ImageSearchStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELED = "canceled"
    STALE = "stale"


class CandidateStatus(StrEnum):
    ACCEPTED = "accepted"
    EXCLUDED = "excluded"


class AcquisitionPlanStatus(StrEnum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"


class AcquisitionPlanItemStatus(StrEnum):
    PLANNED = "planned"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class AcquisitionErrorCode(StrEnum):
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    UNSUPPORTED_RESPONSE_SCHEMA = "UNSUPPORTED_RESPONSE_SCHEMA"
    QUERY_INVALID = "QUERY_INVALID"
    TOO_MANY_TAGS = "TOO_MANY_TAGS"
    POST_METADATA_INVALID = "POST_METADATA_INVALID"
    CURSOR_LOOP_DETECTED = "CURSOR_LOOP_DETECTED"
    REQUEST_LIMIT_EXCEEDED = "REQUEST_LIMIT_EXCEEDED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    CANCELED = "CANCELED"
    UNKNOWN_SOURCE_ERROR = "UNKNOWN_SOURCE_ERROR"


class CandidateExclusionReason(StrEnum):
    MISSING_FILE_URL = "MISSING_FILE_URL"
    INVALID_FILE_URL = "INVALID_FILE_URL"
    INVALID_FILE_HOST = "INVALID_FILE_HOST"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    RATING_NOT_ALLOWED = "RATING_NOT_ALLOWED"
    SCORE_BELOW_MINIMUM = "SCORE_BELOW_MINIMUM"
    WIDTH_BELOW_MINIMUM = "WIDTH_BELOW_MINIMUM"
    HEIGHT_BELOW_MINIMUM = "HEIGHT_BELOW_MINIMUM"
    PIXEL_COUNT_BELOW_MINIMUM = "PIXEL_COUNT_BELOW_MINIMUM"
    POST_DELETED = "POST_DELETED"
    POST_PENDING = "POST_PENDING"
    POST_FLAGGED = "POST_FLAGGED"
    ALREADY_IMPORTED = "ALREADY_IMPORTED"
    ALREADY_PLANNED = "ALREADY_PLANNED"
    INVALID_METADATA = "INVALID_METADATA"


class ImageSearchSort(StrEnum):
    SCORE = "score"
    ID = "id"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class ImageSearchCursor:
    opaque_value: str


@dataclass(frozen=True, slots=True)
class DanbooruSearchCriteria:
    project_id: UUID
    source_type: ImageSourceType = ImageSourceType.DANBOORU
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    ratings: tuple[ImageRating, ...] = ()
    minimum_score: int | None = None
    minimum_width: int | None = None
    minimum_height: int | None = None
    minimum_pixel_count: int | None = None
    required_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    maximum_candidate_count: int = 100
    page_size: int = 20
    sort_rule: ImageSearchSort = ImageSearchSort.SCORE
    query_version: str = "phase8a-query-v1"


@dataclass(frozen=True, slots=True)
class ValidatedImageSearchQuery:
    criteria: DanbooruSearchCriteria
    normalized_query: str
    query_fingerprint: str


ImageSearchQuery = DanbooruSearchCriteria


@dataclass(frozen=True, slots=True)
class ImageSourcePost:
    source_type: ImageSourceType
    external_post_id: str
    post_url: str
    file_url: str | None
    preview_url: str | None
    sample_url: str | None
    width: int | None
    height: int | None
    file_size: int | None
    file_extension: str | None
    rating: ImageRating | None
    score: int | None
    tag_names: tuple[str, ...]
    source_md5: str | None
    created_at: datetime | None
    is_deleted: bool
    is_pending: bool
    is_flagged: bool
    source_metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ImageSearchPage:
    posts: tuple[ImageSourcePost, ...]
    next_cursor: ImageSearchCursor | None
    request_count: int = 1
    retry_count: int = 0
    rate_limit_count: int = 0


@dataclass(frozen=True, slots=True)
class AcquisitionPlanPreviewItem:
    search_result_id: UUID
    external_post_id: str
    metadata_fingerprint: str
    file_url_fingerprint: str | None
    source_md5: str | None
    extension: str | None
    width: int | None
    height: int | None
    already_imported: bool
    already_planned: bool


@dataclass(frozen=True, slots=True)
class AcquisitionPlanPreview:
    project_id: UUID
    search_id: UUID
    source_type: ImageSourceType
    selected_result_ids: tuple[UUID, ...]
    items: tuple[AcquisitionPlanPreviewItem, ...]
    query_fingerprint: str
    adapter_version: str
    plan_fingerprint: str
    generator_version: str


def fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
