from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from runpod_lora_studio.domain.acquisition_models import ImageSourceType


class ImageAcquisitionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    STALE = "stale"


class ImageAcquisitionItemStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VALIDATION_PENDING = "validation_pending"
    VALIDATING = "validating"
    VALIDATED = "validated"
    IMPORTING = "importing"
    IMPORTED = "imported"
    LINKED_EXISTING = "linked_existing"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELED = "canceled"


class DownloadFailureCode(StrEnum):
    PLAN_NOT_CONFIRMED = "PLAN_NOT_CONFIRMED"
    PLAN_METADATA_CHANGED = "PLAN_METADATA_CHANGED"
    SOURCE_POST_NOT_FOUND = "SOURCE_POST_NOT_FOUND"
    SOURCE_POST_UNAVAILABLE = "SOURCE_POST_UNAVAILABLE"
    SOURCE_METADATA_CHANGED = "SOURCE_METADATA_CHANGED"
    FILE_URL_MISSING = "FILE_URL_MISSING"
    FILE_URL_NOT_ALLOWED = "FILE_URL_NOT_ALLOWED"
    REDIRECT_NOT_ALLOWED = "REDIRECT_NOT_ALLOWED"
    HTTP_CLIENT_ERROR = "HTTP_CLIENT_ERROR"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    CONTENT_LENGTH_INVALID = "CONTENT_LENGTH_INVALID"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    RECEIVED_SIZE_MISMATCH = "RECEIVED_SIZE_MISMATCH"
    EMPTY_FILE = "EMPTY_FILE"
    RANGE_NOT_SUPPORTED = "RANGE_NOT_SUPPORTED"
    CONTENT_RANGE_INVALID = "CONTENT_RANGE_INVALID"
    REMOTE_FILE_CHANGED = "REMOTE_FILE_CHANGED"
    SOURCE_MD5_INVALID = "SOURCE_MD5_INVALID"
    SOURCE_MD5_MISMATCH = "SOURCE_MD5_MISMATCH"
    SHA256_FAILED = "SHA256_FAILED"
    UNSUPPORTED_IMAGE_TYPE = "UNSUPPORTED_IMAGE_TYPE"
    IMAGE_FORMAT_MISMATCH = "IMAGE_FORMAT_MISMATCH"
    IMAGE_DIMENSION_MISMATCH = "IMAGE_DIMENSION_MISMATCH"
    IMAGE_PIXEL_LIMIT_EXCEEDED = "IMAGE_PIXEL_LIMIT_EXCEEDED"
    IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
    INSUFFICIENT_STORAGE = "INSUFFICIENT_STORAGE"
    DISK_FULL_DURING_DOWNLOAD = "DISK_FULL_DURING_DOWNLOAD"
    STAGING_PATH_INVALID = "STAGING_PATH_INVALID"
    IMPORT_FAILED = "IMPORT_FAILED"
    THUMBNAIL_FAILED = "THUMBNAIL_FAILED"
    DUPLICATE_SOURCE_CONFLICT = "DUPLICATE_SOURCE_CONFLICT"
    INCOMPLETE_ITEM_STATE = "INCOMPLETE_ITEM_STATE"
    WORKER_CLAIM_LOST = "WORKER_CLAIM_LOST"
    CANCELED = "CANCELED"
    UNKNOWN_DOWNLOAD_ERROR = "UNKNOWN_DOWNLOAD_ERROR"


class PartCleanupWarningCode(StrEnum):
    CLEANUP_FAILED = "PART_CLEANUP_FAILED"
    PATH_INVALID = "PART_PATH_INVALID"
    SYMLINK_REJECTED = "PART_SYMLINK_REJECTED"
    NOT_REGULAR_FILE = "PART_NOT_REGULAR_FILE"


@dataclass(frozen=True, slots=True)
class ImageSourceProvenance:
    source_type: ImageSourceType
    external_post_id: str
    source_post_url: str
    source_md5: str | None
    source_metadata_fingerprint: str
    acquisition_plan_id: UUID
    acquisition_job_id: UUID
    acquisition_job_item_id: UUID


@dataclass(frozen=True, slots=True)
class VerifiedImageFile:
    path: Path
    sha256: str
    md5: str
    file_size: int
    width: int
    height: int
    detected_format: str
    mime_type: str
    extension: str


@dataclass(frozen=True, slots=True)
class AcquisitionJobView:
    id: UUID
    plan_id: UUID
    project_id: UUID
    status: ImageAcquisitionJobStatus
    requested_count: int
    pending_count: int
    downloading_count: int
    downloaded_count: int
    validated_count: int
    imported_count: int
    linked_existing_count: int
    skipped_count: int
    failed_count: int
    received_bytes: int
    expected_bytes: int | None
    current_item_id: UUID | None
    error_code: str | None
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class AcquisitionItemView:
    id: UUID
    external_post_id: str
    status: ImageAcquisitionItemStatus
    attempt_count: int
    received_bytes: int
    expected_file_size: int | None
    detected_format: str | None
    detected_width: int | None
    detected_height: int | None
    sha256_prefix: str | None
    failure_code: str | None
    retryable: bool
    part_cleanup_warning: str | None = None
