from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import case, select, update
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.acquisition_download_models import (
    AcquisitionItemView,
    AcquisitionJobView,
    DownloadFailureCode,
    ImageAcquisitionItemStatus,
    ImageAcquisitionJobStatus,
    ImageSourceProvenance,
    PartCleanupWarningCode,
    VerifiedImageFile,
)
from runpod_lora_studio.domain.acquisition_models import (
    ImageSourcePost,
    ImageSourceType,
    fingerprint,
)
from runpod_lora_studio.external.image_download import (
    DOWNLOAD_TRANSPORT_VERSION,
    DownloadRequest,
    DownloadResponse,
    DownloadTransport,
    DownloadTransportError,
    HttpImageDownloadTransport,
    validate_download_url,
)
from runpod_lora_studio.external.image_sources import (
    DanbooruApiClient,
    DanbooruHttpTransport,
    DanbooruImageSourceAdapter,
    DanbooruSourceError,
    ImageSourceAdapter,
    ImageSourceRequestContext,
    SourceRateLimiter,
    interruptible_sleep,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    ExternalImageAssetLinkRecord,
    ImageAcquisitionAttemptRecord,
    ImageAcquisitionJobItemRecord,
    ImageAcquisitionJobRecord,
    ImageAcquisitionPlanItemRecord,
    ImageAcquisitionPlanRecord,
    ImageAcquisitionReservationRecord,
    ImageAssetRecord,
    ImageSourceSearchRecord,
    ImageSourceSearchResultRecord,
)
from runpod_lora_studio.services.acquisition_service import PLAN_VERSION
from runpod_lora_studio.services.image_ingestion_service import (
    IMPORTER_VERSION,
    VALIDATOR_VERSION,
    ImageVerificationError,
    VerifiedImageIngestionService,
)
from runpod_lora_studio.services.project_service import ProjectService

logger = logging.getLogger("runpod_lora_studio.acquisition_download")
CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)$")
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")
RETRYABLE_CODES = frozenset(
    {
        DownloadFailureCode.RATE_LIMITED,
        DownloadFailureCode.REQUEST_TIMEOUT,
        DownloadFailureCode.CONNECTION_FAILED,
        DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
    }
)
PERMANENT_CODES = frozenset(
    {
        DownloadFailureCode.PLAN_NOT_CONFIRMED,
        DownloadFailureCode.PLAN_METADATA_CHANGED,
        DownloadFailureCode.SOURCE_POST_NOT_FOUND,
        DownloadFailureCode.SOURCE_METADATA_CHANGED,
        DownloadFailureCode.FILE_URL_MISSING,
        DownloadFailureCode.FILE_URL_NOT_ALLOWED,
        DownloadFailureCode.REDIRECT_NOT_ALLOWED,
        DownloadFailureCode.HTTP_CLIENT_ERROR,
        DownloadFailureCode.PERMISSION_DENIED,
        DownloadFailureCode.RETRY_EXHAUSTED,
        DownloadFailureCode.CONTENT_LENGTH_INVALID,
        DownloadFailureCode.FILE_TOO_LARGE,
        DownloadFailureCode.RECEIVED_SIZE_MISMATCH,
        DownloadFailureCode.EMPTY_FILE,
        DownloadFailureCode.RANGE_NOT_SUPPORTED,
        DownloadFailureCode.CONTENT_RANGE_INVALID,
        DownloadFailureCode.REMOTE_FILE_CHANGED,
        DownloadFailureCode.SOURCE_MD5_INVALID,
        DownloadFailureCode.SOURCE_MD5_MISMATCH,
        DownloadFailureCode.UNSUPPORTED_IMAGE_TYPE,
        DownloadFailureCode.IMAGE_FORMAT_MISMATCH,
        DownloadFailureCode.IMAGE_DIMENSION_MISMATCH,
        DownloadFailureCode.IMAGE_PIXEL_LIMIT_EXCEEDED,
        DownloadFailureCode.IMAGE_CORRUPTED,
        DownloadFailureCode.INSUFFICIENT_STORAGE,
        DownloadFailureCode.STAGING_PATH_INVALID,
        DownloadFailureCode.DUPLICATE_SOURCE_CONFLICT,
    }
)
PART_CLEANUP_WARNING_CODES = frozenset(
    code.value
    for code in PartCleanupWarningCode
    if code is not PartCleanupWarningCode.PENDING
)
PART_CLEANUP_ITEM_STATUSES = frozenset(
    {
        ImageAcquisitionItemStatus.IMPORTED.value,
        ImageAcquisitionItemStatus.LINKED_EXISTING.value,
        ImageAcquisitionItemStatus.SKIPPED.value,
        ImageAcquisitionItemStatus.FAILED.value,
    }
)
PART_CLEANUP_BATCH_SIZE = 32
STALE_JOB_BATCH_SIZE = 32


class AcquisitionDownloadError(ValueError):
    def __init__(self, code: DownloadFailureCode, message: str | None = None) -> None:
        super().__init__(message or code.value)
        self.code = code


class _Canceled(Exception):
    pass


class _ClaimLost(Exception):
    pass


class _RetryableDownload(Exception):
    def __init__(
        self, code: DownloadFailureCode, retry_after: float | None = None
    ) -> None:
        self.code = code
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class _PartPathInspection:
    path: Path | None
    warning: PartCleanupWarningCode | None
    parent_components: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _ManifestDirectoryIdentity:
    device: int
    inode: int


@dataclass(slots=True)
class _ManifestDirectoryHandle:
    project_root: Path
    manifest_dir: Path
    fd: int
    projects_root: Path
    component_names: tuple[str, ...]
    identities: tuple[_ManifestDirectoryIdentity, ...]

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> _ManifestDirectoryHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _AcquisitionCountsSnapshot:
    item_count: int
    pending_count: int
    downloading_count: int
    downloaded_count: int
    validated_count: int
    imported_count: int
    linked_existing_count: int
    skipped_count: int
    failed_count: int
    received_bytes: int

    def job_values(self) -> dict[str, int]:
        return {
            "pending_count": self.pending_count,
            "downloading_count": self.downloading_count,
            "downloaded_count": self.downloaded_count,
            "validated_count": self.validated_count,
            "imported_count": self.imported_count,
            "linked_existing_count": self.linked_existing_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "received_bytes": self.received_bytes,
        }


class ImageAcquisitionDownloadService:
    """Run confirmed acquisition plans as restartable, claimed workers."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        projects: ProjectService | None = None,
        adapter: ImageSourceAdapter | None = None,
        adapters: Mapping[ImageSourceType, ImageSourceAdapter] | None = None,
        transport: DownloadTransport | None = None,
        ingestion: VerifiedImageIngestionService | None = None,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        auto_start: bool = True,
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)
        self.ingestion = ingestion or VerifiedImageIngestionService(
            settings, self.projects
        )
        self.disk_usage = disk_usage
        self.clock = clock
        self.sleeper = sleeper
        self.auto_start = auto_start
        self._adapters = dict(adapters or {})
        if adapter is not None:
            self._adapters[ImageSourceType.DANBOORU] = adapter
        if ImageSourceType.DANBOORU not in self._adapters:
            metadata_transport = DanbooruHttpTransport(
                connect_timeout_seconds=settings.image_search_connect_timeout_seconds,
                read_timeout_seconds=settings.image_search_read_timeout_seconds,
                max_response_bytes=settings.image_search_max_response_bytes,
            )
            self._adapters[ImageSourceType.DANBOORU] = DanbooruImageSourceAdapter(
                client=DanbooruApiClient(
                    metadata_transport,
                    limiter=SourceRateLimiter(
                        minimum_interval_seconds=settings.image_search_min_interval_seconds
                    ),
                )
            )
        self.transport = transport or HttpImageDownloadTransport(
            connect_timeout_seconds=settings.image_download_connect_timeout_seconds,
            read_timeout_seconds=settings.image_download_read_timeout_seconds,
            max_header_bytes=settings.image_download_max_header_bytes,
            max_redirects=settings.image_download_max_redirects,
        )

    def start_job(self, plan_id: UUID, *, auto_start: bool | None = None) -> UUID:
        existing = self._active_or_finished_job(plan_id)
        if existing is not None:
            if auto_start is not False and existing.status in {
                ImageAcquisitionJobStatus.QUEUED.value,
                ImageAcquisitionJobStatus.STALE.value,
            }:
                self._start_thread(UUID(existing.id))
            return UUID(existing.id)
        plan, items = self._load_confirmed_plan(plan_id)
        items = self._validate_plan_structure(plan, items)
        job_id = uuid4()
        now = datetime.now(UTC)
        expected_bytes = sum(item.expected_file_size or 0 for item in items) or None
        self._check_job_storage(plan.project_id, items)
        job = ImageAcquisitionJobRecord(
            id=str(job_id),
            project_id=plan.project_id,
            plan_id=plan.id,
            plan_fingerprint=plan.plan_fingerprint,
            source_type=plan.source_type,
            status=ImageAcquisitionJobStatus.QUEUED.value,
            active_key=plan.id,
            requested_count=len(items),
            pending_count=len(items),
            expected_bytes=expected_bytes,
            downloader_version=DOWNLOAD_TRANSPORT_VERSION,
            validator_version=VALIDATOR_VERSION,
            importer_version=IMPORTER_VERSION,
            job_fingerprint=fingerprint(
                {
                    "project_id": plan.project_id,
                    "plan_id": plan.id,
                    "plan_fingerprint": plan.plan_fingerprint,
                    "source_type": plan.source_type,
                    "plan_version": plan.plan_version,
                    "downloader_version": DOWNLOAD_TRANSPORT_VERSION,
                    "validator_version": VALIDATOR_VERSION,
                    "importer_version": IMPORTER_VERSION,
                    "storage_policy_version": "phase8b-storage-v1",
                }
            ),
            created_at=now,
            updated_at=now,
        )
        with self.session_factory() as session:
            session.add(job)
            session.flush()
            for order, item in enumerate(items):
                item_id = str(uuid4())
                session.add(
                    ImageAcquisitionJobItemRecord(
                        id=item_id,
                        job_id=str(job_id),
                        plan_item_id=item.id,
                        source_type=plan.source_type,
                        external_post_id=item.external_post_id,
                        display_order=order,
                        status=ImageAcquisitionItemStatus.PENDING.value,
                        expected_metadata_fingerprint=item.expected_metadata_fingerprint,
                        expected_file_url_fingerprint=item.expected_file_url_fingerprint,
                        expected_md5=item.expected_md5,
                        expected_width=item.expected_width,
                        expected_height=item.expected_height,
                        expected_extension=item.expected_extension,
                        expected_file_size=item.expected_file_size,
                        expected_file_url=None,
                        part_relative_path=f"acquisition/jobs/{job_id}/parts/{item_id}.part",
                        created_at=now,
                        updated_at=now,
                    )
                )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                concurrent = self._active_or_finished_job(plan_id)
                if concurrent is not None:
                    return UUID(concurrent.id)
                raise AcquisitionDownloadError(
                    DownloadFailureCode.DUPLICATE_SOURCE_CONFLICT
                ) from exc
        if auto_start is not False and self.auto_start:
            self._start_thread(job_id)
        return job_id

    def resume_job(self, job_id: UUID, *, auto_start: bool | None = None) -> UUID:
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            if job is None:
                raise AcquisitionDownloadError(
                    DownloadFailureCode.SOURCE_POST_NOT_FOUND
                )
            if job.status not in {
                ImageAcquisitionJobStatus.STALE.value,
                ImageAcquisitionJobStatus.CANCELED.value,
                ImageAcquisitionJobStatus.PARTIALLY_COMPLETED.value,
                ImageAcquisitionJobStatus.FAILED.value,
            }:
                return job_id
            now = datetime.now(UTC)
            existing_items = session.scalars(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id)
                )
            ).all()
            has_work = any(
                item.status
                not in {
                    ImageAcquisitionItemStatus.IMPORTED.value,
                    ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                    ImageAcquisitionItemStatus.SKIPPED.value,
                }
                and (
                    item.status != ImageAcquisitionItemStatus.FAILED.value
                    or item.retryable
                )
                for item in existing_items
            )
            if not has_work:
                if job.status != ImageAcquisitionJobStatus.STALE.value:
                    return job_id
                job.status = ImageAcquisitionJobStatus.QUEUED.value
                job.cancellation_requested = False
                job.worker_id = None
                job.claim_token = None
                job.current_item_id = None
                job.error_code = None
                job.error_summary = None
                job.completed_at = None
                job.heartbeat_at = None
                job.active_key = job.plan_id
                job.updated_at = now
            else:
                job.status = ImageAcquisitionJobStatus.QUEUED.value
                job.cancellation_requested = False
                job.worker_id = None
                job.claim_token = None
                job.current_item_id = None
                job.error_code = None
                job.error_summary = None
                job.completed_at = None
                job.updated_at = now
                for item in session.scalars(
                    select(ImageAcquisitionJobItemRecord).where(
                        ImageAcquisitionJobItemRecord.job_id == str(job_id)
                    )
                ).all():
                    if item.status in {
                        ImageAcquisitionItemStatus.IMPORTED.value,
                        ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                        ImageAcquisitionItemStatus.SKIPPED.value,
                        ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                    }:
                        continue
                    if (
                        item.status == ImageAcquisitionItemStatus.FAILED.value
                        and not item.retryable
                    ):
                        continue
                    item.status = ImageAcquisitionItemStatus.PENDING.value
                    item.failure_code = None
                    item.failure_message = None
                    item.part_cleanup_warning = None
                    item.part_cleanup_claim_token = None
                    item.part_cleanup_claimed_at = None
                    item.updated_at = now
                job.active_key = job.plan_id
            session.commit()
        if auto_start is not False and self.auto_start:
            self._start_thread(job_id)
        return job_id

    retry_job = resume_job

    def cancel_job(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            if job is None:
                return
            job.cancellation_requested = True
            if job.status in {
                ImageAcquisitionJobStatus.QUEUED.value,
                ImageAcquisitionJobStatus.STALE.value,
            }:
                job.status = ImageAcquisitionJobStatus.CANCELED.value
                job.completed_at = datetime.now(UTC)
                job.active_key = None
                job.worker_id = None
                job.claim_token = None
            job.updated_at = datetime.now(UTC)
            session.commit()

    def recover_stale_jobs(self, *, limit: int = STALE_JOB_BATCH_SIZE) -> int:
        if limit <= 0:
            return 0
        threshold = datetime.now(UTC) - timedelta(
            seconds=self.settings.image_download_stale_after_seconds
        )
        queued_job_ids: list[UUID] = []
        cleanup_item_ids: set[str] = set()
        try:
            with self.session_factory() as session:
                candidates = session.execute(
                    select(
                        ImageAcquisitionJobRecord.id,
                        ImageAcquisitionJobRecord.worker_id,
                        ImageAcquisitionJobRecord.claim_token,
                        ImageAcquisitionJobRecord.worker_generation,
                    )
                    .where(
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.heartbeat_at < threshold,
                    )
                    .order_by(ImageAcquisitionJobRecord.heartbeat_at)
                    .limit(limit)
                ).all()
                now = datetime.now(UTC)
                recovered_count = 0
                for job_id, worker_id, claim_token, generation in candidates:
                    if not self._claim_stale_job(
                        session,
                        str(job_id),
                        worker_id,
                        claim_token,
                        generation,
                        threshold,
                        now,
                    ):
                        continue
                    recovered_count += 1
                    job = session.scalar(
                        select(ImageAcquisitionJobRecord).where(
                            ImageAcquisitionJobRecord.id == job_id
                        )
                    )
                    if job is None:
                        continue
                    items = session.scalars(
                        select(ImageAcquisitionJobItemRecord).where(
                            ImageAcquisitionJobItemRecord.job_id == job.id
                        )
                    ).all()
                    for item in items:
                        part_inspection = self._stale_part_path(job.project_id, item)
                        part = part_inspection.path
                        part_matches = self._part_matches_item(part, item)
                        if item.status == ImageAcquisitionItemStatus.IMPORTING.value:
                            if self._recover_importing_item(
                                session, job, item, now, part_inspection
                            ):
                                cleanup_item_ids.add(item.id)
                                continue
                            self._finalize_stale_attempts(
                                session,
                                item,
                                now,
                                DownloadFailureCode.WORKER_CLAIM_LOST,
                                retryable=True,
                            )
                        elif item.status not in {
                            ImageAcquisitionItemStatus.IMPORTED.value,
                            ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                            ImageAcquisitionItemStatus.SKIPPED.value,
                            ImageAcquisitionItemStatus.FAILED.value,
                            ImageAcquisitionItemStatus.CANCELED.value,
                        }:
                            self._finalize_stale_attempts(
                                session,
                                item,
                                now,
                                DownloadFailureCode.WORKER_CLAIM_LOST,
                                retryable=True,
                            )
                        if item.status == ImageAcquisitionItemStatus.DOWNLOADING.value:
                            target = ImageAcquisitionItemStatus.PENDING
                        elif item.status in {
                            ImageAcquisitionItemStatus.DOWNLOADED.value,
                            ImageAcquisitionItemStatus.VALIDATING.value,
                            ImageAcquisitionItemStatus.VALIDATED.value,
                        }:
                            target = (
                                ImageAcquisitionItemStatus.VALIDATION_PENDING
                                if part_matches
                                else ImageAcquisitionItemStatus.PENDING
                            )
                        elif (
                            item.status == ImageAcquisitionItemStatus.FAILED.value
                            and item.retryable
                        ):
                            target = ImageAcquisitionItemStatus.PENDING
                        elif item.status not in {
                            ImageAcquisitionItemStatus.IMPORTED.value,
                            ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                            ImageAcquisitionItemStatus.SKIPPED.value,
                            ImageAcquisitionItemStatus.FAILED.value,
                            ImageAcquisitionItemStatus.CANCELED.value,
                        }:
                            target = ImageAcquisitionItemStatus.PENDING
                        else:
                            continue
                        preserve_part_for_retry = (
                            item.status == ImageAcquisitionItemStatus.FAILED.value
                            and item.retryable
                        )
                        if not part_matches and not preserve_part_for_retry:
                            item.part_cleanup_warning = (
                                PartCleanupWarningCode.PENDING.value
                            )
                            item.part_cleanup_claim_token = None
                            item.part_cleanup_claimed_at = None
                            cleanup_warning = self._cleanup_part_artifact(
                                part_inspection
                            )
                            item.part_cleanup_warning = (
                                cleanup_warning.value
                                if cleanup_warning is not None
                                else None
                            )
                            item.received_bytes = 0
                            item.etag = None
                            item.last_modified = None
                            item.accept_ranges = False
                            item.range_start = None
                        elif preserve_part_for_retry:
                            item.part_cleanup_warning = None
                            item.part_cleanup_claim_token = None
                            item.part_cleanup_claimed_at = None
                        item.status = target.value
                        item.image_asset_id = None
                        item.failure_code = None
                        item.failure_message = None
                        item.retryable = False
                        item.completed_at = None
                        item.updated_at = now
                    job.status = ImageAcquisitionJobStatus.QUEUED.value
                    job.worker_id = None
                    job.claim_token = None
                    job.current_item_id = None
                    job.active_key = job.plan_id
                    job.heartbeat_at = None
                    job.completed_at = None
                    job.error_code = None
                    job.error_summary = None
                    job.manifest_warning = None
                    job.updated_at = now
                    queued_job_ids.append(UUID(job.id))
                session.commit()
            if cleanup_item_ids:
                self.recover_part_cleanup_jobs(item_ids=cleanup_item_ids)
            if self.auto_start:
                for job_id in queued_job_ids:
                    self._start_thread(job_id)
            return recovered_count
        except OperationalError as exc:
            if "no such table: image_acquisition_jobs" not in str(exc):
                raise
            logger.warning("acquisition_jobs_table_not_migrated")
            return 0

    def recover_part_cleanup_jobs(
        self,
        *,
        item_ids: Iterable[str] | None = None,
        limit: int = PART_CLEANUP_BATCH_SIZE,
    ) -> int:
        if limit <= 0:
            return 0
        requested_ids = tuple(item_ids or ())
        if item_ids is not None and not requested_ids:
            return 0
        now = datetime.now(UTC)
        claim_cutoff = now - timedelta(
            seconds=max(self.settings.image_download_stale_after_seconds, 60)
        )
        try:
            with self.session_factory() as session:
                conditions = [
                    ImageAcquisitionJobItemRecord.status.in_(
                        PART_CLEANUP_ITEM_STATUSES
                    ),
                    ImageAcquisitionJobItemRecord.retryable == False,  # noqa: E712
                    ImageAcquisitionJobItemRecord.part_cleanup_warning.in_(
                        [
                            PartCleanupWarningCode.PENDING.value,
                            *PART_CLEANUP_WARNING_CODES,
                        ]
                    ),
                    (
                        ImageAcquisitionJobItemRecord.part_cleanup_claim_token.is_(None)
                        | (
                            ImageAcquisitionJobItemRecord.part_cleanup_claimed_at
                            < claim_cutoff
                        )
                        | ImageAcquisitionJobItemRecord.part_cleanup_claimed_at.is_(
                            None
                        )
                    ),
                ]
                if item_ids is not None:
                    conditions.append(
                        ImageAcquisitionJobItemRecord.id.in_(requested_ids)
                    )
                candidate_ids = session.scalars(
                    select(
                        ImageAcquisitionJobItemRecord.id,
                    )
                    .join(
                        ImageAcquisitionJobRecord,
                        ImageAcquisitionJobRecord.id
                        == ImageAcquisitionJobItemRecord.job_id,
                    )
                    .where(*conditions)
                    .order_by(ImageAcquisitionJobItemRecord.updated_at)
                    .limit(limit)
                ).all()
        except OperationalError as exc:
            if "no such table: image_acquisition_" not in str(exc):
                raise
            logger.warning("acquisition_part_cleanup_tables_not_migrated")
            return 0
        cleaned = 0
        for item_id in candidate_ids:
            if self._cleanup_part_item(item_id):
                cleaned += 1
        return cleaned

    def _cleanup_part_item(
        self,
        item_id: str,
        *,
        job_id: UUID | str | None = None,
        worker: str | None = None,
        token: str | None = None,
        generation: int | None = None,
    ) -> bool:
        cleanup_token = uuid4().hex
        now = datetime.now(UTC)
        try:
            with self.session_factory() as session:
                conditions = [
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.status.in_(
                        [
                            ImageAcquisitionItemStatus.IMPORTED.value,
                            ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                            ImageAcquisitionItemStatus.SKIPPED.value,
                            ImageAcquisitionItemStatus.FAILED.value,
                        ]
                    ),
                    ImageAcquisitionJobItemRecord.retryable == False,  # noqa: E712
                    ImageAcquisitionJobItemRecord.part_cleanup_warning.in_(
                        [
                            PartCleanupWarningCode.PENDING.value,
                            *PART_CLEANUP_WARNING_CODES,
                        ]
                    ),
                    (
                        ImageAcquisitionJobItemRecord.part_cleanup_claim_token.is_(None)
                        | (
                            ImageAcquisitionJobItemRecord.part_cleanup_claimed_at
                            < now
                            - timedelta(
                                seconds=max(
                                    self.settings.image_download_stale_after_seconds,
                                    60,
                                )
                            )
                        )
                        | ImageAcquisitionJobItemRecord.part_cleanup_claimed_at.is_(
                            None
                        )
                    ),
                ]
                if job_id is not None:
                    conditions.append(
                        ImageAcquisitionJobItemRecord.job_id == str(job_id)
                    )
                if worker is not None or token is not None or generation is not None:
                    if worker is None or token is None or generation is None:
                        return False
                    conditions.append(
                        select(ImageAcquisitionJobRecord.id)
                        .where(
                            ImageAcquisitionJobRecord.id
                            == ImageAcquisitionJobItemRecord.job_id,
                            ImageAcquisitionJobRecord.status
                            == ImageAcquisitionJobStatus.RUNNING.value,
                            ImageAcquisitionJobRecord.worker_id == worker,
                            ImageAcquisitionJobRecord.claim_token == token,
                            ImageAcquisitionJobRecord.worker_generation == generation,
                        )
                        .exists()
                    )
                result = session.execute(
                    update(ImageAcquisitionJobItemRecord)
                    .where(*conditions)
                    .values(
                        part_cleanup_claim_token=cleanup_token,
                        part_cleanup_claimed_at=now,
                    )
                    .returning(ImageAcquisitionJobItemRecord.job_id)
                ).scalar_one_or_none()
                if result is None:
                    session.rollback()
                    return False
                item = session.scalar(
                    select(ImageAcquisitionJobItemRecord).where(
                        ImageAcquisitionJobItemRecord.id == item_id
                    )
                )
                job = session.scalar(
                    select(ImageAcquisitionJobRecord).where(
                        ImageAcquisitionJobRecord.id == str(result)
                    )
                )
                if item is None or job is None:
                    session.rollback()
                    return False
                recheck_conditions = [
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == item.job_id,
                    ImageAcquisitionJobItemRecord.part_cleanup_claim_token
                    == cleanup_token,
                    ImageAcquisitionJobItemRecord.status.in_(
                        PART_CLEANUP_ITEM_STATUSES
                    ),
                    ImageAcquisitionJobItemRecord.retryable == False,  # noqa: E712
                ]
                if worker is not None:
                    recheck_conditions.append(
                        select(ImageAcquisitionJobRecord.id)
                        .where(
                            ImageAcquisitionJobRecord.id
                            == ImageAcquisitionJobItemRecord.job_id,
                            ImageAcquisitionJobRecord.status
                            == ImageAcquisitionJobStatus.RUNNING.value,
                            ImageAcquisitionJobRecord.worker_id == worker,
                            ImageAcquisitionJobRecord.claim_token == token,
                            ImageAcquisitionJobRecord.worker_generation == generation,
                        )
                        .exists()
                    )
                if (
                    session.scalar(
                        select(ImageAcquisitionJobItemRecord.id).where(
                            *recheck_conditions
                        )
                    )
                    is None
                ):
                    session.rollback()
                    return False
                warning = self._cleanup_part_artifact(
                    self._stale_part_path(job.project_id, item)
                )
                final_conditions = [
                    *recheck_conditions,
                ]
                updated = session.execute(
                    update(ImageAcquisitionJobItemRecord)
                    .where(*final_conditions)
                    .values(
                        part_cleanup_warning=(
                            warning.value if warning is not None else None
                        ),
                        part_cleanup_claim_token=None,
                        part_cleanup_claimed_at=None,
                        updated_at=datetime.now(UTC),
                    )
                ).rowcount
                if updated != 1:
                    session.rollback()
                    return False
                session.commit()
                return True
        except SQLAlchemyError:
            logger.warning("acquisition_part_cleanup_persist_failed")
            return False

    def _claim_stale_job(
        self,
        session: Any,
        job_id: str,
        worker_id: str | None,
        claim_token: str | None,
        generation: int,
        threshold: datetime,
        now: datetime,
    ) -> bool:
        conditions = [
            ImageAcquisitionJobRecord.id == job_id,
            ImageAcquisitionJobRecord.status == ImageAcquisitionJobStatus.RUNNING.value,
            ImageAcquisitionJobRecord.heartbeat_at < threshold,
            ImageAcquisitionJobRecord.worker_generation == generation,
        ]
        conditions.append(
            ImageAcquisitionJobRecord.worker_id.is_(None)
            if worker_id is None
            else ImageAcquisitionJobRecord.worker_id == worker_id
        )
        conditions.append(
            ImageAcquisitionJobRecord.claim_token.is_(None)
            if claim_token is None
            else ImageAcquisitionJobRecord.claim_token == claim_token
        )
        result = session.execute(
            update(ImageAcquisitionJobRecord)
            .where(*conditions)
            .values(
                status=ImageAcquisitionJobStatus.STALE.value,
                worker_id=None,
                claim_token=None,
                worker_generation=ImageAcquisitionJobRecord.worker_generation + 1,
                current_item_id=None,
                updated_at=now,
            )
            .returning(ImageAcquisitionJobRecord.id)
        ).scalar_one_or_none()
        return result is not None

    def run_job(self, job_id: UUID, *, worker_id: str | None = None) -> None:
        worker = worker_id or f"acquisition-worker-{uuid4().hex[:12]}"
        token = uuid4().hex
        generation = self._claim_job(job_id, worker, token)
        if generation is None:
            return
        try:
            self._run_claimed_job(job_id, worker, token, generation)
        except _ClaimLost:
            logger.warning("acquisition_worker_claim_lost job_id=%s", job_id)
        except Exception:
            logger.exception("acquisition_worker_failed job_id=%s", job_id)
            try:
                self._recompute_counts(job_id, worker, token, generation)
            except _ClaimLost:
                logger.warning("acquisition_worker_claim_lost job_id=%s", job_id)
                return
            self._finish_job(
                job_id,
                worker,
                token,
                generation,
                ImageAcquisitionJobStatus.FAILED,
                DownloadFailureCode.UNKNOWN_DOWNLOAD_ERROR,
            )

    def _start_thread(self, job_id: UUID) -> None:
        threading.Thread(
            target=self.run_job,
            args=(job_id,),
            kwargs={"worker_id": f"acquisition-worker-{uuid4().hex[:12]}"},
            daemon=True,
        ).start()

    def get_job(self, job_id: UUID) -> AcquisitionJobView | None:
        self.recover_stale_jobs()
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            return _job_view(job) if job else None

    def list_items(self, job_id: UUID) -> list[AcquisitionItemView]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(ImageAcquisitionJobItemRecord)
                .where(ImageAcquisitionJobItemRecord.job_id == str(job_id))
                .order_by(ImageAcquisitionJobItemRecord.display_order.asc())
            ).all()
            return [_item_view(row) for row in rows]

    def run_job_sync(self, job_id: UUID) -> None:
        self.run_job(job_id, worker_id="acquisition-test-worker")

    def _run_claimed_job(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> None:
        plan, plan_items = self._load_confirmed_plan(job_id, by_job=True)
        try:
            self._validate_plan_structure(plan, plan_items)
        except AcquisitionDownloadError as exc:
            self._fail_all_unfinished(job_id, worker, token, generation, exc.code)
            counts = self._recompute_counts(job_id, worker, token, generation)
            self._write_manifest(
                job_id,
                worker,
                token,
                generation,
                ImageAcquisitionJobStatus.FAILED,
                counts=counts,
            )
            self._finish_job(
                job_id,
                worker,
                token,
                generation,
                ImageAcquisitionJobStatus.FAILED,
                exc.code,
            )
            return
        while True:
            item_id = self._next_item(job_id)
            if item_id is None:
                break
            if self._is_cancel_requested(job_id):
                self._finish_job(
                    job_id,
                    worker,
                    token,
                    generation,
                    ImageAcquisitionJobStatus.CANCELED,
                    DownloadFailureCode.CANCELED,
                )
                return
            if not self._claim_item(job_id, item_id, worker, token, generation):
                raise _ClaimLost()
            try:
                self._process_item(job_id, item_id, plan, worker, token, generation)
            except _Canceled:
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    DownloadFailureCode.CANCELED,
                    False,
                    canceled=True,
                    generation=generation,
                )
                self._finish_job(
                    job_id,
                    worker,
                    token,
                    generation,
                    ImageAcquisitionJobStatus.CANCELED,
                    DownloadFailureCode.CANCELED,
                )
                return
            except _ClaimLost:
                raise
            except Exception:
                logger.exception("acquisition_item_failed job_id=%s", job_id)
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    DownloadFailureCode.UNKNOWN_DOWNLOAD_ERROR,
                    False,
                    generation=generation,
                )
        counts = self._recompute_counts(job_id, worker, token, generation)
        status, error = self._job_terminal_status(
            job_id, worker=worker, token=token, generation=generation
        )
        self._write_manifest(job_id, worker, token, generation, status, counts=counts)
        self._finish_job(job_id, worker, token, generation, status, error)

    def _process_item(
        self,
        job_id: UUID,
        item_id: str,
        plan: ImageAcquisitionPlanRecord,
        worker: str,
        token: str,
        generation: int,
    ) -> None:
        with self.session_factory() as session:
            item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.id == item_id
                )
            )
            plan_item = (
                session.scalar(
                    select(ImageAcquisitionPlanItemRecord).where(
                        ImageAcquisitionPlanItemRecord.id == item.plan_item_id
                    )
                )
                if item
                else None
            )
            if item is None or plan_item is None:
                raise _ClaimLost()
            source_type = ImageSourceType(item.source_type)
            external_post_id = item.external_post_id
            initial_status = item.status
        self._check_cancel_or_claim(job_id, worker, token, generation)
        adapter = self._adapter(source_type)
        max_attempts = self.settings.image_download_retry_max_attempts
        attempted = self._item_attempt_count(item_id)
        if attempted >= max_attempts:
            self._set_item_failure(
                job_id,
                item_id,
                worker,
                token,
                DownloadFailureCode.RETRY_EXHAUSTED,
                False,
                generation=generation,
            )
            return
        for requested_attempt in range(attempted + 1, max_attempts + 1):
            attempt = self._set_attempt_started(
                job_id,
                item_id,
                worker,
                token,
                generation,
                requested_attempt,
            )
            try:
                post = self._load_current_source_post(
                    adapter,
                    external_post_id,
                    job_id,
                    worker,
                    token,
                    generation,
                )
                if not self._matches_plan_item(item_id, post):
                    raise AcquisitionDownloadError(
                        DownloadFailureCode.SOURCE_METADATA_CHANGED
                    )
                if self._source_link_exists(
                    post.source_type.value, post.external_post_id
                ):
                    self._mark_linked_existing(
                        job_id, item_id, worker, token, generation, attempt
                    )
                    return
                self._check_storage(job_id, item_id, post.file_size)
                if (
                    initial_status
                    == ImageAcquisitionItemStatus.VALIDATION_PENDING.value
                ):
                    verified = self._validate_staged_part(
                        job_id, item_id, post, worker, token, generation
                    )
                else:
                    verified = self._download_and_validate(
                        job_id, item_id, post, worker, token, generation, attempt
                    )
                try:
                    self._import_item(
                        job_id,
                        item_id,
                        plan,
                        post,
                        verified,
                        worker,
                        token,
                        generation,
                        attempt,
                    )
                except OSError as exc:
                    raise AcquisitionDownloadError(
                        DownloadFailureCode.IMPORT_FAILED
                    ) from exc
                return
            except _Canceled:
                raise
            except _ClaimLost:
                raise
            except DanbooruSourceError as exc:
                code = _source_failure(exc)
                if not exc.retryable:
                    self._set_item_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        code,
                        False,
                        generation=generation,
                        http_status=exc.status,
                        retry_after=exc.retry_after,
                    )
                    return
                if attempt >= max_attempts:
                    self._record_attempt_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        generation,
                        attempt,
                        DownloadFailureCode.RETRY_EXHAUSTED,
                        False,
                        retry_after=exc.retry_after,
                        http_status=exc.status,
                    )
                    self._set_item_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        DownloadFailureCode.RETRY_EXHAUSTED,
                        False,
                        generation=generation,
                        http_status=exc.status,
                        retry_after=exc.retry_after,
                    )
                    return
                self._record_attempt_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    generation,
                    attempt,
                    code,
                    True,
                    retry_after=exc.retry_after,
                    http_status=exc.status,
                )
                self._retry_backoff(
                    exc.retry_after, attempt, job_id, worker, token, generation
                )
            except _RetryableDownload as exc:
                if attempt >= max_attempts:
                    self._set_item_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        DownloadFailureCode.RETRY_EXHAUSTED,
                        False,
                        generation=generation,
                    )
                    return
                self._record_attempt_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    generation,
                    attempt,
                    exc.code,
                    True,
                    retry_after=exc.retry_after,
                )
                self._retry_backoff(
                    exc.retry_after, attempt, job_id, worker, token, generation
                )
            except ImageVerificationError as exc:
                code = _failure_code(exc.code)
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    code,
                    False,
                    generation=generation,
                )
                return
            except DownloadTransportError as exc:
                if exc.code in RETRYABLE_CODES:
                    if attempt >= max_attempts:
                        self._set_item_failure(
                            job_id,
                            item_id,
                            worker,
                            token,
                            DownloadFailureCode.RETRY_EXHAUSTED,
                            False,
                            generation=generation,
                            http_status=exc.status,
                            retry_after=exc.retry_after,
                        )
                        return
                    self._record_attempt_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        generation,
                        attempt,
                        exc.code,
                        True,
                        retry_after=exc.retry_after,
                        http_status=exc.status,
                    )
                    self._retry_backoff(
                        exc.retry_after, attempt, job_id, worker, token, generation
                    )
                    continue
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    exc.code,
                    False,
                    generation=generation,
                    http_status=exc.status,
                    retry_after=exc.retry_after,
                )
                return
            except AcquisitionDownloadError as exc:
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    exc.code,
                    False,
                    generation=generation,
                )
                return
            except OSError:
                if attempt >= max_attempts:
                    self._set_item_failure(
                        job_id,
                        item_id,
                        worker,
                        token,
                        DownloadFailureCode.DISK_FULL_DURING_DOWNLOAD,
                        False,
                        generation=generation,
                    )
                    return
                self._record_attempt_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    generation,
                    attempt,
                    DownloadFailureCode.CONNECTION_FAILED,
                    True,
                )
                self._retry_backoff(None, attempt, job_id, worker, token, generation)
            except Exception:
                self._set_item_failure(
                    job_id,
                    item_id,
                    worker,
                    token,
                    DownloadFailureCode.IMPORT_FAILED,
                    False,
                    generation=generation,
                )
                return

    def _validate_staged_part(
        self,
        job_id: UUID,
        item_id: str,
        post: ImageSourcePost,
        worker: str,
        token: str,
        generation: int,
    ) -> VerifiedImageFile:
        path, _ = self._item_path(job_id, item_id)
        self._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.VALIDATING,
            generation=generation,
        )
        self._check_cancel_or_claim(job_id, worker, token, generation)
        verified = self.ingestion.inspect_image(
            path,
            expected_md5=post.source_md5,
            expected_file_size=post.file_size,
            expected_width=post.width,
            expected_height=post.height,
            expected_extension=post.file_extension,
        )
        self._set_verified(job_id, item_id, worker, token, verified, generation)
        return verified

    def _download_and_validate(
        self,
        job_id: UUID,
        item_id: str,
        post: ImageSourcePost,
        worker: str,
        token: str,
        generation: int,
        attempt_number: int,
    ) -> VerifiedImageFile:
        path, item = self._item_path(job_id, item_id)
        self._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.DOWNLOADING,
            generation=generation,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        resume_start = self._resume_start(path, item, post)
        request = DownloadRequest(
            post.file_url or "",
            range_start=resume_start,
            etag=item.etag,
            last_modified=item.last_modified,
        )
        self._check_cancel_or_claim(job_id, worker, token, generation)
        response = self.transport.open(request)
        self._check_cancel_or_claim(job_id, worker, token, generation)
        headers = _normalized_headers(response.headers)
        self._record_attempt_response(
            job_id,
            item_id,
            worker,
            token,
            generation,
            attempt_number,
            response.status,
            resume_start,
            headers,
        )
        actual_start = resume_start
        range_total: int | None = None
        if resume_start is not None and response.status == 200:
            actual_start = None
            self._truncate_part(path)
        elif resume_start is not None and response.status == 206:
            range_total = self._validate_range_response(response, resume_start, item)
        elif response.status == 416 and resume_start is not None:
            response.close()
            self._truncate_part(path)
            return self._download_and_validate(
                job_id,
                item_id,
                post,
                worker,
                token,
                generation,
                attempt_number,
            )
        elif response.status != 200:
            response.close()
            raise DownloadTransportError(
                _status_failure(response.status), status=response.status
            )
        length = _parse_content_length(headers.get("content-length"))
        if headers.get("content-length") is not None and length is None:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_LENGTH_INVALID)
        expected_size = post.file_size
        if (
            length is not None
            and length > self.settings.image_download_max_file_size_bytes
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.FILE_TOO_LARGE)
        if (
            actual_start is None
            and expected_size is not None
            and length is not None
            and length != expected_size
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.RECEIVED_SIZE_MISMATCH)
        if (
            actual_start is not None
            and expected_size is not None
            and length is not None
            and length != expected_size - actual_start
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        mode = "ab" if actual_start is not None else "wb"
        received = path.stat().st_size if mode == "ab" and path.exists() else 0
        self._update_response_metadata(
            job_id,
            item_id,
            worker,
            token,
            generation,
            headers,
            received,
            actual_start,
        )
        try:
            with path.open(mode) as handle:
                for chunk in response.iter_chunks(
                    self.settings.image_download_chunk_size
                ):
                    self._check_cancel_or_claim(job_id, worker, token, generation)
                    if not isinstance(chunk, bytes) or not chunk:
                        continue
                    received += len(chunk)
                    if received > self.settings.image_download_max_file_size_bytes:
                        raise AcquisitionDownloadError(
                            DownloadFailureCode.FILE_TOO_LARGE
                        )
                    try:
                        handle.write(chunk)
                        handle.flush()
                    except OSError as exc:
                        raise OSError("staging write failed") from exc
                    self._update_received(
                        job_id, item_id, worker, token, generation, received
                    )
                os.fsync(handle.fileno())
        except _Canceled:
            raise
        except _ClaimLost:
            raise
        except DownloadTransportError:
            raise
        except OSError as exc:
            raise OSError("download stream interrupted") from exc
        finally:
            response.close()
        if received <= 0:
            raise AcquisitionDownloadError(DownloadFailureCode.EMPTY_FILE)
        if expected_size is not None and received != expected_size:
            raise AcquisitionDownloadError(DownloadFailureCode.RECEIVED_SIZE_MISMATCH)
        if range_total is not None and received != range_total:
            raise AcquisitionDownloadError(DownloadFailureCode.RECEIVED_SIZE_MISMATCH)
        self._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.DOWNLOADED,
            generation=generation,
        )
        self._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.VALIDATING,
            generation=generation,
        )
        self._check_cancel_or_claim(job_id, worker, token, generation)
        verified = self.ingestion.inspect_image(
            path,
            expected_md5=post.source_md5,
            expected_file_size=expected_size,
            expected_width=post.width,
            expected_height=post.height,
            expected_extension=post.file_extension,
        )
        self._set_verified(job_id, item_id, worker, token, verified, generation)
        return verified

    def _import_item(
        self,
        job_id: UUID,
        item_id: str,
        plan: ImageAcquisitionPlanRecord,
        post: ImageSourcePost,
        verified: VerifiedImageFile,
        worker: str,
        token: str,
        generation: int,
        attempt: int,
    ) -> None:
        self._check_cancel_or_claim(job_id, worker, token, generation)
        self._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.IMPORTING,
            generation=generation,
        )
        provenance = ImageSourceProvenance(
            source_type=post.source_type,
            external_post_id=post.external_post_id,
            source_post_url=post.post_url,
            source_md5=post.source_md5,
            source_metadata_fingerprint=_post_metadata_fingerprint(post),
            acquisition_plan_id=UUID(plan.id),
            acquisition_job_id=job_id,
            acquisition_job_item_id=UUID(item_id),
        )
        result = self.ingestion.import_verified(
            UUID(plan.project_id), verified, provenance
        )
        status = (
            ImageAcquisitionItemStatus.LINKED_EXISTING
            if result.linked_existing
            else ImageAcquisitionItemStatus.IMPORTED
        )
        self._set_item_complete(
            job_id,
            item_id,
            worker,
            token,
            status,
            result.image.id,
            generation,
            attempt,
        )

    def _validate_plan_structure(
        self,
        plan: ImageAcquisitionPlanRecord,
        items: Iterable[ImageAcquisitionPlanItemRecord],
    ) -> list[ImageAcquisitionPlanItemRecord]:
        """Validate immutable plan/search/reservation structure without network I/O."""
        if plan.status != "confirmed":
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_NOT_CONFIRMED)
        adapter = self._adapter(ImageSourceType(plan.source_type))
        materialized_items = list(items)
        with self.session_factory() as session:
            search = session.scalar(
                select(ImageSourceSearchRecord).where(
                    ImageSourceSearchRecord.id == plan.source_search_id
                )
            )
            reservations = session.scalars(
                select(ImageAcquisitionReservationRecord).where(
                    ImageAcquisitionReservationRecord.plan_id == plan.id
                )
            ).all()
            result_ids = {item.search_result_id for item in materialized_items}
            results = {
                result.id: result
                for result in session.scalars(
                    select(ImageSourceSearchResultRecord).where(
                        ImageSourceSearchResultRecord.id.in_(result_ids)
                    )
                ).all()
            }
        if search is None:
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_METADATA_CHANGED)
        if (
            plan.project_id != search.project_id
            or plan.source_type != search.source_type
            or plan.source_search_id != search.id
            or plan.query_fingerprint != search.query_fingerprint
            or plan.adapter_version != search.adapter_version
            or plan.plan_version != PLAN_VERSION
            or adapter.adapter_version != plan.adapter_version
            or len(materialized_items) != plan.selected_count
            or [item.display_order for item in materialized_items]
            != list(range(len(materialized_items)))
            or len({item.external_post_id for item in materialized_items})
            != len(materialized_items)
        ):
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_METADATA_CHANGED)
        if any(
            item.plan_id != plan.id or item.planned_status != "planned"
            for item in materialized_items
        ):
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_METADATA_CHANGED)
        expected_reservations = {
            (plan.source_type, item.external_post_id) for item in materialized_items
        }
        actual_reservations = {
            (reservation.source_type, reservation.external_post_id)
            for reservation in reservations
        }
        if actual_reservations != expected_reservations:
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_METADATA_CHANGED)
        plan_items_for_fingerprint: list[dict[str, object]] = []
        for item in materialized_items:
            result = results.get(item.search_result_id)
            if (
                result is None
                or result.search_id != search.id
                or result.external_post_id != item.external_post_id
                or result.candidate_status != "accepted"
                or result.exclusion_reasons_json != "[]"
                or result.metadata_fingerprint_at_search
                != item.expected_metadata_fingerprint
                or result.already_imported
                or result.already_planned
            ):
                raise AcquisitionDownloadError(
                    DownloadFailureCode.PLAN_METADATA_CHANGED
                )
            plan_items_for_fingerprint.append(
                {
                    "search_result_id": result.id,
                    "external_post_id": item.external_post_id,
                    "metadata_fingerprint": item.expected_metadata_fingerprint,
                    "file_url_fingerprint": item.expected_file_url_fingerprint,
                    "source_md5": item.expected_md5,
                    "extension": item.expected_extension,
                    "width": item.expected_width,
                    "height": item.expected_height,
                    "already_imported": bool(result.already_imported),
                    "already_planned": bool(result.already_planned),
                }
            )
        if (
            fingerprint(
                {
                    "project_id": search.project_id,
                    "search_id": search.id,
                    "source_type": search.source_type,
                    "items": plan_items_for_fingerprint,
                    "query_fingerprint": search.query_fingerprint,
                    "adapter_version": search.adapter_version,
                    "plan_version": PLAN_VERSION,
                    "plan_status": "confirmed",
                    "selected_count": len(materialized_items),
                }
            )
            != plan.plan_fingerprint
        ):
            raise AcquisitionDownloadError(DownloadFailureCode.PLAN_METADATA_CHANGED)
        return materialized_items

    def _load_current_source_post(
        self,
        adapter: ImageSourceAdapter,
        external_post_id: str,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
    ) -> ImageSourcePost:
        context = ImageSourceRequestContext(
            cancel_requested=lambda: self._source_cancel_requested(
                job_id, worker, token, generation
            ),
            before_request=lambda: self._source_checkpoint(
                job_id, worker, token, generation
            ),
            after_request=lambda: self._source_checkpoint(
                job_id, worker, token, generation
            ),
            poll_interval_seconds=self.settings.image_download_heartbeat_interval_seconds,
        )
        self._source_checkpoint(job_id, worker, token, generation)
        try:
            post = adapter.get_post(external_post_id, context=context)
        finally:
            self._source_checkpoint(job_id, worker, token, generation)
        if post is None or post.is_deleted:
            raise AcquisitionDownloadError(DownloadFailureCode.SOURCE_POST_NOT_FOUND)
        if post.is_pending or post.is_flagged:
            raise AcquisitionDownloadError(DownloadFailureCode.SOURCE_METADATA_CHANGED)
        if post.file_url is None:
            raise AcquisitionDownloadError(DownloadFailureCode.FILE_URL_MISSING)
        if not validate_download_url(post.file_url):
            raise AcquisitionDownloadError(DownloadFailureCode.FILE_URL_NOT_ALLOWED)
        if post.source_md5 is not None and not _MD5_RE.fullmatch(post.source_md5):
            raise AcquisitionDownloadError(DownloadFailureCode.SOURCE_MD5_INVALID)
        return post

    def _source_cancel_requested(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> bool:
        self._source_checkpoint(job_id, worker, token, generation)
        return False

    def _source_checkpoint(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
            )
            if job is None:
                raise _ClaimLost()
            if job.cancellation_requested:
                raise _Canceled()
            now = datetime.now(UTC)
            updated = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                    ImageAcquisitionJobRecord.cancellation_requested == False,  # noqa: E712
                )
                .values(heartbeat_at=now, updated_at=now)
                .returning(ImageAcquisitionJobRecord.id)
            ).scalar_one_or_none()
            session.commit()
            if updated is None:
                cancellation_requested = session.scalar(
                    select(ImageAcquisitionJobRecord.cancellation_requested).where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                )
                if cancellation_requested:
                    raise _Canceled()
                raise _ClaimLost()

    def _is_cancel_requested(self, job_id: UUID) -> bool:
        with self.session_factory() as session:
            return bool(
                session.scalar(
                    select(ImageAcquisitionJobRecord.cancellation_requested).where(
                        ImageAcquisitionJobRecord.id == str(job_id)
                    )
                )
            )

    def _load_confirmed_plan(
        self, plan_id: UUID, *, by_job: bool = False
    ) -> tuple[ImageAcquisitionPlanRecord, list[ImageAcquisitionPlanItemRecord]]:
        with self.session_factory() as session:
            if by_job:
                job = session.scalar(
                    select(ImageAcquisitionJobRecord).where(
                        ImageAcquisitionJobRecord.id == str(plan_id)
                    )
                )
                if job is None:
                    raise AcquisitionDownloadError(
                        DownloadFailureCode.PLAN_NOT_CONFIRMED
                    )
                actual_plan_id = job.plan_id
            else:
                actual_plan_id = str(plan_id)
            plan = session.scalar(
                select(ImageAcquisitionPlanRecord).where(
                    ImageAcquisitionPlanRecord.id == actual_plan_id
                )
            )
            if plan is None or plan.status != "confirmed":
                raise AcquisitionDownloadError(DownloadFailureCode.PLAN_NOT_CONFIRMED)
            items = session.scalars(
                select(ImageAcquisitionPlanItemRecord)
                .where(
                    ImageAcquisitionPlanItemRecord.plan_id == plan.id,
                    ImageAcquisitionPlanItemRecord.planned_status == "planned",
                )
                .order_by(ImageAcquisitionPlanItemRecord.display_order.asc())
            ).all()
            return plan, items

    def _adapter(self, source_type: ImageSourceType) -> ImageSourceAdapter:
        try:
            return self._adapters[source_type]
        except KeyError as exc:
            raise AcquisitionDownloadError(
                DownloadFailureCode.SOURCE_POST_UNAVAILABLE
            ) from exc

    def _active_or_finished_job(
        self, plan_id: UUID
    ) -> ImageAcquisitionJobRecord | None:
        with self.session_factory() as session:
            result = session.scalar(
                select(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.plan_id == str(plan_id),
                    ImageAcquisitionJobRecord.status.in_(
                        [
                            status.value
                            for status in ImageAcquisitionJobStatus
                            if status != ImageAcquisitionJobStatus.FAILED
                        ]
                    ),
                )
                .order_by(ImageAcquisitionJobRecord.created_at.desc())
            )
            return cast(ImageAcquisitionJobRecord | None, result)

    def _claim_job(self, job_id: UUID, worker: str, token: str) -> int | None:
        with self.session_factory() as session:
            now = datetime.now(UTC)
            result = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status.in_(
                        [
                            ImageAcquisitionJobStatus.QUEUED.value,
                            ImageAcquisitionJobStatus.STALE.value,
                        ]
                    ),
                    ImageAcquisitionJobRecord.cancellation_requested == False,  # noqa: E712
                    ImageAcquisitionJobRecord.worker_id.is_(None),
                    ImageAcquisitionJobRecord.claim_token.is_(None),
                )
                .values(
                    status=ImageAcquisitionJobStatus.RUNNING.value,
                    worker_id=worker,
                    claim_token=token,
                    worker_generation=ImageAcquisitionJobRecord.worker_generation + 1,
                    heartbeat_at=now,
                    started_at=now,
                    updated_at=now,
                )
                .returning(ImageAcquisitionJobRecord.worker_generation)
            ).scalar_one_or_none()
            session.commit()
            return int(result) if result is not None else None

    def _claim_item(
        self, job_id: UUID, item_id: str, worker: str, token: str, generation: int
    ) -> bool:
        with self.session_factory() as session:
            result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    ImageAcquisitionJobItemRecord.status.in_(
                        [
                            ImageAcquisitionItemStatus.PENDING.value,
                            ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                            ImageAcquisitionItemStatus.FAILED.value,
                        ]
                    ),
                    ImageAcquisitionJobItemRecord.retryable.in_([True, False]),
                    select(ImageAcquisitionJobRecord.id)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status=case(
                        (
                            ImageAcquisitionJobItemRecord.status
                            == ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                            ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                        ),
                        else_=ImageAcquisitionItemStatus.DOWNLOADING.value,
                    ),
                    started_at=datetime.now(UTC),
                    part_cleanup_warning=None,
                    part_cleanup_claim_token=None,
                    part_cleanup_claimed_at=None,
                    updated_at=datetime.now(UTC),
                )
                .returning(ImageAcquisitionJobItemRecord.id)
            ).scalar_one_or_none()
            if result is not None:
                session.execute(
                    update(ImageAcquisitionJobRecord)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .values(current_item_id=item_id, updated_at=datetime.now(UTC))
                )
            session.commit()
            return result is not None

    def _next_item(self, job_id: UUID) -> str | None:
        with self.session_factory() as session:
            item = session.scalar(
                select(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    ImageAcquisitionJobItemRecord.status.in_(
                        [
                            ImageAcquisitionItemStatus.PENDING.value,
                            ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                            ImageAcquisitionItemStatus.FAILED.value,
                        ]
                    ),
                    (
                        ImageAcquisitionJobItemRecord.status
                        == ImageAcquisitionItemStatus.PENDING.value
                    )
                    | (
                        ImageAcquisitionJobItemRecord.status
                        == ImageAcquisitionItemStatus.VALIDATION_PENDING.value
                    )
                    | (ImageAcquisitionJobItemRecord.retryable == True),  # noqa: E712
                )
                .order_by(ImageAcquisitionJobItemRecord.display_order.asc())
            )
            return item.id if item else None

    def _check_cancel_or_claim(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
            )
            if job is None:
                raise _ClaimLost()
            if job.cancellation_requested:
                raise _Canceled()

    def _set_item_status(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        status: ImageAcquisitionItemStatus,
        *,
        generation: int,
    ) -> None:
        self._conditional_item_update(
            job_id,
            item_id,
            worker,
            token,
            generation=generation,
            status=status.value,
            updated_at=datetime.now(UTC),
        )

    def _set_item_complete(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        status: ImageAcquisitionItemStatus,
        image_id: UUID | None,
        generation: int,
        attempt: int,
    ) -> None:
        now = datetime.now(UTC)
        with self.session_factory() as session:
            claim = [
                ImageAcquisitionJobRecord.id == str(job_id),
                ImageAcquisitionJobRecord.status
                == ImageAcquisitionJobStatus.RUNNING.value,
                ImageAcquisitionJobRecord.worker_id == worker,
                ImageAcquisitionJobRecord.claim_token == token,
                ImageAcquisitionJobRecord.worker_generation == generation,
            ]
            item_result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    select(ImageAcquisitionJobRecord.id).where(*claim).exists(),
                )
                .values(
                    status=status.value,
                    image_asset_id=str(image_id) if image_id else None,
                    completed_at=now,
                    updated_at=now,
                    failure_code=None,
                    failure_message=None,
                    retryable=False,
                    part_cleanup_warning=PartCleanupWarningCode.PENDING.value,
                    part_cleanup_claim_token=None,
                    part_cleanup_claimed_at=None,
                )
                .returning(ImageAcquisitionJobItemRecord.id)
            ).scalar_one_or_none()
            if item_result is None:
                raise _ClaimLost()
            attempt_result = session.execute(
                update(ImageAcquisitionAttemptRecord)
                .where(
                    ImageAcquisitionAttemptRecord.job_item_id == item_id,
                    ImageAcquisitionAttemptRecord.attempt_number == attempt,
                    select(ImageAcquisitionJobRecord.id)
                    .join(
                        ImageAcquisitionJobItemRecord,
                        ImageAcquisitionJobItemRecord.job_id
                        == ImageAcquisitionJobRecord.id,
                    )
                    .where(*claim, ImageAcquisitionJobItemRecord.id == item_id)
                    .exists(),
                )
                .values(
                    status="succeeded",
                    failure_code=None,
                    received_bytes=(
                        select(ImageAcquisitionJobItemRecord.received_bytes)
                        .where(ImageAcquisitionJobItemRecord.id == item_id)
                        .scalar_subquery()
                    ),
                    completed_at=now,
                )
                .returning(ImageAcquisitionAttemptRecord.id)
            ).scalar_one_or_none()
            if attempt_result is None:
                raise _ClaimLost()
            session.commit()
        self._cleanup_part_item(
            item_id,
            job_id=job_id,
            worker=worker,
            token=token,
            generation=generation,
        )
        self._recompute_counts(job_id, worker, token, generation)

    def _set_verified(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        verified: VerifiedImageFile,
        generation: int,
    ) -> None:
        self._conditional_item_update(
            job_id,
            item_id,
            worker,
            token,
            generation=generation,
            status=ImageAcquisitionItemStatus.VALIDATED.value,
            calculated_md5=verified.md5,
            calculated_sha256=verified.sha256,
            detected_format=verified.detected_format,
            detected_mime_type=verified.mime_type,
            detected_width=verified.width,
            detected_height=verified.height,
            detected_file_size=verified.file_size,
            updated_at=datetime.now(UTC),
        )

    def _conditional_item_update(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        *,
        generation: int,
        **values: Any,
    ) -> None:
        with self.session_factory() as session:
            claim_conditions = [
                ImageAcquisitionJobRecord.id == str(job_id),
                ImageAcquisitionJobRecord.status
                == ImageAcquisitionJobStatus.RUNNING.value,
                ImageAcquisitionJobRecord.worker_id == worker,
                ImageAcquisitionJobRecord.claim_token == token,
            ]
            if generation is not None:
                claim_conditions.append(
                    ImageAcquisitionJobRecord.worker_generation == generation
                )
            result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    select(ImageAcquisitionJobRecord.id)
                    .where(*claim_conditions)
                    .exists(),
                )
                .values(**values)
                .returning(ImageAcquisitionJobItemRecord.id)
            ).scalar_one_or_none()
            session.commit()
            if result is None:
                raise _ClaimLost()
        self._recompute_counts(job_id, worker, token, generation)

    def _update_received(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        received: int,
    ) -> None:
        self._conditional_item_update(
            job_id,
            item_id,
            worker,
            token,
            generation=generation,
            received_bytes=received,
            updated_at=datetime.now(UTC),
        )
        with self.session_factory() as session:
            session.execute(
                update(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
                .values(heartbeat_at=datetime.now(UTC), updated_at=datetime.now(UTC))
            )
            session.commit()

    def _update_response_metadata(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        headers: Mapping[str, str],
        received: int,
        range_start: int | None,
    ) -> None:
        self._conditional_item_update(
            job_id,
            item_id,
            worker,
            token,
            generation=generation,
            etag=headers.get("etag"),
            last_modified=headers.get("last-modified"),
            accept_ranges=headers.get("accept-ranges", "").lower() == "bytes",
            range_start=range_start,
            received_bytes=received,
            updated_at=datetime.now(UTC),
        )

    def _record_attempt_response(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        attempt_number: int,
        status: int,
        requested_range_start: int | None,
        headers: Mapping[str, str],
    ) -> None:
        with self.session_factory() as session:
            result = session.execute(
                update(ImageAcquisitionAttemptRecord)
                .where(
                    ImageAcquisitionAttemptRecord.job_item_id == item_id,
                    ImageAcquisitionAttemptRecord.attempt_number == attempt_number,
                    select(ImageAcquisitionJobRecord.id)
                    .join(
                        ImageAcquisitionJobItemRecord,
                        ImageAcquisitionJobItemRecord.job_id
                        == ImageAcquisitionJobRecord.id,
                    )
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobItemRecord.id == item_id,
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    http_status=status,
                    requested_range_start=requested_range_start,
                    response_etag_fingerprint=(
                        fingerprint(headers["etag"]) if headers.get("etag") else None
                    ),
                    response_last_modified_fingerprint=(
                        fingerprint(headers["last-modified"])
                        if headers.get("last-modified")
                        else None
                    ),
                )
                .returning(ImageAcquisitionAttemptRecord.id)
            ).scalar_one_or_none()
            session.commit()
            if result is None:
                raise _ClaimLost()

    def _set_attempt_started(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        requested_attempt: int,
    ) -> int:
        now = datetime.now(UTC)
        with self.session_factory() as session:
            result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    ImageAcquisitionJobItemRecord.attempt_count
                    == requested_attempt - 1,
                    select(ImageAcquisitionJobRecord.id)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    attempt_count=requested_attempt,
                    last_attempted_at=now,
                    updated_at=now,
                )
                .returning(ImageAcquisitionJobItemRecord.id)
            ).scalar_one_or_none()
            if result is None:
                raise _ClaimLost()
            session.add(
                ImageAcquisitionAttemptRecord(
                    id=str(uuid4()),
                    job_item_id=item_id,
                    attempt_number=requested_attempt,
                    status="running",
                    worker_generation=generation,
                    started_at=now,
                    created_at=now,
                )
            )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise _ClaimLost() from exc
        return requested_attempt

    def _record_attempt_failure(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        attempt: int,
        code: DownloadFailureCode,
        retryable: bool,
        retry_after: float | None = None,
        http_status: int | None = None,
    ) -> None:
        with self.session_factory() as session:
            item_result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    select(ImageAcquisitionJobRecord.id)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status=ImageAcquisitionItemStatus.FAILED.value,
                    failure_code=code.value,
                    failure_message=code.value,
                    retryable=retryable,
                    part_cleanup_warning=(
                        None if retryable else PartCleanupWarningCode.PENDING.value
                    ),
                    part_cleanup_claim_token=None,
                    part_cleanup_claimed_at=None,
                    retry_count=ImageAcquisitionJobItemRecord.retry_count + 1,
                    updated_at=datetime.now(UTC),
                )
                .returning(
                    ImageAcquisitionJobItemRecord.received_bytes,
                    ImageAcquisitionJobItemRecord.attempt_count,
                )
            ).one_or_none()
            if item_result is None:
                raise _ClaimLost()
            received_bytes, _ = item_result
            attempt_result = session.execute(
                update(ImageAcquisitionAttemptRecord)
                .where(
                    ImageAcquisitionAttemptRecord.job_item_id == item_id,
                    ImageAcquisitionAttemptRecord.attempt_number == attempt,
                    select(ImageAcquisitionJobRecord.id)
                    .join(
                        ImageAcquisitionJobItemRecord,
                        ImageAcquisitionJobItemRecord.job_id
                        == ImageAcquisitionJobRecord.id,
                    )
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobItemRecord.id == item_id,
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status="failed",
                    failure_code=code.value,
                    retryable=retryable,
                    retry_after_seconds=retry_after,
                    http_status=http_status,
                    received_bytes=received_bytes,
                    completed_at=datetime.now(UTC),
                )
                .returning(ImageAcquisitionAttemptRecord.id)
            ).scalar_one_or_none()
            if attempt_result is None:
                raise _ClaimLost()
            session.commit()
        self._recompute_counts(job_id, worker, token, generation)

    def _finish_attempt(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        attempt_number: int,
        status: str,
        code: DownloadFailureCode | None,
        *,
        http_status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        with self.session_factory() as session:
            received_bytes = (
                select(ImageAcquisitionJobItemRecord.received_bytes)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                )
                .scalar_subquery()
            )
            result = session.execute(
                update(ImageAcquisitionAttemptRecord)
                .where(
                    ImageAcquisitionAttemptRecord.job_item_id == item_id,
                    ImageAcquisitionAttemptRecord.attempt_number == attempt_number,
                    select(ImageAcquisitionJobRecord.id)
                    .join(
                        ImageAcquisitionJobItemRecord,
                        ImageAcquisitionJobItemRecord.job_id
                        == ImageAcquisitionJobRecord.id,
                    )
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobItemRecord.id == item_id,
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status=status,
                    failure_code=code.value if code else None,
                    http_status=http_status,
                    retry_after_seconds=retry_after,
                    received_bytes=received_bytes,
                    completed_at=datetime.now(UTC),
                )
                .returning(ImageAcquisitionAttemptRecord.id)
            ).scalar_one_or_none()
            if result is None:
                raise _ClaimLost()
            session.commit()

    def _retry_item(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        code: DownloadFailureCode,
        generation: int,
    ) -> None:
        self._set_item_failure(
            job_id,
            item_id,
            worker,
            token,
            code,
            True,
            generation=generation,
        )

    def _item_attempt_count(self, item_id: str) -> int:
        with self.session_factory() as session:
            value = session.scalar(
                select(ImageAcquisitionJobItemRecord.attempt_count).where(
                    ImageAcquisitionJobItemRecord.id == item_id
                )
            )
            return int(value or 0)

    def _set_item_failure(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        code: DownloadFailureCode,
        retryable: bool,
        *,
        canceled: bool = False,
        generation: int,
        http_status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        status = (
            ImageAcquisitionItemStatus.CANCELED
            if canceled
            else ImageAcquisitionItemStatus.FAILED
        )
        now = datetime.now(UTC)
        cleanup_required = not retryable and not canceled
        with self.session_factory() as session:
            item_result = session.execute(
                update(ImageAcquisitionJobItemRecord)
                .where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    select(ImageAcquisitionJobRecord.id)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status=status.value,
                    failure_code=code.value,
                    failure_message=code.value,
                    retryable=retryable,
                    part_cleanup_warning=(
                        PartCleanupWarningCode.PENDING.value
                        if cleanup_required
                        else None
                    ),
                    part_cleanup_claim_token=None,
                    part_cleanup_claimed_at=None,
                    completed_at=now,
                    updated_at=now,
                )
                .returning(ImageAcquisitionJobItemRecord.attempt_count)
            ).scalar_one_or_none()
            if item_result is None:
                raise _ClaimLost()
            attempt_result = session.execute(
                update(ImageAcquisitionAttemptRecord)
                .where(
                    ImageAcquisitionAttemptRecord.job_item_id == item_id,
                    ImageAcquisitionAttemptRecord.attempt_number == item_result,
                    select(ImageAcquisitionJobRecord.id)
                    .join(
                        ImageAcquisitionJobItemRecord,
                        ImageAcquisitionJobItemRecord.job_id
                        == ImageAcquisitionJobRecord.id,
                    )
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobItemRecord.id == item_id,
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                    .exists(),
                )
                .values(
                    status="failed",
                    failure_code=code.value,
                    retryable=retryable,
                    http_status=http_status,
                    retry_after_seconds=retry_after,
                    received_bytes=(
                        select(ImageAcquisitionJobItemRecord.received_bytes)
                        .where(ImageAcquisitionJobItemRecord.id == item_id)
                        .scalar_subquery()
                    ),
                    completed_at=now,
                )
                .returning(ImageAcquisitionAttemptRecord.id)
            ).scalar_one_or_none()
            if attempt_result is None:
                raise _ClaimLost()
            session.commit()
        if cleanup_required:
            self._cleanup_part_item(
                item_id,
                job_id=job_id,
                worker=worker,
                token=token,
                generation=generation,
            )
        self._recompute_counts(job_id, worker, token, generation)

    def _mark_linked_existing(
        self,
        job_id: UUID,
        item_id: str,
        worker: str,
        token: str,
        generation: int,
        attempt: int,
    ) -> None:
        with self.session_factory() as session:
            item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.id == item_id
                )
            )
            if item is None:
                raise _ClaimLost()
            link = session.scalar(
                select(ExternalImageAssetLinkRecord).where(
                    ExternalImageAssetLinkRecord.source_type == item.source_type,
                    ExternalImageAssetLinkRecord.external_post_id
                    == item.external_post_id,
                )
            )
            image_asset_id = link.image_asset_id if link else None
        self._set_item_complete(
            job_id,
            item_id,
            worker,
            token,
            (
                ImageAcquisitionItemStatus.LINKED_EXISTING
                if image_asset_id
                else ImageAcquisitionItemStatus.SKIPPED
            ),
            UUID(image_asset_id) if image_asset_id else None,
            generation,
            attempt,
        )

    def _source_link_exists(self, source_type: str, external_post_id: str) -> bool:
        with self.session_factory() as session:
            return (
                session.scalar(
                    select(ExternalImageAssetLinkRecord.id).where(
                        ExternalImageAssetLinkRecord.source_type == source_type,
                        ExternalImageAssetLinkRecord.external_post_id
                        == external_post_id,
                    )
                )
                is not None
            )

    def _matches_plan_item(self, item_id: str, post: ImageSourcePost) -> bool:
        with self.session_factory() as session:
            item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.id == item_id
                )
            )
            return item is not None and self._item_matches_values(item, post)

    @staticmethod
    def _item_matches_values(item: Any, post: ImageSourcePost) -> bool:
        return bool(
            item.expected_metadata_fingerprint == _post_metadata_fingerprint(post)
            and item.expected_file_url_fingerprint
            == (fingerprint(post.file_url) if post.file_url else None)
            and item.expected_md5 == post.source_md5
            and item.expected_width == post.width
            and item.expected_height == post.height
            and item.expected_extension == post.file_extension
        )

    def _item_path(
        self, job_id: UUID, item_id: str
    ) -> tuple[Path, ImageAcquisitionJobItemRecord]:
        with self.session_factory() as session:
            item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.id == item_id,
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                )
            )
            if item is None:
                raise _ClaimLost()
            root = self.projects.project_root(UUID(self._project_id(job_id))).resolve()
            path = (root / item.part_relative_path).resolve()
            staging = (root / "acquisition" / "jobs").resolve()
            if (
                not path.is_relative_to(staging)
                or path.name != f"{item_id}.part"
                or path.is_symlink()
            ):
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            return path, item

    def _stale_part_path(
        self, project_id: str, item: ImageAcquisitionJobItemRecord
    ) -> _PartPathInspection:
        try:
            root = self.projects.project_root(UUID(project_id))
            projects_root = self.settings.projects_dir.resolve()
            root_mode = os.lstat(root).st_mode
            if stat.S_ISLNK(root_mode):
                return _PartPathInspection(
                    None, PartCleanupWarningCode.SYMLINK_REJECTED
                )
            if not stat.S_ISDIR(root_mode):
                return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)
            resolved_root = root.resolve(strict=True)
            if not resolved_root.is_relative_to(projects_root):
                return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)
            relative = Path(item.part_relative_path)
            if relative.is_absolute() or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)
            staging = resolved_root / "acquisition" / "jobs"
            path = resolved_root / relative
            if not path.is_relative_to(staging) or path.name != f"{item.id}.part":
                return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)

            parent_components_list = [resolved_root]
            current_parent = resolved_root
            for part_name in path.relative_to(resolved_root).parts[:-1]:
                current_parent = current_parent / part_name
                parent_components_list.append(current_parent)
            parent_components = tuple(parent_components_list)
            for parent in parent_components:
                try:
                    mode = os.lstat(parent).st_mode
                except FileNotFoundError:
                    return _PartPathInspection(path, None, parent_components)
                except OSError:
                    return _PartPathInspection(
                        None, PartCleanupWarningCode.PATH_INVALID
                    )
                if stat.S_ISLNK(mode):
                    return _PartPathInspection(
                        None, PartCleanupWarningCode.SYMLINK_REJECTED
                    )
                if not stat.S_ISDIR(mode):
                    return _PartPathInspection(
                        None, PartCleanupWarningCode.PATH_INVALID
                    )

            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                return _PartPathInspection(path, None, parent_components)
            except OSError:
                return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)
            if stat.S_ISLNK(mode):
                return _PartPathInspection(
                    None, PartCleanupWarningCode.SYMLINK_REJECTED
                )
            if not stat.S_ISREG(mode):
                return _PartPathInspection(
                    None, PartCleanupWarningCode.NOT_REGULAR_FILE
                )
            return _PartPathInspection(path, None, parent_components)
        except (OSError, ValueError):
            return _PartPathInspection(None, PartCleanupWarningCode.PATH_INVALID)

    @staticmethod
    def _part_matches_item(
        part: Path | None, item: ImageAcquisitionJobItemRecord
    ) -> bool:
        if part is None or not part.is_file() or part.is_symlink():
            return False
        try:
            return bool(part.stat().st_size == item.received_bytes)
        except OSError:
            return False

    @staticmethod
    def _cleanup_part_artifact(
        inspection: _PartPathInspection,
    ) -> PartCleanupWarningCode | None:
        if inspection.warning is not None:
            return inspection.warning
        path = inspection.path
        if path is None:
            return PartCleanupWarningCode.PATH_INVALID
        try:
            for parent in inspection.parent_components:
                try:
                    mode = os.lstat(parent).st_mode
                except FileNotFoundError:
                    return None
                if stat.S_ISLNK(mode):
                    return PartCleanupWarningCode.SYMLINK_REJECTED
                if not stat.S_ISDIR(mode):
                    return PartCleanupWarningCode.PATH_INVALID

            if os.unlink in getattr(os, "supports_dir_fd", set()):
                flags = os.O_RDONLY
                flags |= getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                parent_fd = os.open(path.parent, flags)
                try:
                    mode = os.stat(
                        path.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    ).st_mode
                    if stat.S_ISLNK(mode):
                        return PartCleanupWarningCode.SYMLINK_REJECTED
                    if not stat.S_ISREG(mode):
                        return PartCleanupWarningCode.NOT_REGULAR_FILE
                    os.unlink(path.name, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
                return None

            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(mode):
                return PartCleanupWarningCode.SYMLINK_REJECTED
            if not stat.S_ISREG(mode):
                return PartCleanupWarningCode.NOT_REGULAR_FILE
            path.unlink()
            return None
        except FileNotFoundError:
            return None
        except OSError:
            return PartCleanupWarningCode.CLEANUP_FAILED

    @staticmethod
    def _finalize_stale_attempts(
        session: Any,
        item: ImageAcquisitionJobItemRecord,
        now: datetime,
        code: DownloadFailureCode,
        *,
        retryable: bool,
        status: str = "failed",
    ) -> None:
        attempts = session.scalars(
            select(ImageAcquisitionAttemptRecord).where(
                ImageAcquisitionAttemptRecord.job_item_id == item.id,
                ImageAcquisitionAttemptRecord.status == "running",
            )
        ).all()
        for attempt in attempts:
            attempt.status = status
            attempt.failure_code = code.value if status == "failed" else None
            attempt.retryable = retryable
            attempt.received_bytes = item.received_bytes
            attempt.completed_at = now

    def _recover_importing_item(
        self,
        session: Any,
        job: ImageAcquisitionJobRecord,
        item: ImageAcquisitionJobItemRecord,
        now: datetime,
        part_inspection: _PartPathInspection,
    ) -> bool:
        link = session.scalar(
            select(ExternalImageAssetLinkRecord).where(
                ExternalImageAssetLinkRecord.source_type == item.source_type,
                ExternalImageAssetLinkRecord.external_post_id == item.external_post_id,
            )
        )
        if link is None or link.project_id != job.project_id:
            return False
        asset = session.scalar(
            select(ImageAssetRecord).where(
                ImageAssetRecord.id == link.image_asset_id,
                ImageAssetRecord.project_id == job.project_id,
            )
        )
        if asset is None:
            return False
        root = self.projects.project_root(UUID(job.project_id)).resolve()
        original = Path(asset.original_path)
        thumbnail = Path(asset.thumbnail_path)
        if not original.is_absolute():
            original = root / original
        if not thumbnail.is_absolute():
            thumbnail = root / thumbnail
        if original.is_symlink() or thumbnail.is_symlink():
            return False
        try:
            original = original.resolve()
            thumbnail = thumbnail.resolve()
        except OSError:
            return False
        if (
            not original.is_relative_to(root)
            or not thumbnail.is_relative_to(root)
            or original.is_symlink()
            or thumbnail.is_symlink()
            or not original.is_file()
            or not thumbnail.is_file()
        ):
            return False
        item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        item.part_cleanup_claim_token = None
        item.part_cleanup_claimed_at = None
        self._finalize_stale_attempts(
            session,
            item,
            now,
            DownloadFailureCode.WORKER_CLAIM_LOST,
            retryable=False,
            status="succeeded",
        )
        item.image_asset_id = asset.id
        item.status = (
            ImageAcquisitionItemStatus.IMPORTED.value
            if link.acquisition_job_item_id == item.id
            else ImageAcquisitionItemStatus.LINKED_EXISTING.value
        )
        item.failure_code = None
        item.failure_message = None
        item.retryable = False
        item.received_bytes = 0
        item.etag = None
        item.last_modified = None
        item.accept_ranges = False
        item.range_start = None
        item.completed_at = now
        item.updated_at = now
        return True

    def _project_id(self, job_id: UUID) -> str:
        with self.session_factory() as session:
            value = session.scalar(
                select(ImageAcquisitionJobRecord.project_id).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            if value is None:
                raise _ClaimLost()
            return str(value)

    def _resume_start(
        self, path: Path, item: ImageAcquisitionJobItemRecord, post: ImageSourcePost
    ) -> int | None:
        if not path.is_file() or path.is_symlink():
            return None
        size = path.stat().st_size
        if (
            size <= 0
            or item.received_bytes != size
            or not item.etag
            and not item.last_modified
            or not item.accept_ranges
        ):
            if size and item.received_bytes != size:
                self._truncate_part(path)
            return None
        if post.file_size is not None and size >= post.file_size:
            return None
        return size

    def _validate_range_response(
        self,
        response: DownloadResponse,
        start: int,
        item: ImageAcquisitionJobItemRecord,
    ) -> int | None:
        headers = _normalized_headers(response.headers)
        value = headers.get("content-range")
        match = CONTENT_RANGE_RE.fullmatch(value or "")
        if response.status != 206 or match is None:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        range_start = int(match.group(1))
        range_end = int(match.group(2))
        if range_start != start or range_end < range_start:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        content_length = _parse_content_length(headers.get("content-length"))
        if content_length is None or content_length != range_end - range_start + 1:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        total = match.group(3)
        total_value: int | None = None
        if total != "*":
            total_value = int(total)
            if range_end >= total_value:
                response.close()
                raise AcquisitionDownloadError(
                    DownloadFailureCode.CONTENT_RANGE_INVALID
                )
        if (
            total_value is not None
            and total_value > self.settings.image_download_max_file_size_bytes
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.FILE_TOO_LARGE)
        if (
            total_value is not None
            and item.expected_file_size is not None
            and total_value != item.expected_file_size
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        if total_value is None and item.expected_file_size is not None:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.CONTENT_RANGE_INVALID)
        if item.etag is None and item.last_modified is None:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.RANGE_NOT_SUPPORTED)
        if item.etag is not None and headers.get("etag") != item.etag:
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.REMOTE_FILE_CHANGED)
        if (
            item.last_modified is not None
            and headers.get("last-modified") != item.last_modified
        ):
            response.close()
            raise AcquisitionDownloadError(DownloadFailureCode.REMOTE_FILE_CHANGED)
        return total_value

    def _truncate_part(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb"):
            pass

    def _check_storage(
        self, job_id: UUID, item_id: str, expected_size: int | None
    ) -> None:
        path, _ = self._item_path(job_id, item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        free = int(self.disk_usage(str(path.parent)).free)
        if expected_size is None:
            expected_size = self.settings.image_download_unknown_size_limit_bytes
        if free < expected_size + self.settings.image_download_disk_safety_margin_bytes:
            raise AcquisitionDownloadError(DownloadFailureCode.INSUFFICIENT_STORAGE)

    def _check_job_storage(
        self, project_id: str, items: Iterable[ImageAcquisitionPlanItemRecord]
    ) -> None:
        project_path = self.projects.project_root(UUID(project_id))
        if project_path.is_symlink():
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        project_root = project_path.resolve()
        projects_root = self.settings.projects_dir.resolve()
        if not project_root.is_relative_to(projects_root) or not project_root.is_dir():
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        staging_root = project_root / "acquisition" / "jobs"
        staging_root.mkdir(parents=True, exist_ok=True)
        materialized_items = list(items)
        expected = sum(
            item.expected_file_size
            if item.expected_file_size is not None
            else self.settings.image_download_unknown_size_limit_bytes
            for item in materialized_items
        )
        free = int(self.disk_usage(str(staging_root)).free)
        if free < expected + self.settings.image_download_disk_safety_margin_bytes:
            raise AcquisitionDownloadError(DownloadFailureCode.INSUFFICIENT_STORAGE)

    def _retry_backoff(
        self,
        retry_after: float | None,
        attempt: int,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
    ) -> None:
        delay = (
            retry_after
            if retry_after is not None
            else min(
                self.settings.image_download_retry_max_backoff_seconds,
                self.settings.image_download_retry_base_backoff_seconds
                * (2 ** (attempt - 1)),
            )
        )
        try:
            interruptible_sleep(
                delay,
                cancel_requested=lambda: self._source_cancel_requested(
                    job_id, worker, token, generation
                ),
                sleeper=self.sleeper,
            )
        except DanbooruSourceError as exc:
            if getattr(exc.code, "value", str(exc.code)) == "CANCELED":
                raise _Canceled() from exc
            raise

    def _recompute_counts(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> _AcquisitionCountsSnapshot:
        with self.session_factory() as session:
            items = session.scalars(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id)
                )
            ).all()
            snapshot = self._counts_snapshot(items)
            now = datetime.now(UTC)
            conditions = [
                ImageAcquisitionJobRecord.id == str(job_id),
                ImageAcquisitionJobRecord.status
                == ImageAcquisitionJobStatus.RUNNING.value,
                ImageAcquisitionJobRecord.worker_id == worker,
                ImageAcquisitionJobRecord.claim_token == token,
                ImageAcquisitionJobRecord.worker_generation == generation,
            ]
            values: dict[str, object] = {**snapshot.job_values()}
            values["updated_at"] = now
            values["heartbeat_at"] = now
            result = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(*conditions)
                .values(**values)
                .returning(ImageAcquisitionJobRecord.id)
            ).scalar_one_or_none()
            if result is None:
                session.rollback()
                raise _ClaimLost()
            session.commit()
            return snapshot

    @staticmethod
    def _counts_snapshot(
        items: Iterable[ImageAcquisitionJobItemRecord],
    ) -> _AcquisitionCountsSnapshot:
        materialized = list(items)
        return _AcquisitionCountsSnapshot(
            item_count=len(materialized),
            pending_count=sum(
                item.status
                in {
                    ImageAcquisitionItemStatus.PENDING.value,
                    ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                }
                for item in materialized
            ),
            downloading_count=sum(
                item.status
                in {
                    ImageAcquisitionItemStatus.DOWNLOADING.value,
                    ImageAcquisitionItemStatus.VALIDATING.value,
                    ImageAcquisitionItemStatus.IMPORTING.value,
                }
                for item in materialized
            ),
            downloaded_count=sum(
                item.status
                in {
                    ImageAcquisitionItemStatus.DOWNLOADED.value,
                    ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
                    ImageAcquisitionItemStatus.VALIDATING.value,
                    ImageAcquisitionItemStatus.VALIDATED.value,
                    ImageAcquisitionItemStatus.IMPORTING.value,
                    ImageAcquisitionItemStatus.IMPORTED.value,
                    ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                }
                for item in materialized
            ),
            validated_count=sum(
                item.status
                in {
                    ImageAcquisitionItemStatus.VALIDATED.value,
                    ImageAcquisitionItemStatus.IMPORTING.value,
                    ImageAcquisitionItemStatus.IMPORTED.value,
                    ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                }
                for item in materialized
            ),
            imported_count=sum(
                item.status == ImageAcquisitionItemStatus.IMPORTED.value
                for item in materialized
            ),
            linked_existing_count=sum(
                item.status == ImageAcquisitionItemStatus.LINKED_EXISTING.value
                for item in materialized
            ),
            skipped_count=sum(
                item.status == ImageAcquisitionItemStatus.SKIPPED.value
                for item in materialized
            ),
            failed_count=sum(
                item.status == ImageAcquisitionItemStatus.FAILED.value
                for item in materialized
            ),
            received_bytes=sum(item.received_bytes for item in materialized),
        )

    def _job_terminal_status(
        self,
        job_id: UUID,
        *,
        worker: str | None = None,
        token: str | None = None,
        generation: int | None = None,
    ) -> tuple[ImageAcquisitionJobStatus, DownloadFailureCode | None]:
        if (worker is None) != (token is None) or (
            worker is not None and generation is None
        ):
            raise ValueError("worker, token, and generation must be provided together")
        with self.session_factory() as session:
            job_conditions = [ImageAcquisitionJobRecord.id == str(job_id)]
            if worker is not None:
                job_conditions.extend(
                    [
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    ]
                )
            if (
                session.scalar(
                    select(ImageAcquisitionJobRecord.id).where(*job_conditions)
                )
                is None
            ):
                if worker is not None:
                    raise _ClaimLost()
                return (
                    ImageAcquisitionJobStatus.FAILED,
                    DownloadFailureCode.INCOMPLETE_ITEM_STATE,
                )
            items = session.scalars(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id)
                )
            ).all()
            terminal = {
                ImageAcquisitionItemStatus.IMPORTED.value,
                ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                ImageAcquisitionItemStatus.SKIPPED.value,
                ImageAcquisitionItemStatus.FAILED.value,
                ImageAcquisitionItemStatus.CANCELED.value,
            }
            if any(item.status not in terminal for item in items):
                return (
                    ImageAcquisitionJobStatus.FAILED,
                    DownloadFailureCode.INCOMPLETE_ITEM_STATE,
                )
            if any(
                item.status == ImageAcquisitionItemStatus.FAILED.value
                and item.retryable
                for item in items
            ):
                return (
                    ImageAcquisitionJobStatus.PARTIALLY_COMPLETED
                    if any(
                        item.status
                        in {
                            ImageAcquisitionItemStatus.IMPORTED.value,
                            ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                            ImageAcquisitionItemStatus.SKIPPED.value,
                        }
                        for item in items
                    )
                    else ImageAcquisitionJobStatus.FAILED,
                    None,
                )
            if any(
                item.status
                in {
                    ImageAcquisitionItemStatus.FAILED.value,
                    ImageAcquisitionItemStatus.CANCELED.value,
                }
                for item in items
            ):
                return (
                    ImageAcquisitionJobStatus.PARTIALLY_COMPLETED
                    if any(
                        item.status
                        in {
                            ImageAcquisitionItemStatus.IMPORTED.value,
                            ImageAcquisitionItemStatus.LINKED_EXISTING.value,
                            ImageAcquisitionItemStatus.SKIPPED.value,
                        }
                        for item in items
                    )
                    else ImageAcquisitionJobStatus.FAILED,
                    None,
                )
            return ImageAcquisitionJobStatus.COMPLETED, None

    def _finish_job(
        self,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
        status: ImageAcquisitionJobStatus,
        error: DownloadFailureCode | None,
    ) -> None:
        with self.session_factory() as session:
            completed_at = datetime.now(UTC)
            result = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
                .values(
                    status=status.value,
                    error_code=error.value if error else None,
                    error_summary=error.value if error else None,
                    completed_at=completed_at,
                    heartbeat_at=completed_at,
                    worker_id=None,
                    claim_token=None,
                    current_item_id=None,
                    active_key=None,
                    updated_at=completed_at,
                )
                .returning(ImageAcquisitionJobRecord.id)
            ).scalar_one_or_none()
            session.commit()
            if result is None:
                return

    def _fail_all_unfinished(
        self,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
        code: DownloadFailureCode,
    ) -> None:
        unfinished_statuses = [
            ImageAcquisitionItemStatus.PENDING.value,
            ImageAcquisitionItemStatus.DOWNLOADING.value,
            ImageAcquisitionItemStatus.DOWNLOADED.value,
            ImageAcquisitionItemStatus.VALIDATION_PENDING.value,
            ImageAcquisitionItemStatus.VALIDATING.value,
            ImageAcquisitionItemStatus.VALIDATED.value,
            ImageAcquisitionItemStatus.IMPORTING.value,
        ]
        with self.session_factory() as session:
            claim = [
                ImageAcquisitionJobRecord.id == str(job_id),
                ImageAcquisitionJobRecord.status
                == ImageAcquisitionJobStatus.RUNNING.value,
                ImageAcquisitionJobRecord.worker_id == worker,
                ImageAcquisitionJobRecord.claim_token == token,
                ImageAcquisitionJobRecord.worker_generation == generation,
            ]
            now = datetime.now(UTC)
            project_id = session.scalar(
                select(ImageAcquisitionJobRecord.project_id).where(*claim)
            )
            if project_id is None:
                session.rollback()
                raise _ClaimLost()
            claim_result = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(*claim)
                .values(updated_at=now)
                .returning(ImageAcquisitionJobRecord.id)
            ).scalar_one_or_none()
            if claim_result is None:
                session.rollback()
                raise _ClaimLost()
            items = session.scalars(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id),
                    ImageAcquisitionJobItemRecord.status.in_(unfinished_statuses),
                )
            ).all()
            cleanup_item_ids: list[str] = []
            for item in items:
                cleanup_item_ids.append(item.id)
                self._finalize_stale_attempts(
                    session,
                    item,
                    now,
                    code,
                    retryable=False,
                )
                item.status = ImageAcquisitionItemStatus.FAILED.value
                item.failure_code = code.value
                item.failure_message = code.value
                item.retryable = False
                item.image_asset_id = None
                item.completed_at = now
                item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
                item.part_cleanup_claim_token = None
                item.part_cleanup_claimed_at = None
                item.updated_at = now
            session.commit()
        for item_id in cleanup_item_ids:
            self._cleanup_part_item(
                item_id,
                job_id=job_id,
                worker=worker,
                token=token,
                generation=generation,
            )

    def _manifest_project_root(self, project_id: str) -> Path:
        try:
            project_path = self.projects.project_root(UUID(project_id))
            project_mode = os.lstat(project_path).st_mode
            if stat.S_ISLNK(project_mode) or not stat.S_ISDIR(project_mode):
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            project_root = project_path.resolve(strict=True)
            projects_root = self.settings.projects_dir.resolve(strict=True)
            if not project_root.is_relative_to(projects_root):
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            return project_root
        except AcquisitionDownloadError:
            raise
        except (OSError, ValueError):
            raise AcquisitionDownloadError(
                DownloadFailureCode.STAGING_PATH_INVALID
            ) from None

    @staticmethod
    def _manifest_fd_traversal_supported() -> bool:
        supported: set[Any] = getattr(os, "supports_dir_fd", set())
        return bool(
            os.name != "nt"
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and all(
                operation in supported
                for operation in (os.open, os.mkdir, os.rename, os.unlink, os.stat)
            )
        )

    @staticmethod
    def _open_manifest_directory_fd(
        name: str | Path, *, parent_fd: int | None = None
    ) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            if parent_fd is None:
                descriptor = os.open(name, flags)
            else:
                descriptor = os.open(name, flags, dir_fd=parent_fd)
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            return descriptor
        except AcquisitionDownloadError:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            raise AcquisitionDownloadError(
                DownloadFailureCode.STAGING_PATH_INVALID
            ) from None

    def _manifest_project_parts(self, project_id: str) -> tuple[Path, Path, str]:
        try:
            project_path = self.projects.project_root(UUID(project_id))
            projects_root = self.settings.projects_dir
            relative = project_path.relative_to(projects_root)
            if len(relative.parts) != 1 or relative.name != project_id:
                raise ValueError
            return project_path, projects_root, relative.name
        except (OSError, ValueError):
            raise AcquisitionDownloadError(
                DownloadFailureCode.STAGING_PATH_INVALID
            ) from None

    @staticmethod
    def _manifest_fd_identity(directory_fd: int) -> _ManifestDirectoryIdentity:
        metadata = os.fstat(directory_fd)
        return _ManifestDirectoryIdentity(metadata.st_dev, metadata.st_ino)

    def _open_manifest_directory_components(
        self,
        project_root: Path,
        projects_root: Path,
        component_names: tuple[str, ...],
        manifest_dir: Path,
        *,
        create_missing: bool,
    ) -> _ManifestDirectoryHandle:
        if not self._manifest_fd_traversal_supported():
            if os.name != "nt":
                logger.warning("acquisition_manifest_fd_traversal_unavailable")
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            logger.warning("acquisition_manifest_fd_traversal_unavailable")
            self._validate_manifest_directory(
                project_root,
                manifest_dir,
                create_missing=create_missing,
                allow_missing=not create_missing,
            )
            return _ManifestDirectoryHandle(
                project_root,
                manifest_dir,
                -1,
                projects_root,
                component_names,
                (),
            )

        projects_fd = -1
        current_fd = -1
        identities: list[_ManifestDirectoryIdentity] = []
        try:
            projects_fd = self._open_manifest_directory_fd(projects_root)
            current_fd = self._open_manifest_directory_fd(
                component_names[0], parent_fd=projects_fd
            )
            identities.append(self._manifest_fd_identity(current_fd))
            os.close(projects_fd)
            projects_fd = -1
            for component in component_names[1:]:
                created = False
                try:
                    child_fd = self._open_manifest_directory_fd(
                        component, parent_fd=current_fd
                    )
                except AcquisitionDownloadError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                    child_fd = self._open_manifest_directory_fd(
                        component, parent_fd=current_fd
                    )
                    created = True
                if created:
                    os.fsync(current_fd)
                identities.append(self._manifest_fd_identity(child_fd))
                os.close(current_fd)
                current_fd = child_fd
            return _ManifestDirectoryHandle(
                project_root,
                manifest_dir,
                current_fd,
                projects_root,
                component_names,
                tuple(identities),
            )
        except Exception:
            if current_fd >= 0:
                os.close(current_fd)
            if projects_fd >= 0:
                os.close(projects_fd)
            raise

    def _open_manifest_directory(
        self, project_id: str, job_id: str, *, create_missing: bool
    ) -> _ManifestDirectoryHandle:
        project_root, projects_root, project_name = self._manifest_project_parts(
            project_id
        )
        manifest_dir = project_root / "acquisition" / "jobs" / job_id / "manifests"
        return self._open_manifest_directory_components(
            project_root,
            projects_root,
            (project_name, "acquisition", "jobs", job_id, "manifests"),
            manifest_dir,
            create_missing=create_missing,
        )

    def _manifest_directory_identity_matches(
        self, directory: _ManifestDirectoryHandle
    ) -> bool:
        if directory.fd < 0:
            return True
        try:
            reopened = self._open_manifest_directory_components(
                directory.project_root,
                directory.projects_root,
                directory.component_names,
                directory.manifest_dir,
                create_missing=False,
            )
        except (AcquisitionDownloadError, OSError, ValueError):
            return False
        try:
            return reopened.identities == directory.identities
        finally:
            reopened.close()

    def _manifest_artifact_matches_handle(
        self, directory: _ManifestDirectoryHandle, name: str
    ) -> bool:
        if directory.fd < 0:
            return True
        try:
            held_identity = self._manifest_artifact_identity_fd(name, directory.fd)
            reopened = self._open_manifest_directory_components(
                directory.project_root,
                directory.projects_root,
                directory.component_names,
                directory.manifest_dir,
                create_missing=False,
            )
        except (AcquisitionDownloadError, OSError, ValueError):
            return False
        try:
            if reopened.identities != directory.identities:
                return False
            current_identity = self._manifest_artifact_identity_fd(name, reopened.fd)
            return current_identity == held_identity
        except (AcquisitionDownloadError, OSError, ValueError):
            return False
        finally:
            reopened.close()

    @staticmethod
    def _validate_manifest_directory(
        project_root: Path,
        manifest_dir: Path,
        *,
        create_missing: bool,
        allow_missing: bool = False,
    ) -> None:
        try:
            relative = manifest_dir.relative_to(project_root)
        except ValueError:
            raise AcquisitionDownloadError(
                DownloadFailureCode.STAGING_PATH_INVALID
            ) from None
        current = project_root
        components = (project_root, *relative.parts)
        for index, component in enumerate(components):
            if index > 0:
                current = current / str(component)
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError:
                if allow_missing and not create_missing:
                    return
                if not create_missing:
                    raise AcquisitionDownloadError(
                        DownloadFailureCode.STAGING_PATH_INVALID
                    ) from None
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
                try:
                    mode = os.lstat(current).st_mode
                except OSError:
                    raise AcquisitionDownloadError(
                        DownloadFailureCode.STAGING_PATH_INVALID
                    ) from None
            except OSError:
                raise AcquisitionDownloadError(
                    DownloadFailureCode.STAGING_PATH_INVALID
                ) from None
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)

    @staticmethod
    def _validate_manifest_file(path: Path, *, regular: bool) -> None:
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            raise OSError("manifest artifact is missing") from None
        except OSError:
            raise OSError("manifest artifact cannot be inspected") from None
        if stat.S_ISLNK(mode):
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        if regular and not stat.S_ISREG(mode):
            raise OSError("manifest artifact is not a regular file")

    def _manifest_paths(
        self, project_id: str, job_id: str, generation: int
    ) -> tuple[Path, Path, Path]:
        project_root = self._manifest_project_root(project_id)
        try:
            if str(UUID(job_id)) != job_id:
                raise ValueError
        except ValueError:
            raise AcquisitionDownloadError(
                DownloadFailureCode.STAGING_PATH_INVALID
            ) from None
        manifest_dir = project_root / "acquisition" / "jobs" / job_id / "manifests"
        self._validate_manifest_directory(
            project_root, manifest_dir, create_missing=False, allow_missing=True
        )
        final_name = f"manifest-g{generation}-{uuid4().hex[:12]}.json"
        final = manifest_dir / final_name
        temporary = manifest_dir / f".{final_name}.tmp"
        for path in (temporary, final):
            try:
                os.lstat(path)
            except FileNotFoundError:
                continue
            except OSError:
                raise OSError("manifest artifact cannot be inspected") from None
            raise OSError("manifest path already exists")
        return manifest_dir, temporary, final

    def _open_manifest_temporary(
        self, path: Path, *, directory_fd: int | None = None
    ) -> Any:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory_fd is not None and directory_fd >= 0:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
            return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        if len(path.parents) < 5:
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        self._validate_manifest_directory(
            path.parents[4], path.parent, create_missing=False
        )
        if os.open in getattr(os, "supports_dir_fd", set()):
            directory_flags = os.O_RDONLY
            directory_flags |= getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(path.parent, directory_flags)
            try:
                descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
        else:
            descriptor = os.open(path, flags, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")

    def _atomic_replace_manifest(
        self,
        temporary: Path,
        final: Path,
        *,
        directory_fd: int | None = None,
    ) -> None:
        if directory_fd is not None and directory_fd >= 0:
            try:
                temporary_mode = os.stat(
                    temporary.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                ).st_mode
                if not stat.S_ISREG(temporary_mode):
                    raise OSError("manifest temporary is not a regular file")
                os.stat(final.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
            os.rename(
                temporary.name,
                final.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            return
        if len(temporary.parents) < 5:
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        self._validate_manifest_directory(
            temporary.parents[4], temporary.parent, create_missing=False
        )
        self._validate_manifest_file(temporary, regular=True)
        try:
            os.lstat(final)
        except FileNotFoundError:
            pass
        else:
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        supports_dir_fd: set[Any] = getattr(os, "supports_dir_fd", set())
        replace_function = (
            os.replace
            if os.replace in supports_dir_fd
            else os.rename
            if os.rename in supports_dir_fd
            else None
        )
        if replace_function is not None:
            flags = os.O_RDONLY
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(temporary.parent, flags)
            try:
                replace_function(
                    temporary.name,
                    final.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                os.close(directory_fd)
            return
        os.replace(temporary, final)

    @staticmethod
    def _manifest_artifact_identity_fd(
        name: str, directory_fd: int
    ) -> _ManifestDirectoryIdentity:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise OSError("manifest artifact is missing") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("manifest artifact is not a regular file")
        return _ManifestDirectoryIdentity(metadata.st_dev, metadata.st_ino)

    @classmethod
    def _validate_manifest_artifact_fd(cls, name: str, directory_fd: int) -> None:
        cls._manifest_artifact_identity_fd(name, directory_fd)

    @staticmethod
    def _fsync_manifest_directory(path: Path, directory_fd: int | None = None) -> None:
        if directory_fd is not None and directory_fd >= 0:
            os.fsync(directory_fd)
            return
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(path, flags)
        except OSError:
            if os.name == "nt":
                logger.warning("acquisition_manifest_directory_fsync_unavailable")
                return
            raise
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _manifest_claim_exists(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> bool:
        with self.session_factory() as session:
            return bool(
                session.scalar(
                    select(ImageAcquisitionJobRecord.id).where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                    )
                )
                is not None
            )

    def _manifest_file_from_relative_path(
        self, project_id: str, relative_path: str
    ) -> tuple[Path, Path, Path]:
        project_root = self._manifest_project_root(project_id)
        relative = Path(relative_path)
        parts = relative.parts
        if (
            relative.is_absolute()
            or len(parts) != 5
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0:2] != ("acquisition", "jobs")
            or parts[4].startswith(".")
            or not re.fullmatch(r"manifest-g[0-9]+-[0-9a-f]{12}\.json", parts[4])
        ):
            raise AcquisitionDownloadError(DownloadFailureCode.STAGING_PATH_INVALID)
        manifest_dir = project_root / Path(*parts[:4])
        manifest_file = project_root / relative
        self._validate_manifest_directory(
            project_root, manifest_dir, create_missing=False
        )
        self._validate_manifest_file(manifest_file, regular=True)
        return project_root, manifest_dir, manifest_file

    def _manifest_db_references(self, job_id: UUID, relative_path: str) -> bool:
        with self.session_factory() as session:
            return bool(
                session.scalar(
                    select(ImageAcquisitionJobRecord.manifest_relative_path).where(
                        ImageAcquisitionJobRecord.id == str(job_id)
                    )
                )
                == relative_path
            )

    def _cleanup_manifest_artifact(
        self, path: Path, *, directory_fd: int | None = None
    ) -> None:
        try:
            if not re.fullmatch(
                r"(?:manifest-g[0-9]+-[0-9a-f]{12}\.json|\.manifest-g[0-9]+-[0-9a-f]{12}\.json\.tmp)",
                path.name,
            ):
                return
            if directory_fd is not None and directory_fd >= 0:
                try:
                    mode = os.stat(
                        path.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    ).st_mode
                except FileNotFoundError:
                    return
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    return
                os.unlink(path.name, dir_fd=directory_fd)
                return
            if len(path.parents) < 5:
                return
            project_root = path.parents[4]
            projects_root = self.settings.projects_dir.resolve(strict=True)
            root_mode = os.lstat(project_root).st_mode
            if (
                stat.S_ISLNK(root_mode)
                or not stat.S_ISDIR(root_mode)
                or not project_root.resolve(strict=True).is_relative_to(projects_root)
            ):
                return
            self._validate_manifest_directory(
                project_root, path.parent, create_missing=False
            )
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                return
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                return
            path.unlink()
        except (AcquisitionDownloadError, OSError, ValueError):
            logger.warning("acquisition_manifest_cleanup_failed error_type=OSError")

    def _record_manifest_warning(
        self, job_id: UUID, worker: str, token: str, generation: int
    ) -> None:
        with self.session_factory() as session:
            result = session.execute(
                update(ImageAcquisitionJobRecord)
                .where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
                .values(
                    manifest_warning="MANIFEST_WRITE_FAILED",
                    updated_at=datetime.now(UTC),
                )
                .returning(ImageAcquisitionJobRecord.id)
            ).scalar_one_or_none()
            if result is None:
                session.rollback()
                raise _ClaimLost()
            session.commit()

    def _clear_manifest_reference_after_commit(
        self,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
        relative_path: str,
    ) -> None:
        try:
            with self.session_factory() as session:
                session.execute(
                    update(ImageAcquisitionJobRecord)
                    .where(
                        ImageAcquisitionJobRecord.id == str(job_id),
                        ImageAcquisitionJobRecord.status
                        == ImageAcquisitionJobStatus.RUNNING.value,
                        ImageAcquisitionJobRecord.worker_id == worker,
                        ImageAcquisitionJobRecord.claim_token == token,
                        ImageAcquisitionJobRecord.worker_generation == generation,
                        ImageAcquisitionJobRecord.manifest_relative_path
                        == relative_path,
                    )
                    .values(
                        manifest_relative_path=None,
                        manifest_warning="MANIFEST_WRITE_FAILED",
                        updated_at=datetime.now(UTC),
                    )
                )
                session.commit()
        except SQLAlchemyError:
            logger.warning("acquisition_manifest_reference_clear_failed")

    def _write_manifest(
        self,
        job_id: UUID,
        worker: str,
        token: str,
        generation: int,
        final_status: ImageAcquisitionJobStatus,
        *,
        counts: _AcquisitionCountsSnapshot | None = None,
    ) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id),
                    ImageAcquisitionJobRecord.status
                    == ImageAcquisitionJobStatus.RUNNING.value,
                    ImageAcquisitionJobRecord.worker_id == worker,
                    ImageAcquisitionJobRecord.claim_token == token,
                    ImageAcquisitionJobRecord.worker_generation == generation,
                )
            )
            if job is None:
                raise _ClaimLost()
            items = session.scalars(
                select(ImageAcquisitionJobItemRecord)
                .where(ImageAcquisitionJobItemRecord.job_id == job.id)
                .order_by(ImageAcquisitionJobItemRecord.display_order)
            ).all()
            current_counts = self._counts_snapshot(items)
            if counts is None:
                counts = current_counts
            elif counts != current_counts:
                raise _ClaimLost()
            completed_at = datetime.now(UTC)
            manifest = {
                "schema_version": "phase8b-manifest-v1",
                "job_id": job.id,
                "plan_id": job.plan_id,
                "project_id": job.project_id,
                "source_type": job.source_type,
                "plan_fingerprint": job.plan_fingerprint,
                "downloader_version": job.downloader_version,
                "validator_version": job.validator_version,
                "importer_version": job.importer_version,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": completed_at.isoformat(),
                "status": final_status.value,
                "item_count": counts.item_count,
                "pending_count": counts.pending_count,
                "downloading_count": counts.downloading_count,
                "downloaded_count": counts.downloaded_count,
                "validated_count": counts.validated_count,
                "imported_count": counts.imported_count,
                "linked_existing_count": counts.linked_existing_count,
                "skipped_count": counts.skipped_count,
                "failed_count": counts.failed_count,
                "received_bytes": counts.received_bytes,
                "part_cleanup_warning_codes": sorted(
                    {
                        item.part_cleanup_warning
                        for item in items
                        if item.part_cleanup_warning
                    }
                ),
                "items": [
                    {
                        "item_id": item.id,
                        "external_post_id": item.external_post_id,
                        "status": item.status,
                        "attempt_count": item.attempt_count,
                        "file_size": item.detected_file_size,
                        "sha256": item.calculated_sha256,
                        "image_asset_id": item.image_asset_id,
                        "source_md5_match": (
                            item.calculated_md5.lower() == item.expected_md5.lower()
                            if item.calculated_md5 and item.expected_md5
                            else None
                        ),
                        "format": item.detected_format,
                        "width": item.detected_width,
                        "height": item.detected_height,
                        "failure_code": item.failure_code,
                        "part_cleanup_warning": item.part_cleanup_warning,
                    }
                    for item in items
                ],
            }
            project_id = job.project_id
            manifest_job_id = job.id

        temporary: Path | None = None
        final: Path | None = None
        operation = "path"
        try:
            project_root = self._manifest_project_root(project_id)
            _, temporary_candidate, final_candidate = self._manifest_paths(
                project_id, manifest_job_id, generation
            )
            with self._open_manifest_directory(
                project_id, manifest_job_id, create_missing=True
            ) as directory:
                manifest_dir = directory.manifest_dir
                temporary = manifest_dir / temporary_candidate.name
                final = manifest_dir / final_candidate.name
                final_written = False
                operation = "write"
                try:
                    if directory.fd >= 0:
                        with self._open_manifest_temporary(
                            temporary, directory_fd=directory.fd
                        ) as handle:
                            json.dump(
                                manifest,
                                handle,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            handle.flush()
                            os.fsync(handle.fileno())
                    else:
                        with self._open_manifest_temporary(temporary) as handle:
                            json.dump(
                                manifest,
                                handle,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            handle.flush()
                            os.fsync(handle.fileno())
                    operation = "recheck_claim"
                    if not self._manifest_claim_exists(
                        job_id, worker, token, generation
                    ):
                        raise _ClaimLost()
                    operation = "replace"
                    if directory.fd >= 0:
                        self._atomic_replace_manifest(
                            temporary, final, directory_fd=directory.fd
                        )
                    else:
                        self._atomic_replace_manifest(temporary, final)
                    final_written = True
                    operation = "directory_fsync"
                    if directory.fd >= 0:
                        self._fsync_manifest_directory(manifest_dir, directory.fd)
                    else:
                        self._fsync_manifest_directory(manifest_dir)
                    operation = "validate"
                    if directory.fd >= 0:
                        self._validate_manifest_artifact_fd(final.name, directory.fd)
                    else:
                        self._validate_manifest_directory(
                            project_root, manifest_dir, create_missing=False
                        )
                        self._validate_manifest_file(final, regular=True)
                    operation = "path_revalidate"
                    self._validate_manifest_directory(
                        project_root, manifest_dir, create_missing=False
                    )
                    operation = "identity_revalidate"
                    if not self._manifest_directory_identity_matches(directory):
                        raise AcquisitionDownloadError(
                            DownloadFailureCode.STAGING_PATH_INVALID
                        )
                except _ClaimLost:
                    self._cleanup_manifest_artifact(
                        temporary, directory_fd=directory.fd
                    )
                    if final_written:
                        self._cleanup_manifest_artifact(
                            final, directory_fd=directory.fd
                        )
                    raise
                except (AcquisitionDownloadError, OSError, ValueError) as exc:
                    self._cleanup_manifest_artifact(
                        temporary, directory_fd=directory.fd
                    )
                    if final_written:
                        self._cleanup_manifest_artifact(
                            final, directory_fd=directory.fd
                        )
                    self._record_manifest_warning(job_id, worker, token, generation)
                    logger.warning(
                        "acquisition_manifest_write_failed "
                        "job_id=%s error_type=%s operation=%s",
                        manifest_job_id,
                        type(exc).__name__,
                        operation,
                    )
                    return

                relative_path = (
                    f"acquisition/jobs/{manifest_job_id}/manifests/{final.name}"
                )
                db_references_own_file = False
                db_committed = False
                try:
                    with self.session_factory() as session:
                        result = session.execute(
                            update(ImageAcquisitionJobRecord)
                            .where(
                                ImageAcquisitionJobRecord.id == str(job_id),
                                ImageAcquisitionJobRecord.status
                                == ImageAcquisitionJobStatus.RUNNING.value,
                                ImageAcquisitionJobRecord.worker_id == worker,
                                ImageAcquisitionJobRecord.claim_token == token,
                                ImageAcquisitionJobRecord.worker_generation
                                == generation,
                            )
                            .values(
                                manifest_relative_path=relative_path,
                                manifest_warning=None,
                                updated_at=datetime.now(UTC),
                            )
                            .returning(ImageAcquisitionJobRecord.id)
                        ).scalar_one_or_none()
                        if result is None:
                            session.rollback()
                            self._cleanup_manifest_artifact(
                                final, directory_fd=directory.fd
                            )
                            raise _ClaimLost()
                        try:
                            session.commit()
                            db_committed = True
                        except BaseException:
                            session.rollback()
                            db_references_own_file = self._manifest_db_references(
                                job_id, relative_path
                            )
                            if not db_references_own_file:
                                self._cleanup_manifest_artifact(
                                    final, directory_fd=directory.fd
                                )
                            raise
                except _ClaimLost:
                    self._cleanup_manifest_artifact(final, directory_fd=directory.fd)
                    raise
                except BaseException:
                    if not db_references_own_file and not db_committed:
                        self._cleanup_manifest_artifact(
                            final, directory_fd=directory.fd
                        )
                    raise
                operation = "post_commit_verify"
                if directory.fd >= 0:
                    if not self._manifest_artifact_matches_handle(
                        directory, final.name
                    ):
                        self._clear_manifest_reference_after_commit(
                            job_id,
                            worker,
                            token,
                            generation,
                            relative_path,
                        )
                        self._cleanup_manifest_artifact(
                            final, directory_fd=directory.fd
                        )
                        logger.warning(
                            "acquisition_manifest_write_failed "
                            "job_id=%s error_type=AcquisitionDownloadError "
                            "operation=%s",
                            manifest_job_id,
                            operation,
                        )
                        return
                else:
                    self._manifest_file_from_relative_path(project_id, relative_path)
        except _ClaimLost:
            raise
        except (AcquisitionDownloadError, OSError, ValueError) as exc:
            self._record_manifest_warning(job_id, worker, token, generation)
            logger.warning(
                "acquisition_manifest_write_failed "
                "job_id=%s error_type=%s operation=%s",
                manifest_job_id,
                type(exc).__name__,
                operation,
            )
            return


def _job_view(job: ImageAcquisitionJobRecord) -> AcquisitionJobView:
    return AcquisitionJobView(
        UUID(job.id),
        UUID(job.plan_id),
        UUID(job.project_id),
        ImageAcquisitionJobStatus(job.status),
        job.requested_count,
        job.pending_count,
        job.downloading_count,
        job.downloaded_count,
        job.validated_count,
        job.imported_count,
        job.linked_existing_count,
        job.skipped_count,
        job.failed_count,
        job.received_bytes,
        job.expected_bytes,
        UUID(job.current_item_id) if job.current_item_id else None,
        job.error_code,
        job.started_at,
        job.completed_at,
    )


def _item_view(item: ImageAcquisitionJobItemRecord) -> AcquisitionItemView:
    return AcquisitionItemView(
        UUID(item.id),
        item.external_post_id,
        ImageAcquisitionItemStatus(item.status),
        item.attempt_count,
        item.received_bytes,
        item.expected_file_size,
        item.detected_format,
        item.detected_width,
        item.detected_height,
        item.calculated_sha256[:12] if item.calculated_sha256 else None,
        item.failure_code,
        bool(item.retryable),
        item.part_cleanup_warning,
    )


def _post_metadata_fingerprint(post: ImageSourcePost) -> str:
    return fingerprint(
        {
            "source_type": post.source_type.value,
            "external_post_id": post.external_post_id,
            "file_url": post.file_url,
            "preview_url": post.preview_url,
            "sample_url": post.sample_url,
            "width": post.width,
            "height": post.height,
            "file_size": post.file_size,
            "file_extension": post.file_extension,
            "rating": post.rating.value if post.rating else None,
            "score": post.score,
            "tag_names": post.tag_names,
            "source_md5": post.source_md5,
            "is_deleted": post.is_deleted,
            "is_pending": post.is_pending,
            "is_flagged": post.is_flagged,
        }
    )


def _normalized_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _status_failure(status: int) -> DownloadFailureCode:
    mapped = {
        400: DownloadFailureCode.HTTP_CLIENT_ERROR,
        401: DownloadFailureCode.AUTHENTICATION_FAILED,
        403: DownloadFailureCode.PERMISSION_DENIED,
        404: DownloadFailureCode.SOURCE_POST_NOT_FOUND,
        408: DownloadFailureCode.REQUEST_TIMEOUT,
        429: DownloadFailureCode.RATE_LIMITED,
    }.get(status)
    if mapped is not None:
        return mapped
    if 400 <= status < 500:
        return DownloadFailureCode.HTTP_CLIENT_ERROR
    return {
        500: DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
        501: DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
        502: DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
        503: DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
        504: DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
    }.get(status, DownloadFailureCode.CONNECTION_FAILED)


def _source_failure(error: DanbooruSourceError) -> DownloadFailureCode:
    if error.status is not None:
        status_code = {
            400: DownloadFailureCode.HTTP_CLIENT_ERROR,
            401: DownloadFailureCode.AUTHENTICATION_FAILED,
            403: DownloadFailureCode.PERMISSION_DENIED,
            404: DownloadFailureCode.SOURCE_POST_NOT_FOUND,
            408: DownloadFailureCode.REQUEST_TIMEOUT,
            429: DownloadFailureCode.RATE_LIMITED,
        }.get(error.status)
        if status_code is not None:
            return status_code
        if 400 <= error.status < 500:
            return DownloadFailureCode.HTTP_CLIENT_ERROR
    return {
        "AUTHENTICATION_FAILED": DownloadFailureCode.AUTHENTICATION_FAILED,
        "PERMISSION_DENIED": DownloadFailureCode.PERMISSION_DENIED,
        "RATE_LIMITED": DownloadFailureCode.RATE_LIMITED,
        "REQUEST_TIMEOUT": DownloadFailureCode.REQUEST_TIMEOUT,
        "CONNECTION_FAILED": DownloadFailureCode.CONNECTION_FAILED,
        "SOURCE_UNAVAILABLE": DownloadFailureCode.SOURCE_POST_UNAVAILABLE,
    }.get(error.code.value, DownloadFailureCode.SOURCE_POST_UNAVAILABLE)


def _failure_code(value: str) -> DownloadFailureCode:
    try:
        return DownloadFailureCode(value)
    except ValueError:
        return DownloadFailureCode.IMAGE_CORRUPTED
