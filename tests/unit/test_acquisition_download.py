from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.acquisition_download_models import (
    DownloadFailureCode,
    ImageAcquisitionItemStatus,
    ImageAcquisitionJobStatus,
    PartCleanupWarningCode,
)
from runpod_lora_studio.domain.acquisition_models import (
    AcquisitionErrorCode,
    DanbooruSearchCriteria,
    ImageRating,
    ImageSearchPage,
    ImageSearchSort,
    ImageSourcePost,
    ImageSourceType,
)
from runpod_lora_studio.external.image_download import (
    FakeDownloadSpec,
    FakeDownloadTransport,
    validate_download_url,
)
from runpod_lora_studio.external.image_sources import (
    DanbooruApiClient,
    DanbooruImageSourceAdapter,
    DanbooruSourceError,
    FakeImageSourceAdapter,
    HttpResponse,
    ImageSourceRequestContext,
    SourceRateLimiter,
)
from runpod_lora_studio.persistence.database import (
    create_engine_for_settings,
    create_session_factory,
)
from runpod_lora_studio.persistence.models import (
    Base,
    ImageAcquisitionAttemptRecord,
    ImageAcquisitionJobItemRecord,
    ImageAcquisitionJobRecord,
)
from runpod_lora_studio.services import (
    acquisition_download_service as acquisition_download_module,
)
from runpod_lora_studio.services.acquisition_download_service import (
    AcquisitionDownloadError,
    ImageAcquisitionDownloadService,
    _ClaimLost,
)
from runpod_lora_studio.services.acquisition_service import ImageAcquisitionService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService


def _png_bytes(color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    output = io.BytesIO()
    with Image.new("RGB", (32, 24), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def _post(external_id: str, body: bytes, *, url_suffix: str = "") -> ImageSourcePost:
    return ImageSourcePost(
        source_type=ImageSourceType.DANBOORU,
        external_post_id=external_id,
        post_url=f"https://danbooru.donmai.us/posts/{external_id}",
        file_url=f"https://cdn.donmai.us/original/{external_id}{url_suffix}.png",
        preview_url=None,
        sample_url=None,
        width=32,
        height=24,
        file_size=len(body),
        file_extension=".png",
        rating=ImageRating.GENERAL,
        score=10,
        tag_names=("solo",),
        source_md5=hashlib.md5(body, usedforsecurity=False).hexdigest(),
        created_at=None,
        is_deleted=False,
        is_pending=False,
        is_flagged=False,
        source_metadata={"tag_names": ["solo"]},
    )


def _settings(
    workspace: Path,
    *,
    retry_base: float = 0.0,
    retry_max_attempts: int = 4,
    cleanup_retry_base: float = 30.0,
) -> AppSettings:
    settings = AppSettings(
        workspace_root=workspace / "runtime",
        projects_dir=workspace / "runtime" / "projects",
        models_dir=workspace / "runtime" / "models",
        outputs_dir=workspace / "runtime" / "outputs",
        logs_dir=workspace / "runtime" / "logs",
        temp_dir=workspace / "runtime" / "tmp",
        database_path=workspace / "runtime" / "database" / "studio.sqlite3",
        image_download_disk_safety_margin_bytes=0,
        image_download_retry_max_attempts=retry_max_attempts,
        image_download_retry_base_backoff_seconds=retry_base,
        image_download_retry_max_backoff_seconds=retry_base,
        image_download_cleanup_retry_base_backoff_seconds=cleanup_retry_base,
        image_download_cleanup_retry_max_backoff_seconds=cleanup_retry_base,
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    return settings


class _MutablePostAdapter(FakeImageSourceAdapter):
    def __init__(self, pages: dict[str | None, ImageSearchPage]) -> None:
        super().__init__(pages)
        self.missing_ids: set[str] = set()
        self.post_failures: dict[str, DanbooruSourceError] = {}
        self.post_overrides: dict[str, ImageSourcePost] = {}
        self.get_post_calls = 0

    def get_post(
        self,
        external_post_id: str,
        *,
        context: ImageSourceRequestContext | None = None,
    ) -> ImageSourcePost | None:
        self.get_post_calls += 1
        if external_post_id in self.post_failures:
            raise self.post_failures[external_post_id]
        if external_post_id in self.missing_ids:
            return None
        if external_post_id in self.post_overrides:
            return self.post_overrides[external_post_id]
        return super().get_post(external_post_id, context=context)


def _make_plan(
    settings: AppSettings,
    posts: tuple[ImageSourcePost, ...],
    adapter: FakeImageSourceAdapter | None = None,
) -> tuple[UUID, FakeImageSourceAdapter]:
    project = ProjectService(settings).create(ProjectInput(name="download-test"))
    adapter = adapter or FakeImageSourceAdapter(
        {None: ImageSearchPage(posts=posts, next_cursor=None)}
    )
    acquisition = ImageAcquisitionService(settings, adapter=adapter)
    search_id = acquisition.start_search(
        DanbooruSearchCriteria(
            project_id=project.id,
            include_tags=("solo",),
            exclude_tags=(),
            ratings=(ImageRating.GENERAL,),
            required_extensions=(".png",),
            maximum_candidate_count=len(posts),
            page_size=len(posts),
            sort_rule=ImageSearchSort.ID,
        )
    )
    for _ in range(100):
        current = acquisition.get_search(search_id)
        if current and current.status == "completed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("metadata search worker did not complete")
    candidates = acquisition.list_candidates(search_id)
    plan = acquisition.preview_plan(
        search_id, [candidate.result_id for candidate in candidates]
    )
    return acquisition.confirm_plan(plan), adapter


def test_download_validates_imports_and_writes_safe_manifest(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes()
    post = _post("1001", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {post.file_url or "": FakeDownloadSpec(body, headers={"etag": "v1"})}
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )

    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    assert job.imported_count == 1
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.IMPORTED
    assert item.sha256_prefix
    assert transport.requests[0].range_start is None

    project_root = settings.projects_dir / str(job.project_id)
    original_files = list((project_root / "originals").glob("*.png"))
    assert len(original_files) == 1
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None
        manifest = project_root / stored_job.manifest_relative_path
        assert manifest.is_file()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["status"] == "completed"
    assert "file_url" not in json.dumps(manifest_data)
    assert "path" not in json.dumps(manifest_data)


def test_normal_import_cleanup_failure_is_audited(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((10, 140, 220))
    post = _post("1001-cleanup", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {post.file_url or "": FakeDownloadSpec(body, headers={"etag": "v1"})}
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    monkeypatch.setattr(
        service,
        "_cleanup_part_artifact",
        lambda _: PartCleanupWarningCode.CLEANUP_FAILED,
    )

    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.IMPORTED
    assert item.part_cleanup_warning == PartCleanupWarningCode.CLEANUP_FAILED.value
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert stored_job is not None
        assert attempt is not None and attempt.status == "succeeded"
        assert stored_job.manifest_relative_path is not None
        manifest_path = (
            settings.projects_dir
            / stored_job.project_id
            / stored_job.manifest_relative_path
        )
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["part_cleanup_warning_codes"] == [
        PartCleanupWarningCode.CLEANUP_FAILED.value
    ]
    assert manifest_data["items"][0]["part_cleanup_warning"] == (
        PartCleanupWarningCode.CLEANUP_FAILED.value
    )


def test_download_resumes_partial_stream_and_links_same_sha(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((100, 20, 60))
    first = _post("1002", body)
    second = _post("1003", body, url_suffix="-same")
    plan_id, adapter = _make_plan(settings, (first, second))
    transport = FakeDownloadTransport(
        {
            first.file_url or "": [
                FakeDownloadSpec(
                    body,
                    chunk_size=8,
                    fail_after_chunks=1,
                    headers={"etag": "same", "accept-ranges": "bytes"},
                ),
                FakeDownloadSpec(body, headers={"etag": "same"}),
            ],
            second.file_url or "": FakeDownloadSpec(body),
        }
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    statuses = [item.status for item in service.list_items(job_id)]
    assert statuses == [
        ImageAcquisitionItemStatus.IMPORTED,
        ImageAcquisitionItemStatus.LINKED_EXISTING,
    ]
    assert any(request.range_start is not None for request in transport.requests), (
        transport.requests
    )


def test_download_records_verification_failure_without_importing(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    expected = _png_bytes()
    post = _post("1004", expected)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {post.file_url or "": FakeDownloadSpec(b"not-an-image")}
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.FAILED
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.FAILED
    assert item.failure_code == DownloadFailureCode.RECEIVED_SIZE_MISMATCH.value


def test_source_revalidation_is_item_scoped_and_allows_partial_success(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_max_attempts=2)
    bodies = tuple(
        _png_bytes(((index * 31) % 256, (index * 53) % 256, (index * 79) % 256))
        for index in range(1, 11)
    )
    posts = tuple(_post(str(1100 + index), body) for index, body in enumerate(bodies))
    adapter = _MutablePostAdapter(
        {None: ImageSearchPage(posts=posts, next_cursor=None)}
    )
    plan_id, adapter = _make_plan(settings, posts, adapter)
    transport = FakeDownloadTransport(
        {
            post.file_url or "": FakeDownloadSpec(body)
            for post, body in zip(posts, bodies, strict=True)
        }
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    adapter.missing_ids.add(posts[1].external_post_id)
    adapter.post_overrides[posts[2].external_post_id] = replace(posts[2], width=31)
    adapter.post_overrides[posts[3].external_post_id] = replace(
        posts[3], source_md5="not-an-md5"
    )
    adapter.post_failures[posts[4].external_post_id] = DanbooruSourceError(
        AcquisitionErrorCode.REQUEST_TIMEOUT,
        status=503,
    )
    job_id = service.start_job(plan_id, auto_start=False)
    assert adapter.get_post_calls == 0

    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.PARTIALLY_COMPLETED
    assert job.imported_count == 6
    assert job.failed_count == 4
    items = service.list_items(job_id)
    failures = {item.external_post_id: item.failure_code for item in items}
    assert failures[posts[1].external_post_id] == (
        DownloadFailureCode.SOURCE_POST_NOT_FOUND.value
    )
    assert failures[posts[2].external_post_id] == (
        DownloadFailureCode.SOURCE_METADATA_CHANGED.value
    )
    assert failures[posts[3].external_post_id] == (
        DownloadFailureCode.SOURCE_MD5_INVALID.value
    )
    assert failures[posts[4].external_post_id] == (
        DownloadFailureCode.RETRY_EXHAUSTED.value
    )
    assert adapter.get_post_calls == len(posts) + 1
    assert (
        sum(item.status is ImageAcquisitionItemStatus.IMPORTED for item in items) == 6
    )


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, DownloadFailureCode.HTTP_CLIENT_ERROR),
        (401, DownloadFailureCode.AUTHENTICATION_FAILED),
        (403, DownloadFailureCode.PERMISSION_DENIED),
        (404, DownloadFailureCode.SOURCE_POST_NOT_FOUND),
    ],
)
def test_http_client_errors_are_recorded_without_retry(
    test_workspace: Path,
    status: int,
    expected_code: DownloadFailureCode,
) -> None:
    settings = _settings(test_workspace, retry_max_attempts=4)
    body = _png_bytes((10, 140, 220))
    post = _post("1201", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {post.file_url or "": FakeDownloadSpec(body, status=status)}
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.FAILED
    assert item.failure_code == expected_code.value
    assert item.attempt_count == 1
    assert len(transport.requests) == 1
    with create_session_factory(settings)() as session:
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert attempt is not None
        assert attempt.attempt_number == 1
        assert attempt.http_status == status
        assert not attempt.retryable


def test_http_503_retries_and_records_each_attempt(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_max_attempts=2)
    body = _png_bytes((10, 150, 220))
    post = _post("1202", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {
            post.file_url or "": [
                FakeDownloadSpec(body, status=503),
                FakeDownloadSpec(body),
            ]
        }
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    item = service.list_items(job_id)[0]
    assert item.attempt_count == 2
    with create_session_factory(settings)() as session:
        attempts = session.scalars(
            select(ImageAcquisitionAttemptRecord).order_by(
                ImageAcquisitionAttemptRecord.attempt_number
            )
        ).all()
        assert [attempt.attempt_number for attempt in attempts] == [1, 2]
        assert attempts[0].http_status == 503
        assert attempts[0].retryable
        assert attempts[1].status == "succeeded"


def test_source_400_is_nonretryable(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_max_attempts=4)
    body = _png_bytes((30, 160, 230))
    post = _post("1205", body)
    adapter = _MutablePostAdapter(
        {None: ImageSearchPage(posts=(post,), next_cursor=None)}
    )
    plan_id, adapter = _make_plan(settings, (post,), adapter)
    transport = FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)})
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    adapter.post_failures[post.external_post_id] = DanbooruSourceError(
        AcquisitionErrorCode.SOURCE_UNAVAILABLE,
        status=400,
    )
    service.run_job_sync(job_id)

    item = service.list_items(job_id)[0]
    assert item.failure_code == DownloadFailureCode.HTTP_CLIENT_ERROR.value
    assert item.attempt_count == 1


def test_resume_continues_cumulative_attempt_numbers_after_cancel(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_base=1.0, retry_max_attempts=4)
    body = _png_bytes((200, 30, 90))
    post = _post("1203", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {
            post.file_url or "": [
                FakeDownloadSpec(
                    body,
                    chunk_size=8,
                    fail_after_chunks=1,
                    headers={"etag": "resume-v1", "accept-ranges": "bytes"},
                ),
                FakeDownloadSpec(body, headers={"etag": "resume-v1"}),
            ]
        }
    )
    service: ImageAcquisitionDownloadService
    canceled = False

    def cancel_during_backoff(_: float) -> None:
        nonlocal canceled
        if not canceled:
            canceled = True
            service.cancel_job(job_id)

    service = ImageAcquisitionDownloadService(
        settings,
        adapter=adapter,
        transport=transport,
        sleeper=cancel_during_backoff,
        auto_start=False,
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    canceled_item = service.list_items(job_id)[0]
    assert service.get_job(job_id).status is ImageAcquisitionJobStatus.CANCELED  # type: ignore[union-attr]
    assert canceled_item.attempt_count == 1
    service.resume_job(job_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.IMPORTED
    assert item.attempt_count == 2
    with create_session_factory(settings)() as session:
        attempts = session.scalars(
            select(ImageAcquisitionAttemptRecord).order_by(
                ImageAcquisitionAttemptRecord.attempt_number
            )
        ).all()
        assert [attempt.attempt_number for attempt in attempts] == [1, 2]
        assert attempts[0].status == "failed"
        assert attempts[1].status == "succeeded"


@pytest.mark.parametrize(
    ("item_status", "job_status", "retryable"),
    [
        (
            ImageAcquisitionItemStatus.FAILED,
            ImageAcquisitionJobStatus.FAILED,
            True,
        ),
        (
            ImageAcquisitionItemStatus.CANCELED,
            ImageAcquisitionJobStatus.CANCELED,
            False,
        ),
    ],
)
def test_get_job_and_resume_preserve_retryable_part_and_range_state(
    test_workspace: Path,
    item_status: ImageAcquisitionItemStatus,
    job_status: ImageAcquisitionJobStatus,
    retryable: bool,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((220, 40, 100))
    post = _post(f"1203-{item_status.value}", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport(
        {post.file_url or "": FakeDownloadSpec(body, headers={"etag": "resume-v1"})}
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    partial = body[:16]
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = item_status.value
        item.failure_code = DownloadFailureCode.REQUEST_TIMEOUT.value
        item.failure_message = DownloadFailureCode.REQUEST_TIMEOUT.value
        item.retryable = retryable
        item.attempt_count = 1
        item.retry_count = 1
        item.received_bytes = len(partial)
        item.etag = "resume-v1"
        item.accept_ranges = True
        item.range_start = 0
        job.status = job_status.value
        job.cancellation_requested = job_status is ImageAcquisitionJobStatus.CANCELED
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(partial)
        session.commit()

    for _ in range(2):
        job_view = service.get_job(job_id)
        assert job_view is not None and job_view.status is job_status
        assert part.exists()
        assert part.read_bytes() == partial

    service.resume_job(job_id, auto_start=False)
    with session_factory() as session:
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert item is not None
        assert item.status == ImageAcquisitionItemStatus.PENDING.value
        assert bool(item.retryable) is retryable
        assert item.received_bytes == len(partial)
        assert item.etag == "resume-v1"
        assert bool(item.accept_ranges) is True
        assert item.range_start == 0
        assert item.part_cleanup_warning is None

    service.run_job_sync(job_id)
    assert any(request.range_start == len(partial) for request in transport.requests)
    assert not part.exists()


def test_range_validation_keeps_etag_and_last_modified_independent(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    service = ImageAcquisitionDownloadService(settings, auto_start=False)
    item = SimpleNamespace(expected_file_size=10, etag="etag-v1", last_modified=None)

    response = SimpleNamespace(
        status=206,
        headers={
            "Content-Range": "bytes 4-9/10",
            "Content-Length": "6",
            "ETag": "etag-v1",
            "Last-Modified": "changed-but-not-used",
        },
        close=lambda: None,
    )
    assert service._validate_range_response(response, 4, item) == 10  # type: ignore[arg-type]

    last_modified_only = SimpleNamespace(
        expected_file_size=10,
        etag=None,
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
    )
    response = SimpleNamespace(
        status=206,
        headers={
            "content-range": "bytes 4-9/10",
            "content-length": "6",
            "last-modified": "Wed, 01 Jan 2025 00:00:00 GMT",
        },
        close=lambda: None,
    )
    assert service._validate_range_response(response, 4, last_modified_only) == 10  # type: ignore[arg-type]

    with pytest.raises(AcquisitionDownloadError) as changed:
        service._validate_range_response(
            SimpleNamespace(
                status=206,
                headers={
                    "content-range": "bytes 4-9/10",
                    "content-length": "6",
                    "etag": "etag-v1",
                    "last-modified": "different",
                },
                close=lambda: None,
            ),
            4,
            SimpleNamespace(
                expected_file_size=10,
                etag="etag-v1",
                last_modified="expected",
            ),
        )
    assert changed.value.code is DownloadFailureCode.REMOTE_FILE_CHANGED

    with pytest.raises(AcquisitionDownloadError) as missing_validator:
        service._validate_range_response(
            SimpleNamespace(
                status=206,
                headers={
                    "content-range": "bytes 4-9/10",
                    "content-length": "6",
                },
                close=lambda: None,
            ),
            4,
            SimpleNamespace(expected_file_size=10, etag=None, last_modified=None),
        )
    assert missing_validator.value.code is DownloadFailureCode.RANGE_NOT_SUPPORTED

    with pytest.raises(AcquisitionDownloadError) as invalid_range:
        service._validate_range_response(
            SimpleNamespace(
                status=206,
                headers={
                    "content-range": "bytes 4-10/10",
                    "content-length": "7",
                    "etag": "etag-v1",
                },
                close=lambda: None,
            ),
            4,
            item,
        )
    assert invalid_range.value.code is DownloadFailureCode.CONTENT_RANGE_INVALID


@pytest.mark.parametrize(
    ("status", "expected_status", "write_part"),
    [
        (
            ImageAcquisitionItemStatus.DOWNLOADING,
            ImageAcquisitionItemStatus.PENDING,
            True,
        ),
        (
            ImageAcquisitionItemStatus.DOWNLOADED,
            ImageAcquisitionItemStatus.VALIDATION_PENDING,
            True,
        ),
        (
            ImageAcquisitionItemStatus.VALIDATING,
            ImageAcquisitionItemStatus.VALIDATION_PENDING,
            True,
        ),
        (
            ImageAcquisitionItemStatus.VALIDATED,
            ImageAcquisitionItemStatus.VALIDATION_PENDING,
            True,
        ),
        (
            ImageAcquisitionItemStatus.IMPORTING,
            ImageAcquisitionItemStatus.PENDING,
            False,
        ),
    ],
)
def test_stale_recovery_requeues_interrupted_item_states(
    test_workspace: Path,
    status: ImageAcquisitionItemStatus,
    expected_status: ImageAcquisitionItemStatus,
    write_part: bool,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((50, 180, 80))
    post = _post("1301", body)
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        partial = body[:16]
        item.status = status.value
        item.received_bytes = len(partial) if write_part else 0
        item.etag = "stale-etag" if write_part else None
        item.accept_ranges = write_part
        if write_part:
            part = settings.projects_dir / str(job.project_id) / item.part_relative_path
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(partial)
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "stale-worker"
        job.claim_token = "stale-token"
        job.worker_generation = 7
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        job.current_item_id = item.id
        session.commit()

    assert service.recover_stale_jobs() == 1

    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        assert job.status == ImageAcquisitionJobStatus.QUEUED.value
        assert job.worker_id is None
        assert job.claim_token is None
        assert job.current_item_id is None
        assert job.worker_generation == 8
        assert item.status == expected_status.value
        assert item.attempt_count == 0


def test_stale_importing_state_is_completed_from_existing_idempotent_import(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((80, 80, 210))
    post = _post("1302", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)})
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        attempt = session.scalar(
            select(ImageAcquisitionAttemptRecord).where(
                ImageAcquisitionAttemptRecord.job_item_id == item.id
            )
        )
        assert attempt is not None
        item.status = ImageAcquisitionItemStatus.IMPORTING.value
        item.image_asset_id = None
        item.received_bytes = len(body)
        item.completed_at = None
        part = settings.projects_dir / str(job.project_id) / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(body)
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "stale-worker"
        job.claim_token = "stale-token"
        job.worker_generation = 5
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        attempt.status = "running"
        attempt.worker_generation = 5
        attempt.received_bytes = len(body)
        attempt.completed_at = None
        session.commit()

    assert service.recover_stale_jobs() == 1
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.IMPORTED
    assert not part.exists()
    with create_session_factory(settings)() as session:
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert attempt is not None
        assert attempt.status == "succeeded"
        assert attempt.completed_at is not None
        assert attempt.received_bytes == len(body)
        assert attempt.failure_code is None
        assert attempt.worker_generation == 5

    service.run_job_sync(job_id)
    job_view = service.get_job(job_id)
    assert job_view is not None
    assert job_view.status is ImageAcquisitionJobStatus.COMPLETED
    assert job_view.imported_count == 1
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        assert job.active_key is None
        assert job.completed_at is not None
        assert job.manifest_relative_path is not None
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None
        manifest = (
            settings.projects_dir
            / str(job_view.project_id)
            / stored_job.manifest_relative_path
        )
        assert manifest.is_file()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["status"] == ImageAcquisitionJobStatus.COMPLETED.value
    assert manifest_data["item_count"] == 1
    assert manifest_data["imported_count"] == 1
    assert manifest_data["linked_existing_count"] == 0
    assert manifest_data["failed_count"] == 0
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.imported_count == manifest_data["imported_count"]
        assert (
            stored_job.linked_existing_count == manifest_data["linked_existing_count"]
        )
        assert stored_job.failed_count == manifest_data["failed_count"]


@pytest.mark.parametrize(
    "interrupted_status",
    [
        ImageAcquisitionItemStatus.DOWNLOADED,
        ImageAcquisitionItemStatus.VALIDATING,
        ImageAcquisitionItemStatus.VALIDATED,
    ],
)
def test_validation_pending_recovery_validates_part_without_download(
    test_workspace: Path,
    interrupted_status: ImageAcquisitionItemStatus,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((90, 110, 220))
    post = _post("1305", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)})
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = interrupted_status.value
        item.received_bytes = len(body)
        item.etag = "stale-etag"
        item.accept_ranges = True
        part = settings.projects_dir / str(job.project_id) / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(body)
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "stale-worker"
        job.claim_token = "stale-token"
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

    assert service.recover_stale_jobs() == 1
    assert (
        service.list_items(job_id)[0].status
        is ImageAcquisitionItemStatus.VALIDATION_PENDING
    )
    service.resume_job(job_id, auto_start=False)
    assert (
        service.list_items(job_id)[0].status
        is ImageAcquisitionItemStatus.VALIDATION_PENDING
    )
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    assert service.list_items(job_id)[0].status is ImageAcquisitionItemStatus.IMPORTED
    assert transport.requests == []


def test_stale_importing_without_import_is_audited_then_reprocessed(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((190, 110, 40))
    post = _post("1306", body)
    plan_id, adapter = _make_plan(settings, (post,))
    transport = FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)})
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        part = settings.projects_dir / str(job.project_id) / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(body[:16])
        item.status = ImageAcquisitionItemStatus.IMPORTING.value
        item.received_bytes = 16
        item.attempt_count = 1
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "stale-worker"
        job.claim_token = "stale-token"
        job.worker_generation = 6
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.add(
            ImageAcquisitionAttemptRecord(
                id="stale-attempt-1306",
                job_item_id=item.id,
                attempt_number=1,
                status="running",
                worker_generation=6,
                started_at=datetime.now(UTC) - timedelta(hours=1),
                created_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )
        session.commit()

    assert service.recover_stale_jobs() == 1
    with create_session_factory(settings)() as session:
        old_attempt = session.scalar(
            select(ImageAcquisitionAttemptRecord).where(
                ImageAcquisitionAttemptRecord.id == "stale-attempt-1306"
            )
        )
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        assert old_attempt.failure_code == DownloadFailureCode.WORKER_CLAIM_LOST.value
        assert old_attempt.completed_at is not None

    service.resume_job(job_id, auto_start=False)
    service.run_job_sync(job_id)
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.IMPORTED
    with create_session_factory(settings)() as session:
        attempts = session.scalars(
            select(ImageAcquisitionAttemptRecord).order_by(
                ImageAcquisitionAttemptRecord.attempt_number
            )
        ).all()
        assert [attempt.attempt_number for attempt in attempts] == [1, 2]
        assert all(attempt.status != "running" for attempt in attempts)


def test_stale_importing_partial_recovery_finalizes_job_and_manifest(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    bodies = (_png_bytes((20, 100, 180)), _png_bytes((180, 100, 20)))
    posts = tuple(_post(str(1310 + index), body) for index, body in enumerate(bodies))
    plan_id, adapter = _make_plan(settings, posts)
    transport = FakeDownloadTransport(
        {
            post.file_url or "": FakeDownloadSpec(body)
            for post, body in zip(posts, bodies, strict=True)
        }
    )
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, transport=transport, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        items = session.scalars(
            select(ImageAcquisitionJobItemRecord)
            .where(ImageAcquisitionJobItemRecord.job_id == str(job_id))
            .order_by(ImageAcquisitionJobItemRecord.display_order)
        ).all()
        assert job is not None and len(items) == 2
        recovered_item, failed_item = items
        part = (
            settings.projects_dir
            / str(job.project_id)
            / recovered_item.part_relative_path
        )
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"orphaned part")
        recovered_item.status = ImageAcquisitionItemStatus.IMPORTING.value
        recovered_item.image_asset_id = None
        recovered_item.completed_at = None
        failed_item.status = ImageAcquisitionItemStatus.FAILED.value
        failed_item.failure_code = DownloadFailureCode.SOURCE_POST_NOT_FOUND.value
        failed_item.failure_message = DownloadFailureCode.SOURCE_POST_NOT_FOUND.value
        failed_item.retryable = False
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "stale-worker"
        job.claim_token = "stale-token"
        job.worker_generation = 11
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

    assert service.recover_stale_jobs() == 1
    assert service.list_items(job_id)[0].status is ImageAcquisitionItemStatus.IMPORTED
    assert not part.exists()
    assert service.list_items(job_id)[1].failure_code == (
        DownloadFailureCode.SOURCE_POST_NOT_FOUND.value
    )

    service.run_job_sync(job_id)
    job_view = service.get_job(job_id)
    assert job_view is not None
    assert job_view.status is ImageAcquisitionJobStatus.PARTIALLY_COMPLETED
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        assert job.active_key is None
        assert job.completed_at is not None
        assert job.manifest_relative_path is not None


def test_source_metadata_retry_honors_cancel_during_backoff(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_base=10.0, retry_max_attempts=4)
    post = _post("1320", _png_bytes())
    adapter = _MutablePostAdapter(
        {None: ImageSearchPage(posts=(post,), next_cursor=None)}
    )
    adapter.post_failures[post.external_post_id] = DanbooruSourceError(
        AcquisitionErrorCode.SOURCE_UNAVAILABLE,
        status=503,
        retry_after=10.0,
    )
    plan_id, adapter = _make_plan(settings, (post,), adapter)
    job_id_holder: list[UUID] = []
    canceled = False

    def sleeper(_: float) -> None:
        nonlocal canceled
        if not canceled:
            canceled = True
            service.cancel_job(job_id_holder[0])

    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, sleeper=sleeper, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    job_id_holder.append(job_id)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.CANCELED
    assert service.list_items(job_id)[0].status is ImageAcquisitionItemStatus.CANCELED
    assert adapter.get_post_calls == 1


def test_source_metadata_retry_heartbeats_during_backoff(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace, retry_base=1.0, retry_max_attempts=2)
    settings.image_download_stale_after_seconds = 0.1
    body = _png_bytes((10, 120, 200))
    post = _post("1321", body)
    adapter = _MutablePostAdapter(
        {None: ImageSearchPage(posts=(post,), next_cursor=None)}
    )
    adapter.post_failures[post.external_post_id] = DanbooruSourceError(
        AcquisitionErrorCode.SOURCE_UNAVAILABLE,
        status=503,
    )
    plan_id, adapter = _make_plan(settings, (post,), adapter)
    recoveries: list[int] = []
    service: ImageAcquisitionDownloadService

    def sleeper(_: float) -> None:
        recoveries.append(service.recover_stale_jobs())
        adapter.post_failures.pop(post.external_post_id, None)

    service = ImageAcquisitionDownloadService(
        settings,
        adapter=adapter,
        transport=FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)}),
        sleeper=sleeper,
        auto_start=False,
    )
    job_id = service.start_job(plan_id, auto_start=False)
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.COMPLETED
    assert recoveries and all(value == 0 for value in recoveries)


def test_blocking_metadata_request_heartbeats_and_cancels_within_bound(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    settings.image_download_stale_after_seconds = 0.1
    settings.image_download_heartbeat_interval_seconds = 0.02
    body = _png_bytes((30, 140, 210))
    post = _post("1324", body)

    class BlockingMetadataTransport:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def get(self, params: dict[str, str]) -> HttpResponse:
            del params
            self.started.set()
            assert self.release.wait(5.0)
            payload = {
                "id": int(post.external_post_id),
                "file_url": post.file_url,
                "preview_file_url": post.preview_url,
                "large_file_url": post.sample_url,
                "image_width": post.width,
                "image_height": post.height,
                "file_size": post.file_size,
                "file_ext": "png",
                "rating": "g",
                "score": post.score,
                "tag_string": " ".join(post.tag_names),
                "md5": post.source_md5,
            }
            return HttpResponse(
                200,
                {"Content-Type": "application/json"},
                json.dumps([payload]).encode("utf-8"),
            )

    metadata_transport = BlockingMetadataTransport()
    metadata_adapter = DanbooruImageSourceAdapter(
        client=DanbooruApiClient(
            metadata_transport,
            limiter=SourceRateLimiter(minimum_interval_seconds=0),
        )
    )

    class BlockingMetadataAdapter(FakeImageSourceAdapter):
        def __init__(self) -> None:
            super().__init__({None: ImageSearchPage(posts=(post,), next_cursor=None)})
            self.metadata = metadata_adapter

        def get_post(
            self,
            external_post_id: str,
            *,
            context: ImageSourceRequestContext | None = None,
        ) -> ImageSourcePost | None:
            return self.metadata.get_post(external_post_id, context=context)

    adapter = BlockingMetadataAdapter()
    plan_id, _ = _make_plan(settings, (post,), adapter=adapter)
    service = ImageAcquisitionDownloadService(
        settings,
        adapter=adapter,
        transport=FakeDownloadTransport({post.file_url or "": FakeDownloadSpec(body)}),
        auto_start=False,
    )
    job_id = service.start_job(plan_id, auto_start=False)
    worker_thread = threading.Thread(target=service.run_job_sync, args=(job_id,))
    worker_thread.start()
    assert metadata_transport.started.wait(1.0)

    time.sleep(0.25)
    assert service.recover_stale_jobs() == 0

    service.cancel_job(job_id)
    worker_thread.join(1.0)
    assert not worker_thread.is_alive()
    metadata_transport.release.set()
    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.CANCELED


def test_stale_claim_does_not_revoke_worker_after_heartbeat_refresh(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1322", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    stale_threshold = datetime.now(UTC) - timedelta(minutes=5)
    with session_factory() as setup:
        job = setup.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        job.status = ImageAcquisitionJobStatus.RUNNING.value
        job.worker_id = "live-worker"
        job.claim_token = "live-token"
        job.worker_generation = 3
        job.heartbeat_at = stale_threshold - timedelta(minutes=1)
        setup.commit()

    with session_factory() as stale_session:
        candidate = stale_session.execute(
            select(
                ImageAcquisitionJobRecord.worker_id,
                ImageAcquisitionJobRecord.claim_token,
                ImageAcquisitionJobRecord.worker_generation,
            ).where(ImageAcquisitionJobRecord.id == str(job_id))
        ).one()
        with session_factory() as live_session:
            live_job = live_session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            assert live_job is not None
            live_job.heartbeat_at = datetime.now(UTC)
            live_session.commit()
        assert not service._claim_stale_job(
            stale_session,
            str(job_id),
            candidate.worker_id,
            candidate.claim_token,
            candidate.worker_generation,
            stale_threshold,
            datetime.now(UTC),
        )
        stale_session.rollback()

    with session_factory() as verify:
        job = verify.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        assert job.status == ImageAcquisitionJobStatus.RUNNING.value
        assert job.worker_id == "live-worker"
        assert job.claim_token == "live-token"
        assert job.worker_generation == 3


def test_old_worker_counter_update_is_rejected_after_stale_reclaim(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1323", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    old_generation = service._claim_job(job_id, "old-worker", "old-token")
    assert old_generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

    snapshot_selected = threading.Event()
    release_old_worker = threading.Event()
    old_error: list[BaseException] = []
    original_counts_snapshot = service._counts_snapshot
    first_snapshot = True

    def pause_old_snapshot(items: object) -> object:
        nonlocal first_snapshot
        if first_snapshot:
            first_snapshot = False
            snapshot_selected.set()
            assert release_old_worker.wait(5.0)
        return original_counts_snapshot(items)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "_counts_snapshot", pause_old_snapshot)

    def old_worker() -> None:
        try:
            service._recompute_counts(job_id, "old-worker", "old-token", old_generation)
        except BaseException as exc:
            old_error.append(exc)

    worker_thread = threading.Thread(target=old_worker)
    worker_thread.start()
    assert snapshot_selected.wait(1.0)

    with session_factory() as session:
        assert service._claim_stale_job(
            session,
            str(job_id),
            "old-worker",
            "old-token",
            old_generation,
            datetime.now(UTC) - timedelta(minutes=5),
            datetime.now(UTC),
        )
        session.commit()

    new_generation = service._claim_job(job_id, "new-worker", "new-token")
    assert new_generation is not None
    with session_factory() as session:
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert item is not None
        item.status = ImageAcquisitionItemStatus.IMPORTED.value
        session.commit()
    new_counts = service._recompute_counts(
        job_id, "new-worker", "new-token", new_generation
    )
    assert new_counts is not None
    assert new_counts.imported_count == 1

    release_old_worker.set()
    worker_thread.join(1.0)
    assert not worker_thread.is_alive()
    assert old_error and isinstance(old_error[0], _ClaimLost)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        assert job.status == ImageAcquisitionJobStatus.RUNNING.value
        assert job.worker_id == "new-worker"
        assert job.imported_count == 1


@pytest.mark.parametrize(
    ("final_status", "item_statuses"),
    [
        (
            ImageAcquisitionJobStatus.COMPLETED,
            (ImageAcquisitionItemStatus.IMPORTED,),
        ),
        (
            ImageAcquisitionJobStatus.PARTIALLY_COMPLETED,
            (
                ImageAcquisitionItemStatus.IMPORTED,
                ImageAcquisitionItemStatus.FAILED,
            ),
        ),
        (
            ImageAcquisitionJobStatus.FAILED,
            (ImageAcquisitionItemStatus.FAILED,),
        ),
    ],
)
def test_old_worker_cannot_remove_new_manifest_for_any_terminal_status(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_status: ImageAcquisitionJobStatus,
    item_statuses: tuple[ImageAcquisitionItemStatus, ...],
) -> None:
    settings = _settings(test_workspace)
    posts = tuple(
        _post(str(1330 + index), _png_bytes((20 + index, 100, 180)))
        for index in range(len(item_statuses))
    )
    plan_id, adapter = _make_plan(settings, posts)
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    old_generation = service._claim_job(job_id, "old-worker", "old-token")
    assert old_generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        items = session.scalars(
            select(ImageAcquisitionJobItemRecord)
            .where(ImageAcquisitionJobItemRecord.job_id == str(job_id))
            .order_by(ImageAcquisitionJobItemRecord.display_order)
        ).all()
        assert job is not None
        project_id = job.project_id
        assert len(items) == len(item_statuses)
        for item, status in zip(items, item_statuses, strict=True):
            item.status = status.value
            if status is ImageAcquisitionItemStatus.FAILED:
                item.failure_code = DownloadFailureCode.PLAN_METADATA_CHANGED.value
                item.failure_message = DownloadFailureCode.PLAN_METADATA_CHANGED.value
                item.retryable = False
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

    old_counts = service._recompute_counts(
        job_id, "old-worker", "old-token", old_generation
    )
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()
    paused_after_replace = threading.Event()
    release_old_worker = threading.Event()
    old_error: list[BaseException] = []
    replace_calls = 0
    original_atomic_replace = service._atomic_replace_manifest

    def pause_after_replace(
        source: Path,
        destination: Path,
        *,
        directory_fd: int | None = None,
    ) -> None:
        nonlocal replace_calls
        original_atomic_replace(
            source,
            destination,
            directory_fd=directory_fd,
        )
        replace_calls += 1
        if replace_calls == 1:
            paused_after_replace.set()
            assert release_old_worker.wait(5.0)

    monkeypatch.setattr(service, "_atomic_replace_manifest", pause_after_replace)

    def old_worker() -> None:
        try:
            service._write_manifest(
                job_id,
                "old-worker",
                "old-token",
                old_generation,
                final_status,
                counts=old_counts,
            )
        except BaseException as exc:
            old_error.append(exc)

    old_thread = threading.Thread(target=old_worker)
    old_thread.start()
    assert paused_after_replace.wait(1.0)
    manifest_dir = (
        settings.projects_dir
        / project_id
        / "acquisition"
        / "jobs"
        / str(job_id)
        / "manifests"
    )
    old_manifest_files = list(manifest_dir.glob("manifest-*.json"))
    assert len(old_manifest_files) == 1

    with session_factory() as session:
        assert service._claim_stale_job(
            session,
            str(job_id),
            "old-worker",
            "old-token",
            old_generation,
            datetime.now(UTC) - timedelta(minutes=5),
            datetime.now(UTC),
        )
        session.commit()
    new_generation = service._claim_job(job_id, "new-worker", "new-token")
    assert new_generation is not None
    new_counts = service._recompute_counts(
        job_id, "new-worker", "new-token", new_generation
    )
    service._write_manifest(
        job_id,
        "new-worker",
        "new-token",
        new_generation,
        final_status,
        counts=new_counts,
    )
    service._finish_job(
        job_id,
        "new-worker",
        "new-token",
        new_generation,
        final_status,
        None,
    )
    with session_factory() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None
        new_manifest = (
            settings.projects_dir
            / stored_job.project_id
            / stored_job.manifest_relative_path
        )
        assert new_manifest.is_file()
        assert json.loads(new_manifest.read_text(encoding="utf-8"))["status"] == (
            final_status.value
        )

    release_old_worker.set()
    old_thread.join(1.0)
    assert not old_thread.is_alive()
    assert old_error and isinstance(old_error[0], _ClaimLost)
    assert old_manifest_files[0] != new_manifest
    assert not old_manifest_files[0].exists()
    assert list(manifest_dir.glob("manifest-*.json")) == [new_manifest]
    assert not list(manifest_dir.glob(".*.tmp"))


@pytest.mark.parametrize(
    "unfinished_status",
    [
        ImageAcquisitionItemStatus.PENDING,
        ImageAcquisitionItemStatus.DOWNLOADING,
        ImageAcquisitionItemStatus.DOWNLOADED,
        ImageAcquisitionItemStatus.VALIDATION_PENDING,
        ImageAcquisitionItemStatus.VALIDATING,
        ImageAcquisitionItemStatus.VALIDATED,
        ImageAcquisitionItemStatus.IMPORTING,
    ],
)
def test_plan_validation_failure_finishes_every_unfinished_item(
    test_workspace: Path,
    unfinished_status: ImageAcquisitionItemStatus,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((90, 20, 180))
    post = _post("1340", body)
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "validation-worker", "validation-token")
    assert generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = unfinished_status.value
        item.received_bytes = 1
        item.attempt_count = 1
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"x")
        now = datetime.now(UTC)
        session.add(
            ImageAcquisitionAttemptRecord(
                id=f"validation-attempt-{unfinished_status.value}",
                job_item_id=item.id,
                attempt_number=1,
                status="running",
                worker_generation=generation,
                started_at=now,
                created_at=now,
            )
        )
        session.commit()

    service._fail_all_unfinished(
        job_id,
        "validation-worker",
        "validation-token",
        generation,
        DownloadFailureCode.PLAN_METADATA_CHANGED,
    )
    counts = service._recompute_counts(
        job_id, "validation-worker", "validation-token", generation
    )
    service._write_manifest(
        job_id,
        "validation-worker",
        "validation-token",
        generation,
        ImageAcquisitionJobStatus.FAILED,
        counts=counts,
    )
    service._finish_job(
        job_id,
        "validation-worker",
        "validation-token",
        generation,
        ImageAcquisitionJobStatus.FAILED,
        DownloadFailureCode.PLAN_METADATA_CHANGED,
    )

    item_view = service.list_items(job_id)[0]
    assert item_view.status is ImageAcquisitionItemStatus.FAILED
    assert item_view.failure_code == DownloadFailureCode.PLAN_METADATA_CHANGED.value
    assert item_view.part_cleanup_warning is None
    assert not part.exists()
    with session_factory() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None
        assert stored_job.pending_count == 0
        assert stored_job.downloading_count == 0
        assert attempt is not None
        assert attempt.status == "failed"
        assert attempt.failure_code == DownloadFailureCode.PLAN_METADATA_CHANGED.value
        manifest = (
            settings.projects_dir
            / stored_job.project_id
            / stored_job.manifest_relative_path
        )
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_data["status"] == ImageAcquisitionJobStatus.FAILED.value
        assert manifest_data["part_cleanup_warning_codes"] == []
        assert manifest_data["items"][0]["part_cleanup_warning"] is None
        assert manifest_data["pending_count"] == 0
        assert manifest_data["downloading_count"] == 0
        assert manifest_data["failed_count"] == 1


@pytest.mark.parametrize(
    ("cleanup_kind", "expected_warning"),
    [
        ("absent", None),
        ("invalid", PartCleanupWarningCode.PATH_INVALID.value),
        ("directory", PartCleanupWarningCode.NOT_REGULAR_FILE.value),
        ("unlink_failed", PartCleanupWarningCode.CLEANUP_FAILED.value),
        ("symlink", PartCleanupWarningCode.SYMLINK_REJECTED.value),
    ],
)
def test_plan_validation_cleanup_result_is_audited(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_kind: str,
    expected_warning: str | None,
) -> None:
    if cleanup_kind == "directory" and os.name == "nt":
        pytest.skip("Windows test workspace path is too long for directory fixture")
    settings = _settings(test_workspace)
    post = _post("1342", _png_bytes((30, 90, 160)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "cleanup-worker", "cleanup-token")
    assert generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = ImageAcquisitionItemStatus.DOWNLOADING.value
        item.received_bytes = 1
        item.attempt_count = 1
        if cleanup_kind == "invalid":
            item.part_relative_path = "../outside-part.part"
        elif cleanup_kind == "directory":
            item.part_relative_path = f"acquisition/jobs/{job_id}/{item.id}.part"
        part = settings.projects_dir / job.project_id / item.part_relative_path
        now = datetime.now(UTC)
        session.add(
            ImageAcquisitionAttemptRecord(
                id=f"cleanup-attempt-{cleanup_kind}",
                job_item_id=item.id,
                attempt_number=1,
                status="running",
                worker_generation=generation,
                started_at=now,
                created_at=now,
            )
        )
        session.commit()

    if cleanup_kind == "directory":
        part.parent.mkdir(parents=True, exist_ok=True)
        part.mkdir()
    elif cleanup_kind == "unlink_failed":
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"x")

        def fail_unlink(_: Path, *, missing_ok: bool = False) -> None:
            del missing_ok
            raise OSError("intentional test failure")

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    elif cleanup_kind == "symlink":
        part.parent.mkdir(parents=True, exist_ok=True)
        target = test_workspace / "outside-part-target"
        target.write_bytes(b"must remain")
        try:
            part.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")

    service._fail_all_unfinished(
        job_id,
        "cleanup-worker",
        "cleanup-token",
        generation,
        DownloadFailureCode.PLAN_METADATA_CHANGED,
    )
    counts = service._recompute_counts(
        job_id, "cleanup-worker", "cleanup-token", generation
    )
    service._write_manifest(
        job_id,
        "cleanup-worker",
        "cleanup-token",
        generation,
        ImageAcquisitionJobStatus.FAILED,
        counts=counts,
    )
    service._finish_job(
        job_id,
        "cleanup-worker",
        "cleanup-token",
        generation,
        ImageAcquisitionJobStatus.FAILED,
        DownloadFailureCode.PLAN_METADATA_CHANGED,
    )

    item_view = service.list_items(job_id)[0]
    assert item_view.status is ImageAcquisitionItemStatus.FAILED
    assert item_view.part_cleanup_warning == expected_warning
    with session_factory() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        stored_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert stored_job is not None and stored_item is not None
        assert stored_item.part_cleanup_warning == expected_warning
        assert attempt is not None and attempt.status == "failed"
        assert stored_job.manifest_relative_path is not None
        manifest_path = (
            settings.projects_dir
            / stored_job.project_id
            / stored_job.manifest_relative_path
        )
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["part_cleanup_warning_codes"] == (
        [expected_warning] if expected_warning else []
    )
    assert manifest_data["items"][0]["part_cleanup_warning"] == expected_warning
    assert str(test_workspace) not in json.dumps(manifest_data)
    assert "intentional test failure" not in json.dumps(manifest_data)


def test_pending_part_cleanup_is_recovered_after_worker_stops(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1348", _png_bytes((40, 110, 190)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "stopping-worker", "stopping-token")
    assert generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = ImageAcquisitionItemStatus.DOWNLOADING.value
        item.received_bytes = 1
        item.attempt_count = 1
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"orphan")
        now = datetime.now(UTC)
        session.add(
            ImageAcquisitionAttemptRecord(
                id="stopping-cleanup-attempt",
                job_item_id=item.id,
                attempt_number=1,
                status="running",
                worker_generation=generation,
                started_at=now,
                created_at=now,
            )
        )
        session.commit()

    def stop_before_cleanup(_: object) -> PartCleanupWarningCode:
        raise KeyboardInterrupt

    monkeypatch.setattr(service, "_cleanup_part_artifact", stop_before_cleanup)
    with pytest.raises(KeyboardInterrupt):
        service._fail_all_unfinished(
            job_id,
            "stopping-worker",
            "stopping-token",
            generation,
            DownloadFailureCode.PLAN_METADATA_CHANGED,
        )
    monkeypatch.setattr(
        service,
        "_cleanup_part_artifact",
        ImageAcquisitionDownloadService._cleanup_part_artifact,
    )
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        assert item.status == ImageAcquisitionItemStatus.FAILED.value
        assert item.part_cleanup_warning == PartCleanupWarningCode.PENDING.value
        session.commit()

    original_recover_cleanup = service.recover_part_cleanup_jobs

    def verify_stale_commit_before_cleanup(
        *, item_ids: object = None, **kwargs: object
    ) -> int:
        assert item_ids
        with session_factory() as session:
            recovered_job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            recovered_item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.job_id == str(job_id)
                )
            )
            assert recovered_job is not None and recovered_item is not None
            assert recovered_job.status == ImageAcquisitionJobStatus.QUEUED.value
            assert recovered_job.worker_id is None
            assert recovered_job.claim_token is None
            assert part.exists()
        return original_recover_cleanup(item_ids=item_ids, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        service, "recover_part_cleanup_jobs", verify_stale_commit_before_cleanup
    )
    assert service.recover_stale_jobs() == 1
    monkeypatch.setattr(service, "recover_part_cleanup_jobs", original_recover_cleanup)
    assert service.recover_part_cleanup_jobs() == 0
    item_view = service.list_items(job_id)[0]
    assert item_view.status is ImageAcquisitionItemStatus.FAILED
    assert item_view.part_cleanup_warning is None
    assert not part.exists()
    with session_factory() as session:
        attempt = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert attempt is not None and attempt.status == "failed"


def test_part_cleanup_warning_survives_retry_and_clears_only_after_cleanup(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace, cleanup_retry_base=0.0)
    post = _post("1349", _png_bytes((70, 130, 210)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        item.status = ImageAcquisitionItemStatus.FAILED.value
        item.failure_code = DownloadFailureCode.PLAN_METADATA_CHANGED.value
        item.failure_message = DownloadFailureCode.PLAN_METADATA_CHANGED.value
        item.retryable = False
        item.part_cleanup_warning = PartCleanupWarningCode.CLEANUP_FAILED.value
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"retry-me")
        session.commit()

    monkeypatch.setattr(
        service,
        "_cleanup_part_artifact",
        lambda _: PartCleanupWarningCode.CLEANUP_FAILED,
    )
    service.recover_part_cleanup_jobs()
    with session_factory() as session:
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert item is not None
        assert item.status == ImageAcquisitionItemStatus.FAILED.value
        assert item.part_cleanup_warning == PartCleanupWarningCode.CLEANUP_FAILED.value
    assert part.exists()

    monkeypatch.setattr(
        service,
        "_cleanup_part_artifact",
        ImageAcquisitionDownloadService._cleanup_part_artifact,
    )
    service.recover_part_cleanup_jobs()
    item_view = service.list_items(job_id)[0]
    assert item_view.status is ImageAcquisitionItemStatus.FAILED
    assert item_view.part_cleanup_warning is None
    assert not part.exists()


def test_get_job_does_not_recover_terminal_part_cleanup(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1350", _png_bytes((90, 150, 210)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        item.status = ImageAcquisitionItemStatus.FAILED.value
        item.retryable = False
        item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"keep-until-recovery")
        session.commit()

    cleanup_calls = 0

    def fail_if_called(_: object) -> PartCleanupWarningCode:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise AssertionError("get_job must not perform terminal cleanup")

    monkeypatch.setattr(service, "_cleanup_part_artifact", fail_if_called)
    assert service.get_job(job_id) is not None
    assert service.get_job(job_id) is not None
    assert cleanup_calls == 0
    assert part.exists()
    assert service.list_items(job_id)[0].part_cleanup_warning == (
        PartCleanupWarningCode.PENDING.value
    )


def test_part_cleanup_recovery_is_bounded(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    posts = (
        _post("1351", _png_bytes((10, 50, 90))),
        _post("1352", _png_bytes((20, 60, 100))),
    )
    plan_id, adapter = _make_plan(settings, posts)
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    parts: list[Path] = []
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        items = session.scalars(
            select(ImageAcquisitionJobItemRecord)
            .where(ImageAcquisitionJobItemRecord.job_id == str(job_id))
            .order_by(ImageAcquisitionJobItemRecord.display_order)
        ).all()
        assert job is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        for item in items:
            item.status = ImageAcquisitionItemStatus.FAILED.value
            item.retryable = False
            item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
            part = settings.projects_dir / job.project_id / item.part_relative_path
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(item.id.encode())
            parts.append(part)
        session.commit()

    assert service.recover_part_cleanup_jobs(limit=1) == 1
    assert sum(part.exists() for part in parts) == 1
    assert service.recover_part_cleanup_jobs(limit=1) == 1
    assert not any(part.exists() for part in parts)


def test_part_cleanup_drain_processes_backlog_and_does_not_repeat_warning(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    posts = tuple(
        _post(f"1351-backlog-{index:02d}", _png_bytes((index, 80, 140)))
        for index in range(65)
    )
    plan_id, adapter = _make_plan(settings, posts)
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    parts: dict[str, Path] = {}
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        items = session.scalars(
            select(ImageAcquisitionJobItemRecord)
            .where(ImageAcquisitionJobItemRecord.job_id == str(job_id))
            .order_by(ImageAcquisitionJobItemRecord.display_order)
        ).all()
        assert job is not None
        assert len(items) == 65
        job.status = ImageAcquisitionJobStatus.FAILED.value
        for item in items:
            item.status = ImageAcquisitionItemStatus.FAILED.value
            item.retryable = False
            item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
            part = settings.projects_dir / job.project_id / item.part_relative_path
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(item.id.encode())
            parts[item.id] = part
        failed_item_id = items[0].id
        session.commit()

    original_cleanup = service._cleanup_part_artifact
    cleanup_calls: list[str] = []

    def fail_one_cleanup(inspection: object) -> PartCleanupWarningCode | None:
        path = getattr(inspection, "path", None)
        if path is not None:
            cleanup_calls.append(path.stem)
        if path == parts[failed_item_id]:
            return PartCleanupWarningCode.CLEANUP_FAILED
        return original_cleanup(inspection)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "_cleanup_part_artifact", fail_one_cleanup)
    assert (
        service.recover_part_cleanup_jobs(max_batches=4, time_budget_seconds=5.0) == 65
    )
    assert len(cleanup_calls) == 65
    assert parts[failed_item_id].exists()
    assert (
        sum(
            part.exists()
            for item_id, part in parts.items()
            if item_id != failed_item_id
        )
        == 0
    )
    with session_factory() as session:
        failed_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.id == failed_item_id
            )
        )
        assert failed_item is not None
        assert failed_item.part_cleanup_warning == (
            PartCleanupWarningCode.CLEANUP_FAILED.value
        )
        assert failed_item.part_cleanup_next_retry_at is not None

    assert (
        service.recover_part_cleanup_jobs(max_batches=4, time_budget_seconds=5.0) == 0
    )
    assert len(cleanup_calls) == 65


def test_part_cleanup_claim_is_committed_before_file_operation(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1351-claim-stop", _png_bytes((10, 50, 90)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        item.status = ImageAcquisitionItemStatus.FAILED.value
        item.retryable = False
        item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"claim-before-unlink")
        item_id = item.id
        session.commit()

    original_factory = service.session_factory

    def fail_after_claim_commit(*args: object, **kwargs: object) -> object:
        session = original_factory(*args, **kwargs)
        original_commit = session.commit

        def commit_then_stop() -> None:
            original_commit()
            raise KeyboardInterrupt

        session.commit = commit_then_stop  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(service, "session_factory", fail_after_claim_commit)
    with pytest.raises(KeyboardInterrupt):
        service._cleanup_part_item(item_id)

    assert part.exists()
    with session_factory() as session:
        stored_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.id == item_id
            )
        )
        assert stored_item is not None
        assert stored_item.part_cleanup_claim_token is not None


def test_part_cleanup_restarts_after_unlink_before_result_commit(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace, cleanup_retry_base=0.0)
    post = _post("1351-unlink-stop", _png_bytes((15, 55, 95)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        item.status = ImageAcquisitionItemStatus.FAILED.value
        item.retryable = False
        item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"unlink-before-result")
        item_id = item.id
        session.commit()

    original_cleanup = service._cleanup_part_artifact

    def unlink_then_stop(inspection: object) -> PartCleanupWarningCode | None:
        warning = original_cleanup(inspection)  # type: ignore[arg-type]
        assert warning is None
        raise KeyboardInterrupt

    monkeypatch.setattr(service, "_cleanup_part_artifact", unlink_then_stop)
    with pytest.raises(KeyboardInterrupt):
        service.recover_part_cleanup_jobs()
    assert not part.exists()

    monkeypatch.setattr(
        service,
        "_cleanup_part_artifact",
        ImageAcquisitionDownloadService._cleanup_part_artifact,
    )
    with session_factory() as session:
        stored_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.id == item_id
            )
        )
        assert stored_item is not None
        stored_item.part_cleanup_claimed_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()
    assert service.recover_part_cleanup_jobs() == 1
    assert service.list_items(job_id)[0].part_cleanup_warning is None

    with session_factory() as session:
        stored_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.id == item_id
            )
        )
        assert stored_item is not None
        stored_item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        part.write_bytes(b"result-commit-failure")
        session.commit()

    original_factory = service.session_factory
    commit_count = 0

    def fail_result_commit(*args: object, **kwargs: object) -> object:
        session = original_factory(*args, **kwargs)
        original_commit = session.commit

        def commit() -> None:
            nonlocal commit_count
            commit_count += 1
            if commit_count == 2:
                raise SQLAlchemyError("intentional result commit failure")
            original_commit()

        session.commit = commit  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(service, "session_factory", fail_result_commit)
    assert service.recover_part_cleanup_jobs() == 0
    assert not part.exists()
    with session_factory() as session:
        stored_item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.id == item_id
            )
        )
        assert stored_item is not None
        assert stored_item.part_cleanup_claim_token is not None
        stored_item.part_cleanup_claimed_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()
    monkeypatch.setattr(service, "session_factory", original_factory)
    assert service.recover_part_cleanup_jobs() == 1
    assert service.list_items(job_id)[0].part_cleanup_warning is None


def test_part_cleanup_and_resume_race_rejects_terminal_resume(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1353", _png_bytes((30, 70, 110)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        job.status = ImageAcquisitionJobStatus.FAILED.value
        item.status = ImageAcquisitionItemStatus.FAILED.value
        item.retryable = False
        item.part_cleanup_warning = PartCleanupWarningCode.PENDING.value
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"old-part")
        session.commit()

    cleanup_started = threading.Event()
    claim_visible = threading.Event()
    release_cleanup = threading.Event()
    original_cleanup = service._cleanup_part_artifact

    def pause_cleanup(inspection: object) -> PartCleanupWarningCode | None:
        cleanup_started.set()
        with session_factory() as session:
            claimed_item = session.scalar(
                select(ImageAcquisitionJobItemRecord).where(
                    ImageAcquisitionJobItemRecord.id == item.id
                )
            )
            claimed_job = session.scalar(
                select(ImageAcquisitionJobRecord).where(
                    ImageAcquisitionJobRecord.id == str(job_id)
                )
            )
            assert claimed_item is not None and claimed_job is not None
            assert claimed_item.part_cleanup_claim_token is not None
            claimed_job.updated_at = datetime.now(UTC)
            session.commit()
        claim_visible.set()
        assert release_cleanup.wait(5.0)
        return original_cleanup(inspection)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "_cleanup_part_artifact", pause_cleanup)
    cleanup_result: list[int] = []
    cleanup_thread = threading.Thread(
        target=lambda: cleanup_result.append(service.recover_part_cleanup_jobs())
    )
    cleanup_thread.start()
    assert cleanup_started.wait(5.0)
    assert claim_visible.wait(5.0)

    resume_thread = threading.Thread(
        target=lambda: service.resume_job(job_id, auto_start=False)
    )
    resume_thread.start()
    time.sleep(0.1)
    release_cleanup.set()
    cleanup_thread.join(5.0)
    resume_thread.join(5.0)
    assert not cleanup_thread.is_alive()
    assert not resume_thread.is_alive()
    assert cleanup_result == [1]

    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        assert job.status == ImageAcquisitionJobStatus.FAILED.value
        assert item.status == ImageAcquisitionItemStatus.FAILED.value
        assert item.part_cleanup_warning is None
    assert not part.exists()


@pytest.mark.parametrize("component", ["acquisition", "jobs", "job", "manifests"])
def test_manifest_rejects_symlink_parent_components(
    test_workspace: Path,
    component: str,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1343", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "symlink-worker", "symlink-token")
    assert generation is not None
    counts = service._recompute_counts(
        job_id, "symlink-worker", "symlink-token", generation
    )
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        project_id = job.project_id
    project_root = settings.projects_dir / project_id
    target = project_root / "manifest-symlink-target"
    target.mkdir(parents=True)
    link = project_root / "acquisition"
    if component == "jobs":
        link = project_root / "acquisition" / "jobs"
        link.parent.mkdir(parents=True, exist_ok=True)
    elif component == "job":
        link = project_root / "acquisition" / "jobs" / str(job_id)
        link.parent.mkdir(parents=True, exist_ok=True)
    elif component == "manifests":
        link = project_root / "acquisition" / "jobs" / str(job_id) / "manifests"
        link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        shutil.rmtree(link)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    service._write_manifest(
        job_id,
        "symlink-worker",
        "symlink-token",
        generation,
        ImageAcquisitionJobStatus.COMPLETED,
        counts=counts,
    )
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is None
        assert stored_job.manifest_warning == "MANIFEST_WRITE_FAILED"
    assert not list(target.glob("manifest-*.json"))
    assert str(target) not in stored_job.manifest_warning


@pytest.mark.parametrize("artifact_kind", ["temporary", "final"])
def test_manifest_rejects_existing_artifact_symlink(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1344", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "artifact-worker", "artifact-token")
    assert generation is not None
    counts = service._recompute_counts(
        job_id, "artifact-worker", "artifact-token", generation
    )
    fixed_uuid = UUID("11111111-1111-4111-8111-111111111111")
    monkeypatch.setattr(acquisition_download_module, "uuid4", lambda: fixed_uuid)
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        project_id = job.project_id
    project_root = settings.projects_dir / project_id
    manifest_dir = project_root / "acquisition" / "jobs" / str(job_id) / "manifests"
    manifest_dir.mkdir(parents=True)
    final = manifest_dir / "manifest-g1-111111111111.json"
    artifact = (
        manifest_dir / ".manifest-g1-111111111111.json.tmp"
        if artifact_kind == "temporary"
        else final
    )
    target = test_workspace / f"{artifact_kind}-target"
    target.write_bytes(b"must remain")
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    service._write_manifest(
        job_id,
        "artifact-worker",
        "artifact-token",
        generation,
        ImageAcquisitionJobStatus.COMPLETED,
        counts=counts,
    )
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is None
        assert stored_job.manifest_warning == "MANIFEST_WRITE_FAILED"
    assert target.read_bytes() == b"must remain"


@pytest.mark.skipif(
    os.name == "nt",
    reason="manifest fd race coverage requires POSIX directory descriptors",
)
@pytest.mark.parametrize(
    "race_point",
    [
        "validation",
        "open",
        "temporary",
        "rename",
        "fsync",
        "db-before",
        "db-after",
        "cleanup",
    ],
)
@pytest.mark.parametrize("target_scope", ["same-project", "outside-project"])
@pytest.mark.parametrize("replacement_kind", ["symlink", "directory"])
def test_manifest_fd_operations_reject_directory_swap_races(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_point: str,
    target_scope: str,
    replacement_kind: str,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1348", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "race-worker", "race-token")
    assert generation is not None
    counts = service._recompute_counts(job_id, "race-worker", "race-token", generation)
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        project_id = job.project_id
    project_root = settings.projects_dir / project_id
    manifest_dir = project_root / "acquisition" / "jobs" / str(job_id) / "manifests"
    manifest_dir.mkdir(parents=True)
    same_project_target = project_root / "manifest-race-same-project"
    outside_project_target = test_workspace / "manifest-race-outside-project"
    target = (
        same_project_target
        if target_scope == "same-project"
        else outside_project_target
    )
    target.mkdir()
    backup = manifest_dir.with_name("manifests-race-backup")
    swapped = False

    def swap_manifest_path() -> None:
        nonlocal swapped
        if swapped:
            return
        manifest_dir.rename(backup)
        if replacement_kind == "symlink":
            manifest_dir.symlink_to(target, target_is_directory=True)
        else:
            manifest_dir.mkdir()
        swapped = True

    try:
        if race_point == "validation":
            original_validate = service._validate_manifest_directory

            def validate(
                project_root_value: Path,
                manifest_dir_value: Path,
                *,
                create_missing: bool,
                allow_missing: bool = False,
            ) -> None:
                original_validate(
                    project_root_value,
                    manifest_dir_value,
                    create_missing=create_missing,
                    allow_missing=allow_missing,
                )
                if not create_missing:
                    swap_manifest_path()

            monkeypatch.setattr(service, "_validate_manifest_directory", validate)
        elif race_point == "open":
            original_open_directory = service._open_manifest_directory

            def open_directory(
                project_id_value: str,
                job_id_value: str,
                *,
                create_missing: bool,
            ) -> object:
                handle = original_open_directory(
                    project_id_value,
                    job_id_value,
                    create_missing=create_missing,
                )
                swap_manifest_path()
                return handle

            monkeypatch.setattr(service, "_open_manifest_directory", open_directory)
        elif race_point == "temporary":
            original_open_temporary = service._open_manifest_temporary

            def open_temporary(
                path: Path, *, directory_fd: int | None = None
            ) -> object:
                swap_manifest_path()
                return original_open_temporary(path, directory_fd=directory_fd)

            monkeypatch.setattr(service, "_open_manifest_temporary", open_temporary)
        elif race_point == "rename":
            original_replace = service._atomic_replace_manifest

            def replace(
                temporary: Path,
                final: Path,
                *,
                directory_fd: int | None = None,
            ) -> None:
                original_replace(temporary, final, directory_fd=directory_fd)
                swap_manifest_path()

            monkeypatch.setattr(service, "_atomic_replace_manifest", replace)
        elif race_point == "fsync":
            original_directory_fsync = service._fsync_manifest_directory

            def fsync_directory(path: Path, directory_fd: int | None = None) -> None:
                original_directory_fsync(path, directory_fd)
                swap_manifest_path()

            monkeypatch.setattr(service, "_fsync_manifest_directory", fsync_directory)
        elif race_point == "db-before":
            original_identity_match = service._manifest_directory_identity_matches

            def identity_match(handle: object) -> bool:
                swap_manifest_path()
                return original_identity_match(handle)  # type: ignore[arg-type]

            monkeypatch.setattr(
                service, "_manifest_directory_identity_matches", identity_match
            )
        elif race_point == "db-after":
            original_artifact_match = service._manifest_artifact_matches_handle

            def artifact_match(handle: object, name: str) -> bool:
                swap_manifest_path()
                return original_artifact_match(  # type: ignore[arg-type]
                    handle, name
                )

            monkeypatch.setattr(
                service, "_manifest_artifact_matches_handle", artifact_match
            )
        else:
            original_cleanup = service._cleanup_manifest_artifact

            def fail_fsync(path: Path, directory_fd: int | None = None) -> None:
                raise OSError("intentional directory fsync failure")

            def cleanup(path: Path, *, directory_fd: int | None = None) -> None:
                swap_manifest_path()
                original_cleanup(path, directory_fd=directory_fd)

            monkeypatch.setattr(service, "_fsync_manifest_directory", fail_fsync)
            monkeypatch.setattr(service, "_cleanup_manifest_artifact", cleanup)

        service._write_manifest(
            job_id,
            "race-worker",
            "race-token",
            generation,
            ImageAcquisitionJobStatus.COMPLETED,
            counts=counts,
        )
    finally:
        if manifest_dir.is_symlink():
            manifest_dir.unlink()
        elif manifest_dir.is_dir():
            shutil.rmtree(manifest_dir)
        if backup.is_dir():
            backup.rename(manifest_dir)

    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is None
        assert stored_job.manifest_warning == "MANIFEST_WRITE_FAILED"
    assert not list(target.glob("manifest-*.json"))
    assert not list(manifest_dir.glob("manifest-*.json"))
    assert not list(manifest_dir.glob(".*.tmp"))


def test_manifest_operations_sync_directory_before_db_update(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1345", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "order-worker", "order-token")
    assert generation is not None
    counts = service._recompute_counts(
        job_id, "order-worker", "order-token", generation
    )
    events: list[str] = []
    original_open = service._open_manifest_temporary
    original_fsync = acquisition_download_module.os.fsync
    original_replace = service._atomic_replace_manifest
    original_directory_fsync = service._fsync_manifest_directory

    def open_temporary(path: Path, *, directory_fd: int | None = None) -> object:
        events.append("temporary_write")
        return original_open(path, directory_fd=directory_fd)

    def fsync(descriptor: int) -> None:
        events.append("file_fsync")
        original_fsync(descriptor)

    def replace(
        temporary: Path,
        final: Path,
        *,
        directory_fd: int | None = None,
    ) -> None:
        events.append("atomic_replace")
        original_replace(temporary, final, directory_fd=directory_fd)

    def fsync_directory(path: Path, directory_fd: int | None = None) -> None:
        events.append("directory_fsync")
        original_directory_fsync(path, directory_fd)

    monkeypatch.setattr(service, "_open_manifest_temporary", open_temporary)
    monkeypatch.setattr(acquisition_download_module.os, "fsync", fsync)
    monkeypatch.setattr(service, "_atomic_replace_manifest", replace)
    monkeypatch.setattr(service, "_fsync_manifest_directory", fsync_directory)

    service._write_manifest(
        job_id,
        "order-worker",
        "order-token",
        generation,
        ImageAcquisitionJobStatus.COMPLETED,
        counts=counts,
    )
    assert events.index("temporary_write") < events.index("file_fsync")
    assert events.index("file_fsync") < events.index("atomic_replace")
    assert events.index("atomic_replace") < events.index("directory_fsync")
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None


def test_manifest_directory_fsync_failure_does_not_update_db(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1346", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "fsync-worker", "fsync-token")
    assert generation is not None
    counts = service._recompute_counts(
        job_id, "fsync-worker", "fsync-token", generation
    )

    def fail_directory_fsync(_: Path, _directory_fd: int | None = None) -> None:
        raise OSError("intentional fsync failure")

    monkeypatch.setattr(service, "_fsync_manifest_directory", fail_directory_fsync)
    service._write_manifest(
        job_id,
        "fsync-worker",
        "fsync-token",
        generation,
        ImageAcquisitionJobStatus.COMPLETED,
        counts=counts,
    )
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is None
        assert stored_job.manifest_warning == "MANIFEST_WRITE_FAILED"
        project_id = stored_job.project_id
    assert not list(
        (
            settings.projects_dir
            / project_id
            / "acquisition"
            / "jobs"
            / str(job_id)
            / "manifests"
        ).glob("manifest-*.json")
    )


def test_manifest_ambiguous_commit_preserves_db_referenced_file(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1347", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "commit-worker", "commit-token")
    assert generation is not None
    counts = service._recompute_counts(
        job_id, "commit-worker", "commit-token", generation
    )
    original_factory = service.session_factory

    def ambiguous_factory(*args: object, **kwargs: object) -> object:
        session = original_factory(*args, **kwargs)
        original_commit = session.commit

        def ambiguous_commit() -> None:
            original_commit()
            raise RuntimeError("ambiguous commit")

        session.commit = ambiguous_commit  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(service, "session_factory", ambiguous_factory)
    with pytest.raises(RuntimeError, match="ambiguous commit"):
        service._write_manifest(
            job_id,
            "commit-worker",
            "commit-token",
            generation,
            ImageAcquisitionJobStatus.COMPLETED,
            counts=counts,
        )
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path is not None
        manifest = (
            settings.projects_dir
            / stored_job.project_id
            / stored_job.manifest_relative_path
        )
    assert manifest.is_file()


@pytest.mark.parametrize(
    ("failure_mode", "expected_result"),
    [
        ("claim_lost", "still_referenced"),
        ("database_error", "database_error"),
    ],
)
def test_manifest_reference_clear_failure_preserves_referenced_file(
    test_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_result: str,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1347-reference-failure", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "reference-worker", "reference-token")
    assert generation is not None
    relative_path = f"acquisition/jobs/{job_id}/manifests/manifest-g1-aaaaaaaaaaaa.json"
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        job.manifest_relative_path = relative_path
        project_id = job.project_id
        session.commit()
    manifest_dir = (
        settings.projects_dir
        / project_id
        / "acquisition"
        / "jobs"
        / str(job_id)
        / "manifests"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "manifest-g1-aaaaaaaaaaaa.json"
    manifest.write_text("keep", encoding="utf-8")

    if failure_mode == "database_error":

        def database_error(*_: object, **__: object) -> str | None:
            raise SQLAlchemyError("intentional database error")

        monkeypatch.setattr(service, "_manifest_reference_state", database_error)
        clear_token = "wrong-token"
    else:
        clear_token = "wrong-token"

    with service._open_manifest_directory(
        project_id, str(job_id), create_missing=True
    ) as directory:
        result = service._handle_manifest_post_commit_failure(
            job_id,
            "reference-worker",
            clear_token,
            generation,
            relative_path,
            manifest,
            directory,
        )
    assert result.value == expected_result
    assert manifest.is_file()
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path == relative_path


def test_manifest_reference_clear_allows_safe_old_artifact_cleanup(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1347-reference-owner", _png_bytes())
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "reference-owner", "reference-token")
    assert generation is not None
    relative_path = f"acquisition/jobs/{job_id}/manifests/manifest-g1-bbbbbbbbbbbb.json"
    other_path = f"acquisition/jobs/{job_id}/manifests/manifest-g1-cccccccccccc.json"
    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert job is not None
        project_id = job.project_id
        job.manifest_relative_path = other_path
        session.commit()
    manifest_dir = (
        settings.projects_dir
        / project_id
        / "acquisition"
        / "jobs"
        / str(job_id)
        / "manifests"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "manifest-g1-bbbbbbbbbbbb.json"
    manifest.write_text("old", encoding="utf-8")

    with service._open_manifest_directory(
        project_id, str(job_id), create_missing=True
    ) as directory:
        result = service._handle_manifest_post_commit_failure(
            job_id,
            "reference-owner",
            "reference-token",
            generation,
            relative_path,
            manifest,
            directory,
        )
    assert result.value == "ownership_changed"
    assert not manifest.exists()
    with create_session_factory(settings)() as session:
        stored_job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        assert stored_job is not None
        assert stored_job.manifest_relative_path == other_path


def test_stale_validation_pending_plan_change_finishes_job_as_failed(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((100, 40, 200))
    post = _post("1341", body)
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    generation = service._claim_job(job_id, "stale-worker", "stale-token")
    assert generation is not None
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        item = session.scalar(
            select(ImageAcquisitionJobItemRecord).where(
                ImageAcquisitionJobItemRecord.job_id == str(job_id)
            )
        )
        assert job is not None and item is not None
        item.status = ImageAcquisitionItemStatus.DOWNLOADED.value
        item.received_bytes = len(body)
        part = settings.projects_dir / job.project_id / item.part_relative_path
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(body)
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

    assert service.recover_stale_jobs() == 1
    assert (
        service.list_items(job_id)[0].status
        is ImageAcquisitionItemStatus.VALIDATION_PENDING
    )
    adapter.adapter_version = "changed-after-stale-recovery"  # type: ignore[misc]
    service.run_job_sync(job_id)

    job = service.get_job(job_id)
    assert job is not None
    assert job.status is ImageAcquisitionJobStatus.FAILED
    item = service.list_items(job_id)[0]
    assert item.status is ImageAcquisitionItemStatus.FAILED
    assert item.failure_code == DownloadFailureCode.PLAN_METADATA_CHANGED.value
    assert not part.exists()


def test_terminal_status_rejects_nonterminal_items(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    post = _post("1303", _png_bytes((180, 180, 20)))
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)

    status, error = service._job_terminal_status(job_id)
    assert status is ImageAcquisitionJobStatus.FAILED
    assert error is DownloadFailureCode.INCOMPLETE_ITEM_STATE


def test_attempt_audit_rejects_old_worker_generation(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    body = _png_bytes((20, 220, 120))
    post = _post("1304", body)
    plan_id, adapter = _make_plan(settings, (post,))
    service = ImageAcquisitionDownloadService(
        settings, adapter=adapter, auto_start=False
    )
    job_id = service.start_job(plan_id, auto_start=False)
    worker = "generation-worker"
    token = "generation-token"
    generation = service._claim_job(job_id, worker, token)
    assert generation is not None
    item_id = service._next_item(job_id)
    assert item_id is not None
    assert service._claim_item(job_id, item_id, worker, token, generation)
    attempt = service._set_attempt_started(
        job_id, item_id, worker, token, generation, requested_attempt=1
    )

    with pytest.raises(_ClaimLost):
        service._record_attempt_response(
            job_id,
            item_id,
            worker,
            token,
            generation - 1,
            attempt,
            200,
            None,
            {"etag": "old-worker"},
        )
    with pytest.raises(_ClaimLost):
        service._set_item_status(
            job_id,
            item_id,
            worker,
            token,
            ImageAcquisitionItemStatus.VALIDATING,
            generation=generation - 1,
        )
    with pytest.raises(_ClaimLost):
        service._finish_attempt(
            job_id,
            item_id,
            worker,
            token,
            generation - 1,
            attempt,
            "succeeded",
            None,
        )
    service._finish_job(
        job_id,
        worker,
        token,
        generation - 1,
        ImageAcquisitionJobStatus.FAILED,
        DownloadFailureCode.UNKNOWN_DOWNLOAD_ERROR,
    )

    with create_session_factory(settings)() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == str(job_id)
            )
        )
        row = session.scalar(select(ImageAcquisitionAttemptRecord))
        assert job is not None
        assert job.status == ImageAcquisitionJobStatus.RUNNING.value
        assert row is not None
        assert row.http_status is None
        assert row.status == "running"


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.donmai.us/original/1.png",
        "https://user:pass@cdn.donmai.us/original/1.png",
        "https://cdn.donmai.us:444/original/1.png",
        "https://cdn.donmai.us/original/1.png#fragment",
        "https://127.0.0.1/original/1.png",
    ],
)
def test_download_url_validation_rejects_unsafe_urls(url: str) -> None:
    assert not validate_download_url(url)
