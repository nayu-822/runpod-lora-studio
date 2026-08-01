from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.acquisition_download_models import (
    DownloadFailureCode,
    ImageAcquisitionItemStatus,
    ImageAcquisitionJobStatus,
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
    DanbooruSourceError,
    FakeImageSourceAdapter,
    ImageSourceRequestContext,
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
    workspace: Path, *, retry_base: float = 0.0, retry_max_attempts: int = 4
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
    manifest = (
        project_root
        / "acquisition"
        / "jobs"
        / str(job_id)
        / "manifests"
        / "manifest.json"
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["status"] == "completed"
    assert "file_url" not in json.dumps(manifest_data)
    assert "path" not in json.dumps(manifest_data)


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
    manifest = (
        settings.projects_dir
        / str(job_view.project_id)
        / "acquisition"
        / "jobs"
        / str(job_id)
        / "manifests"
        / "manifest.json"
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["status"] == ImageAcquisitionJobStatus.COMPLETED.value
    assert manifest_data["item_count"] == 1


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
