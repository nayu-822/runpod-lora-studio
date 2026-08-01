from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.acquisition_download_models import (
    DownloadFailureCode,
    ImageAcquisitionItemStatus,
    ImageAcquisitionJobStatus,
)
from runpod_lora_studio.domain.acquisition_models import (
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
from runpod_lora_studio.external.image_sources import FakeImageSourceAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.acquisition_download_service import (
    ImageAcquisitionDownloadService,
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


def _settings(workspace: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=workspace / "runtime",
        projects_dir=workspace / "runtime" / "projects",
        models_dir=workspace / "runtime" / "models",
        outputs_dir=workspace / "runtime" / "outputs",
        logs_dir=workspace / "runtime" / "logs",
        temp_dir=workspace / "runtime" / "tmp",
        database_path=workspace / "runtime" / "database" / "studio.sqlite3",
        image_download_disk_safety_margin_bytes=0,
        image_download_retry_base_backoff_seconds=0,
        image_download_retry_max_backoff_seconds=0,
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    return settings


def _make_plan(
    settings: AppSettings, posts: tuple[ImageSourcePost, ...]
) -> tuple[UUID, FakeImageSourceAdapter]:
    project = ProjectService(settings).create(ProjectInput(name="download-test"))
    adapter = FakeImageSourceAdapter(
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
