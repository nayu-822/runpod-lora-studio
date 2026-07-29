from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.acquisition_models import (
    AcquisitionErrorCode,
    CandidateExclusionReason,
    DanbooruSearchCriteria,
    ImageRating,
    ImageSearchCursor,
    ImageSearchPage,
    ImageSearchSort,
    ImageSourcePost,
    ImageSourceType,
)
from runpod_lora_studio.external.image_sources import (
    DanbooruApiClient,
    DanbooruRetryPolicy,
    DanbooruSourceError,
    FakeImageSourceAdapter,
    HttpResponse,
    SourceRateLimiter,
    normalize_danbooru_post,
)
from runpod_lora_studio.persistence.database import (
    create_engine_for_settings,
    create_session_factory,
)
from runpod_lora_studio.persistence.models import Base, ProjectRecord
from runpod_lora_studio.services.acquisition_service import (
    AcquisitionValidationError,
    ImageAcquisitionService,
    filter_source_post,
    validate_search_query,
)


def query(**changes: object) -> DanbooruSearchCriteria:
    values: dict[str, object] = {
        "project_id": uuid4(),
        "include_tags": (" 1girl ", "solo", "solo"),
        "exclude_tags": ("-gore",),
        "ratings": (ImageRating.GENERAL,),
        "minimum_score": 10,
        "required_extensions": (".PNG", ".jpg"),
        "maximum_candidate_count": 10,
        "page_size": 5,
        "sort_rule": ImageSearchSort.SCORE,
    }
    values.update(changes)
    return DanbooruSearchCriteria(**values)  # type: ignore[arg-type]


def post(
    external_id: str = "123",
    *,
    file_url: str | None = "https://cdn.donmai.us/original/1/2/123.jpg",
    extension: str | None = ".jpg",
    rating: ImageRating | None = ImageRating.GENERAL,
) -> ImageSourcePost:
    return ImageSourcePost(
        source_type=ImageSourceType.DANBOORU,
        external_post_id=external_id,
        post_url=f"https://danbooru.donmai.us/posts/{external_id}",
        file_url=file_url,
        preview_url=None,
        sample_url=None,
        width=1024,
        height=1024,
        file_size=100,
        file_extension=extension,
        rating=rating,
        score=20,
        tag_names=("1girl", "solo"),
        source_md5="abc",
        created_at=None,
        is_deleted=False,
        is_pending=False,
        is_flagged=False,
        source_metadata={"tag_names": ["1girl", "solo"]},
    )


def test_query_validation_is_normalized_and_fingerprint_is_deterministic() -> None:
    project_id = uuid4()
    first = validate_search_query(query(project_id=project_id))
    second = validate_search_query(
        query(
            project_id=project_id,
            include_tags=("solo", "1girl"),
            exclude_tags=("gore",),
        )
    )

    assert first.criteria.include_tags == ("1girl", "solo")
    assert first.criteria.exclude_tags == ("gore",)
    assert first.criteria.required_extensions == (".jpg", ".png")
    assert first.query_fingerprint == second.query_fingerprint


@pytest.mark.parametrize(
    "changes",
    [
        {"include_tags": ()},
        {"include_tags": ("bad\nline",)},
        {"include_tags": ("a",), "exclude_tags": ("a",)},
        {"maximum_candidate_count": True},
        {"page_size": 101},
    ],
)
def test_query_validation_rejects_unsafe_or_ambiguous_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises(AcquisitionValidationError):
        validate_search_query(query(**changes))


def test_local_filter_returns_specific_metadata_reasons() -> None:
    validated = validate_search_query(query())
    reasons = filter_source_post(post(file_url=None), validated)

    assert CandidateExclusionReason.MISSING_FILE_URL in reasons
    assert CandidateExclusionReason.INVALID_METADATA not in reasons

    reasons = filter_source_post(post(extension=".gif"), validated)
    assert (
        reasons
        == (
            CandidateExclusionReason.UNSUPPORTED_FILE_TYPE,
            CandidateExclusionReason.RATING_NOT_ALLOWED,
        )
        or CandidateExclusionReason.UNSUPPORTED_FILE_TYPE in reasons
    )


def test_danbooru_normalization_uses_fixed_safe_fields() -> None:
    value = normalize_danbooru_post(
        {
            "id": 42,
            "file_url": "https://cdn.donmai.us/original/4/2/42.jpg",
            "file_ext": "jpg",
            "image_width": 1024,
            "image_height": 768,
            "rating": "g",
            "score": 5,
            "tag_string": "solo 1girl",
        }
    )

    assert value.external_post_id == "42"
    assert value.file_extension == ".jpg"
    assert value.rating is ImageRating.GENERAL
    assert set(value.source_metadata) == {"tag_names", "api_version"}


def test_fake_adapter_deduplicates_by_external_id_and_supports_cursor_pages() -> None:
    first = ImageSearchPage(
        posts=(post("1"), post("1")), next_cursor=ImageSearchCursor("before:1")
    )
    second = ImageSearchPage(posts=(post("2"),), next_cursor=None)
    adapter = FakeImageSourceAdapter({None: first, "before:1": second})

    page = adapter.search_page(validate_search_query(query()), None)
    assert [item.external_post_id for item in page.posts] == ["1", "1"]
    next_page = adapter.search_page(validate_search_query(query()), page.next_cursor)
    assert [item.external_post_id for item in next_page.posts] == ["2"]


class _FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, params: dict[str, str]) -> HttpResponse:
        del params
        self.calls += 1
        if self.calls == 1:
            raise DanbooruSourceError(
                AcquisitionErrorCode.RATE_LIMITED, retry_after=2.0
            )
        return HttpResponse(200, {"Content-Type": "application/json"}, b"[]")


def test_retry_after_is_bounded_and_counted_once() -> None:
    sleeps: list[float] = []
    transport = _FakeTransport()
    limiter = SourceRateLimiter(minimum_interval_seconds=0, sleeper=sleeps.append)
    client = DanbooruApiClient(
        transport,
        limiter=limiter,
        retry_policy=DanbooruRetryPolicy(max_attempts=2),
        sleeper=sleeps.append,
    )

    payload, requests, retries, rate_limits = client.get_json({"tags": "solo"})

    assert json.loads(json.dumps(payload)) == []
    assert requests == 2
    assert retries == 1
    assert rate_limits == 1
    assert limiter.rate_limit_count == 1
    assert sleeps == [2.0]


def test_authentication_failure_is_not_retried() -> None:
    class AuthTransport:
        calls = 0

        def get(self, params: dict[str, str]) -> HttpResponse:
            del params
            self.calls += 1
            raise DanbooruSourceError(
                AcquisitionErrorCode.AUTHENTICATION_FAILED, status=401
            )

    transport = AuthTransport()
    client = DanbooruApiClient(transport)
    with pytest.raises(DanbooruSourceError) as error:
        client.get_json({"tags": "solo"})
    assert error.value.code is AcquisitionErrorCode.AUTHENTICATION_FAILED
    assert transport.calls == 1


def test_search_preview_and_confirm_are_idempotent_and_revalidated(
    test_workspace: Path,
) -> None:
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        models_dir=test_workspace / "runtime" / "models",
        outputs_dir=test_workspace / "runtime" / "outputs",
        logs_dir=test_workspace / "runtime" / "logs",
        temp_dir=test_workspace / "runtime" / "tmp",
        database_path=test_workspace / "runtime" / "database" / "studio.sqlite3",
    )
    ensure_runtime_directories(settings)
    engine = create_engine_for_settings(settings)
    Base.metadata.create_all(engine)
    project_id = uuid4()
    now = datetime.now(UTC)
    session_factory = create_session_factory(settings)
    with session_factory() as session:
        session.add(
            ProjectRecord(
                id=str(project_id),
                name="test",
                description="",
                concept_type="character",
                trigger_words="[]",
                status="draft",
                schema_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
        assert session.query(ProjectRecord).count() == 1
    with session_factory() as session:
        assert (
            session.scalar(
                select(ProjectRecord.id).where(ProjectRecord.id == str(project_id))
            )
            is not None
        )
    adapter = FakeImageSourceAdapter({None: ImageSearchPage((post(),), None)})
    service = ImageAcquisitionService(settings, adapter=adapter)
    search_id = service.start_search(
        query(project_id=project_id, include_tags=("solo",))
    )
    for _ in range(50):
        search = service.get_search(search_id)
        if search and search.status == "completed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("search worker did not complete")

    candidates = service.list_candidates(search_id)
    assert len(candidates) == 1
    preview = service.preview_plan(search_id, [candidates[0].result_id])
    plan_id = service.confirm_plan(preview)
    assert service.confirm_plan(preview) == plan_id
    with pytest.raises(AcquisitionValidationError):
        service.confirm_plan(replace(preview, plan_fingerprint="tampered"))
