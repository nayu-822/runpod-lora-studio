from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.acquisition_models import (
    AcquisitionErrorCode,
    AcquisitionPlanItemStatus,
    AcquisitionPlanPreview,
    AcquisitionPlanPreviewItem,
    AcquisitionPlanStatus,
    CandidateExclusionReason,
    CandidateStatus,
    DanbooruSearchCriteria,
    ImageRating,
    ImageSearchSort,
    ImageSearchStatus,
    ImageSourcePost,
    ImageSourceType,
    ValidatedImageSearchQuery,
    fingerprint,
)
from runpod_lora_studio.external.image_sources import (
    ADAPTER_VERSION,
    SUPPORTED_EXTENSIONS,
    DanbooruApiClient,
    DanbooruHttpTransport,
    DanbooruImageSourceAdapter,
    DanbooruRetryPolicy,
    DanbooruSourceError,
    ImageSourceAdapter,
    SourceRateLimiter,
    validate_source_url,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    ExternalImageAssetLinkRecord,
    ExternalImagePostRecord,
    ImageAcquisitionPlanItemRecord,
    ImageAcquisitionPlanRecord,
    ImageSourceSearchRecord,
    ImageSourceSearchResultRecord,
    ProjectRecord,
)

logger = logging.getLogger("runpod_lora_studio.acquisition")

MAX_CANDIDATES = 1000
MAX_PAGE_SIZE = 100
MAX_MINIMUM_VALUE = 100_000
MAX_TAGS = 40
MAX_TAG_LENGTH = 100
MAX_PLAN_ITEMS = 1000
SEARCH_STALE_SECONDS = 300.0
PLAN_VERSION = "phase8a-plan-v1"


class AcquisitionValidationError(ValueError):
    """Safe validation error that contains no source credentials or raw response."""


@dataclass(frozen=True, slots=True)
class SearchCandidateView:
    result_id: UUID
    external_post_id: str
    rating: str
    score: str
    resolution: str
    extension: str
    tags: str
    exclusion_reasons: tuple[str, ...]
    already_imported: bool
    already_planned: bool
    selected: bool


def validate_search_query(query: DanbooruSearchCriteria) -> ValidatedImageSearchQuery:
    if query.source_type is not ImageSourceType.DANBOORU:
        raise AcquisitionValidationError(AcquisitionErrorCode.QUERY_INVALID.value)
    include = _normalize_tags(query.include_tags, required=True)
    exclude_raw = _normalize_tags(query.exclude_tags, required=False)
    exclude = tuple(tag.lstrip("-") for tag in exclude_raw)
    if set(include) & set(exclude):
        raise AcquisitionValidationError("include and exclude tags overlap")
    if len(exclude) > MAX_TAGS:
        raise AcquisitionValidationError(AcquisitionErrorCode.TOO_MANY_TAGS.value)
    raw_ratings = query.ratings
    if any(not isinstance(rating, ImageRating) for rating in raw_ratings):
        raise AcquisitionValidationError("invalid rating")
    ratings = tuple(sorted(set(raw_ratings), key=lambda item: item.value))
    if not isinstance(query.sort_rule, ImageSearchSort):
        raise AcquisitionValidationError("invalid sort rule")
    _validate_number(query.minimum_score, "minimum_score", minimum=-1_000_000)
    _validate_number(query.minimum_width, "minimum_width", minimum=0)
    _validate_number(query.minimum_height, "minimum_height", minimum=0)
    _validate_number(query.minimum_pixel_count, "minimum_pixel_count", minimum=0)
    extensions = tuple(sorted(set(_normalize_extensions(query.required_extensions))))
    if not extensions or not set(extensions) <= SUPPORTED_EXTENSIONS:
        raise AcquisitionValidationError("unsupported file extension")
    _validate_int_range(
        query.maximum_candidate_count,
        "candidate count",
        minimum=1,
        maximum=MAX_CANDIDATES,
    )
    _validate_int_range(query.page_size, "page size", minimum=1, maximum=MAX_PAGE_SIZE)
    values = {
        "project_id": str(query.project_id),
        "source_type": query.source_type.value,
        "include_tags": include,
        "exclude_tags": exclude,
        "ratings": tuple(rating.value for rating in ratings),
        "minimum_score": query.minimum_score,
        "minimum_width": query.minimum_width,
        "minimum_height": query.minimum_height,
        "minimum_pixel_count": query.minimum_pixel_count,
        "required_extensions": extensions,
        "maximum_candidate_count": query.maximum_candidate_count,
        "page_size": query.page_size,
        "sort_rule": query.sort_rule.value,
        "query_version": query.query_version,
        "adapter_version": ADAPTER_VERSION,
    }
    normalized = " ".join(
        [
            *include,
            *(f"-{tag}" for tag in exclude),
            *(f"rating:{item.value}" for item in ratings),
        ]
    )
    normalized_query = json.dumps(values, ensure_ascii=False, sort_keys=True)
    normalized_query = f"{normalized_query}|tags={normalized}"
    normalized_query_fingerprint = fingerprint(values)
    return ValidatedImageSearchQuery(
        DanbooruSearchCriteria(
            project_id=query.project_id,
            source_type=query.source_type,
            include_tags=include,
            exclude_tags=exclude,
            ratings=ratings,
            minimum_score=query.minimum_score,
            minimum_width=query.minimum_width,
            minimum_height=query.minimum_height,
            minimum_pixel_count=query.minimum_pixel_count,
            required_extensions=extensions,
            maximum_candidate_count=query.maximum_candidate_count,
            page_size=query.page_size,
            sort_rule=query.sort_rule,
            query_version=query.query_version,
        ),
        normalized_query,
        normalized_query_fingerprint,
    )


def filter_source_post(
    post: ImageSourcePost,
    query: ValidatedImageSearchQuery,
    *,
    already_imported: bool = False,
    already_planned: bool = False,
) -> tuple[CandidateExclusionReason, ...]:
    criteria = query.criteria
    reasons: list[CandidateExclusionReason] = []
    if post.file_url is None:
        reasons.append(CandidateExclusionReason.MISSING_FILE_URL)
    elif not validate_source_url(post.file_url):
        parsed_host = _host(post.file_url)
        reasons.append(
            CandidateExclusionReason.INVALID_FILE_URL
            if parsed_host is None
            else CandidateExclusionReason.INVALID_FILE_HOST
        )
    if post.file_extension not in criteria.required_extensions:
        reasons.append(CandidateExclusionReason.UNSUPPORTED_FILE_TYPE)
    if post.rating is None or (
        criteria.ratings and post.rating not in criteria.ratings
    ):
        reasons.append(CandidateExclusionReason.RATING_NOT_ALLOWED)
    if criteria.minimum_score is not None and (
        post.score is None or post.score < criteria.minimum_score
    ):
        reasons.append(CandidateExclusionReason.SCORE_BELOW_MINIMUM)
    if criteria.minimum_width is not None and (
        post.width is None or post.width < criteria.minimum_width
    ):
        reasons.append(CandidateExclusionReason.WIDTH_BELOW_MINIMUM)
    if criteria.minimum_height is not None and (
        post.height is None or post.height < criteria.minimum_height
    ):
        reasons.append(CandidateExclusionReason.HEIGHT_BELOW_MINIMUM)
    if criteria.minimum_pixel_count is not None and (
        post.width is None
        or post.height is None
        or post.width * post.height < criteria.minimum_pixel_count
    ):
        reasons.append(CandidateExclusionReason.PIXEL_COUNT_BELOW_MINIMUM)
    if post.is_deleted:
        reasons.append(CandidateExclusionReason.POST_DELETED)
    if post.is_pending:
        reasons.append(CandidateExclusionReason.POST_PENDING)
    if post.is_flagged:
        reasons.append(CandidateExclusionReason.POST_FLAGGED)
    if already_imported:
        reasons.append(CandidateExclusionReason.ALREADY_IMPORTED)
    if already_planned:
        reasons.append(CandidateExclusionReason.ALREADY_PLANNED)
    if not post.external_post_id or not post.post_url:
        reasons.append(CandidateExclusionReason.INVALID_METADATA)
    return tuple(dict.fromkeys(reasons))


class ImageAcquisitionService:
    """Persist source search runs and immutable plans without downloading files."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: ImageSourceAdapter | None = None,
        adapters: Mapping[ImageSourceType, ImageSourceAdapter] | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self._adapters = dict(adapters or {})
        default_adapter = adapter or DanbooruImageSourceAdapter(
            client=DanbooruApiClient(
                DanbooruHttpTransport(
                    connect_timeout_seconds=settings.image_search_connect_timeout_seconds,
                    read_timeout_seconds=settings.image_search_read_timeout_seconds,
                    max_response_bytes=settings.image_search_max_response_bytes,
                ),
                limiter=SourceRateLimiter(
                    minimum_interval_seconds=settings.image_search_min_interval_seconds
                ),
                retry_policy=DanbooruRetryPolicy(
                    max_attempts=settings.image_search_retry_max_attempts,
                    base_backoff_seconds=settings.image_search_retry_base_backoff_seconds,
                    max_backoff_seconds=settings.image_search_retry_max_backoff_seconds,
                ),
            )
        )
        self._adapters.setdefault(ImageSourceType.DANBOORU, default_adapter)

    def validate(self, query: DanbooruSearchCriteria) -> ValidatedImageSearchQuery:
        adapter = self._adapter(query.source_type)
        return adapter.validate_query(query)

    def start_search(self, query: DanbooruSearchCriteria) -> UUID:
        validated = self.validate(query)
        search_id = uuid4()
        now = datetime.now(UTC)
        with self.session_factory() as session:
            if (
                session.scalar(
                    select(ProjectRecord.id).where(
                        ProjectRecord.id == str(query.project_id)
                    )
                )
                is None
            ):
                raise AcquisitionValidationError("project not found")
            session.add(
                ImageSourceSearchRecord(
                    id=str(search_id),
                    project_id=str(query.project_id),
                    source_type=query.source_type.value,
                    normalized_query=validated.normalized_query,
                    query_fingerprint=validated.query_fingerprint,
                    query_version=query.query_version,
                    adapter_version=self._adapter(query.source_type).adapter_version,
                    status=ImageSearchStatus.QUEUED.value,
                    requested_candidate_count=query.maximum_candidate_count,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
        worker_id = f"search-worker-{uuid4().hex[:12]}"
        import threading

        threading.Thread(
            target=self.run_search,
            args=(search_id, validated),
            kwargs={"worker_id": worker_id},
            daemon=True,
        ).start()
        return search_id

    def run_search(
        self,
        search_id: UUID,
        validated: ValidatedImageSearchQuery,
        *,
        worker_id: str = "synchronous-search",
    ) -> None:
        if not self._claim_search(search_id, worker_id):
            return
        adapter = self._adapter(validated.criteria.source_type)
        cursor = None
        seen_cursors: set[str] = set()
        seen_posts: set[str] = set()
        try:
            while True:
                if self._is_cancel_requested(search_id):
                    raise DanbooruSourceError(AcquisitionErrorCode.CANCELED)
                if cursor is not None:
                    if cursor.opaque_value in seen_cursors:
                        raise DanbooruSourceError(
                            AcquisitionErrorCode.CURSOR_LOOP_DETECTED
                        )
                    seen_cursors.add(cursor.opaque_value)
                page = adapter.search_page(
                    validated,
                    cursor,
                    cancel_requested=lambda: self._is_cancel_requested(search_id),
                )
                self._increment_search(
                    search_id,
                    returned=len(page.posts),
                    pages=1,
                    requests=page.request_count,
                    retries=page.retry_count,
                    rate_limits=page.rate_limit_count,
                    cursor=page.next_cursor.opaque_value if page.next_cursor else None,
                )
                if not page.posts:
                    break
                for post in page.posts:
                    if post.external_post_id in seen_posts:
                        continue
                    seen_posts.add(post.external_post_id)
                    self._save_candidate(search_id, post, validated)
                    if len(seen_posts) >= validated.criteria.maximum_candidate_count:
                        cursor = None
                        break
                if len(seen_posts) >= validated.criteria.maximum_candidate_count:
                    break
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
                if self._search_page_count(search_id) >= getattr(
                    self.settings, "image_search_max_pages", 100
                ):
                    raise DanbooruSourceError(
                        AcquisitionErrorCode.REQUEST_LIMIT_EXCEEDED
                    )
                if self._search_request_count(search_id) >= getattr(
                    self.settings, "image_search_max_requests", 200
                ):
                    raise DanbooruSourceError(
                        AcquisitionErrorCode.REQUEST_LIMIT_EXCEEDED
                    )
            self._finish_search(search_id, ImageSearchStatus.COMPLETED)
        except DanbooruSourceError as exc:
            status = (
                ImageSearchStatus.CANCELED
                if exc.code is AcquisitionErrorCode.CANCELED
                else ImageSearchStatus.PARTIALLY_FAILED
                if self._search_returned_count(search_id) > 0
                else ImageSearchStatus.FAILED
            )
            self._finish_search(search_id, status, exc.code.value)
        except Exception:
            logger.exception("image_source_search_failed search_id=%s", search_id)
            self._finish_search(
                search_id,
                ImageSearchStatus.PARTIALLY_FAILED,
                AcquisitionErrorCode.UNKNOWN_SOURCE_ERROR.value,
            )

    def cancel_search(self, search_id: UUID) -> None:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            if record is not None:
                record.cancellation_requested = True
                record.updated_at = datetime.now(UTC)
                session.commit()

    def get_search(self, search_id: UUID) -> ImageSourceSearchRecord | None:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            if record and self._is_stale(record):
                record.status = ImageSearchStatus.STALE.value
                record.updated_at = datetime.now(UTC)
                session.commit()
            return record

    def list_candidates(self, search_id: UUID) -> list[SearchCandidateView]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(ImageSourceSearchResultRecord)
                .where(ImageSourceSearchResultRecord.search_id == str(search_id))
                .order_by(ImageSourceSearchResultRecord.result_order.asc())
            ).all()
            result: list[SearchCandidateView] = []
            for row in rows:
                post = session.scalar(
                    select(ExternalImagePostRecord).where(
                        ExternalImagePostRecord.source_type
                        == ImageSourceType.DANBOORU.value,
                        ExternalImagePostRecord.external_post_id
                        == row.external_post_id,
                    )
                )
                if post is None:
                    continue
                result.append(
                    SearchCandidateView(
                        result_id=UUID(row.id),
                        external_post_id=row.external_post_id,
                        rating=post.rating or "unknown",
                        score=str(post.score if post.score is not None else ""),
                        resolution=f"{post.width or '?'}×{post.height or '?'}",
                        extension=post.file_extension or "unknown",
                        tags=", ".join(json.loads(post.normalized_tags_json or "[]")),
                        exclusion_reasons=tuple(
                            json.loads(row.exclusion_reasons_json or "[]")
                        ),
                        already_imported=bool(row.already_imported),
                        already_planned=bool(row.already_planned),
                        selected=bool(row.selected),
                    )
                )
            return result

    def set_selection(self, result_ids: Iterable[UUID], selected: bool) -> None:
        ids = [str(item) for item in result_ids]
        if not ids:
            return
        with self.session_factory() as session:
            rows = session.scalars(
                select(ImageSourceSearchResultRecord).where(
                    ImageSourceSearchResultRecord.id.in_(ids)
                )
            ).all()
            for row in rows:
                if not row.exclusion_reasons_json or row.exclusion_reasons_json == "[]":
                    row.selected = selected
                row.updated_at = datetime.now(UTC)
            session.commit()

    def select_all_available(self, search_id: UUID, selected: bool = True) -> None:
        with self.session_factory() as session:
            rows = session.scalars(
                select(ImageSourceSearchResultRecord).where(
                    ImageSourceSearchResultRecord.search_id == str(search_id),
                    ImageSourceSearchResultRecord.exclusion_reasons_json == "[]",
                )
            ).all()
            for row in rows:
                row.selected = selected
                row.updated_at = datetime.now(UTC)
            session.commit()

    def preview_plan(
        self, search_id: UUID, result_ids: Iterable[UUID] | None = None
    ) -> AcquisitionPlanPreview:
        with self.session_factory() as session:
            search = self._get_search_record(session, search_id)
            if search is None:
                raise AcquisitionValidationError("search not found")
            selected_ids = {str(item) for item in result_ids or ()}
            rows = session.scalars(
                select(ImageSourceSearchResultRecord)
                .where(ImageSourceSearchResultRecord.search_id == str(search_id))
                .order_by(ImageSourceSearchResultRecord.result_order.asc())
            ).all()
            if not selected_ids:
                selected_ids = {row.id for row in rows if row.selected}
            known_ids = {row.id for row in rows}
            if selected_ids - known_ids:
                raise AcquisitionValidationError(
                    "plan candidate does not belong to search"
                )
            items: list[AcquisitionPlanPreviewItem] = []
            for row in rows:
                if row.id not in selected_ids:
                    continue
                reasons = tuple(json.loads(row.exclusion_reasons_json or "[]"))
                if reasons:
                    continue
                post = session.scalar(
                    select(ExternalImagePostRecord).where(
                        ExternalImagePostRecord.source_type == search.source_type,
                        ExternalImagePostRecord.external_post_id
                        == row.external_post_id,
                    )
                )
                if post is None:
                    continue
                items.append(
                    AcquisitionPlanPreviewItem(
                        search_result_id=UUID(row.id),
                        external_post_id=row.external_post_id,
                        metadata_fingerprint=post.metadata_fingerprint,
                        file_url_fingerprint=fingerprint(post.file_url)
                        if post.file_url
                        else None,
                        source_md5=post.source_md5,
                        extension=post.file_extension,
                        width=post.width,
                        height=post.height,
                        already_imported=bool(row.already_imported),
                        already_planned=bool(row.already_planned),
                    )
                )
            if len(items) > MAX_PLAN_ITEMS:
                raise AcquisitionValidationError("plan item count is too large")
            plan_values = self._plan_fingerprint_values(search, items)
            return AcquisitionPlanPreview(
                project_id=UUID(search.project_id),
                search_id=UUID(search.id),
                source_type=ImageSourceType(search.source_type),
                selected_result_ids=tuple(item.search_result_id for item in items),
                items=tuple(items),
                query_fingerprint=search.query_fingerprint,
                adapter_version=search.adapter_version,
                plan_fingerprint=fingerprint(plan_values),
                generator_version=PLAN_VERSION,
            )

    def confirm_plan(self, preview: AcquisitionPlanPreview) -> UUID:
        with self.session_factory() as session:
            existing = session.scalar(
                select(ImageAcquisitionPlanRecord).where(
                    ImageAcquisitionPlanRecord.plan_fingerprint
                    == preview.plan_fingerprint
                )
            )
            if existing is not None:
                return UUID(existing.id)
            search = self._get_search_record(session, preview.search_id)
            if search is None or search.project_id != str(preview.project_id):
                raise AcquisitionValidationError("search does not belong to project")
            if search.status != ImageSearchStatus.COMPLETED.value or self._is_stale(
                search
            ):
                raise AcquisitionValidationError("search is not confirmable")
            if search.adapter_version != preview.adapter_version:
                raise AcquisitionValidationError("adapter version changed")
            if search.source_type != preview.source_type.value:
                raise AcquisitionValidationError("source type changed")
            if search.query_fingerprint != preview.query_fingerprint:
                raise AcquisitionValidationError("query fingerprint changed")
            if (
                fingerprint(self._plan_values_from_preview(preview))
                != preview.plan_fingerprint
            ):
                raise AcquisitionValidationError("plan fingerprint is invalid")
            if len({str(item.search_result_id) for item in preview.items}) != len(
                preview.items
            ):
                raise AcquisitionValidationError("duplicate plan candidate")
            if set(preview.selected_result_ids) != {
                item.search_result_id for item in preview.items
            }:
                raise AcquisitionValidationError("plan selection is inconsistent")
            plan_items: list[ImageAcquisitionPlanItemRecord] = []
            for order, item in enumerate(preview.items):
                row = session.scalar(
                    select(ImageSourceSearchResultRecord).where(
                        ImageSourceSearchResultRecord.id == str(item.search_result_id),
                        ImageSourceSearchResultRecord.search_id == search.id,
                    )
                )
                post = session.scalar(
                    select(ExternalImagePostRecord).where(
                        ExternalImagePostRecord.source_type == search.source_type,
                        ExternalImagePostRecord.external_post_id
                        == item.external_post_id,
                    )
                )
                if row is None or post is None or row.exclusion_reasons_json != "[]":
                    raise AcquisitionValidationError("plan candidate changed")
                if post.metadata_fingerprint != item.metadata_fingerprint:
                    raise AcquisitionValidationError("candidate metadata changed")
                if not post.file_url or not validate_source_url(post.file_url):
                    raise AcquisitionValidationError(
                        "candidate file URL is not allowed"
                    )
                if post.file_extension not in SUPPORTED_EXTENSIONS:
                    raise AcquisitionValidationError(
                        "candidate file type is not allowed"
                    )
                if post.is_deleted or post.is_pending or post.is_flagged:
                    raise AcquisitionValidationError("candidate is no longer available")
                if self._already_imported(
                    session, search.source_type, post.external_post_id
                ):
                    raise AcquisitionValidationError("candidate is already imported")
                if self._already_planned(
                    session, search.source_type, post.external_post_id
                ):
                    raise AcquisitionValidationError("candidate is already planned")
                plan_items.append(
                    ImageAcquisitionPlanItemRecord(
                        id=str(uuid4()),
                        plan_id="",
                        external_post_id=post.external_post_id,
                        search_result_id=row.id,
                        display_order=order,
                        planned_status=AcquisitionPlanItemStatus.PLANNED.value,
                        expected_metadata_fingerprint=post.metadata_fingerprint,
                        expected_file_url_fingerprint=fingerprint(post.file_url),
                        expected_md5=post.source_md5,
                        expected_width=post.width,
                        expected_height=post.height,
                        expected_extension=post.file_extension,
                        created_at=datetime.now(UTC),
                    )
                )
            plan_id = uuid4()
            now = datetime.now(UTC)
            plan = ImageAcquisitionPlanRecord(
                id=str(plan_id),
                project_id=search.project_id,
                source_type=search.source_type,
                source_search_id=search.id,
                status=AcquisitionPlanStatus.CONFIRMED.value,
                selected_count=len(plan_items),
                skipped_existing_count=0,
                blocked_count=0,
                plan_fingerprint=preview.plan_fingerprint,
                plan_version=preview.generator_version,
                query_fingerprint=search.query_fingerprint,
                adapter_version=search.adapter_version,
                created_at=now,
                updated_at=now,
            )
            session.add(plan)
            session.flush()
            for item in plan_items:
                item.plan_id = str(plan_id)
                session.add(item)
            session.commit()
            return plan_id

    def get_plan(self, plan_id: UUID) -> ImageAcquisitionPlanRecord | None:
        with self.session_factory() as session:
            return self._get_plan_record(session, plan_id)

    @staticmethod
    def _plan_fingerprint_values(
        search: ImageSourceSearchRecord,
        items: Iterable[AcquisitionPlanPreviewItem],
    ) -> dict[str, object]:
        return {
            "project_id": search.project_id,
            "search_id": search.id,
            "source_type": search.source_type,
            "items": [asdict(item) for item in items],
            "query_fingerprint": search.query_fingerprint,
            "adapter_version": search.adapter_version,
            "generator_version": PLAN_VERSION,
        }

    @staticmethod
    def _plan_values_from_preview(preview: AcquisitionPlanPreview) -> dict[str, object]:
        return {
            "project_id": str(preview.project_id),
            "search_id": str(preview.search_id),
            "source_type": preview.source_type.value,
            "items": [asdict(item) for item in preview.items],
            "query_fingerprint": preview.query_fingerprint,
            "adapter_version": preview.adapter_version,
            "generator_version": preview.generator_version,
        }

    def _adapter(self, source_type: ImageSourceType) -> ImageSourceAdapter:
        try:
            return self._adapters[source_type]
        except KeyError as exc:
            raise AcquisitionValidationError("source is not enabled") from exc

    def _claim_search(self, search_id: UUID, worker_id: str) -> bool:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            if record is None or record.status != ImageSearchStatus.QUEUED.value:
                return False
            now = datetime.now(UTC)
            record.status = ImageSearchStatus.RUNNING.value
            record.worker_id = worker_id
            record.started_at = now
            record.heartbeat_at = now
            record.updated_at = now
            session.commit()
            return True

    def _save_candidate(
        self,
        search_id: UUID,
        post: ImageSourcePost,
        query: ValidatedImageSearchQuery,
    ) -> None:
        with self.session_factory() as session:
            search = self._get_search_record(session, search_id)
            if search is None:
                return
            now = datetime.now(UTC)
            existing = session.scalar(
                select(ExternalImagePostRecord).where(
                    ExternalImagePostRecord.source_type == post.source_type.value,
                    ExternalImagePostRecord.external_post_id == post.external_post_id,
                )
            )
            metadata = _metadata_fingerprint(post)
            if existing is None:
                existing = ExternalImagePostRecord(
                    id=str(uuid4()),
                    source_type=post.source_type.value,
                    external_post_id=post.external_post_id,
                    post_url=post.post_url,
                    file_url=post.file_url,
                    preview_url=post.preview_url,
                    sample_url=post.sample_url,
                    width=post.width,
                    height=post.height,
                    file_size=post.file_size,
                    file_extension=post.file_extension,
                    rating=post.rating.value if post.rating else None,
                    score=post.score,
                    source_md5=post.source_md5,
                    normalized_tags_json=json.dumps(post.tag_names, sort_keys=True),
                    metadata_fingerprint=metadata,
                    source_metadata_json=json.dumps(
                        post.source_metadata, sort_keys=True
                    ),
                    is_deleted=post.is_deleted,
                    is_pending=post.is_pending,
                    is_flagged=post.is_flagged,
                    first_seen_at=now,
                    last_seen_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(existing)
            else:
                for key, value in _external_post_values(post, metadata, now).items():
                    setattr(existing, key, value)
            imported = self._already_imported(
                session, post.source_type.value, post.external_post_id
            )
            planned = self._already_planned(
                session, post.source_type.value, post.external_post_id
            )
            reasons = filter_source_post(
                post, query, already_imported=imported, already_planned=planned
            )
            result = ImageSourceSearchResultRecord(
                id=str(uuid4()),
                search_id=str(search_id),
                external_post_id=post.external_post_id,
                result_order=search.returned_post_count,
                candidate_status=(
                    CandidateStatus.ACCEPTED
                    if not reasons
                    else CandidateStatus.EXCLUDED
                ).value,
                exclusion_reasons_json=json.dumps(
                    [reason.value for reason in reasons], sort_keys=True
                ),
                already_imported=imported,
                already_planned=planned,
                selected=False,
                metadata_fingerprint_at_search=metadata,
                created_at=now,
                updated_at=now,
            )
            session.add(result)
            search.returned_post_count += 1
            search.accepted_candidate_count += int(not reasons)
            search.excluded_candidate_count += int(bool(reasons))
            search.heartbeat_at = now
            search.updated_at = now
            session.commit()

    def _increment_search(self, search_id: UUID, **values: object) -> None:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            if record is None:
                return
            record.page_count += _integer_value(values.get("pages", 0))
            record.api_request_count += _integer_value(values.get("requests", 0))
            record.retry_count += _integer_value(values.get("retries", 0))
            record.rate_limit_count += _integer_value(values.get("rate_limits", 0))
            record.current_cursor = (
                values.get("cursor") if isinstance(values.get("cursor"), str) else None
            )
            record.heartbeat_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)
            session.commit()

    def _finish_search(
        self, search_id: UUID, status: ImageSearchStatus, error_code: str | None = None
    ) -> None:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            if record is None:
                return
            now = datetime.now(UTC)
            record.status = status.value
            record.error_code = error_code
            record.completed_at = now
            record.heartbeat_at = now
            record.updated_at = now
            session.commit()

    def _is_cancel_requested(self, search_id: UUID) -> bool:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            return bool(record and record.cancellation_requested)

    def _search_page_count(self, search_id: UUID) -> int:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            return record.page_count if record else 0

    def _search_returned_count(self, search_id: UUID) -> int:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            return record.returned_post_count if record else 0

    def _search_request_count(self, search_id: UUID) -> int:
        with self.session_factory() as session:
            record = self._get_search_record(session, search_id)
            return record.api_request_count if record else 0

    def _is_stale(self, record: ImageSourceSearchRecord) -> bool:
        if record.status != ImageSearchStatus.RUNNING.value or not record.heartbeat_at:
            return bool(record.status == ImageSearchStatus.STALE.value)
        threshold = getattr(
            self.settings, "image_search_stale_after_seconds", SEARCH_STALE_SECONDS
        )
        heartbeat = cast(datetime, record.heartbeat_at)
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=UTC)
        return (datetime.now(UTC) - heartbeat).total_seconds() > threshold

    @staticmethod
    def _get_search_record(
        session: Any, search_id: UUID
    ) -> ImageSourceSearchRecord | None:
        return cast(
            ImageSourceSearchRecord | None,
            session.scalar(
                select(ImageSourceSearchRecord).where(
                    ImageSourceSearchRecord.id == str(search_id)
                )
            ),
        )

    @staticmethod
    def _get_plan_record(
        session: Any, plan_id: UUID
    ) -> ImageAcquisitionPlanRecord | None:
        return cast(
            ImageAcquisitionPlanRecord | None,
            session.scalar(
                select(ImageAcquisitionPlanRecord).where(
                    ImageAcquisitionPlanRecord.id == str(plan_id)
                )
            ),
        )

    @staticmethod
    def _already_imported(
        session: Any, source_type: str, external_post_id: str
    ) -> bool:
        return bool(
            session.scalar(
                select(ExternalImageAssetLinkRecord.id).where(
                    ExternalImageAssetLinkRecord.source_type == source_type,
                    ExternalImageAssetLinkRecord.external_post_id == external_post_id,
                )
            )
            is not None
        )

    @staticmethod
    def _already_planned(session: Any, source_type: str, external_post_id: str) -> bool:
        return bool(
            session.scalar(
                select(ImageAcquisitionPlanItemRecord.id)
                .join(
                    ImageAcquisitionPlanRecord,
                    ImageAcquisitionPlanRecord.id
                    == ImageAcquisitionPlanItemRecord.plan_id,
                )
                .where(
                    ImageAcquisitionPlanRecord.source_type == source_type,
                    ImageAcquisitionPlanRecord.status
                    == AcquisitionPlanStatus.CONFIRMED.value,
                    ImageAcquisitionPlanItemRecord.external_post_id == external_post_id,
                )
            )
            is not None
        )


def _normalize_tags(values: Iterable[str], *, required: bool) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise AcquisitionValidationError("tags must be a sequence")
    if required and not values:
        raise AcquisitionValidationError("at least one include tag is required")
    if len(values) > MAX_TAGS:
        raise AcquisitionValidationError(AcquisitionErrorCode.TOO_MANY_TAGS.value)
    normalized: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise AcquisitionValidationError("tag must be text")
        value = raw.strip()
        if not value or len(value) > MAX_TAG_LENGTH or "\n" in value or "\r" in value:
            raise AcquisitionValidationError("invalid tag")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise AcquisitionValidationError("invalid tag")
        normalized.add(value)
    return tuple(sorted(normalized))


def _normalize_extensions(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise AcquisitionValidationError("extension must be text")
        normalized = value.strip().lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        result.append(normalized)
    return tuple(result)


def _validate_number(value: object, name: str, *, minimum: int) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= MAX_MINIMUM_VALUE
    ):
        raise AcquisitionValidationError(f"{name} is out of range")


def _validate_int_range(
    value: object, name: str, *, minimum: int, maximum: int
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcquisitionValidationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise AcquisitionValidationError(f"{name} is out of range")


def _integer_value(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _metadata_fingerprint(post: ImageSourcePost) -> str:
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


def _external_post_values(
    post: ImageSourcePost, metadata: str, now: datetime
) -> dict[str, object]:
    return {
        "post_url": post.post_url,
        "file_url": post.file_url,
        "preview_url": post.preview_url,
        "sample_url": post.sample_url,
        "width": post.width,
        "height": post.height,
        "file_size": post.file_size,
        "file_extension": post.file_extension,
        "rating": post.rating.value if post.rating else None,
        "score": post.score,
        "source_md5": post.source_md5,
        "normalized_tags_json": json.dumps(post.tag_names, sort_keys=True),
        "metadata_fingerprint": metadata,
        "source_metadata_json": json.dumps(post.source_metadata, sort_keys=True),
        "is_deleted": post.is_deleted,
        "is_pending": post.is_pending,
        "is_flagged": post.is_flagged,
        "last_seen_at": now,
        "updated_at": now,
    }


def _host(value: str) -> str | None:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    return parsed.hostname.lower() if parsed.hostname else None
