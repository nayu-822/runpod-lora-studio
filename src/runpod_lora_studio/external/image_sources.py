from __future__ import annotations

import base64
import email.utils
import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import ip_address
from typing import Any, Protocol

from runpod_lora_studio.domain.acquisition_models import (
    AcquisitionErrorCode,
    DanbooruSearchCriteria,
    ImageRating,
    ImageSearchCursor,
    ImageSearchPage,
    ImageSearchSort,
    ImageSourcePost,
    ImageSourceType,
    JsonValue,
    ValidatedImageSearchQuery,
)

DANBOORU_API_HOST = "danbooru.donmai.us"
DANBOORU_API_ENDPOINT = "https://danbooru.donmai.us/posts.json"
DANBOORU_ALLOWED_FILE_HOSTS = frozenset({"cdn.donmai.us", "danbooru.donmai.us"})
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
ADAPTER_VERSION = "phase8a-danbooru-v1"
MAX_TAGS = 40
MAX_TAG_LENGTH = 100
MAX_CURSOR_LENGTH = 128
MAX_METADATA_BYTES = 32 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImageSourceRequestContext:
    """Callbacks used to keep a long metadata request under worker control."""

    cancel_requested: Callable[[], bool] | None = None
    before_request: Callable[[], None] | None = None
    after_request: Callable[[], None] | None = None
    poll_interval_seconds: float | None = None


class ImageSourceAdapter(Protocol):
    source_type: ImageSourceType
    adapter_version: str

    def validate_query(
        self, query: DanbooruSearchCriteria
    ) -> ValidatedImageSearchQuery: ...

    def search_page(
        self,
        query: ValidatedImageSearchQuery,
        cursor: ImageSearchCursor | None,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ImageSearchPage: ...

    def get_post(
        self,
        external_post_id: str,
        *,
        context: ImageSourceRequestContext | None = None,
    ) -> ImageSourcePost | None: ...


class DanbooruSourceError(RuntimeError):
    def __init__(
        self,
        code: AcquisitionErrorCode,
        message: str = "source request failed",
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        if self.status is not None:
            return self.status in {408, 429, 500, 501, 502, 503, 504}
        return self.code in {
            AcquisitionErrorCode.RATE_LIMITED,
            AcquisitionErrorCode.REQUEST_TIMEOUT,
            AcquisitionErrorCode.CONNECTION_FAILED,
            AcquisitionErrorCode.SOURCE_UNAVAILABLE,
        }


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def get(self, params: Mapping[str, str]) -> HttpResponse: ...


class DanbooruHttpTransport:
    def __init__(
        self,
        *,
        endpoint: str = DANBOORU_API_ENDPOINT,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        user_agent: str = "RunPod-LoRA-Studio/phase8a",
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "https" or parsed.hostname != DANBOORU_API_HOST:
            raise ValueError("Danbooru endpoint is fixed to the approved HTTPS host")
        self.endpoint = endpoint
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.user_agent = user_agent
        self.opener = opener or urllib.request.build_opener(_NoRedirectHandler())

    def get(self, params: Mapping[str, str]) -> HttpResponse:
        query = urllib.parse.urlencode(dict(params), doseq=True)
        url = f"{self.endpoint}?{query}"
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", self.user_agent)
        login = os.getenv("DANBOORU_LOGIN", "").strip()
        api_key = os.getenv("DANBOORU_API_KEY", "").strip()
        if login and api_key:
            token = base64.b64encode(f"{login}:{api_key}".encode()).decode()
            request.add_header("Authorization", f"Basic {token}")
        try:
            with self.opener.open(
                request,
                timeout=max(self.connect_timeout_seconds, self.read_timeout_seconds),
            ) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "json" not in content_type:
                    raise DanbooruSourceError(
                        AcquisitionErrorCode.INVALID_RESPONSE,
                        "source returned a non-JSON response",
                        status=getattr(response, "status", None),
                    )
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise DanbooruSourceError(
                        AcquisitionErrorCode.INVALID_RESPONSE,
                        "source response is too large",
                    )
                return HttpResponse(
                    status=int(getattr(response, "status", 200)),
                    headers={str(k): str(v) for k, v in response.headers.items()},
                    body=body,
                )
        except DanbooruSourceError:
            raise
        except urllib.error.HTTPError as exc:
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            code = {
                401: AcquisitionErrorCode.AUTHENTICATION_FAILED,
                403: AcquisitionErrorCode.PERMISSION_DENIED,
                429: AcquisitionErrorCode.RATE_LIMITED,
            }.get(exc.code, AcquisitionErrorCode.SOURCE_UNAVAILABLE)
            raise DanbooruSourceError(
                code,
                "Danbooru request failed",
                status=exc.code,
                retry_after=retry_after,
            ) from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise DanbooruSourceError(
                AcquisitionErrorCode.REQUEST_TIMEOUT
                if isinstance(exc, TimeoutError)
                else AcquisitionErrorCode.CONNECTION_FAILED,
                "Danbooru connection failed",
            ) from exc


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise DanbooruSourceError(
            AcquisitionErrorCode.INVALID_RESPONSE, "redirect is not allowed"
        )


class SourceRateLimiter:
    version = "phase8a-rate-limit-v1"

    def __init__(
        self,
        *,
        minimum_interval_seconds: float = 1.0,
        max_concurrency: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if minimum_interval_seconds < 0 or max_concurrency != 1:
            raise ValueError("Phase 8A permits one bounded source worker")
        self.minimum_interval_seconds = minimum_interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self.last_request_at: float | None = None
        self.rate_limit_count = 0
        self.backoff_active = False

    def acquire(
        self,
        retry_after: float | None = None,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._semaphore.acquire()
        try:
            now = self.clock()
            wait = 0.0
            if self.last_request_at is not None:
                wait = max(
                    0.0, self.minimum_interval_seconds - (now - self.last_request_at)
                )
            if retry_after is not None:
                wait = max(wait, max(0.0, retry_after))
                self.backoff_active = retry_after > 0
            if wait:
                interruptible_sleep(
                    wait,
                    cancel_requested=cancel_requested,
                    sleeper=self.sleeper,
                )
            self.last_request_at = self.clock()
        except Exception:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()

    def record_rate_limit(self) -> None:
        self.rate_limit_count += 1


@dataclass(frozen=True, slots=True)
class DanbooruRetryPolicy:
    max_attempts: int = 4
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    def delay(self, attempt: int) -> float:
        return float(
            min(self.max_backoff_seconds, self.base_backoff_seconds * (2**attempt))
        )


class DanbooruTagQueryBuilder:
    def build(
        self,
        query: ValidatedImageSearchQuery,
        cursor: ImageSearchCursor | None,
    ) -> dict[str, str]:
        criteria = query.criteria
        tags = list(criteria.include_tags)
        tags.extend(f"-{tag}" for tag in criteria.exclude_tags)
        tags.extend(f"rating:{rating.value}" for rating in criteria.ratings)
        if criteria.minimum_score is not None:
            tags.append(f"score:>={criteria.minimum_score}")
        if criteria.sort_rule is ImageSearchSort.SCORE:
            tags.append("order:score")
        elif criteria.sort_rule is ImageSearchSort.ID:
            tags.append("order:id")
        elif criteria.sort_rule is ImageSearchSort.RANDOM:
            tags.append("order:random")
        params = {"tags": " ".join(tags), "limit": str(criteria.page_size)}
        if cursor is not None:
            if len(cursor.opaque_value) > MAX_CURSOR_LENGTH or not re.fullmatch(
                r"before:[0-9]+", cursor.opaque_value
            ):
                raise DanbooruSourceError(
                    AcquisitionErrorCode.INVALID_RESPONSE, "invalid source cursor"
                )
            params["page"] = cursor.opaque_value
        return params


class DanbooruApiClient:
    def __init__(
        self,
        transport: HttpTransport,
        *,
        limiter: SourceRateLimiter | None = None,
        retry_policy: DanbooruRetryPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.limiter = limiter or SourceRateLimiter()
        self.retry_policy = retry_policy or DanbooruRetryPolicy()
        self.sleeper = sleeper

    def get_json(
        self,
        params: Mapping[str, str],
        *,
        cancel_requested: Callable[[], bool] | None = None,
        retry_policy: DanbooruRetryPolicy | None = None,
        before_request: Callable[[], None] | None = None,
        after_request: Callable[[], None] | None = None,
        poll_interval_seconds: float | None = None,
    ) -> tuple[object, int, int, int]:
        policy = retry_policy or self.retry_policy
        retries = 0
        rate_limits = 0
        for attempt in range(policy.max_attempts):
            if cancel_requested and cancel_requested():
                raise DanbooruSourceError(
                    AcquisitionErrorCode.CANCELED, "search canceled"
                )
            try:
                response = self._request_with_controls(
                    params,
                    cancel_requested=cancel_requested,
                    before_request=before_request,
                    after_request=after_request,
                    poll_interval_seconds=poll_interval_seconds,
                )
            except DanbooruSourceError as exc:
                if exc.code is AcquisitionErrorCode.RATE_LIMITED:
                    self.limiter.record_rate_limit()
                    rate_limits += 1
                if not exc.retryable or attempt + 1 >= policy.max_attempts:
                    raise
                retries += 1
                retry_after = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else policy.delay(attempt)
                )
                if cancel_requested and cancel_requested():
                    raise DanbooruSourceError(
                        AcquisitionErrorCode.CANCELED, "search canceled"
                    ) from exc
                interruptible_sleep(
                    retry_after,
                    cancel_requested=cancel_requested,
                    sleeper=self.sleeper,
                )
                continue
            try:
                payload = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DanbooruSourceError(
                    AcquisitionErrorCode.INVALID_RESPONSE, "invalid JSON response"
                ) from exc
            return payload, retries + 1, retries, rate_limits
        raise DanbooruSourceError(AcquisitionErrorCode.UNKNOWN_SOURCE_ERROR)

    def _request_with_controls(
        self,
        params: Mapping[str, str],
        *,
        cancel_requested: Callable[[], bool] | None,
        before_request: Callable[[], None] | None,
        after_request: Callable[[], None] | None,
        poll_interval_seconds: float | None,
    ) -> HttpResponse:
        acquired = False
        primary_error: BaseException | None = None
        try:
            self.limiter.acquire(cancel_requested=cancel_requested)
            acquired = True
            try:
                if before_request is not None:
                    before_request()
                response = self._get_transport_response(
                    params,
                    cancel_requested=cancel_requested,
                    before_request=before_request,
                    poll_interval_seconds=poll_interval_seconds,
                )
            except BaseException as exc:
                primary_error = exc
                if after_request is not None:
                    try:
                        after_request()
                    except BaseException as after_error:
                        raise after_error from exc
                raise
            else:
                try:
                    if after_request is not None:
                        after_request()
                except BaseException as exc:
                    primary_error = exc
                    raise
            return response
        finally:
            if acquired:
                try:
                    self.limiter.release()
                except Exception as release_error:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "source_rate_limiter_release_failed error_type=%s",
                        type(release_error).__name__,
                    )

    def _get_transport_response(
        self,
        params: Mapping[str, str],
        *,
        cancel_requested: Callable[[], bool] | None,
        before_request: Callable[[], None] | None,
        poll_interval_seconds: float | None,
    ) -> HttpResponse:
        if (
            poll_interval_seconds is None
            or poll_interval_seconds <= 0
            or (cancel_requested is None and before_request is None)
        ):
            return self.transport.get(params)

        completed = threading.Event()
        response: list[HttpResponse] = []
        errors: list[BaseException] = []

        def run_transport() -> None:
            try:
                response.append(self.transport.get(params))
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        threading.Thread(
            target=run_transport,
            name="danbooru-metadata-request",
            daemon=True,
        ).start()
        while not completed.wait(poll_interval_seconds):
            if before_request is not None:
                before_request()
            if cancel_requested is not None and cancel_requested():
                raise DanbooruSourceError(
                    AcquisitionErrorCode.CANCELED, "source request canceled"
                )
        if errors:
            raise errors[0]
        if not response:
            raise DanbooruSourceError(
                AcquisitionErrorCode.UNKNOWN_SOURCE_ERROR,
                "source request returned no response",
            )
        return response[0]


class DanbooruImageSourceAdapter:
    source_type = ImageSourceType.DANBOORU
    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        *,
        client: DanbooruApiClient | None = None,
        query_builder: DanbooruTagQueryBuilder | None = None,
    ) -> None:
        self.client = client or DanbooruApiClient(DanbooruHttpTransport())
        self.query_builder = query_builder or DanbooruTagQueryBuilder()

    def validate_query(
        self, query: DanbooruSearchCriteria
    ) -> ValidatedImageSearchQuery:
        from runpod_lora_studio.services.acquisition_service import (
            validate_search_query,
        )

        return validate_search_query(query)

    def search_page(
        self,
        query: ValidatedImageSearchQuery,
        cursor: ImageSearchCursor | None,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ImageSearchPage:
        payload, requests, retries, rate_limits = self.client.get_json(
            self.query_builder.build(query, cursor), cancel_requested=cancel_requested
        )
        if not isinstance(payload, list):
            raise DanbooruSourceError(
                AcquisitionErrorCode.UNSUPPORTED_RESPONSE_SCHEMA,
                "posts response must be an array",
            )
        posts: list[ImageSourcePost] = []
        for raw in payload:
            if not isinstance(raw, dict):
                raise DanbooruSourceError(
                    AcquisitionErrorCode.UNSUPPORTED_RESPONSE_SCHEMA,
                    "post entry must be an object",
                )
            posts.append(normalize_danbooru_post(raw))
        next_cursor = None
        if posts and posts[-1].external_post_id.isdigit():
            next_cursor = ImageSearchCursor(f"before:{posts[-1].external_post_id}")
        return ImageSearchPage(
            tuple(posts), next_cursor, requests, retries, rate_limits
        )

    def get_post(
        self,
        external_post_id: str,
        *,
        context: ImageSourceRequestContext | None = None,
    ) -> ImageSourcePost | None:
        if not re.fullmatch(r"[0-9]{1,32}", external_post_id):
            return None
        payload, _, _, _ = self.client.get_json(
            {"search[id]": external_post_id},
            cancel_requested=context.cancel_requested if context else None,
            retry_policy=DanbooruRetryPolicy(max_attempts=1),
            before_request=context.before_request if context else None,
            after_request=context.after_request if context else None,
            poll_interval_seconds=context.poll_interval_seconds if context else None,
        )
        if not isinstance(payload, list) or not payload:
            return None
        if not isinstance(payload[0], dict):
            raise DanbooruSourceError(AcquisitionErrorCode.UNSUPPORTED_RESPONSE_SCHEMA)
        return normalize_danbooru_post(payload[0])


class FakeImageSourceAdapter:
    source_type = ImageSourceType.DANBOORU
    adapter_version = "phase8a-fake-v1"

    def __init__(
        self,
        pages: Mapping[str | None, ImageSearchPage],
        *,
        failures: Mapping[str | None, DanbooruSourceError] | None = None,
    ) -> None:
        self.pages = dict(pages)
        self.failures = dict(failures or {})
        self.calls = 0

    def validate_query(
        self, query: DanbooruSearchCriteria
    ) -> ValidatedImageSearchQuery:
        from runpod_lora_studio.services.acquisition_service import (
            validate_search_query,
        )

        return validate_search_query(query)

    def search_page(
        self,
        query: ValidatedImageSearchQuery,
        cursor: ImageSearchCursor | None,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ImageSearchPage:
        del query
        if cancel_requested and cancel_requested():
            raise DanbooruSourceError(AcquisitionErrorCode.CANCELED)
        key = cursor.opaque_value if cursor else None
        self.calls += 1
        if key in self.failures:
            raise self.failures[key]
        return self.pages.get(key, ImageSearchPage((), None))

    def get_post(
        self,
        external_post_id: str,
        *,
        context: ImageSourceRequestContext | None = None,
    ) -> ImageSourcePost | None:
        del context
        for page in self.pages.values():
            for post in page.posts:
                if post.external_post_id == external_post_id:
                    return post
        return None


def normalize_danbooru_post(raw: Mapping[str, Any]) -> ImageSourcePost:
    post_id = _bounded_text(raw.get("id"), 32)
    if not post_id:
        raise DanbooruSourceError(AcquisitionErrorCode.POST_METADATA_INVALID)
    tags = _normalize_tags(raw)
    extension = _normalize_extension(raw.get("file_ext"))
    rating = _rating(raw.get("rating"))
    created_at = _parse_datetime(raw.get("created_at"))
    source_metadata: dict[str, JsonValue] = {
        "tag_names": list(tags),
        "api_version": "posts.json",
    }
    encoded_metadata = json.dumps(source_metadata, ensure_ascii=False).encode()
    if len(encoded_metadata) > MAX_METADATA_BYTES:
        raise DanbooruSourceError(AcquisitionErrorCode.POST_METADATA_INVALID)
    return ImageSourcePost(
        source_type=ImageSourceType.DANBOORU,
        external_post_id=post_id,
        post_url=f"https://{DANBOORU_API_HOST}/posts/{post_id}",
        file_url=_bounded_text(raw.get("file_url"), 2048),
        preview_url=_bounded_text(raw.get("preview_file_url"), 2048),
        sample_url=_bounded_text(raw.get("large_file_url"), 2048),
        width=_bounded_int(raw.get("image_width"), 1, 100_000),
        height=_bounded_int(raw.get("image_height"), 1, 100_000),
        file_size=_bounded_int(raw.get("file_size"), 0, 10 * 1024**3),
        file_extension=extension,
        rating=rating,
        score=_bounded_int(raw.get("score"), -1_000_000, 1_000_000),
        tag_names=tags,
        source_md5=_bounded_text(raw.get("md5"), 128),
        created_at=created_at,
        is_deleted=bool(raw.get("is_deleted", False)),
        is_pending=bool(raw.get("is_pending", False)),
        is_flagged=bool(raw.get("is_flagged", False)),
        source_metadata=source_metadata,
    )


def _normalize_tags(raw: Mapping[str, Any]) -> tuple[str, ...]:
    value = raw.get("tag_string")
    if isinstance(value, str):
        tags = value.split()
    else:
        tags = []
        for key in (
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
            "tag_string_meta",
        ):
            item = raw.get(key)
            if isinstance(item, str):
                tags.extend(item.split())
    normalized = sorted(
        {tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()}
    )
    return tuple(tag for tag in normalized if len(tag) <= MAX_TAG_LENGTH)[:MAX_TAGS]


def _normalize_extension(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    extension = value.strip().lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def _rating(value: object) -> ImageRating | None:
    try:
        return ImageRating(str(value).lower())
    except ValueError:
        return None


def _bounded_text(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and len(text) <= maximum else None


def _bounded_int(value: object, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if minimum <= value <= maximum else None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _retry_after_seconds(
    value: str | None, *, now: datetime | None = None
) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return max(0.0, min(numeric, 300.0))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        reference = now or datetime.now(UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        seconds = (parsed.astimezone(UTC) - reference.astimezone(UTC)).total_seconds()
        return max(0.0, min(seconds, 300.0))


def interruptible_sleep(
    seconds: float,
    *,
    cancel_requested: Callable[[], bool] | None,
    sleeper: Callable[[float], None],
    interval_seconds: float = 0.25,
) -> None:
    """Sleep in bounded chunks so cancellation does not wait for backoff expiry."""
    if seconds < 0 or interval_seconds <= 0:
        raise ValueError("sleep duration must be non-negative")
    remaining = seconds
    while remaining > 0:
        if cancel_requested and cancel_requested():
            raise DanbooruSourceError(AcquisitionErrorCode.CANCELED, "search canceled")
        chunk = min(interval_seconds, remaining)
        sleeper(chunk)
        remaining -= chunk
    if cancel_requested and cancel_requested():
        raise DanbooruSourceError(AcquisitionErrorCode.CANCELED, "search canceled")


def validate_source_url(value: str | None, *, allow_post_url: bool = False) -> bool:
    if not value or len(value) > 2048:
        return False
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port not in (None, 443)
    ):
        return False
    host = parsed.hostname.lower().rstrip(".")
    if allow_post_url and host == DANBOORU_API_HOST:
        return True
    if host not in DANBOORU_ALLOWED_FILE_HOSTS:
        return False
    try:
        address = ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )
