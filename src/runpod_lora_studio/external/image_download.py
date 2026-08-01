from __future__ import annotations

import email.utils
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from runpod_lora_studio.domain.acquisition_download_models import DownloadFailureCode
from runpod_lora_studio.external.image_sources import (
    DANBOORU_ALLOWED_FILE_HOSTS,
    validate_source_url,
)

DOWNLOAD_TRANSPORT_VERSION = "phase8b-urllib-stream-v1"


class DownloadTransportError(RuntimeError):
    def __init__(
        self,
        code: DownloadFailureCode,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    url: str
    range_start: int | None = None
    etag: str | None = None
    last_modified: str | None = None


class DownloadResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class DownloadTransport(Protocol):
    version: str

    def open(self, request: DownloadRequest) -> DownloadResponse: ...


@dataclass(slots=True)
class _UrlLibResponse:
    response: object
    status: int
    headers: Mapping[str, str]

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        reader = self.response
        while True:
            chunk = reader.read(chunk_size)  # type: ignore[attr-defined]
            if not chunk:
                return
            yield bytes(chunk)

    def close(self) -> None:
        self.response.close()  # type: ignore[attr-defined]


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


class HttpImageDownloadTransport:
    version = DOWNLOAD_TRANSPORT_VERSION

    def __init__(
        self,
        *,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 60.0,
        max_header_bytes: int = 64 * 1024,
        max_redirects: int = 3,
        user_agent: str = "RunPod-LoRA-Studio/phase8b",
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self.timeout_seconds = max(connect_timeout_seconds, read_timeout_seconds)
        self.max_header_bytes = max_header_bytes
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        self.opener = opener or urllib.request.build_opener(_RejectRedirectHandler())

    def open(self, request: DownloadRequest) -> DownloadResponse:
        url = request.url
        for redirect_count in range(self.max_redirects + 1):
            if not validate_download_url(url):
                raise DownloadTransportError(DownloadFailureCode.FILE_URL_NOT_ALLOWED)
            http_request = urllib.request.Request(url, method="GET")
            http_request.add_header("Accept", "image/jpeg,image/png,image/webp")
            http_request.add_header("User-Agent", self.user_agent)
            if request.range_start is not None:
                http_request.add_header("Range", f"bytes={request.range_start}-")
                if request.etag:
                    http_request.add_header("If-Range", request.etag)
                elif request.last_modified:
                    http_request.add_header("If-Range", request.last_modified)
            try:
                response = self.opener.open(http_request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                headers = _headers(exc.headers)
                if 300 <= exc.code < 400:
                    location = headers.get("location")
                    exc.close()
                    if not location or redirect_count >= self.max_redirects:
                        raise DownloadTransportError(
                            DownloadFailureCode.REDIRECT_NOT_ALLOWED
                        ) from exc
                    url = urllib.parse.urljoin(url, location)
                    if not validate_download_url(url):
                        raise DownloadTransportError(
                            DownloadFailureCode.REDIRECT_NOT_ALLOWED
                        ) from exc
                    continue
                raise DownloadTransportError(
                    _status_code(exc.code),
                    status=exc.code,
                    retry_after=parse_retry_after(headers.get("retry-after")),
                ) from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                raise DownloadTransportError(
                    DownloadFailureCode.REQUEST_TIMEOUT
                ) from exc

            headers = _headers(response.headers)
            if _header_size(headers) > self.max_header_bytes:
                response.close()
                raise DownloadTransportError(DownloadFailureCode.CONTENT_LENGTH_INVALID)
            status = int(getattr(response, "status", 200))
            if 300 <= status < 400:
                location = headers.get("location")
                response.close()
                if not location or redirect_count >= self.max_redirects:
                    raise DownloadTransportError(
                        DownloadFailureCode.REDIRECT_NOT_ALLOWED
                    )
                url = urllib.parse.urljoin(url, location)
                if not validate_download_url(url):
                    raise DownloadTransportError(
                        DownloadFailureCode.REDIRECT_NOT_ALLOWED
                    )
                continue
            return _UrlLibResponse(response, status, headers)
        raise DownloadTransportError(DownloadFailureCode.REDIRECT_NOT_ALLOWED)


@dataclass(frozen=True, slots=True)
class FakeDownloadSpec:
    body: bytes
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    chunk_size: int = 16 * 1024
    fail_after_chunks: int | None = None


class _FakeResponse:
    def __init__(self, spec: FakeDownloadSpec, headers: Mapping[str, str]) -> None:
        self.status = spec.status
        self.headers: Mapping[str, str] = {
            key.lower(): value for key, value in headers.items()
        }
        self._body = spec.body
        self._chunk_size = max(1, spec.chunk_size)
        self._fail_after_chunks = spec.fail_after_chunks

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        size = min(max(1, chunk_size), self._chunk_size)
        emitted = 0
        for offset in range(0, len(self._body), size):
            if (
                self._fail_after_chunks is not None
                and emitted >= self._fail_after_chunks
            ):
                raise OSError("simulated connection interruption")
            emitted += 1
            yield self._body[offset : offset + size]

    def close(self) -> None:
        return None


class FakeDownloadTransport:
    version = "phase8b-fake-stream-v1"

    def __init__(
        self,
        responses: Mapping[str, FakeDownloadSpec | list[FakeDownloadSpec]],
        *,
        support_range: bool = True,
    ) -> None:
        self.responses = {
            url: list(spec) if isinstance(spec, list) else [spec]
            for url, spec in responses.items()
        }
        self.support_range = support_range
        self.requests: list[DownloadRequest] = []

    def open(self, request: DownloadRequest) -> DownloadResponse:
        self.requests.append(request)
        if not validate_download_url(request.url):
            raise DownloadTransportError(DownloadFailureCode.FILE_URL_NOT_ALLOWED)
        candidates = self.responses.get(request.url)
        if not candidates:
            raise DownloadTransportError(
                DownloadFailureCode.SOURCE_POST_UNAVAILABLE, status=404
            )
        spec = candidates.pop(0) if len(candidates) > 1 else candidates[0]
        body = spec.body
        headers = dict(spec.headers)
        if request.range_start is not None:
            if not self.support_range:
                return _FakeResponse(spec, headers)
            if request.range_start >= len(body):
                return _FakeResponse(
                    FakeDownloadSpec(b"", status=416, headers=headers), headers
                )
            body = body[request.range_start :]
            headers.setdefault(
                "content-range",
                f"bytes {request.range_start}-{len(spec.body) - 1}/{len(spec.body)}",
            )
            headers.setdefault("content-length", str(len(body)))
            headers.setdefault("accept-ranges", "bytes")
            return _FakeResponse(
                FakeDownloadSpec(
                    body, status=206, headers=headers, chunk_size=spec.chunk_size
                ),
                headers,
            )
        headers.setdefault("content-length", str(len(body)))
        return _FakeResponse(
            FakeDownloadSpec(
                body,
                status=spec.status,
                headers=headers,
                chunk_size=spec.chunk_size,
                fail_after_chunks=spec.fail_after_chunks,
            ),
            headers,
        )


def validate_download_url(value: str | None) -> bool:
    if not validate_source_url(value):
        return False
    if not value:
        return False
    parsed = urllib.parse.urlparse(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    return host in DANBOORU_ALLOWED_FILE_HOSTS


def _headers(headers: object) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _header_size(headers: Mapping[str, str]) -> int:
    return sum(
        len(key.encode()) + len(value.encode()) for key, value in headers.items()
    )


def _status_code(status: int) -> DownloadFailureCode:
    return {
        401: DownloadFailureCode.AUTHENTICATION_FAILED,
        403: DownloadFailureCode.PERMISSION_DENIED,
        404: DownloadFailureCode.SOURCE_POST_NOT_FOUND,
        408: DownloadFailureCode.REQUEST_TIMEOUT,
        429: DownloadFailureCode.RATE_LIMITED,
    }.get(status, DownloadFailureCode.CONNECTION_FAILED)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 3600.0))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, min((parsed - datetime.now(UTC)).total_seconds(), 3600.0))
        except (TypeError, ValueError, OverflowError):
            return None
