from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path

from runpod_lora_studio.domain.storage_models import (
    OverwritePolicy,
    StorageEntry,
    StorageRemote,
    StorageRemotePath,
    StorageValidationCheck,
    StorageValidationResult,
    TransferItemPlan,
    TransferPlan,
    TransferProgress,
    VerificationPolicy,
    VerificationResult,
)
from runpod_lora_studio.external.rclone import (
    CancelToken,
    CommandResult,
    CopyOptions,
    ListOptions,
    ProcessCallback,
    ProgressCallback,
)


class FakeStorageTransferAdapter:
    """In-memory remote used by unit tests; never contacts Google Drive.

    The knobs on this adapter deliberately model the failure boundaries of the
    real transfer adapter.  Tests can mutate ``files`` directly for remote
    changes, or use the injection attributes to make a copy/read fail, block,
    or raise after a remote side effect.
    """

    def __init__(
        self,
        *,
        remote_name: str = "gdrive",
        entries: dict[str, bytes] | None = None,
        fail_on_copy_number: int | None = None,
        raise_on_copy_number: int | None = None,
        crash_after_copy_number: int | None = None,
    ) -> None:
        self.remote_name = remote_name
        self.files = dict(entries or {})
        self.copy_calls: list[tuple[str, str]] = []
        self.copy_count = 0
        self.read_remote_file_calls: list[StorageRemotePath] = []
        self._modified_at = datetime.now(UTC)
        self.fail_on_copy_number = fail_on_copy_number
        self.raise_on_copy_number = raise_on_copy_number
        self.crash_after_copy_number = crash_after_copy_number
        self.copy_blocker: threading.Event | None = None
        self.read_blocker: threading.Event | None = None
        self.copy_started = threading.Event()
        self.read_started = threading.Event()
        self.cancel_after_bytes: int | None = None
        self.remote_hash_overrides: dict[str, tuple[str, str] | None] = {}

    def validate_environment(self) -> StorageValidationResult:
        return StorageValidationResult(
            True,
            (
                StorageValidationCheck("rclone", True, "ok"),
                StorageValidationCheck("remote", True, "ok"),
            ),
            "fake-rclone",
        )

    def list_remotes(self) -> tuple[StorageRemote, ...]:
        return (StorageRemote(self.remote_name, "drive"),)

    def list_entries(
        self, remote_path: StorageRemotePath, options: ListOptions
    ) -> tuple[StorageEntry, ...]:
        prefix = remote_path.relative_path.strip("/")
        values: list[StorageEntry] = []
        for value, content in sorted(self.files.items()):
            if not value.startswith(prefix + "/") and value != prefix:
                continue
            relative = value[len(prefix) :].strip("/") if prefix else value
            if not options.recursive and "/" in relative:
                continue
            name = Path(relative).name
            if options.query and options.query.casefold() not in name.casefold():
                continue
            if options.extension and not name.casefold().endswith(
                options.extension.casefold()
            ):
                continue
            override = self.remote_hash_overrides.get(value)
            if value not in self.remote_hash_overrides:
                hash_type: str | None = "md5"
                hash_value: str | None = hashlib.md5(content).hexdigest()
            elif override is None:
                hash_type = None
                hash_value = None
            else:
                hash_type, hash_value = override
            values.append(
                StorageEntry(
                    remote_path=remote_path.child(relative),
                    name=name,
                    size_bytes=len(content),
                    modified_at=self._modified_at,
                    hash_type=hash_type,
                    hash_value=hash_value,
                )
            )
        start = max(options.page - 1, 0) * options.page_size
        return tuple(values[start : start + options.page_size])

    def dry_run_copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
    ) -> TransferPlan:
        source_value = _value(source)
        destination_value = _value(destination)
        size = (
            source.stat().st_size
            if isinstance(source, Path) and source.is_file()
            else 0
        )
        return TransferPlan(
            token=f"fake:{source_value}:{destination_value}:{size}",
            source=source_value,
            destination=destination_value,
            items=(TransferItemPlan(source_value, size, "copy"),),
            total_bytes=size,
            available_bytes=None,
        )

    def copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
        process_callback: ProcessCallback | None = None,
    ) -> CommandResult:
        source_value = _value(source)
        destination_value = _value(destination)
        self.copy_calls.append((source_value, destination_value))
        self.copy_count += 1
        copy_number = self.copy_count
        self.copy_started.set()
        self._wait(self.copy_blocker)
        if process_callback:
            process_callback(99999)
        try:
            if self.raise_on_copy_number == copy_number:
                raise RuntimeError("fake injected copy exception")
            if self.fail_on_copy_number == copy_number:
                return CommandResult(1, "", "fake injected copy failure")
            if cancel_token and cancel_token.cancelled:
                return CommandResult(130, "", "canceled")
            if options.dry_run:
                return CommandResult(0, "", "")

            if isinstance(source, Path) and source.is_dir():
                destination_root = self._remote_key(destination_value)
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        relative = path.relative_to(source).as_posix()
                        if not self._copy_local_file(
                            path,
                            f"{destination_root}/{relative}".strip("/"),
                            options,
                            progress_callback,
                            cancel_token,
                        ):
                            return CommandResult(1, "", "remote overwrite conflict")
            elif isinstance(source, Path) and source.is_file():
                if not destination_value.startswith(self.remote_name + ":"):
                    return CommandResult(1, "", "unsupported local destination")
                key = self._remote_key(destination_value)
                if not self._copy_local_file(
                    source, key, options, progress_callback, cancel_token
                ):
                    return CommandResult(1, "", "remote overwrite conflict")
            elif source_value.startswith(self.remote_name + ":"):
                remote_key = self._remote_key(source_value)
                content = self.files.get(remote_key)
                if content is None:
                    return CommandResult(1, "", "missing")
                if options.max_bytes is not None and len(content) > options.max_bytes:
                    return CommandResult(1, "", "remote file exceeds maximum size")
                target = Path(destination_value)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                if progress_callback:
                    progress_callback(
                        TransferProgress(
                            bytes_transferred=len(content),
                            total_bytes=len(content),
                            transfers=1,
                        )
                    )
            else:
                return CommandResult(1, "", "unsupported source")
            if self.crash_after_copy_number == copy_number:
                raise RuntimeError("fake injected crash after copy")
            return CommandResult(0, "", "")
        finally:
            if process_callback:
                process_callback(None)

    def verify(
        self,
        source: Path,
        expected_size: int,
        expected_hash: str | None,
        policy: VerificationPolicy,
    ) -> VerificationResult:
        if not source.is_file():
            return VerificationResult(
                False, policy, expected_size, None, expected_hash, None, "missing"
            )
        actual_size = source.stat().st_size
        actual_hash = (
            hashlib.sha256(source.read_bytes()).hexdigest() if expected_hash else None
        )
        return VerificationResult(
            actual_size == expected_size
            and (expected_hash is None or actual_hash == expected_hash),
            policy,
            expected_size,
            actual_size,
            expected_hash,
            actual_hash,
        )

    def read_remote_file(
        self, remote_path: StorageRemotePath, max_bytes: int | None = None
    ) -> bytes:
        self.read_remote_file_calls.append(remote_path)
        self.read_started.set()
        self._wait(self.read_blocker)
        key = remote_path.relative_path.strip("/")
        if key not in self.files:
            raise RuntimeError("missing")
        content = self.files[key]
        if max_bytes is not None and len(content) > max_bytes:
            raise ValueError("remote file exceeds maximum size")
        return content

    def set_remote_bytes(self, relative_path: str, content: bytes | None) -> None:
        if content is None:
            self.files.pop(relative_path.strip("/"), None)
        else:
            self.files[relative_path.strip("/")] = content
        self._modified_at = datetime.now(UTC)

    def set_remote_hash(
        self, relative_path: str, hash_type: str, hash_value: str
    ) -> None:
        self.remote_hash_overrides[relative_path.strip("/")] = (
            hash_type,
            hash_value,
        )

    @staticmethod
    def _wait(blocker: threading.Event | None) -> None:
        if blocker is not None:
            blocker.wait()

    def _copy_local_file(
        self,
        source: Path,
        key: str,
        options: CopyOptions,
        progress_callback: ProgressCallback | None,
        cancel_token: CancelToken | None,
    ) -> bool:
        content = source.read_bytes()
        existing = self.files.get(key)
        if existing is not None:
            if options.overwrite_policy is OverwritePolicy.FAIL_IF_EXISTS:
                return False
            if options.overwrite_policy is OverwritePolicy.COPY_MISSING:
                return False
            if options.overwrite_policy is OverwritePolicy.SKIP_IDENTICAL:
                if existing == content:
                    if progress_callback:
                        progress_callback(
                            TransferProgress(
                                bytes_transferred=len(content),
                                total_bytes=len(content),
                                transfers=1,
                            )
                        )
                    return True
                return False
        chunk_size = 1024 * 1024
        written = bytearray()
        for start in range(0, len(content), chunk_size):
            if cancel_token and cancel_token.cancelled:
                return False
            chunk = content[start : start + chunk_size]
            written.extend(chunk)
            if (
                self.cancel_after_bytes is not None
                and len(written) >= self.cancel_after_bytes
                and cancel_token is not None
            ):
                cancel_token.cancel()
                return False
            if progress_callback:
                progress_callback(
                    TransferProgress(
                        bytes_transferred=len(written),
                        total_bytes=len(content),
                        transfers=1,
                    )
                )
        self.files[key] = bytes(written)
        return True

    def _remote_key(self, value: str) -> str:
        if not value.startswith(self.remote_name + ":"):
            return value.strip("/")
        return value.split(":", 1)[1].strip("/")


def _value(value: str | Path | StorageRemotePath) -> str:
    return value.rclone_value if isinstance(value, StorageRemotePath) else str(value)
