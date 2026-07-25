from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from runpod_lora_studio.domain.storage_models import (
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
    ProgressCallback,
)


class FakeStorageTransferAdapter:
    """In-memory remote used by unit tests; never contacts Google Drive."""

    def __init__(
        self,
        *,
        remote_name: str = "gdrive",
        entries: dict[str, bytes] | None = None,
    ) -> None:
        self.remote_name = remote_name
        self.files = dict(entries or {})
        self.copy_calls: list[tuple[str, str]] = []
        self._modified_at = datetime.now(UTC)

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
            digest = hashlib.md5(content).hexdigest()
            values.append(
                StorageEntry(
                    remote_path=remote_path.child(relative),
                    name=name,
                    size_bytes=len(content),
                    modified_at=self._modified_at,
                    hash_type="md5",
                    hash_value=digest,
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
    ) -> CommandResult:
        source_value = _value(source)
        destination_value = _value(destination)
        self.copy_calls.append((source_value, destination_value))
        if cancel_token and cancel_token.cancelled:
            return CommandResult(130, "", "canceled")
        if isinstance(source, Path):
            if source.is_dir():
                for path in source.rglob("*"):
                    if path.is_file():
                        relative = path.relative_to(source).as_posix()
                        destination_root = destination_value.removeprefix(
                            self.remote_name + ":"
                        ).strip("/")
                        self.files[f"{destination_root}/{relative}"] = path.read_bytes()
            elif source.is_file() and destination_value.startswith(
                self.remote_name + ":"
            ):
                self.files[destination_value.split(":", 1)[1].strip("/")] = (
                    source.read_bytes()
                )
        elif source_value.startswith(self.remote_name + ":"):
            remote_key = source_value.split(":", 1)[1].strip("/")
            content = self.files.get(remote_key)
            if content is None:
                return CommandResult(1, "", "missing")
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
        return CommandResult(0, "", "")

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

    def read_remote_file(self, remote_path: StorageRemotePath) -> bytes:
        key = remote_path.relative_path.strip("/")
        if key not in self.files:
            raise RuntimeError("missing")
        return self.files[key]


def _value(value: str | Path | StorageRemotePath) -> str:
    return value.rclone_value if isinstance(value, StorageRemotePath) else str(value)
