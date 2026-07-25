from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID


class ModelType(StrEnum):
    BASE_MODEL = "base_model"
    CHECKPOINT = "checkpoint"
    VAE = "vae"
    UNKNOWN = "unknown"


class ManagedModelStatus(StrEnum):
    REMOTE_ONLY = "remote_only"
    DOWNLOADING = "downloading"
    AVAILABLE = "available"
    VERIFICATION_FAILED = "verification_failed"
    MISSING_LOCAL = "missing_local"
    MISSING_REMOTE = "missing_remote"
    FAILED = "failed"


class TransferDirection(StrEnum):
    DOWNLOAD = "download"
    UPLOAD = "upload"


class TransferStatus(StrEnum):
    PENDING = "pending"
    DRY_RUN = "dry_run"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELED = "canceled"
    STALE = "stale"


class StorageTransferType(StrEnum):
    MODEL_DOWNLOAD = "model_download"
    SNAPSHOT_UPLOAD = "snapshot_upload"
    ARTIFACT_UPLOAD = "artifact_upload"
    VERIFICATION = "verification"


class StorageKind(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class OverwritePolicy(StrEnum):
    FAIL_IF_EXISTS = "fail_if_exists"
    SKIP_IDENTICAL = "skip_identical"
    COPY_MISSING = "copy_missing"
    OVERWRITE_CHANGED = "overwrite_changed"


class VerificationPolicy(StrEnum):
    FULL_CHECKSUM = "full_checksum"
    REMOTE_HASH_AND_SIZE = "remote_hash_and_size"
    SIZE_AND_MANIFEST = "size_and_manifest"
    EXISTENCE_ONLY = "existence_only"


@dataclass(frozen=True, slots=True)
class StorageRemotePath:
    """Validated rclone path; local paths must never be passed here."""

    remote_name: str
    relative_path: str = ""

    def __post_init__(self) -> None:
        remote_name = self.remote_name.strip()
        if (
            not remote_name
            or ":" in remote_name
            or "/" in remote_name
            or "\\" in remote_name
            or any(ord(char) < 32 for char in remote_name)
        ):
            raise ValueError("remote名が不正です")
        raw_relative = self.relative_path.replace("\\", "/").strip()
        if Path(raw_relative).is_absolute() or raw_relative.startswith("/"):
            raise ValueError("絶対パスはremoteパスに使用できません")
        normalized = raw_relative.strip("/")
        parts = tuple(part for part in normalized.split("/") if part)
        if any(part in {".", ".."} for part in parts):
            raise ValueError("remoteパスに親ディレクトリ指定は使用できません")
        if any(":" in part or any(ord(char) < 32 for char in part) for part in parts):
            raise ValueError("remoteパスに制御文字は使用できません")
        object.__setattr__(self, "remote_name", remote_name)
        object.__setattr__(self, "relative_path", "/".join(parts))

    @property
    def rclone_value(self) -> str:
        return (
            f"{self.remote_name}:{self.relative_path}"
            if self.relative_path
            else f"{self.remote_name}:"
        )

    def child(self, *parts: str) -> StorageRemotePath:
        return StorageRemotePath(
            self.remote_name,
            "/".join(filter(None, (self.relative_path, *parts))),
        )

    def is_under(self, root: StorageRemotePath) -> bool:
        if self.remote_name != root.remote_name:
            return False
        path = PurePosixPath(self.relative_path or ".")
        root_path = PurePosixPath(root.relative_path or ".")
        return path == root_path or root_path in path.parents


@dataclass(frozen=True, slots=True)
class StorageRemote:
    name: str
    remote_type: str


@dataclass(frozen=True, slots=True)
class StorageEntry:
    remote_path: StorageRemotePath
    name: str
    size_bytes: int
    modified_at: datetime | None
    hash_type: str | None = None
    hash_value: str | None = None
    is_directory: bool = False


@dataclass(frozen=True, slots=True)
class StorageValidationCheck:
    name: str
    ok: bool
    message: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class StorageValidationResult:
    ok: bool
    checks: tuple[StorageValidationCheck, ...]
    rclone_version: str | None = None

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            check.message for check in self.checks if not check.ok and check.required
        )


@dataclass(frozen=True, slots=True)
class TransferItemPlan:
    relative_path: str
    size_bytes: int
    action: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TransferPlan:
    token: str
    source: str
    destination: str
    items: tuple[TransferItemPlan, ...]
    total_bytes: int
    available_bytes: int | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def copy_count(self) -> int:
        return sum(item.action == "copy" for item in self.items)

    @property
    def skip_count(self) -> int:
        return sum(item.action == "skip" for item in self.items)

    @property
    def conflict_count(self) -> int:
        return sum(item.action == "conflict" for item in self.items)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    level: VerificationPolicy
    expected_size: int | None
    actual_size: int | None
    expected_hash: str | None
    actual_hash: str | None
    message: str = ""


@dataclass(frozen=True, slots=True)
class TransferProgress:
    bytes_transferred: int = 0
    total_bytes: int = 0
    checks: int = 0
    transfers: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0
    speed_bytes_per_second: float = 0.0
    eta_seconds: float | None = None
    current_path: str | None = None
    checking: tuple[str, ...] = ()
    transferring: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManagedModel:
    id: UUID
    display_name: str
    model_type: ModelType
    remote_name: str
    remote_relative_path: str
    remote_file_name: str
    remote_size_bytes: int
    remote_modified_at: datetime | None
    remote_hash_type: str | None
    remote_hash_value: str | None
    local_path: Path | None
    local_size_bytes: int | None
    local_sha256: str | None
    status: ManagedModelStatus
    source: str
    rclone_version: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    downloaded_at: datetime | None
    verified_at: datetime | None
    error_summary: str | None


@dataclass(frozen=True, slots=True)
class ProjectStorageSettings:
    project_id: UUID
    project_remote_root: str
    snapshot_remote_root: str
    training_remote_root: str
    artifact_remote_root: str
    selected_managed_model_id: UUID | None
    overwrite_policy: OverwritePolicy
    verification_policy: VerificationPolicy


@dataclass(frozen=True, slots=True)
class StorageTransferJob:
    id: UUID
    project_id: UUID | None
    snapshot_id: UUID | None
    training_run_id: UUID | None
    transfer_type: StorageTransferType
    source_kind: StorageKind
    destination_kind: StorageKind
    status: TransferStatus
    current_step: str
    item_count: int
    processed_item_count: int
    succeeded_item_count: int
    failed_item_count: int
    skipped_item_count: int
    total_bytes: int
    transferred_bytes: int
    cancel_requested: bool
    pid: int | None
    worker_id: str | None
    heartbeat_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    error_summary: str | None


@dataclass(frozen=True, slots=True)
class TransferManifest:
    schema_version: str
    transfer_job_id: UUID
    transfer_type: StorageTransferType
    project_id: UUID | None
    snapshot_id: UUID | None
    managed_model_id: UUID | None
    source: str
    destination: str
    started_at: datetime | None
    completed_at: datetime | None
    rclone_version: str | None
    settings: dict[str, Any]
    item_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    total_bytes: int
    transferred_bytes: int
    verification_level: VerificationPolicy
    status: TransferStatus
    items: tuple[dict[str, Any], ...]
    error_summary: str | None = None
