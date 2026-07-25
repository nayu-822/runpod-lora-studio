from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.storage_models import (
    OverwritePolicy,
    StorageEntry,
    StorageKind,
    StorageRemotePath,
    StorageTransferType,
    TransferProgress,
    TransferStatus,
    VerificationPolicy,
)
from runpod_lora_studio.external.fake_storage import FakeStorageTransferAdapter
from runpod_lora_studio.external.rclone import CancelToken, CopyOptions, ListOptions
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.persistence.storage_repository import StorageRepository
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)
from runpod_lora_studio.services.storage_service import (
    StorageService,
    _error_classification,
    _manifest_verification_level,
)


class NoHashFakeAdapter(FakeStorageTransferAdapter):
    def list_entries(
        self, remote_path: StorageRemotePath, options: ListOptions
    ) -> tuple[StorageEntry, ...]:
        return tuple(
            replace(entry, hash_type=None, hash_value=None)
            for entry in super().list_entries(remote_path, options)
        )


class Sha256FakeAdapter(FakeStorageTransferAdapter):
    def list_entries(
        self, remote_path: StorageRemotePath, options: ListOptions
    ) -> tuple[StorageEntry, ...]:
        return tuple(
            replace(
                entry,
                hash_type="sha256",
                hash_value=hashlib.sha256(
                    self.files[entry.remote_path.relative_path]
                ).hexdigest(),
            )
            for entry in super().list_entries(remote_path, options)
        )


class MutatingFakeAdapter(FakeStorageTransferAdapter):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.mutate_after_copy = False

    def copy(self, *args: object, **kwargs: object):
        result = super().copy(*args, **kwargs)  # type: ignore[arg-type]
        if self.mutate_after_copy:
            self.mutate_after_copy = False
            key = "lora-studio/models/sdxl-base.safetensors"
            self.files[key] = b"changed-model"
            self._modified_at = datetime.now(UTC) + timedelta(seconds=1)
        return result


class MissingFinalManifestAdapter(FakeStorageTransferAdapter):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.manifest_copy_count = 0

    def copy(self, *args: object, **kwargs: object):
        result = super().copy(*args, **kwargs)  # type: ignore[arg-type]
        destination = args[1] if len(args) > 1 else None
        if isinstance(
            destination, StorageRemotePath
        ) and destination.relative_path.endswith("transfer-manifest.json"):
            self.manifest_copy_count += 1
            if self.manifest_copy_count == 2:
                self.files.pop(destination.relative_path.strip("/"), None)
        return result


def _settings(test_workspace: Path) -> AppSettings:
    runtime = test_workspace / "runtime"
    settings = AppSettings(
        workspace_root=runtime,
        projects_dir=runtime / "projects",
        models_dir=runtime / "models",
        outputs_dir=runtime / "outputs",
        logs_dir=runtime / "logs",
        temp_dir=runtime / "tmp",
        database_path=runtime / "database" / "studio.sqlite3",
        model_cache_dir=runtime / "models" / "base",
        transfer_temp_dir=runtime / "tmp" / "transfers",
        model_disk_safety_margin_bytes=0,
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    return settings


def _remote_verification_fixture(
    test_workspace: Path, adapter: FakeStorageTransferAdapter
) -> tuple[StorageService, StorageRemotePath, list[tuple[str, Path, int, str]]]:
    local = test_workspace / "verification.bin"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"abc")
    target = StorageRemotePath("gdrive", "verification")
    adapter.files["verification/file.bin"] = b"abc"
    modified = adapter._modified_at.isoformat()
    adapter.files["verification/transfer-manifest.json"] = json.dumps(
        {
            "settings": {"snapshot_content_sha256": "content"},
            "items": [
                {
                    "relative_path": "file.bin",
                    "remote_size": 3,
                    "remote_modified_at": modified,
                    "remote_hash_type": None,
                    "remote_hash": None,
                }
            ],
        }
    ).encode()
    settings = _settings(test_workspace)
    service = StorageService(settings, adapter=adapter)
    return service, target, [("file.bin", local, 3, hashlib.sha256(b"abc").hexdigest())]


def test_model_list_download_and_reuse_with_fake_adapter(test_workspace: Path) -> None:
    settings = _settings(test_workspace)
    remote_path = "lora-studio/models/sdxl-base.safetensors"
    adapter = FakeStorageTransferAdapter(entries={remote_path: b"model-bytes"})
    service = StorageService(settings, adapter=adapter)

    models = service.list_models()
    assert len(models) == 1
    model = models[0]
    plan = service.dry_run_model_download(model.id)
    assert plan.copy_count == 1
    service.download_model(model.id, plan_token=plan.token)
    downloaded = service.get_model(model.id)
    assert downloaded.status.value == "available"
    assert (
        downloaded.local_path is not None
        and downloaded.local_path.read_bytes() == b"model-bytes"
    )
    calls_after_download = len(adapter.copy_calls)

    service.download_model(model.id)
    assert len(adapter.copy_calls) == calls_after_download


def test_model_download_stale_plan_is_rejected_before_job_creation(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    adapter = FakeStorageTransferAdapter(
        entries={"lora-studio/models/sdxl-base.safetensors": b"model-bytes"}
    )
    service = StorageService(settings, adapter=adapter)
    model = service.list_models()[0]

    with pytest.raises(UserFacingError):
        service.start_model_download(model.id, plan_token="stale")

    assert service.list_jobs() == []


def test_same_size_remote_content_change_is_not_reused(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    adapter = FakeStorageTransferAdapter(
        entries={"lora-studio/models/sdxl-base.safetensors": b"model-bytes"}
    )
    service = StorageService(settings, adapter=adapter)
    model = service.list_models()[0]
    service.download_model(model.id)
    adapter.files["lora-studio/models/sdxl-base.safetensors"] = b"other-bytes"
    calls = len(adapter.copy_calls)

    service.download_model(model.id)

    assert len(adapter.copy_calls) == calls + 1
    assert service.get_model(model.id).local_path is not None
    assert service.get_model(model.id).local_path.read_bytes() == b"other-bytes"


def test_remote_change_during_copy_preserves_existing_model(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    adapter = MutatingFakeAdapter(
        entries={"lora-studio/models/sdxl-base.safetensors": b"model-bytes"}
    )
    service = StorageService(settings, adapter=adapter)
    model = service.list_models()[0]
    service.download_model(model.id)
    before = service.get_model(model.id).local_path.read_bytes()  # type: ignore[union-attr]
    adapter.files["lora-studio/models/sdxl-base.safetensors"] = b"new-content"
    adapter._modified_at = datetime.now(UTC) + timedelta(seconds=1)
    adapter.mutate_after_copy = True

    with pytest.raises(UserFacingError, match="remoteモデルが変更されました"):
        service.download_model(model.id)

    restored = service.get_model(model.id)
    assert restored.status.value == "verification_failed"
    assert (
        restored.local_path is not None and restored.local_path.read_bytes() == before
    )


def test_missing_saved_sha256_is_fully_verified_and_saved(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    adapter = FakeStorageTransferAdapter(
        entries={"lora-studio/models/sdxl-base.safetensors": b"model-bytes"}
    )
    service = StorageService(settings, adapter=adapter)
    model = service.list_models()[0]
    service.download_model(model.id)
    with service.session_factory() as session:
        record = StorageRepository(session).get_model(model.id)
        assert record is not None
        record.local_sha256 = None
        session.commit()
    calls = len(adapter.copy_calls)

    service.download_model(model.id)

    assert len(adapter.copy_calls) == calls
    assert service.get_model(model.id).local_sha256 is not None


@pytest.mark.parametrize(
    ("message", "classification"),
    [
        ("401 unauthorized", "authentication"),
        ("permission denied", "permission"),
        ("checksum mismatch", "checksum"),
        ("429 rate limit", "rate_limit"),
        ("connection reset", "network"),
    ],
)
def test_retry_error_classification(message: str, classification: str) -> None:
    assert _error_classification(message) == classification


def test_retry_backoff_uses_injected_sleeper_and_cancellation(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    delays: list[float] = []
    service = StorageService(
        settings,
        adapter=FakeStorageTransferAdapter(),
        sleeper=delays.append,
    )
    token = CancelToken()

    service._sleep_before_retry(2.5, token)

    assert delays == [2.5]


def test_full_checksum_requires_remote_sha256(test_workspace: Path) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, Sha256FakeAdapter()
    )
    assert service._remote_verification_statuses(
        target, files, VerificationPolicy.FULL_CHECKSUM
    ) == {"file.bin": "full_checksum"}

    md5_service, md5_target, md5_files = _remote_verification_fixture(
        test_workspace / "md5", FakeStorageTransferAdapter()
    )
    with pytest.raises(UserFacingError):
        md5_service._remote_verification_statuses(
            md5_target, md5_files, VerificationPolicy.FULL_CHECKSUM
        )


def test_remote_hash_and_size_accepts_md5_and_rejects_mismatch(
    test_workspace: Path,
) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, FakeStorageTransferAdapter()
    )
    assert service._remote_verification_statuses(
        target, files, VerificationPolicy.REMOTE_HASH_AND_SIZE
    ) == {"file.bin": "remote_hash_and_size"}
    service.adapter.files["verification/file.bin"] = b"abd"  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.REMOTE_HASH_AND_SIZE
        )


def test_size_and_manifest_checks_metadata_without_remote_hash(
    test_workspace: Path,
) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, NoHashFakeAdapter()
    )
    assert service._remote_verification_statuses(
        target, files, VerificationPolicy.SIZE_AND_MANIFEST
    ) == {"file.bin": "manifest_metadata_and_size"}

    service.adapter.files["verification/file.bin"] = b"abcd"  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.SIZE_AND_MANIFEST
        )

    service.adapter.files["verification/file.bin"] = b"abc"  # type: ignore[attr-defined]
    service.adapter._modified_at += timedelta(seconds=1)  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.SIZE_AND_MANIFEST
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_manifest",
        "remove_item",
        "remove_size",
        "remove_modified",
    ],
)
def test_size_and_manifest_requires_complete_manifest_item(
    test_workspace: Path, mutation: str
) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, NoHashFakeAdapter()
    )
    adapter = service.adapter
    if mutation == "remove_manifest":
        adapter.files.pop("verification/transfer-manifest.json")  # type: ignore[attr-defined]
    else:
        payload = json.loads(adapter.files["verification/transfer-manifest.json"])  # type: ignore[attr-defined]
        item = payload["items"][0]
        if mutation == "remove_item":
            payload["items"] = []
        elif mutation == "remove_size":
            item.pop("remote_size")
        else:
            item.pop("remote_modified_at")
        adapter.files["verification/transfer-manifest.json"] = json.dumps(
            payload
        ).encode()  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.SIZE_AND_MANIFEST
        )


def test_size_and_manifest_requires_matching_hash_metadata_and_content(
    test_workspace: Path,
) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, FakeStorageTransferAdapter()
    )
    adapter = service.adapter
    payload = json.loads(adapter.files["verification/transfer-manifest.json"])  # type: ignore[attr-defined]
    item = payload["items"][0]
    item["remote_hash_type"] = "md5"
    item["remote_hash"] = hashlib.md5(b"abc").hexdigest()
    adapter.files["verification/transfer-manifest.json"] = json.dumps(payload).encode()  # type: ignore[attr-defined]
    assert service._remote_verification_statuses(
        target, files, VerificationPolicy.SIZE_AND_MANIFEST, "content"
    ) == {"file.bin": "manifest_metadata_and_size"}

    item["remote_hash"] = "0" * 32
    adapter.files["verification/transfer-manifest.json"] = json.dumps(payload).encode()  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.SIZE_AND_MANIFEST, "content"
        )
    item["remote_hash"] = hashlib.md5(b"abc").hexdigest()
    payload["settings"]["snapshot_content_sha256"] = "other-content"
    adapter.files["verification/transfer-manifest.json"] = json.dumps(payload).encode()  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.SIZE_AND_MANIFEST, "content"
        )


def test_existence_only_and_remote_hash_fallback_are_explicit(
    test_workspace: Path,
) -> None:
    service, target, files = _remote_verification_fixture(
        test_workspace, NoHashFakeAdapter()
    )
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.REMOTE_HASH_AND_SIZE
        )
    assert service._remote_verification_statuses(
        target, files, VerificationPolicy.EXISTENCE_ONLY
    ) == {"file.bin": "existence_only"}
    service.adapter.files.pop("verification/file.bin")  # type: ignore[attr-defined]
    with pytest.raises(UserFacingError):
        service._remote_verification_statuses(
            target, files, VerificationPolicy.EXISTENCE_ONLY
        )

    fallback_service, fallback_target, fallback_files = _remote_verification_fixture(
        test_workspace / "fallback", NoHashFakeAdapter()
    )
    fallback_service.settings.storage_remote_hash_fallback = "size_and_manifest"
    assert fallback_service._remote_verification_statuses(
        fallback_target, fallback_files, VerificationPolicy.REMOTE_HASH_AND_SIZE
    ) == {"file.bin": "manifest_metadata_and_size"}
    existence_service, existence_target, existence_files = _remote_verification_fixture(
        test_workspace / "existence-fallback", NoHashFakeAdapter()
    )
    existence_service.settings.storage_remote_hash_fallback = "existence_only"
    assert existence_service._remote_verification_statuses(
        existence_target, existence_files, VerificationPolicy.REMOTE_HASH_AND_SIZE
    ) == {"file.bin": "existence_only"}
    entry = next(
        entry
        for entry in existence_service.adapter.list_entries(
            existence_target, ListOptions(recursive=True)
        )
        if entry.remote_path.relative_path.endswith("file.bin")
    )
    assert existence_service._fallback_identity(
        entry,
        3,
        {"remote_size": 3, "remote_modified_at": entry.modified_at.isoformat()},
    ) == (False, "not_verified")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["full_checksum"], VerificationPolicy.FULL_CHECKSUM),
        (
            ["full_checksum", "remote_hash_and_size"],
            VerificationPolicy.REMOTE_HASH_AND_SIZE,
        ),
        (
            ["remote_hash_and_size", "manifest_metadata_and_size"],
            VerificationPolicy.SIZE_AND_MANIFEST,
        ),
        (
            ["manifest_metadata_and_size", "existence_only"],
            VerificationPolicy.EXISTENCE_ONLY,
        ),
        (["full_checksum", "existence_only"], VerificationPolicy.EXISTENCE_ONLY),
        (["existence_only"], VerificationPolicy.EXISTENCE_ONLY),
        (["verification_failed"], "verification_failed"),
        (["not_verified"], "not_verified"),
    ],
)
def test_manifest_verification_level_reflects_item_results(
    statuses: list[str], expected: VerificationPolicy | str
) -> None:
    items = [{"verification_status": status} for status in statuses]
    assert (
        _manifest_verification_level(items, VerificationPolicy.FULL_CHECKSUM)
        == expected
    )


def test_stale_running_job_is_recovered_idempotently(test_workspace: Path) -> None:
    settings = _settings(test_workspace)
    service = StorageService(settings, adapter=FakeStorageTransferAdapter())
    with service.session_factory() as session:
        job = StorageRepository(session).create_job(
            project_id=None,
            snapshot_id=None,
            transfer_type=StorageTransferType.MODEL_DOWNLOAD,
            source_kind=StorageKind.REMOTE,
            destination_kind=StorageKind.LOCAL,
            item_count=1,
            total_bytes=1,
        )
        job.status = TransferStatus.RUNNING.value
        job.heartbeat_at = datetime.now(UTC) - timedelta(seconds=600)
        session.commit()

    assert service.recover_stale_jobs() == 1
    assert service.recover_stale_jobs() == 0
    assert service.list_jobs()[0].status is TransferStatus.STALE


def test_live_rclone_process_is_not_recovered_as_stale(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(test_workspace)
    service = StorageService(settings, adapter=FakeStorageTransferAdapter())
    with service.session_factory() as session:
        job = StorageRepository(session).create_job(
            project_id=None,
            snapshot_id=None,
            transfer_type=StorageTransferType.MODEL_DOWNLOAD,
            source_kind=StorageKind.REMOTE,
            destination_kind=StorageKind.LOCAL,
            item_count=1,
            total_bytes=1,
        )
        job.status = TransferStatus.RUNNING.value
        job.pid = 1234
        job.heartbeat_at = datetime.now(UTC) - timedelta(seconds=600)
        session.commit()
    monkeypatch.setattr(
        "runpod_lora_studio.services.storage_service._pid_exists", lambda _pid: True
    )
    monkeypatch.setattr(
        "runpod_lora_studio.services.storage_service._is_expected_rclone_process",
        lambda _pid, _executable: True,
    )

    assert service.recover_stale_jobs() == 0
    assert service.list_jobs()[0].status is TransferStatus.RUNNING

    monkeypatch.setattr(
        "runpod_lora_studio.services.storage_service._is_expected_rclone_process",
        lambda _pid, _executable: False,
    )
    assert service.recover_stale_jobs() == 1
    assert service.list_jobs()[0].status is TransferStatus.STALE


def test_transfer_progress_is_cumulative_and_monotonic(test_workspace: Path) -> None:
    settings = _settings(test_workspace)
    service = StorageService(settings, adapter=FakeStorageTransferAdapter())
    with service.session_factory() as session:
        job = StorageRepository(session).create_job(
            project_id=None,
            snapshot_id=None,
            transfer_type=StorageTransferType.MODEL_DOWNLOAD,
            source_kind=StorageKind.REMOTE,
            destination_kind=StorageKind.LOCAL,
            item_count=2,
            total_bytes=12,
        )
        session.commit()
        job_id = UUID(job.id)
    service._set_job_running(job_id)
    service._set_current_file(job_id, 10)
    service._progress_job(job_id, TransferProgress(4, 10, 1, "first"))
    service._progress_job(job_id, TransferProgress(3, 10, 1, "first"))
    first = service.list_jobs()[0]
    assert first.transferred_bytes == 4
    service._complete_current_file(job_id, 10)
    service._set_current_file(job_id, 2)
    service._progress_job(job_id, TransferProgress(2, 2, 1, "second"))
    second = service.list_jobs()[0]
    assert second.transferred_bytes == 12
    assert second.completed_transferred_bytes == 10
    assert second.current_file_transferred_bytes == 2
    service._finish_job(job_id, TransferStatus.CANCELED, None)


@pytest.mark.parametrize(
    "remote_name,relative_path",
    [
        ("gdrive:other", "models/model.safetensors"),
        ("gdrive", "../model.safetensors"),
        ("gdrive", "/absolute/model.safetensors"),
        ("gdrive", "models/bad\u0000name.safetensors"),
    ],
)
def test_remote_paths_reject_escape_and_remote_switching(
    remote_name: str, relative_path: str
) -> None:
    with pytest.raises(ValueError):
        StorageRemotePath(remote_name, relative_path)


def test_completed_snapshot_upload_is_verified_and_manifested(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("storage-test"))
    from PIL import Image

    source = test_workspace / "source.png"
    Image.new("RGB", (128, 96), "red").save(source)
    from runpod_lora_studio.domain.models import SelectionState
    from runpod_lora_studio.services.image_service import ImageService

    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.ACCEPTED
    )
    from runpod_lora_studio.services.caption_service import CaptionEditingService

    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "character"
    )
    datasets = DatasetSnapshotService(settings, projects)
    snapshot = datasets.create_snapshot_sync(
        datasets.preview(project.id), name="upload"
    )
    adapter = FakeStorageTransferAdapter()
    service = StorageService(settings, adapter=adapter, datasets=datasets)

    plan = service.dry_run_snapshot_upload(snapshot.id)
    assert plan.errors == ()
    job_id = service.upload_snapshot(snapshot.id, plan_token=plan.token)

    job = next(job for job in service.list_jobs(project.id) if job.id == job_id)
    assert job.status.value == "completed"
    assert any(path.endswith("transfer-manifest.json") for path in adapter.files)
    assert (snapshot.snapshot_root / "manifest.json").is_file()

    identical_plan = service.dry_run_snapshot_upload(snapshot.id)
    assert identical_plan.errors == ()
    assert identical_plan.skip_count == len(identical_plan.items)

    remote_root = identical_plan.destination.removeprefix("gdrive:").strip("/")
    content_key = next(
        key
        for key in adapter.files
        if key.startswith(remote_root + "/")
        and not key.endswith("transfer-manifest.json")
    )
    original = adapter.files[content_key]
    adapter.files[content_key] = b"x" * len(original)
    changed_plan = service.dry_run_snapshot_upload(snapshot.id)
    assert changed_plan.conflict_count > 0
    assert service.dry_run_snapshot_upload(
        snapshot.id, overwrite_policy=OverwritePolicy.FAIL_IF_EXISTS
    ).conflict_count == len(changed_plan.items)
    assert (
        service.dry_run_snapshot_upload(
            snapshot.id, overwrite_policy=OverwritePolicy.COPY_MISSING
        ).conflict_count
        > 0
    )
    assert (
        service.dry_run_snapshot_upload(
            snapshot.id, overwrite_policy=OverwritePolicy.OVERWRITE_CHANGED
        ).copy_count
        > 0
    )


def _final_manifest_fixture(
    test_workspace: Path,
) -> tuple[
    StorageService,
    FakeStorageTransferAdapter,
    Path,
    StorageRemotePath,
    UUID,
    UUID,
    UUID,
    list[dict[str, Any]],
    str,
]:
    settings = _settings(test_workspace)
    adapter = FakeStorageTransferAdapter()
    service = StorageService(settings, adapter=adapter)
    job_id = uuid4()
    project_id = uuid4()
    snapshot_id = uuid4()
    target = StorageRemotePath("gdrive", "final-manifest")
    digest = hashlib.sha256(b"abc").hexdigest()
    items = [
        service._manifest_item(
            "image.png", 3, digest, "completed", "remote_hash_and_size"
        )
    ]
    project_settings = service.get_project_storage_settings(project_id)
    local_manifest = service._write_transfer_manifest(
        job_id,
        StorageTransferType.SNAPSHOT_UPLOAD,
        project_id,
        snapshot_id,
        target,
        items,
        TransferStatus.COMPLETED,
        project_settings,
        "snapshot-content",
    )
    adapter.copy(
        local_manifest,
        target.child("transfer-manifest.json"),
        CopyOptions(overwrite_policy=OverwritePolicy.OVERWRITE_CHANGED, checksum=True),
    )
    return (
        service,
        adapter,
        local_manifest,
        target,
        job_id,
        project_id,
        snapshot_id,
        items,
        "snapshot-content",
    )


def test_final_manifest_remote_readback_succeeds(
    test_workspace: Path,
) -> None:
    (
        service,
        adapter,
        local_manifest,
        target,
        job_id,
        project_id,
        snapshot_id,
        items,
        content_sha256,
    ) = _final_manifest_fixture(test_workspace)
    service._validate_final_remote_manifest(
        target,
        local_manifest,
        job_id,
        project_id,
        snapshot_id,
        items,
        VerificationPolicy.REMOTE_HASH_AND_SIZE,
        content_sha256,
    )
    assert "final-manifest/transfer-manifest.json" in adapter.files


@pytest.mark.parametrize(
    "mutation",
    ["missing", "status", "level", "content", "count", "path", "item_status", "job"],
)
def test_final_manifest_remote_readback_rejects_stale_or_invalid_manifest(
    test_workspace: Path, mutation: str
) -> None:
    (
        service,
        adapter,
        local_manifest,
        target,
        job_id,
        project_id,
        snapshot_id,
        items,
        content_sha256,
    ) = _final_manifest_fixture(test_workspace)
    key = "final-manifest/transfer-manifest.json"
    if mutation == "missing":
        del adapter.files[key]
    else:
        payload = json.loads(adapter.files[key])
        if mutation == "status":
            payload["status"] = "running"
        elif mutation == "level":
            payload["verification_level"] = "full_checksum"
        elif mutation == "content":
            payload["settings"]["snapshot_content_sha256"] = "other"
        elif mutation == "count":
            payload["item_count"] = 2
        elif mutation == "path":
            payload["items"][0]["relative_path"] = "other.png"
        elif mutation == "item_status":
            payload["items"][0]["verification_status"] = "not_verified"
        elif mutation == "job":
            payload["transfer_job_id"] = str(uuid4())
        adapter.files[key] = json.dumps(payload).encode()

    with pytest.raises(UserFacingError, match="最終転送マニフェストを確認できません"):
        service._validate_final_remote_manifest(
            target,
            local_manifest,
            job_id,
            project_id,
            snapshot_id,
            items,
            VerificationPolicy.REMOTE_HASH_AND_SIZE,
            content_sha256,
        )
    if mutation != "missing":
        assert key in adapter.files


def test_missing_final_manifest_fails_without_deleting_snapshot_files(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    settings.storage_verification_policy = VerificationPolicy.SIZE_AND_MANIFEST
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("final-manifest-failure"))
    from PIL import Image

    source = test_workspace / "source.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    from runpod_lora_studio.domain.models import SelectionState
    from runpod_lora_studio.services.caption_service import CaptionEditingService
    from runpod_lora_studio.services.image_service import ImageService

    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.ACCEPTED
    )
    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "purple"
    )
    datasets = DatasetSnapshotService(settings, projects)
    snapshot = datasets.create_snapshot_sync(
        datasets.preview(project.id), name="final-manifest-failure"
    )
    adapter = MissingFinalManifestAdapter()
    service = StorageService(settings, adapter=adapter, datasets=datasets)

    with pytest.raises(UserFacingError, match="最終転送マニフェストを確認できません"):
        service.upload_snapshot(snapshot.id)

    job = service.list_jobs(project.id)[0]
    assert job.status is TransferStatus.FAILED
    assert job.error_summary is not None
    assert "最終転送マニフェストを確認できません" in job.error_summary
    assert adapter.manifest_copy_count == 2
    assert any(not key.endswith("transfer-manifest.json") for key in adapter.files)


def test_old_manifest_sha256_does_not_allow_skip_without_remote_hash(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("no-hash"))
    from PIL import Image

    source = test_workspace / "source.png"
    Image.new("RGB", (64, 64), "blue").save(source)
    from runpod_lora_studio.domain.models import SelectionState
    from runpod_lora_studio.services.caption_service import CaptionEditingService
    from runpod_lora_studio.services.image_service import ImageService

    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.ACCEPTED
    )
    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "blue"
    )
    datasets = DatasetSnapshotService(settings, projects)
    snapshot = datasets.create_snapshot_sync(
        datasets.preview(project.id), name="no-hash"
    )
    adapter = FakeStorageTransferAdapter()
    service = StorageService(settings, adapter=adapter, datasets=datasets)
    service.upload_snapshot(snapshot.id)
    no_hash_adapter = NoHashFakeAdapter()
    no_hash_adapter.files = adapter.files.copy()
    service.adapter = no_hash_adapter
    remote_root = (
        service.dry_run_snapshot_upload(snapshot.id)
        .destination.removeprefix("gdrive:")
        .strip("/")
    )
    service.settings.storage_remote_hash_fallback = "existence_only"
    existence_plan = service.dry_run_snapshot_upload(snapshot.id)
    assert existence_plan.skip_count == 0
    assert existence_plan.conflict_count == len(existence_plan.items)
    assert (
        service.dry_run_snapshot_upload(
            snapshot.id, overwrite_policy=OverwritePolicy.OVERWRITE_CHANGED
        ).copy_count
        > 0
    )
    content_key = next(
        key
        for key in no_hash_adapter.files
        if key.startswith(remote_root + "/")
        and not key.endswith("transfer-manifest.json")
    )
    no_hash_adapter.files[content_key] = b"z" * len(no_hash_adapter.files[content_key])

    plan = service.dry_run_snapshot_upload(snapshot.id)
    assert plan.skip_count == 0
    assert plan.conflict_count == len(plan.items)


def test_transfer_manifest_contains_remote_metadata_and_verification(
    test_workspace: Path,
) -> None:
    settings = _settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("manifest-fields"))
    from PIL import Image

    source = test_workspace / "source.png"
    Image.new("RGB", (64, 64), "green").save(source)
    from runpod_lora_studio.domain.models import SelectionState
    from runpod_lora_studio.services.caption_service import CaptionEditingService
    from runpod_lora_studio.services.image_service import ImageService

    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.ACCEPTED
    )
    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "green"
    )
    datasets = DatasetSnapshotService(settings, projects)
    snapshot = datasets.create_snapshot_sync(
        datasets.preview(project.id), name="manifest-fields"
    )
    adapter = FakeStorageTransferAdapter()
    service = StorageService(settings, adapter=adapter, datasets=datasets)
    service.upload_snapshot(snapshot.id)
    manifest_key = next(
        key for key in adapter.files if key.endswith("transfer-manifest.json")
    )
    payload = json.loads(adapter.files[manifest_key])
    item = payload["items"][0]
    assert item["remote_hash_type"] == "md5"
    assert item["remote_hash"]
    assert item["remote_size"] == item["size"]
    assert item["remote_modified_at"]
    assert item["verification_status"] == "remote_hash_and_size"
