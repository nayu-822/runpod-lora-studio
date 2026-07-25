from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.storage_models import (
    StorageRemotePath,
    StorageTransferType,
    StorageKind,
    TransferStatus,
)
from runpod_lora_studio.external.fake_storage import FakeStorageTransferAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.persistence.storage_repository import StorageRepository
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.services.storage_service import _error_classification


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
    adapter.mutate_after_copy = True

    with pytest.raises(UserFacingError, match="remoteモデルが変更されました"):
        service.download_model(model.id)

    restored = service.get_model(model.id)
    assert restored.status.value == "verification_failed"
    assert (
        restored.local_path is not None and restored.local_path.read_bytes() == before
    )


def test_missing_saved_sha256_forces_a_new_download(
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

    assert len(adapter.copy_calls) == calls + 1
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
