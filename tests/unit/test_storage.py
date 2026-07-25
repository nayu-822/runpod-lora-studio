from __future__ import annotations

from pathlib import Path

import pytest

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.storage_models import StorageRemotePath
from runpod_lora_studio.external.fake_storage import FakeStorageTransferAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)
from runpod_lora_studio.services.storage_service import StorageService


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
