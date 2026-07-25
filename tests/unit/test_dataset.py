from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import (
    DatasetSettings,
    DatasetSnapshotStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.caption_service import CaptionEditingService
from runpod_lora_studio.services.dataset_config_service import DatasetConfigService
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)


def _fixture(test_workspace: Path, *, caption: str | None = "character, blue_hair"):
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        models_dir=test_workspace / "runtime" / "models",
        outputs_dir=test_workspace / "runtime" / "outputs",
        logs_dir=test_workspace / "runtime" / "logs",
        temp_dir=test_workspace / "runtime" / "tmp",
        database_path=test_workspace / "runtime" / "database" / "db.sqlite3",
        tagger_model_dir=test_workspace / "runtime" / "models" / "taggers" / "wd14",
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("dataset"))
    source = test_workspace / "source.png"
    Image.new("RGB", (128, 96), "red").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    image_service = ImageService(settings, projects)
    image_service.change_state(project.id, [image.id], SelectionState.ACCEPTED)
    if caption is not None:
        CaptionEditingService(settings, projects).save_image_caption(
            project.id, image.id, caption
        )
    return settings, projects, project, image


def test_snapshot_copies_current_dataset_and_is_immutable(test_workspace: Path) -> None:
    settings, projects, project, image = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)

    preview = service.preview(project.id)
    assert preview.summary.target_image_count == 1
    assert preview.summary.error_count == 0
    assert preview.images[0].selection_state is SelectionState.ACCEPTED

    snapshot = service.create_snapshot_sync(preview, name="first")
    assert snapshot.status is DatasetSnapshotStatus.COMPLETED
    root = snapshot.snapshot_root
    assert (root / "images").is_dir()
    assert (root / "captions").is_dir()
    assert (root / "configs" / "dataset.toml").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "reports" / "dataset_report.json").is_file()

    caption_path = next((root / "captions").glob("*.txt"))
    assert caption_path.read_bytes().endswith(b"\n")
    assert b"\r" not in caption_path.read_bytes()
    assert caption_path.read_text(encoding="utf-8") == "character, blue_hair\n"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_count"] == 1
    assert manifest["items"][0]["image_id"] == str(image.id)
    tomllib.loads((root / "configs" / "dataset.toml").read_text(encoding="utf-8"))

    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "changed"
    )
    assert caption_path.read_text(encoding="utf-8") == "character, blue_hair\n"
    assert service.revalidate(snapshot.id) is DatasetSnapshotStatus.COMPLETED


def test_preview_token_rejects_caption_or_selection_changes(
    test_workspace: Path,
) -> None:
    settings, projects, project, image = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    preview = service.preview(project.id)

    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "edited"
    )
    with pytest.raises(UserFacingError, match="有効期限"):
        service.create_snapshot_sync(preview, name="stale")
    assert service.list_snapshots(project.id) == []

    preview = service.preview(project.id)
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.EXCLUDED
    )
    with pytest.raises(UserFacingError, match="有効期限"):
        service.create_snapshot_sync(preview, name="state-stale")
    assert service.list_snapshots(project.id) == []


def test_required_file_or_caption_error_blocks_creation(test_workspace: Path) -> None:
    settings, projects, project, image = _fixture(test_workspace, caption=None)
    service = DatasetSnapshotService(settings, projects)
    preview = service.preview(project.id)
    assert preview.summary.caption_missing_count == 1
    assert preview.summary.error_count > 0
    with pytest.raises(UserFacingError, match="エラー"):
        service.create_snapshot_sync(preview, name="missing-caption")
    assert service.list_snapshots(project.id) == []

    source = image.original_path
    source.unlink()
    missing_preview = service.preview(project.id)
    assert missing_preview.summary.missing_file_count == 1
    with pytest.raises(UserFacingError, match="エラー"):
        service.create_snapshot_sync(missing_preview, name="missing-file")


def test_warning_requires_confirmation_and_reports_are_deterministic(
    test_workspace: Path,
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    projects.update(project.id, ProjectInput("dataset", trigger_words=("trigger",)))
    service = DatasetSnapshotService(settings, projects)
    preview = service.preview(project.id)
    assert preview.summary.trigger_missing_count == 1
    with pytest.raises(UserFacingError, match="警告"):
        service.create_snapshot_sync(preview, name="without-confirmation")

    snapshot = service.create_snapshot_sync(
        preview, name="with-confirmation", confirm_warnings=True
    )
    report = json.loads(
        (snapshot.snapshot_root / "reports" / "dataset_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["image_count"] == 1
    assert report["trigger_word_rate"] == 0
    assert "blue_hair" in (
        snapshot.snapshot_root / "reports" / "tag_frequency.csv"
    ).read_text(encoding="utf-8")


def test_revalidate_marks_corruption_without_deleting_snapshot(
    test_workspace: Path,
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    snapshot = service.create_snapshot_sync(service.preview(project.id), name="check")
    image_path = next((snapshot.snapshot_root / "images").glob("*"))
    image_path.write_bytes(image_path.read_bytes() + b"tampered")

    assert service.revalidate(snapshot.id) is DatasetSnapshotStatus.CORRUPTED
    assert image_path.exists()
    assert (
        service.list_snapshots(project.id)[0].status is DatasetSnapshotStatus.CORRUPTED
    )


def test_dataset_config_rejects_invalid_values_and_writes_safe_toml() -> None:
    config = DatasetConfigService()
    issues = config.validate(
        DatasetSettings(resolution=0, min_bucket_reso=1024, max_bucket_reso=256)
    )
    assert any(issue.issue_code == "resolution_invalid" for issue in issues)
    assert any(issue.issue_code == "bucket_range_invalid" for issue in issues)
    text = config.to_toml(DatasetSettings(), image_dir="images")
    parsed = tomllib.loads(text)
    assert parsed["general"]["resolution"] == 1024
    assert parsed["datasets"][0]["subsets"][0]["image_dir"] == "images"
