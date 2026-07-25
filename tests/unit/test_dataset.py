from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import (
    DatasetSettings,
    DatasetSnapshotStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.dataset_repository import DatasetRepository
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.caption_service import CaptionEditingService
from runpod_lora_studio.services.dataset_config_service import DatasetConfigService
from runpod_lora_studio.services.dataset_snapshot_service import (
    DatasetSnapshotService,
    _content_sha256,
)
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


def test_db_failure_after_rename_is_recoverable_and_idempotent(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    original_add_item = DatasetRepository.add_item

    def fail_add_item(self: DatasetRepository, item: object) -> None:
        raise RuntimeError("simulated item commit failure")

    monkeypatch.setattr(DatasetRepository, "add_item", fail_add_item)
    with pytest.raises(UserFacingError, match="DB確定"):
        service.create_snapshot_sync(service.preview(project.id), name="pending")
    pending = service.list_snapshots(project.id)[0]
    assert pending.status is DatasetSnapshotStatus.DB_FINALIZATION_PENDING
    assert pending.snapshot_root.is_dir()

    monkeypatch.setattr(DatasetRepository, "add_item", original_add_item)
    assert service.recover_finalized_snapshots(project.id) == 1
    assert service.recover_finalized_snapshots(project.id) == 0
    recovered = service.list_snapshots(project.id)[0]
    assert recovered.status is DatasetSnapshotStatus.COMPLETED

    with service.session_factory() as session:
        assert len(DatasetRepository(session).list_items(recovered.id)) == 1


def test_db_commit_failure_after_rename_is_recovered(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    original_finish = DatasetRepository.finish
    failed = False

    def fail_once(self: DatasetRepository, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated commit failure")
        original_finish(self, *args, **kwargs)

    monkeypatch.setattr(DatasetRepository, "finish", fail_once)
    with pytest.raises(UserFacingError, match="DB確定"):
        service.create_snapshot_sync(service.preview(project.id), name="commit-fail")
    assert (
        service.list_snapshots(project.id)[0].status
        is DatasetSnapshotStatus.DB_FINALIZATION_PENDING
    )
    assert service.recover_finalized_snapshots(project.id) == 1
    assert (
        service.list_snapshots(project.id)[0].status is DatasetSnapshotStatus.COMPLETED
    )


def test_corrupt_finalized_directory_does_not_recover(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)

    def fail_add_item(self: DatasetRepository, item: object) -> None:
        raise RuntimeError("simulated item failure")

    monkeypatch.setattr(DatasetRepository, "add_item", fail_add_item)
    with pytest.raises(UserFacingError):
        service.create_snapshot_sync(service.preview(project.id), name="corrupt")
    pending = service.list_snapshots(project.id)[0]
    manifest = pending.snapshot_root / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "tamper", encoding="utf-8"
    )
    assert service.recover_finalized_snapshots(project.id) == 0
    assert (
        service.list_snapshots(project.id)[0].status is DatasetSnapshotStatus.CORRUPTED
    )


def test_completed_snapshot_is_not_touched_by_recovery(test_workspace: Path) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    completed = service.create_snapshot_sync(service.preview(project.id), name="done")
    assert service.recover_finalized_snapshots(project.id) == 0
    assert service.list_snapshots(project.id)[0].id == completed.id
    assert (
        service.list_snapshots(project.id)[0].status is DatasetSnapshotStatus.COMPLETED
    )


def test_content_hash_includes_relative_paths_settings_and_is_order_independent() -> (
    None
):
    settings_snapshot = DatasetConfigService.settings_snapshot(DatasetSettings())
    first = SimpleNamespace(
        sequence_number=1,
        snapshot_image_relative_path="images/000001.png",
        caption_relative_path="captions/000001.txt",
        snapshot_image_sha256="a" * 64,
        caption_sha256="b" * 64,
    )
    second = SimpleNamespace(
        sequence_number=2,
        snapshot_image_relative_path="images/000002.png",
        caption_relative_path="captions/000002.txt",
        snapshot_image_sha256="c" * 64,
        caption_sha256="d" * 64,
    )
    baseline = _content_sha256([second, first], "t" * 64, settings_snapshot)
    assert baseline == _content_sha256([first, second], "t" * 64, settings_snapshot)
    assert baseline != _content_sha256(
        [
            SimpleNamespace(**{**first.__dict__, "snapshot_image_sha256": "z" * 64}),
            second,
        ],
        "t" * 64,
        settings_snapshot,
    )
    assert baseline != _content_sha256(
        [
            SimpleNamespace(**{**first.__dict__, "caption_sha256": "z" * 64}),
            second,
        ],
        "t" * 64,
        settings_snapshot,
    )
    assert baseline != _content_sha256([first, second], "u" * 64, settings_snapshot)
    assert baseline != _content_sha256(
        [
            SimpleNamespace(
                **{**first.__dict__, "snapshot_image_relative_path": "x.png"}
            ),
            second,
        ],
        "t" * 64,
        settings_snapshot,
    )
    changed_settings = DatasetConfigService.settings_snapshot(
        DatasetSettings(resolution=768, num_repeats=2)
    )
    assert baseline != _content_sha256([first, second], "t" * 64, changed_settings)


def test_capacity_is_checked_again_before_db_creation(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    preview = service.preview(project.id)
    usage = SimpleNamespace(total=100, used=100, free=0)
    with patch(
        "runpod_lora_studio.services.dataset_snapshot_service.shutil.disk_usage",
        return_value=usage,
    ):
        with pytest.raises(UserFacingError, match="空き容量"):
            service.create_snapshot_sync(preview, name="no-space")
    assert service.list_snapshots(project.id) == []


def test_streaming_copy_uses_copyfileobj_and_report_has_distribution_stats(
    test_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, projects, project, _ = _fixture(test_workspace)
    service = DatasetSnapshotService(settings, projects)
    called = False
    original_copyfileobj = shutil.copyfileobj

    def spy_copy(source: object, target: object, length: int = 0) -> None:
        nonlocal called
        called = True
        original_copyfileobj(source, target, length=length)

    monkeypatch.setattr(
        "runpod_lora_studio.services.dataset_snapshot_service.shutil.copyfileobj",
        spy_copy,
    )
    snapshot = service.create_snapshot_sync(service.preview(project.id), name="stream")
    report = json.loads(
        (snapshot.snapshot_root / "reports" / "dataset_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert called
    assert "resolution_stats" in report
    assert "aspect_ratio_stats" in report
    assert report["resolution_stats"]["short_edge"]["p10"] == 96
