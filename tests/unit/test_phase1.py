from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import ConceptType, ProjectStatus, SelectionState
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.image_service import (
    ImageService,
    UploadFailure,
    UploadResult,
)
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)


def phase1_settings(test_workspace: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        models_dir=test_workspace / "runtime" / "models",
        outputs_dir=test_workspace / "runtime" / "outputs",
        logs_dir=test_workspace / "runtime" / "logs",
        temp_dir=test_workspace / "runtime" / "tmp",
        database_path=test_workspace / "runtime" / "database" / "studio.sqlite3",
    )
    ensure_runtime_directories(settings)
    engine = create_engine(f"sqlite:///{settings.database_path.as_posix()}")
    Base.metadata.create_all(engine)
    return settings


def test_project_create_list_and_trigger_word_normalization(
    test_workspace: Path,
) -> None:
    service = ProjectService(phase1_settings(test_workspace))

    project = service.create(
        ProjectInput(
            "  My Project  ",
            " description ",
            ConceptType.CHARACTER,
            (" trigger ", "trigger", "", "second"),
        )
    )

    assert project.name == "My Project"
    assert project.description == "description"
    assert project.trigger_words == ("trigger", "second")
    assert project.status is ProjectStatus.DRAFT
    listed = service.list_projects()
    assert listed[0].id == project.id
    assert listed[0].image_counts[SelectionState.PENDING] == 0


def test_image_upload_is_verified_hashed_thumbnailed_and_persisted(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Images"))
    source = test_workspace / "unsafe name.png"
    Image.new("RGBA", (640, 320), (255, 0, 0, 128)).save(source)

    result = ImageService(settings, projects).register_uploads(project.id, [source])

    assert len(result.successes) == 1
    image = result.successes[0]
    assert image.selection_state is SelectionState.PENDING
    assert image.original_path.exists()
    assert image.thumbnail_path.exists()
    assert image.stored_filename != source.name
    assert image.mime_type == "image/png"
    assert image.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(image.thumbnail_path) as thumbnail:
        assert thumbnail.width <= 320
        assert thumbnail.height <= 320

    image_service = ImageService(settings, projects)
    listed, total = image_service.list_images(project.id)
    assert total == 1
    assert listed[0].id == image.id
    assert (
        image_service.change_state(project.id, [image.id], SelectionState.ACCEPTED) == 1
    )
    assert (
        image_service.list_images(project.id)[0][0].selection_state
        is SelectionState.ACCEPTED
    )


def test_invalid_image_is_reported_without_persisting_files(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Images"))
    source = test_workspace / "not-image.png"
    source.write_bytes(b"not an image")

    result = ImageService(settings, projects).register_uploads(project.id, [source])

    assert len(result.successes) == 0
    assert len(result.failures) == 1
    assert list((projects.project_root(project.id) / "originals").iterdir()) == []


def test_image_state_cannot_be_changed_from_another_project(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    first = projects.create(ProjectInput("First"))
    second = projects.create(ProjectInput("Second"))
    source = test_workspace / "image.jpg"
    Image.new("RGB", (10, 10), "blue").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(first.id, [source])
        .successes[0]
    )

    with pytest.raises(UserFacingError):
        ImageService(settings, projects).change_state(
            second.id, [image.id], SelectionState.EXCLUDED
        )


@pytest.mark.parametrize(
    ("extension", "image_format", "mime_type"),
    [
        ("jpg", "JPEG", "image/jpeg"),
        ("png", "PNG", "image/png"),
        ("webp", "WEBP", "image/webp"),
    ],
)
def test_supported_image_formats_are_registered(
    test_workspace: Path, extension: str, image_format: str, mime_type: str
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Formats"))
    source = test_workspace / f"image.{extension}"
    Image.new("RGB", (20, 10), "green").save(source, format=image_format)

    result = ImageService(settings, projects).register_uploads(project.id, [source])

    assert len(result.successes) == 1
    assert result.successes[0].mime_type == mime_type


def test_mixed_upload_keeps_success_and_reports_duplicate_warning(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Mixed"))
    valid = test_workspace / "valid.png"
    duplicate = test_workspace / "duplicate.png"
    broken = test_workspace / "broken.png"
    Image.new("RGB", (10, 10), "red").save(valid)
    duplicate.write_bytes(valid.read_bytes())
    broken.write_bytes(b"broken")

    result = ImageService(settings, projects).register_uploads(
        project.id, [valid, duplicate, broken]
    )

    assert len(result.successes) == 2
    assert result.duplicate_warning_count == 1
    assert len(result.failures) == 1

    failed = ImageService(settings, projects).register_uploads(project.id, [broken])
    assert len(failed.failures) == 1
    assert "破損" in failed.failures[0].reason
    assert not list(settings.temp_dir.glob("*.upload"))


def test_all_uploads_can_fail_without_leaving_temporary_files(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Failures"))
    files = [test_workspace / "broken-a.png", test_workspace / "broken-b.webp"]
    for file in files:
        file.write_bytes(b"not an image")

    result = ImageService(settings, projects).register_uploads(project.id, files)

    assert result.successes == ()
    assert len(result.failures) == 2
    assert not list(settings.temp_dir.glob("*.upload"))


def test_upload_limits_are_rejected(test_workspace: Path) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Limits"))
    source = test_workspace / "large.png"
    Image.new("RGB", (20, 20), "black").save(source)

    small_file_settings = settings.model_copy(update={"max_upload_file_size_bytes": 1})
    size_result = ImageService(small_file_settings, projects).register_uploads(
        project.id, [source]
    )
    assert "ファイルサイズ上限" in size_result.failures[0].reason

    pixel_settings = settings.model_copy(update={"max_image_pixels": 10})
    pixel_result = ImageService(pixel_settings, projects).register_uploads(
        project.id, [source]
    )
    assert "ピクセル数上限" in pixel_result.failures[0].reason

    count_settings = settings.model_copy(update={"max_upload_files": 1})
    with pytest.raises(UserFacingError, match="1枚まで"):
        ImageService(count_settings, projects).register_uploads(
            project.id, [source, source]
        )


def test_exif_orientation_is_applied_to_thumbnail_without_changing_original(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Exif"))
    source = test_workspace / "oriented.jpg"
    image = Image.new("RGB", (100, 50), "blue")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif.tobytes())
    original_bytes = source.read_bytes()

    result = ImageService(settings, projects).register_uploads(project.id, [source])

    assert source.read_bytes() == original_bytes
    with Image.open(result.successes[0].thumbnail_path) as thumbnail:
        assert thumbnail.height > thumbnail.width


def test_same_sha_is_not_deleted_or_excluded(test_workspace: Path) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Duplicates"))
    first = test_workspace / "first.png"
    second = test_workspace / "second.png"
    Image.new("RGB", (10, 10), "black").save(first)
    second.write_bytes(first.read_bytes())

    result = ImageService(settings, projects).register_uploads(
        project.id, [first, second]
    )
    listed, total = ImageService(settings, projects).list_images(project.id)

    assert len(result.successes) == 2
    assert result.duplicate_warning_count == 1
    assert total == 2
    assert all(image.selection_state is SelectionState.PENDING for image in listed)


def test_upload_result_format_is_safe_and_human_readable() -> None:
    from runpod_lora_studio.ui.phase1 import format_upload_result

    message = format_upload_result(
        UploadResult(
            successes=(),
            failures=(
                UploadFailure("broken.png", "画像が破損しています。"),
                UploadFailure("secret.png", r"C:\private\secret.png"),
            ),
            duplicate_warning_count=2,
        )
    )

    assert "成功: 0件" in message
    assert "失敗: 2件" in message
    assert "同一内容の可能性がある画像: 2件" in message
    assert "broken.png: 画像が破損しています。" in message
    assert "C:\\private" not in message


def test_gallery_selection_uses_internal_ids() -> None:
    from runpod_lora_studio.ui.phase1 import selected_gallery_ids

    assert selected_gallery_ids(1, ["first", "second"]) == ["second"]
    assert selected_gallery_ids((0,), ["first", "second"]) == ["first"]
    assert selected_gallery_ids(5, ["first"]) == []


def test_image_listing_supports_paging_filters_search_and_boundaries(
    test_workspace: Path,
) -> None:
    settings = phase1_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Paging"))
    service = ImageService(settings, projects)
    for index in range(5):
        source = test_workspace / f"image-{index}.png"
        Image.new("RGB", (10, 10), (index, index, index)).save(source)
        service.register_uploads(project.id, [source])

    first, total = service.list_images(project.id, page=1, page_size=2)
    last, _ = service.list_images(project.id, page=3, page_size=2)
    beyond, _ = service.list_images(project.id, page=99, page_size=2)
    negative, _ = service.list_images(project.id, page=0, page_size=0)
    searched, searched_total = service.list_images(
        project.id, search="image-3", page=1, page_size=10
    )
    changed = service.change_state(project.id, [first[0].id], SelectionState.ACCEPTED)
    accepted, accepted_total = service.list_images(
        project.id, state=SelectionState.ACCEPTED
    )

    assert total == 5
    assert len(first) == 2
    assert len(last) == 1
    assert beyond == []
    assert len(negative) == 1
    assert searched_total == 1 and searched[0].original_filename == "image-3.png"
    assert changed == 1
    assert (
        accepted_total == 1 and accepted[0].selection_state is SelectionState.ACCEPTED
    )
