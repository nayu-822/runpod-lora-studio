from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import ConceptType, ProjectStatus, SelectionState
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.image_service import ImageService
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
