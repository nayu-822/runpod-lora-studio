from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageFilter
from sqlalchemy import create_engine, func, select

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import (
    InspectionRule,
    InspectionStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.models import (
    Base,
    ImageInspectionResultRecord,
)
from runpod_lora_studio.services.image_inspection_service import (
    ImageInspectionService,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService


def inspection_settings(test_workspace: Path) -> AppSettings:
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
    Base.metadata.create_all(
        create_engine(f"sqlite:///{settings.database_path.as_posix()}")
    )
    return settings


def test_inspection_persists_duplicate_and_quality_results_without_state_change(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Quality"))
    image_service = ImageService(settings, projects)

    clear = test_workspace / "clear.png"
    Image.new("RGB", (640, 640), "red").save(clear)
    duplicate = test_workspace / "duplicate.png"
    duplicate.write_bytes(clear.read_bytes())
    result = image_service.register_uploads(project.id, [clear, duplicate])
    first, second = result.successes
    image_service.change_state(project.id, [second.id], SelectionState.ACCEPTED)

    service = ImageInspectionService(settings, projects)
    run = service.inspect_project(project.id)
    assert run.inspected_image_count == 2
    assert run.summary.exact_duplicate_count == 1
    second_results = {item.rule: item for item in service.get_results(second.id)}
    assert (
        second_results[InspectionRule.EXACT_DUPLICATE].status
        is InspectionStatus.WARNING
    )
    assert (
        second_results[InspectionRule.LOW_INFORMATION].status
        is InspectionStatus.WARNING
    )
    assert projects.get(project.id).image_counts[SelectionState.ACCEPTED] == 1

    service.inspect_project(project.id)
    engine = create_engine(f"sqlite:///{settings.database_path.as_posix()}")
    with engine.connect() as connection:
        count = connection.scalar(
            select(func.count())
            .select_from(ImageInspectionResultRecord)
            .where(ImageInspectionResultRecord.image_id == str(first.id))
        )
    assert count == len(InspectionRule)


def test_inspection_detects_resolution_aspect_and_keeps_scores_finite(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Dimensions"))
    source = test_workspace / "wide.png"
    Image.new("RGB", (32, 1024), "white").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )

    service = ImageInspectionService(settings, projects)
    service.inspect_image(project.id, image.id)
    results = {item.rule: item for item in service.get_results(image.id)}
    assert (
        results[InspectionRule.RESOLUTION_TOO_SMALL].status is InspectionStatus.WARNING
    )
    assert (
        results[InspectionRule.ASPECT_RATIO_EXTREME].status is InspectionStatus.WARNING
    )
    assert all(
        item.score is None or math.isfinite(item.score) for item in results.values()
    )


def test_missing_and_corrupt_images_are_failed_independently(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Failures"))
    valid = test_workspace / "valid.png"
    second = test_workspace / "second.png"
    Image.new("RGB", (640, 640), "blue").save(valid)
    Image.new("RGB", (640, 640), "green").save(second)
    registered = (
        ImageService(settings, projects)
        .register_uploads(project.id, [valid, second])
        .successes
    )
    registered[0].original_path.unlink()
    registered[1].original_path.write_bytes(b"not an image")

    service = ImageInspectionService(settings, projects)
    run = service.inspect_project(project.id)
    assert run.failed_image_count == 2
    assert all(
        item.status is InspectionStatus.FAILED
        for item in service.get_results(registered[0].id)
    )


def test_blur_score_is_higher_for_sharp_image(test_workspace: Path) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Blur"))
    sharp_path = test_workspace / "sharp.png"
    blurred_path = test_workspace / "blurred.png"
    sharp = Image.effect_noise((640, 640), 80).convert("RGB")
    sharp.save(sharp_path)
    sharp.filter(ImageFilter.GaussianBlur(12)).save(blurred_path)
    images = (
        ImageService(settings, projects)
        .register_uploads(project.id, [sharp_path, blurred_path])
        .successes
    )

    service = ImageInspectionService(settings, projects)
    service.inspect_project(project.id)
    sharp_result = next(
        item
        for item in service.get_results(images[0].id)
        if item.rule is InspectionRule.BLUR_SCORE
    )
    blurred_result = next(
        item
        for item in service.get_results(images[1].id)
        if item.rule is InspectionRule.BLUR_SCORE
    )
    assert sharp_result.score is not None and blurred_result.score is not None
    assert sharp_result.score > blurred_result.score
