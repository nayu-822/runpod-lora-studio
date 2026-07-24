from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageFilter
from sqlalchemy import create_engine, func, select

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import (
    ImageInspectionResult,
    InspectionRule,
    InspectionStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.models import (
    Base,
    ImageInspectionResultRecord,
)
from runpod_lora_studio.persistence.repositories import ImageInspectionRepository
from runpod_lora_studio.services.image_inspection_service import (
    ImageInspectionService,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService


def make_results(
    image_id, statuses: dict[InspectionRule, InspectionStatus], version: str
) -> tuple[ImageInspectionResult, ...]:
    now = datetime.now(UTC)
    return tuple(
        ImageInspectionResult(
            image_id=image_id,
            rule=rule,
            status=statuses[rule],
            score=1.0,
            threshold=0.0,
            reason="test",
            detector_version=version,
            inspected_at=now,
        )
        for rule in InspectionRule
    )


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


def test_summary_uses_one_exclusive_state_per_image(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Summary"))
    image_service = ImageService(settings, projects)
    sources = []
    for index in range(4):
        source = test_workspace / f"summary-{index}.png"
        Image.new("RGB", (640, 640), (index * 40, 80, 120)).save(source)
        sources.append(source)
    images = image_service.register_uploads(project.id, sources).successes
    service = ImageInspectionService(settings, projects)
    passed = {rule: InspectionStatus.PASS for rule in InspectionRule}
    warning = passed | {InspectionRule.BLUR_SCORE: InspectionStatus.WARNING}
    failed = warning | {InspectionRule.LOW_INFORMATION: InspectionStatus.FAILED}
    service._save_results(
        images[0].id, make_results(images[0].id, passed, service.detector_version)
    )
    service._save_results(
        images[1].id, make_results(images[1].id, warning, service.detector_version)
    )
    service._save_results(
        images[2].id, make_results(images[2].id, failed, service.detector_version)
    )
    service._save_results(
        images[3].id, make_results(images[3].id, passed, service.detector_version)
    )

    summary = service.get_summary(project.id)

    assert summary.pass_count == 2
    assert summary.warning_count == 1
    assert summary.failed_count == 1
    assert summary.inspected_images == 4
    assert (
        summary.pass_count + summary.warning_count + summary.failed_count
        == summary.inspected_images
    )


def test_summary_safely_classifies_unexpected_status_as_failed(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Incomplete"))
    source = test_workspace / "incomplete.png"
    Image.new("RGB", (640, 640), "white").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    service = ImageInspectionService(settings, projects)
    results = make_results(
        image.id,
        {rule: InspectionStatus.PASS for rule in InspectionRule},
        service.detector_version,
    )[:-1]
    with service.session_factory() as session:
        repository = ImageInspectionRepository(session)
        repository.replace_for_image(image.id, results, service.detector_version)
        session.commit()

    summary = service.get_summary(project.id)
    assert summary.inspected_images == 1
    assert summary.failed_count == 1
    assert summary.pass_count + summary.warning_count + summary.failed_count == 1


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


def test_inspection_uses_exif_transposed_dimensions_without_changing_db_metadata(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("EXIF"))
    oriented_source = test_workspace / "oriented.jpg"
    oriented = Image.new("RGB", (1024, 400), "blue")
    exif = oriented.getexif()
    exif[274] = 6
    oriented.save(oriented_source, format="JPEG", exif=exif.tobytes())
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [oriented_source])
        .successes[0]
    )

    service = ImageInspectionService(settings, projects)
    service.inspect_image(project.id, image.id)
    results = {item.rule: item for item in service.get_results(image.id)}

    # The file metadata remains the physical 1024x400 dimensions, while the
    # inspection sees the EXIF-corrected 400x1024 image.
    assert (image.width, image.height) == (1024, 400)
    resolution = results[InspectionRule.RESOLUTION_TOO_SMALL]
    aspect = results[InspectionRule.ASPECT_RATIO_EXTREME]
    assert resolution.status is InspectionStatus.WARNING
    assert "幅" in resolution.reason
    assert "高さ" not in resolution.reason
    assert aspect.score is not None
    assert math.isclose(aspect.score, 1024 / 400, rel_tol=1e-6)


def test_inspection_dimensions_without_exif_keep_existing_behavior(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("No EXIF"))
    source = test_workspace / "plain.jpg"
    Image.new("RGB", (400, 1024), "blue").save(source, format="JPEG")
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
    assert results[InspectionRule.ASPECT_RATIO_EXTREME].score == 1024 / 400


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
    assert run.inspected_image_count == 2
    assert run.failed_image_count == 2
    assert all(
        item.status is InspectionStatus.FAILED
        for item in service.get_results(registered[0].id)
    )


def test_save_failure_counts_once_and_continues_with_other_images(
    test_workspace: Path,
    monkeypatch,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Save failures"))
    sources = []
    for index in range(2):
        source = test_workspace / f"save-{index}.png"
        Image.new("RGB", (640, 640), (index * 30, 30, 30)).save(source)
        sources.append(source)
    images = (
        ImageService(settings, projects).register_uploads(project.id, sources).successes
    )
    service = ImageInspectionService(settings, projects)
    original = service._save_results
    calls = 0

    def fail_first(image_id, results):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated save failure")
        original(image_id, results)

    monkeypatch.setattr(service, "_save_results", fail_first)
    run = service.inspect_project(project.id)

    assert run.inspected_image_count == 1
    assert run.failed_image_count == 1
    assert service.get_results(images[1].id)


def test_failed_result_save_failure_is_not_double_counted(
    test_workspace: Path,
    monkeypatch,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Failed save"))
    source = test_workspace / "missing.png"
    Image.new("RGB", (640, 640), "black").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    image.original_path.unlink()
    service = ImageInspectionService(settings, projects)
    monkeypatch.setattr(
        service,
        "_save_results",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("simulated save failure")),
    )

    run = service.inspect_project(project.id)

    assert run.inspected_image_count == 0
    assert run.failed_image_count == 1


def test_reinspection_keeps_previous_detector_version_and_summary_uses_current(
    test_workspace: Path,
) -> None:
    settings = inspection_settings(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Versions"))
    source = test_workspace / "versioned.png"
    Image.effect_noise((640, 640), 80).convert("RGB").save(source)
    image = (
        ImageService(settings, projects)
        .register_uploads(project.id, [source])
        .successes[0]
    )
    service = ImageInspectionService(settings, projects)
    old_version = "phase2a-v0"
    old_results = make_results(
        image.id,
        {rule: InspectionStatus.WARNING for rule in InspectionRule},
        old_version,
    )
    with service.session_factory() as session:
        ImageInspectionRepository(session).replace_for_image(
            image.id, old_results, old_version
        )
        session.commit()

    service.inspect_image(project.id, image.id)
    with service.session_factory() as session:
        all_results = (
            session.query(ImageInspectionResultRecord)
            .filter(ImageInspectionResultRecord.image_id == str(image.id))
            .all()
        )

    assert {result.detector_version for result in all_results} == {
        old_version,
        service.detector_version,
    }
    assert service.get_summary(project.id).warning_count == 0


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
