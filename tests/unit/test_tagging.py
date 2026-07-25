from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.models import (
    ManualCaptionPolicy,
    SelectionState,
    TagCategory,
    TaggerRunMode,
    TagPrediction,
)
from runpod_lora_studio.external.fake_tagger import FakeTaggerAdapter
from runpod_lora_studio.external.wd_tagger import WDTaggerAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.caption_service import (
    CaptionEditingService,
    TagFrequencyService,
    format_caption,
    normalize_trigger_words,
    parse_caption_tags,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService
from runpod_lora_studio.services.tagging_service import TaggingService


def _fixture(
    test_workspace: Path,
    count: int = 2,
    predictions: dict[str, tuple[TagPrediction, ...]] | None = None,
    failing: set[str] | None = None,
) -> tuple[
    AppSettings, ProjectService, object, list[object], FakeTaggerAdapter, TaggingService
]:
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
    project = projects.create(ProjectInput("tagging"))
    source = Image.new("RGB", (128, 128), "red")
    paths = []
    for index in range(count):
        path = test_workspace / f"image-{index}.png"
        source.save(path)
        paths.append(path)
    uploads = ImageService(settings, projects).register_uploads(project.id, paths)
    assert len(uploads.successes) == count
    image_service = ImageService(settings, projects)
    image_service.change_state(
        project.id, [image.id for image in uploads.successes], SelectionState.ACCEPTED
    )
    resolved_predictions = None
    if predictions is not None:
        resolved_predictions = {
            uploads.successes[index].original_path.name: predictions.get(
                f"image-{index}.png", ()
            )
            for index in range(len(uploads.successes))
        }
    resolved_failing = {
        uploads.successes[index].original_path.name
        for index in range(len(uploads.successes))
        if f"image-{index}.png" in (failing or set())
    }
    fake = FakeTaggerAdapter(resolved_predictions, resolved_failing)
    service = TaggingService(settings, projects, lambda: fake)
    return settings, projects, project, list(uploads.successes), fake, service


def test_fake_tagger_targets_accepted_images_and_unloads_once(
    test_workspace: Path,
) -> None:
    settings, projects, project, images, fake, service = _fixture(
        test_workspace, count=2
    )
    ImageService(settings, projects).change_state(
        project.id, [images[1].id], SelectionState.PENDING
    )

    run = service.run_sync(project.id)

    assert run.target_image_count == 1
    assert run.succeeded_image_count == 1
    assert run.skipped_image_count == 0
    assert fake.load_count == 1
    assert fake.unload_count == 1
    assert service.get_run(run.id) is not None


def test_wd_adapter_records_identity_and_fails_safely_without_model(
    test_workspace: Path,
) -> None:
    settings, _, _, _, _, _ = _fixture(test_workspace, count=0)
    adapter = WDTaggerAdapter(settings)

    identity = adapter.model_identity()
    validation = adapter.validate_environment()

    assert identity.adapter_name == "wd14"
    assert identity.model_identifier == settings.tagger_model_identifier
    assert identity.model_revision == settings.tagger_model_revision
    assert validation.ok is False
    assert "WD Tagger" in validation.message


def test_image_failure_continues_and_final_caption_is_not_created(
    test_workspace: Path,
) -> None:
    settings, projects, project, images, fake, service = _fixture(
        test_workspace, count=2
    )
    fake.failing_filenames.add(images[0].original_path.name)

    run = service.run_sync(project.id, TaggerRunMode.ALL_ACCEPTED)

    assert run.status.value == "partially_failed"
    assert run.failed_image_count == 1
    assert run.succeeded_image_count == 1
    assert (
        CaptionEditingService(settings, projects).get_caption(project.id, images[0].id)
        is None
    )


def test_untagged_mode_skips_matching_result_and_all_mode_creates_new_run(
    test_workspace: Path,
) -> None:
    _, _, project, _, _, service = _fixture(test_workspace, count=1)

    first = service.run_sync(project.id, TaggerRunMode.ALL_ACCEPTED)
    skipped = service.run_sync(project.id, TaggerRunMode.UNTAGGED_ONLY)
    rerun = service.run_sync(project.id, TaggerRunMode.ALL_ACCEPTED)

    assert first.succeeded_image_count == 1
    assert skipped.target_image_count == 0
    assert skipped.skipped_image_count == 1
    assert rerun.target_image_count == 1
    assert rerun.id != first.id


def test_frequency_deduplicates_tags_per_image_and_applies_deterministic_order(
    test_workspace: Path,
) -> None:
    predictions = {
        "image-0.png": (
            TagPrediction("blue_hair", "blue_hair", TagCategory.GENERAL, 0.8, 0),
            TagPrediction("blue_hair", "blue_hair", TagCategory.GENERAL, 0.9, 1),
            TagPrediction("character", "character", TagCategory.CHARACTER, 0.99, 2),
        ),
        "image-1.png": (
            TagPrediction("blue_hair", "blue_hair", TagCategory.GENERAL, 0.7, 0),
        ),
    }
    _, projects, project, _, _, service = _fixture(test_workspace, 2, predictions)
    service.run_sync(project.id)

    page = TagFrequencyService(service.settings, projects).list_frequencies(project.id)

    assert page.target_image_count == 2
    assert page.items[0].tag_name_normalized == "blue_hair"
    assert page.items[0].image_count == 2
    assert page.items[0].occurrence_rate == 1.0
    assert page.items[0].average_confidence == pytest.approx(0.75)


def test_preview_apply_trigger_and_stale_detection(test_workspace: Path) -> None:
    settings, projects, project, images, _, service = _fixture(test_workspace, 1)
    service.run_sync(project.id)
    captions = CaptionEditingService(settings, projects)
    preview = captions.build_preview(
        project.id,
        keep_states={"character": True, "blue_hair": False},
        trigger_words="subject, subject\nstyle",
        policy=ManualCaptionPolicy.KEEP_MANUAL,
    )

    assert preview.changes[0].after == "subject, style, character"
    captions.apply_preview(preview)
    assert (
        captions.get_caption(project.id, images[0].id).caption_text
        == "subject, style, character"
    )
    captions.save_image_caption(project.id, images[0].id, "manual_change")
    with pytest.raises(ValueError, match="プレビューの有効期限"):
        captions.apply_preview(preview)


def test_manual_edit_restore_and_history(test_workspace: Path) -> None:
    settings, projects, project, images, _, service = _fixture(test_workspace, 1)
    service.run_sync(project.id)
    captions = CaptionEditingService(settings, projects)
    captions.save_image_caption(project.id, images[0].id, "manual_tag, another_tag")
    assert (
        captions.get_caption(project.id, images[0].id).caption_text
        == "manual_tag, another_tag"
    )
    captions.restore_from_source(project.id, images[0].id)
    restored = captions.get_caption(project.id, images[0].id)
    assert restored is not None
    assert restored.caption_text == "character, blue_hair"
    assert len(captions.history(project.id, images[0].id)) == 2


def test_normalization_trigger_and_caption_input_rules() -> None:
    assert normalize_trigger_words("one, two\none,three") == ("one", "two", "three")
    tags = parse_caption_tags(" blue_hair, blue_hair\ncharacter ")
    assert format_caption(tags) == "blue_hair, character"
    with pytest.raises(ValueError):
        parse_caption_tags("bad\x00tag")
