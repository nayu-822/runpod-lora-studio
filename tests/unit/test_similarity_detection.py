from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import select

pytest.importorskip("imagehash")

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
)
from runpod_lora_studio.domain.models import ImageAsset, Project
from runpod_lora_studio.persistence.database import (
    create_engine_for_settings,
    create_session_factory,
)
from runpod_lora_studio.persistence.models import Base, ImagePerceptualHashRecord
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.perceptual_hash_service import PerceptualHashService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService
from runpod_lora_studio.services.similarity_detection_service import (
    SimilarityDetectionService,
)


def test_similar_group_and_manual_rejection_survive_rebuild(
    test_workspace: Path,
) -> None:
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        database_path=test_workspace / "runtime" / "db.sqlite3",
        temp_dir=test_workspace / "runtime" / "tmp",
        phash_distance_threshold=8,
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("pHash"))
    first = test_workspace / "first.png"
    source = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((24, 24, 104, 104), fill="red")
    source.save(first)
    second = test_workspace / "second.jpg"
    source.save(second, quality=92)
    images = ImageService(settings, projects).register_uploads(
        project.id, [first, second]
    )
    assert len(images.successes) == 2

    service = SimilarityDetectionService(settings, projects)
    result = service.run_project(project.id)
    assert result.group_count == 1
    groups, _ = service.list_groups(project.id)
    assert groups[0].group_type == "approximate"

    service.review_group(groups[0].id, similar=False)
    result = service.run_project(project.id)
    assert result.group_count == 0


def _similarity_fixture(
    test_workspace: Path, count: int = 2
) -> tuple[
    AppSettings, ProjectService, Project, list[ImageAsset], SimilarityDetectionService
]:
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        database_path=test_workspace / "runtime" / "db.sqlite3",
        temp_dir=test_workspace / "runtime" / "tmp",
        phash_distance_threshold=8,
    )
    ensure_runtime_directories(settings)
    Base.metadata.create_all(create_engine_for_settings(settings))
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("pHash"))
    source = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(source).rectangle((24, 24, 104, 104), fill="red")
    paths: list[Path] = []
    for index in range(count):
        path = test_workspace / f"image-{index}.jpg"
        source.save(path, quality=max(70, 95 - index * 10))
        paths.append(path)
    images = ImageService(settings, projects).register_uploads(project.id, paths)
    assert len(images.successes) == count
    service = SimilarityDetectionService(settings, projects)
    return settings, projects, project, list(images.successes), service


def test_manual_representative_recalculates_all_distances_and_survives_rebuild(
    test_workspace: Path,
) -> None:
    settings, _, project, images, service = _similarity_fixture(test_workspace)
    service.run_project(project.id)
    group = service.list_groups(project.id)[0][0]
    selected = images[1].id

    service.change_representative(group.id, selected)
    changed = service.get_group(group.id)
    assert changed is not None
    assert changed.representative_image_id == selected
    values = {
        value.image_id: value for value in service.hashes.get_project_hashes(project.id)
    }
    expected = {
        image_id: PerceptualHashService.hamming_distance(
            values[image_id].hash_value,
            values[image_id].hash_size,
            values[selected].hash_value,
            values[selected].hash_size,
        )
        for image_id in values
        if image_id in {member.image_id for member in changed.members}
    }
    assert {
        member.image_id: member.representative_distance for member in changed.members
    } == expected
    assert (
        next(
            member.representative_distance
            for member in changed.members
            if member.image_id == selected
        )
        == 0
    )

    service.run_project(project.id)
    rebuilt = service.list_groups(project.id)[0][0]
    assert rebuilt.representative_image_id == selected
    assert {
        member.image_id: member.representative_distance for member in rebuilt.members
    } == expected


def test_missing_representative_phash_rolls_back_manual_change(
    test_workspace: Path,
) -> None:
    settings, _, project, images, service = _similarity_fixture(test_workspace)
    service.run_project(project.id)
    group = service.list_groups(project.id)[0][0]
    before = service.get_group(group.id)
    assert before is not None
    with create_session_factory(settings)() as session:
        record = session.scalar(
            select(ImagePerceptualHashRecord).where(
                ImagePerceptualHashRecord.image_id == str(images[1].id)
            )
        )
        assert record is not None
        session.delete(record)
        session.commit()

    with pytest.raises(ValueError, match="pHash is unavailable"):
        service.change_representative(group.id, images[1].id)
    after = service.get_group(group.id)
    assert after is not None
    assert after.representative_image_id == before.representative_image_id
    assert [member.representative_distance for member in after.members] == [
        member.representative_distance for member in before.members
    ]


def test_new_member_rebuild_is_unreviewed_but_all_confirmed_pairs_stay_confirmed(
    test_workspace: Path,
) -> None:
    settings, _, project, _, service = _similarity_fixture(test_workspace, count=2)
    service.run_project(project.id)
    group = service.list_groups(project.id)[0][0]
    service.review_group(group.id, similar=True)
    service.run_project(project.id)
    confirmed = service.list_groups(project.id)[0][0]
    assert all(
        member.review_status.value == "confirmed_similar"
        for member in confirmed.members
    )

    extra = test_workspace / "new-image.jpg"
    source = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(source).rectangle((24, 24, 104, 104), fill="red")
    source.save(extra, quality=80)
    # The existing project service is intentionally reused so the new pair is
    # absent from similarity_pair_reviews until the next group review.
    projects = ProjectService(settings)
    upload = ImageService(settings, projects).register_uploads(project.id, [extra])
    assert len(upload.successes) == 1
    service.run_project(project.id)
    rebuilt = service.list_groups(project.id)[0][0]
    assert len(rebuilt.members) == 3
    assert all(member.review_status.value == "unreviewed" for member in rebuilt.members)


def test_manual_representative_conflict_is_deterministic() -> None:
    first = uuid4()
    second = uuid4()
    component = {str(first), str(second)}
    old_groups = [
        SimpleNamespace(
            representative_source="manual",
            representative_image_id=str(second),
            created_at=datetime.now(UTC) + timedelta(seconds=1),
            members=[SimpleNamespace(image_id=str(second), image=None)],
        ),
        SimpleNamespace(
            representative_source="manual",
            representative_image_id=str(first),
            created_at=datetime.now(UTC),
            members=[SimpleNamespace(image_id=str(first), image=None)],
        ),
    ]
    assert (
        SimilarityDetectionService._manual_representative_for_component(
            component, old_groups
        )
        == first
    )
