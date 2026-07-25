from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("imagehash")

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import Base
from runpod_lora_studio.services.image_service import ImageService
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
