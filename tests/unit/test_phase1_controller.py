from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from runpod_lora_studio.domain.models import (
    ConceptType,
    ImageAsset,
    Project,
    ProjectStatus,
    SelectionState,
)
from runpod_lora_studio.ui.phase1_controller import (
    ImageController,
    ProjectController,
    normalize_page,
    synchronize_project_selection,
)


def make_project(name: str) -> Project:
    now = datetime.now(UTC)
    return Project(
        id=uuid4(),
        name=name,
        description="",
        concept_type=ConceptType.OTHER,
        trigger_words=(),
        status=ProjectStatus.DRAFT,
        schema_version=1,
        created_at=now,
        updated_at=now,
    )


def test_project_selection_sync_handles_empty_single_and_multiple_projects() -> None:
    assert synchronize_project_selection([], None).selected_id is None

    single = make_project("single")
    single_view = synchronize_project_selection([single], None)
    assert single_view.selected_id == str(single.id)
    assert single_view.choices[0][1] == single_view.selected_id

    second = make_project("second")
    multiple = synchronize_project_selection([single, second], str(second.id))
    assert multiple.selected_id == str(second.id)
    assert multiple.choices[1][1] == multiple.selected_id


def test_project_selection_sync_falls_back_when_current_project_is_unknown() -> None:
    first = make_project("first")
    second = make_project("second")

    view = synchronize_project_selection([first, second], str(uuid4()))

    assert view.selected_id == str(first.id)
    assert view.choices[0][1] == view.selected_id


def test_page_state_has_zero_pages_for_empty_results() -> None:
    empty = normalize_page(0, 99, 30)
    last = normalize_page(5, 99, 2)
    first = normalize_page(5, 0, 2)

    assert empty.page == 0
    assert empty.total_pages == 0
    assert empty.label == "0 / 0ページ、全0件"
    assert last.page == 3
    assert first.page == 1


class FakeProjectService:
    def __init__(self) -> None:
        self.projects: list[Project] = []

    def list_projects(self) -> list[Project]:
        return list(self.projects)

    def create(self, data) -> Project:
        project = make_project(data.name)
        self.projects.append(project)
        return project

    def update(self, project_id, data) -> Project:
        current = next(project for project in self.projects if project.id == project_id)
        updated = replace(current, name=data.name)
        self.projects[self.projects.index(current)] = updated
        return updated


def test_project_controller_create_update_and_selection_flow() -> None:
    controller = ProjectController(FakeProjectService())

    created, created_view = controller.create(
        "created", "", "その他", "trigger, trigger"
    )
    updated, updated_view = controller.update(
        str(created.id), "renamed", "", "その他", "trigger"
    )

    assert created_view.selected_id == str(created.id)
    assert updated.name == "renamed"
    assert updated_view.selected_id == str(created.id)
    assert updated_view.choices[0][0] == "renamed"


def make_image(project_id, index: int) -> ImageAsset:
    now = datetime.now(UTC)
    return ImageAsset(
        id=uuid4(),
        project_id=project_id,
        original_filename=f"image-{index}.png",
        stored_filename=f"image-{index}.png",
        original_path=Path(f"image-{index}.png"),
        thumbnail_path=Path(f"thumbnail-{index}.png"),
        sha256=str(index),
        width=10,
        height=10,
        file_size=100,
        mime_type="image/png",
        selection_state=SelectionState.PENDING,
        exclusion_reasons=(),
        source_type="upload",
        created_at=now,
        updated_at=now,
    )


class FakeImageService:
    def __init__(self, images: list[ImageAsset]) -> None:
        self.images = images
        self.calls: list[tuple[int, int]] = []

    def list_images(self, project_id, **kwargs):
        page = kwargs["page"]
        page_size = kwargs["page_size"]
        self.calls.append((page, page_size))
        start = max(page - 1, 0) * page_size
        return self.images[start : start + page_size], len(self.images)


def test_image_controller_corrects_normal_boundary_empty_and_invalid_pages() -> None:
    project_id = uuid4()
    service = FakeImageService([make_image(project_id, index) for index in range(5)])
    controller = ImageController(service)

    first = controller.list_page(project_id, None, "", 1, 2)
    last = controller.list_page(project_id, None, "", 99, 2)
    beginning = controller.list_page(project_id, None, "", -3, 2)
    minimum_size = controller.list_page(project_id, None, "", 1, 0)
    empty_service = FakeImageService([])
    empty = ImageController(empty_service).list_page(project_id, None, "", 99, 2)

    assert first.page.page == 1 and len(first.images) == 2
    assert last.page.page == 3 and len(last.images) == 1
    assert service.calls[:3] == [(1, 2), (99, 2), (3, 2)]
    assert service.calls[3:5] == [(-3, 2), (1, 2)]
    assert beginning.page.page == 1
    assert minimum_size.page.page == 1 and minimum_size.page.total_pages == 5
    assert empty.page.label == "0 / 0ページ、全0件"


def test_image_controller_refresh_clears_selection_after_state_change() -> None:
    project_id = uuid4()
    service = FakeImageService([make_image(project_id, index) for index in range(5)])
    service.images = service.images[:4]
    view, selected_ids = ImageController(service).refresh_after_change(
        project_id, SelectionState.PENDING, "", 3, 2
    )

    assert view.page.page == 2
    assert selected_ids == []
    assert view.gallery_ids == []
