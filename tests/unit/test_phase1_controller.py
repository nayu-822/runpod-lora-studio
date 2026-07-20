from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from runpod_lora_studio.domain.models import ConceptType, Project, ProjectStatus
from runpod_lora_studio.ui.phase1_controller import (
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
