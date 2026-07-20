from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_lora_studio.domain.models import (
    ConceptType,
    ImageAsset,
    Project,
    SelectionState,
)
from runpod_lora_studio.services.image_service import ImageService, UploadResult
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    normalize_trigger_words,
)

CONCEPT_LABELS = {
    ConceptType.CHARACTER.value: "キャラクター",
    ConceptType.STYLE.value: "画風",
    ConceptType.COSTUME.value: "衣装",
    ConceptType.OBJECT.value: "物体",
    ConceptType.OTHER.value: "その他",
}


@dataclass(frozen=True, slots=True)
class ProjectSelectionView:
    rows: list[list[str | int]]
    choices: list[tuple[str, str]]
    selected_id: str | None


@dataclass(frozen=True, slots=True)
class PageState:
    page: int
    total_pages: int
    label: str


@dataclass(frozen=True, slots=True)
class ImagePageView:
    images: list[ImageAsset]
    total: int
    page: PageState
    gallery_ids: list[str]


class ProjectController:
    def __init__(self, service: ProjectService) -> None:
        self.service = service

    def reload(self, current_id: str | None) -> ProjectSelectionView:
        return synchronize_project_selection(self.service.list_projects(), current_id)

    def create(
        self, name: str, description: str, concept: str, triggers: str
    ) -> tuple[Project, ProjectSelectionView]:
        project = self.service.create(
            ProjectInput(
                name,
                description,
                ConceptType(
                    next(
                        key for key, value in CONCEPT_LABELS.items() if value == concept
                    )
                ),
                normalize_trigger_words(triggers),
            )
        )
        return project, self.reload(str(project.id))

    def update(
        self,
        project_id: str,
        name: str,
        description: str,
        concept: str,
        triggers: str,
    ) -> tuple[Project, ProjectSelectionView]:
        project = self.service.update(
            UUID(project_id),
            ProjectInput(
                name,
                description,
                ConceptType(
                    next(
                        key for key, value in CONCEPT_LABELS.items() if value == concept
                    )
                ),
                normalize_trigger_words(triggers),
            ),
        )
        return project, self.reload(str(project.id))


class ImageController:
    def __init__(self, service: ImageService) -> None:
        self.service = service

    def list_page(
        self,
        project_id: UUID,
        state: SelectionState | None,
        search: str,
        page: int,
        page_size: int,
    ) -> ImagePageView:
        safe_page_size = max(page_size, 1)
        images, total = self.service.list_images(
            project_id,
            state=state,
            search=search,
            page=page,
            page_size=safe_page_size,
        )
        page_state = normalize_page(total, page, safe_page_size)
        if page_state.page != page and total > 0:
            images, _ = self.service.list_images(
                project_id,
                state=state,
                search=search,
                page=page_state.page,
                page_size=safe_page_size,
            )
        gallery_ids = [
            str(image.id) for image in images if image.thumbnail_path.is_file()
        ]
        return ImagePageView(images, total, page_state, gallery_ids)

    def refresh_after_change(
        self,
        project_id: UUID,
        state: SelectionState | None,
        search: str,
        page: int,
        page_size: int,
    ) -> tuple[ImagePageView, list[str]]:
        view = self.list_page(project_id, state, search, page, page_size)
        return view, []

    def register(self, project_id: UUID, files: list[str]) -> UploadResult:
        return self.service.register_uploads(project_id, files)

    def change_state(
        self, project_id: UUID, image_ids: list[UUID], state: SelectionState
    ) -> int:
        return self.service.change_state(project_id, image_ids, state)


def project_table_rows(projects: list[Project]) -> list[list[str | int]]:
    return [
        [
            str(project.id),
            project.name,
            CONCEPT_LABELS[project.concept_type.value],
            project.status.value,
            project.image_counts.get(SelectionState.ACCEPTED, 0),
            project.image_counts.get(SelectionState.PENDING, 0),
            project.image_counts.get(SelectionState.EXCLUDED, 0),
            project.created_at.isoformat(),
            project.updated_at.isoformat(),
        ]
        for project in projects
    ]


def synchronize_project_selection(
    projects: list[Project], current_id: str | None
) -> ProjectSelectionView:
    choices = [(project.name, str(project.id)) for project in projects]
    valid_ids = {project_id for _, project_id in choices}
    selected_id = current_id if current_id in valid_ids else None
    if selected_id is None and choices:
        selected_id = choices[0][1]
    return ProjectSelectionView(project_table_rows(projects), choices, selected_id)


def normalize_page(total: int, requested_page: int, page_size: int) -> PageState:
    safe_size = max(page_size, 1)
    total_pages = 0 if total == 0 else (total + safe_size - 1) // safe_size
    page = 0 if total_pages == 0 else min(max(requested_page, 1), total_pages)
    return PageState(page, total_pages, f"{page} / {total_pages}ページ、全{total}件")
