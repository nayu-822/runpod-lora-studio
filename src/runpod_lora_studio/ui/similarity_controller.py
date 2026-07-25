from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_lora_studio.domain.models import (
    SelectionState,
    SimilarityGroup,
    SimilarityRunResult,
    SimilaritySummary,
)
from runpod_lora_studio.services.similarity_detection_service import (
    SimilarityDetectionService,
)


@dataclass(frozen=True, slots=True)
class SimilarityPageView:
    groups: list[SimilarityGroup]
    total: int
    page: int
    total_pages: int


class SimilarityController:
    def __init__(self, service: SimilarityDetectionService) -> None:
        self.service = service

    def run(
        self, project_id: UUID, image_ids: list[UUID] | None = None
    ) -> SimilarityRunResult:
        return self.service.run_project(project_id, image_ids)

    def summary(self, project_id: UUID) -> SimilaritySummary:
        return self.service.get_summary(project_id)

    def list_page(
        self, project_id: UUID, page: int, page_size: int
    ) -> SimilarityPageView:
        groups, total = self.service.list_groups(project_id, page, page_size)
        total_pages = 0 if total == 0 else (total + page_size - 1) // page_size
        safe_page = 0 if total_pages == 0 else min(max(page, 1), total_pages)
        if safe_page != page and total:
            groups, _ = self.service.list_groups(project_id, safe_page, page_size)
        return SimilarityPageView(groups, total, safe_page, total_pages)

    def group(self, group_id: UUID) -> SimilarityGroup | None:
        return self.service.get_group(group_id)

    def set_representative(self, group_id: UUID, image_id: UUID) -> None:
        self.service.change_representative(group_id, image_id)

    def review(self, group_id: UUID, similar: bool) -> None:
        self.service.review_group(group_id, similar)

    def change_state(
        self, project_id: UUID, image_ids: list[UUID], state: SelectionState
    ) -> int:
        return self.service.change_image_state(project_id, image_ids, state)


def similarity_summary_markdown(summary: SimilaritySummary) -> str:
    return "\n".join(
        [
            f"- pHash計算済み: **{summary.calculated_count}**",
            f"- pHash未計算: **{summary.uncalculated_count}**",
            f"- pHash計算失敗: **{summary.failed_count}**",
            f"- 類似グループ: **{summary.group_count}**",
            f"- 候補画像: **{summary.candidate_image_count}**",
            f"- 完全重複だけのグループ: **{summary.exact_only_group_count}**",
            f"- 手動未確認グループ: **{summary.unreviewed_group_count}**",
        ]
    )


def similarity_group_rows(groups: list[SimilarityGroup]) -> list[list[str | int]]:
    rows: list[list[str | int]] = []
    for group in groups:
        representative = next(
            (item for item in group.members if item.is_representative), None
        )
        distances = [
            item.representative_distance
            for item in group.members
            if item.representative_distance is not None
        ]
        statuses = {item.review_status.value for item in group.members}
        review = "未確認" if "unreviewed" in statuses else ",".join(sorted(statuses))
        rows.append(
            [
                str(group.id)[:8],
                len(group.members),
                representative.image.original_filename
                if representative and representative.image
                else "不在",
                group.group_type,
                max(distances, default=0),
                review,
                group.representative_source.value,
            ]
        )
    return rows


def similarity_detail_rows(group: SimilarityGroup) -> list[list[str | int | float]]:
    rows: list[list[str | int | float]] = []
    for member in group.members:
        image = member.image
        if image is None:
            continue
        rows.append(
            [
                str(member.image_id)[:8],
                image.original_filename,
                f"{image.width}x{image.height}",
                image.selection_state.value,
                member.representative_distance
                if member.representative_distance is not None
                else "代表",
                member.minimum_distance if member.minimum_distance is not None else "-",
                f"{member.representative_candidate_score:.2f}",
                "代表" if member.is_representative else "",
                member.review_status.value,
                ", ".join(image.exclusion_reasons) or "なし",
            ]
        )
    return rows


def similarity_gallery(group: SimilarityGroup) -> list[tuple[str, str]]:
    return [
        (str(member.image.thumbnail_path), member.image.original_filename)
        for member in group.members
        if member.image is not None and member.image.thumbnail_path.is_file()
    ]
