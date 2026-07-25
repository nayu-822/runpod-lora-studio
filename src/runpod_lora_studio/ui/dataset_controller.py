from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_lora_studio.domain.models import (
    DatasetPreview,
    DatasetSettings,
    DatasetSnapshotSummary,
)
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService


@dataclass(frozen=True, slots=True)
class DatasetController:
    service: DatasetSnapshotService

    def preview(self, project_id: UUID, settings: DatasetSettings) -> DatasetPreview:
        return self.service.preview(project_id, settings)

    def create(
        self,
        preview: DatasetPreview,
        name: str,
        description: str,
        confirm_warnings: bool,
    ) -> str:
        snapshot_id = self.service.start_snapshot(
            preview,
            name=name,
            description=description,
            confirm_warnings=confirm_warnings,
        )
        return str(snapshot_id)

    def cancel(self, snapshot_id: UUID) -> None:
        self.service.cancel(snapshot_id)

    def snapshots(self, project_id: UUID) -> list[DatasetSnapshotSummary]:
        return self.service.list_snapshots(project_id)

    def revalidate(self, snapshot_id: UUID) -> str:
        return self.service.revalidate(snapshot_id).value


def preview_summary(preview: DatasetPreview) -> str:
    summary = preview.summary
    return "\n".join(
        [
            f"- 対象画像: **{summary.target_image_count}**",
            f"- キャプションあり/なし: **{summary.caption_present_count} / "
            f"{summary.caption_missing_count}**",
            f"- ファイル欠落/破損: **{summary.missing_file_count} / "
            f"{summary.corrupt_file_count}**",
            f"- 品質警告/失敗: **{summary.quality_warning_image_count} / "
            f"{summary.quality_failed_image_count}**",
            f"- 完全重複/近似重複: **{summary.exact_duplicate_count} / "
            f"{summary.approximate_duplicate_count}**",
            f"- トリガー未付与: **{summary.trigger_missing_count}**",
            f"- 警告: **{summary.warning_count}**",
            f"- エラー: **{summary.error_count}**",
            f"- 推定容量: **{summary.estimated_size_bytes} bytes**",
            f"- 利用可能容量: **{summary.available_disk_bytes} bytes**",
            f"- 推定作成後空き容量: **{summary.estimated_free_bytes} bytes**",
        ]
    )


def preview_rows(
    preview: DatasetPreview,
    page: int | None = None,
    page_size: int = 20,
) -> list[list[str | int | float]]:
    images = list(preview.images)
    if page is not None:
        page = max(page, 1)
        start = (page - 1) * max(page_size, 1)
        images = images[start : start + max(page_size, 1)]
    return [
        [
            item.original_filename,
            str(item.image_id),
            f"{item.width}x{item.height}",
            round(item.aspect_ratio, 3),
            item.file_size,
            item.selection_state.value,
            item.caption_revision or "-",
            len(item.caption_text),
            item.tag_count,
            item.trigger_word_count,
            item.quality_status,
            item.exact_duplicate_status,
            "可" if item.can_include else "不可",
            " / ".join(issue.message for issue in (*item.errors, *item.warnings)),
        ]
        for item in images
    ]


def snapshot_rows(
    service: DatasetSnapshotService, project_id: UUID
) -> list[list[str | int]]:
    return [
        [
            str(item.id)[:8],
            item.name,
            item.status.value,
            item.created_at.isoformat(),
            item.target_image_count,
            item.total_size_bytes,
            item.warning_count,
            (item.content_sha256 or "")[:12],
            str(item.source_tagger_run_id)[:8] if item.source_tagger_run_id else "-",
        ]
        for item in service.list_snapshots(project_id)
    ]
