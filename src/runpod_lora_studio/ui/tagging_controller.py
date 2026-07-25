from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_lora_studio.domain.models import (
    CaptionPreview,
    ManualCaptionPolicy,
    TagCategory,
    TaggerRunMode,
    TaggerRunSummary,
)
from runpod_lora_studio.services.caption_service import (
    CaptionEditingService,
    TagFrequencyPage,
    TagFrequencyService,
)
from runpod_lora_studio.services.tagging_service import TaggingService


@dataclass(frozen=True, slots=True)
class TaggingController:
    tagging: TaggingService
    frequency: TagFrequencyService
    captions: CaptionEditingService

    def validate(self) -> str:
        return self.tagging.validate_environment()

    def start(self, project_id: UUID, mode: TaggerRunMode) -> TaggerRunSummary:
        return self.tagging.start_run(project_id, mode)

    def runs(self, project_id: UUID) -> list[TaggerRunSummary]:
        return self.tagging.list_runs(project_id)

    def cancel(self, run_id: UUID) -> None:
        self.tagging.cancel_run(run_id)

    def frequencies(
        self,
        project_id: UUID,
        run_id: UUID | None = None,
        search: str = "",
        category: str = "all",
        minimum_rate: float = 0.0,
        page: int = 1,
    ) -> TagFrequencyPage:
        categories = None if category == "all" else {TagCategory(category)}
        return self.frequency.list_frequencies(
            project_id,
            run_id=run_id,
            search=search,
            categories=categories,
            minimum_rate=minimum_rate,
            page=page,
        )

    def preview(
        self,
        project_id: UUID,
        run_id: UUID | None,
        draft: dict[str, bool],
        triggers: str,
        policy: ManualCaptionPolicy,
    ) -> CaptionPreview:
        return self.captions.build_preview(
            project_id,
            run_id=run_id,
            keep_states=draft,
            trigger_words=triggers,
            policy=policy,
        )

    def apply(self, preview: CaptionPreview) -> CaptionPreview:
        return self.captions.apply_preview(preview)

    def restore_source(self, project_id: UUID, image_id: UUID) -> str:
        return self.captions.restore_from_source(project_id, image_id).caption_text

    def restore_previous(self, project_id: UUID, image_id: UUID) -> str:
        return self.captions.restore_previous(project_id, image_id).caption_text


def tagger_run_rows(runs: list[TaggerRunSummary]) -> list[list[str | int]]:
    return [
        [
            str(run.id)[:8],
            run.status.value,
            run.device,
            run.model_identifier,
            run.target_image_count,
            run.processed_image_count,
            run.succeeded_image_count,
            run.failed_image_count,
            run.skipped_image_count,
            run.error_summary or "",
        ]
        for run in runs
    ]


def tag_frequency_rows(page: TagFrequencyPage) -> list[list[str | int | float]]:
    return [
        [
            item.tag_name_normalized,
            item.display_name,
            item.category.value,
            item.image_count,
            item.target_image_count,
            round(item.occurrence_rate * 100, 2),
            round(item.average_confidence, 4)
            if item.average_confidence is not None
            else "-",
            round(item.minimum_confidence, 4)
            if item.minimum_confidence is not None
            else "-",
            round(item.maximum_confidence, 4)
            if item.maximum_confidence is not None
            else "-",
            "保持" if item.keep else "削除",
            item.rule_origin,
        ]
        for item in page.items
    ]


def preview_summary(preview: CaptionPreview) -> str:
    return "\n".join(
        [
            f"- 対象画像数: **{preview.target_image_count}**",
            f"- 保持タグ数: **{preview.keep_tag_count}**",
            f"- 削除タグ数: **{preview.remove_tag_count}**",
            f"- 変更画像数: **{preview.changed_image_count}**",
            f"- 空キャプション数: **{preview.empty_caption_count}**",
            f"- トリガーワード追加画像数: **{preview.trigger_image_count}**",
            f"- 手動編集済み画像数: **{preview.manual_image_count}**",
        ]
    )


def preview_rows(preview: CaptionPreview) -> list[list[str]]:
    return [
        [
            change.filename,
            change.before,
            change.after,
            ", ".join(change.added_tags),
            ", ".join(change.removed_tags),
            ", ".join(change.trigger_words),
            change.manual_policy.value,
            change.warning or "",
        ]
        for change in preview.changes
    ]
