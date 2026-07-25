from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    CaptionChange,
    CaptionEditSource,
    CaptionPreview,
    CaptionTagValue,
    ManualCaptionPolicy,
    ProjectTagRule,
    StoredCaption,
    StoredTaggingResult,
    TagCategory,
    TagFrequency,
    TaggerRunStatus,
    TaggerRunSummary,
    TagSource,
)
from runpod_lora_studio.external.tagger import normalize_tag_name
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    CaptionEditHistoryRecord,
    ImageAssetRecord,
)
from runpod_lora_studio.persistence.tagging_repository import TaggingRepository
from runpod_lora_studio.services.project_service import (
    ProjectService,
    UserFacingError,
)


@dataclass(frozen=True, slots=True)
class TagFrequencyPage:
    items: tuple[TagFrequency, ...]
    total: int
    target_image_count: int
    run_id: UUID | None


class TagFrequencyService:
    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    def list_frequencies(
        self,
        project_id: UUID,
        *,
        run_id: UUID | None = None,
        search: str = "",
        categories: set[TagCategory] | None = None,
        minimum_rate: float = 0.0,
        keep_filter: bool | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> TagFrequencyPage:
        with self.session_factory() as session:
            repository = TaggingRepository(session)
            selected_run = self._select_run(repository, project_id, run_id)
            if selected_run is None:
                return TagFrequencyPage((), 0, 0, None)
            results = repository.results_for_run(project_id, selected_run.id)
            results = [result for result in results if result.tags]
            target_count = len(results)
            rules = {
                rule.normalized_tag_name: rule
                for rule in repository.get_rules(project_id)
            }
            frequencies = self._calculate(results, target_count, rules)
            query = search.strip().casefold()
            filtered = [
                item
                for item in frequencies
                if (
                    not query
                    or query in item.tag_name_normalized.casefold()
                    or query in item.display_name.casefold()
                )
                and (categories is None or item.category in categories)
                and item.occurrence_rate >= max(minimum_rate, 0.0)
                and (keep_filter is None or item.keep is keep_filter)
            ]
            size = max(page_size or self.settings.tagger_caption_page_size, 1)
            start = max(page - 1, 0) * size
            return TagFrequencyPage(
                tuple(filtered[start : start + size]),
                len(filtered),
                target_count,
                selected_run.id,
            )

    def rules(self, project_id: UUID) -> dict[str, bool]:
        with self.session_factory() as session:
            return {
                rule.normalized_tag_name: rule.action == "keep"
                for rule in TaggingRepository(session).get_rules(project_id)
            }

    @staticmethod
    def initial_draft(items: Iterable[TagFrequency]) -> dict[str, bool]:
        return {item.tag_name_normalized: item.keep for item in items}

    @staticmethod
    def set_visible(
        draft: Mapping[str, bool], names: Iterable[str], keep: bool
    ) -> dict[str, bool]:
        result = dict(draft)
        for name in names:
            result[name] = keep
        return result

    @staticmethod
    def set_rate(
        draft: Mapping[str, bool],
        items: Iterable[TagFrequency],
        minimum_rate: float,
        keep: bool,
    ) -> dict[str, bool]:
        result = dict(draft)
        for item in items:
            if item.occurrence_rate >= minimum_rate:
                result[item.tag_name_normalized] = keep
        return result

    @staticmethod
    def set_category(
        draft: Mapping[str, bool],
        items: Iterable[TagFrequency],
        category: TagCategory,
        keep: bool,
    ) -> dict[str, bool]:
        result = dict(draft)
        for item in items:
            if item.category is category:
                result[item.tag_name_normalized] = keep
        return result

    @staticmethod
    def _select_run(
        repository: TaggingRepository, project_id: UUID, run_id: UUID | None
    ) -> TaggerRunSummary | None:
        if run_id is not None:
            run = repository.get_run(run_id)
            if run is None or run.project_id != project_id:
                raise UserFacingError("指定されたTaggerRunが見つかりません。")
            return run if run.status is TaggerRunStatus.COMPLETED else None
        return next(
            (
                run
                for run in repository.list_runs(project_id)
                if run.status is TaggerRunStatus.COMPLETED
            ),
            None,
        )

    @staticmethod
    def _calculate(
        results: Sequence[StoredTaggingResult],
        target_count: int,
        rules: Mapping[str, ProjectTagRule],
    ) -> list[TagFrequency]:
        occurrences: dict[str, set[UUID]] = defaultdict(set)
        confidences: dict[str, list[float]] = defaultdict(list)
        metadata: dict[str, tuple[str, TagCategory]] = {}
        for result in results:
            seen: set[str] = set()
            for tag in result.tags:
                if tag.tag_name_normalized in seen:
                    continue
                seen.add(tag.tag_name_normalized)
                occurrences[tag.tag_name_normalized].add(result.image_id)
                metadata.setdefault(
                    tag.tag_name_normalized,
                    (tag.tag_name_normalized.replace("_", " "), tag.category),
                )
                if tag.confidence is not None:
                    confidences[tag.tag_name_normalized].append(tag.confidence)
        values: list[TagFrequency] = []
        for normalized, image_ids in occurrences.items():
            confidence = confidences.get(normalized, [])
            display, category = metadata[normalized]
            rule = rules.get(normalized)
            values.append(
                TagFrequency(
                    tag_name_normalized=normalized,
                    display_name=display,
                    category=category,
                    image_count=len(image_ids),
                    target_image_count=target_count,
                    occurrence_rate=(
                        len(image_ids) / target_count if target_count else 0.0
                    ),
                    average_confidence=(
                        sum(confidence) / len(confidence) if confidence else None
                    ),
                    minimum_confidence=min(confidence) if confidence else None,
                    maximum_confidence=max(confidence) if confidence else None,
                    keep=rule is None or rule.action == "keep",
                    rule_origin=(rule.updated_by if rule else "initial"),
                )
            )
        return sorted(
            values,
            key=lambda item: (
                -item.image_count,
                -item.occurrence_rate,
                item.category.value,
                item.tag_name_normalized,
            ),
        )


class CaptionEditingService:
    caption_format_version = "phase3-v1"

    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    def build_preview(
        self,
        project_id: UUID,
        *,
        run_id: UUID | None = None,
        keep_states: Mapping[str, bool] | None = None,
        trigger_words: Iterable[str] | str | None = None,
        policy: ManualCaptionPolicy = ManualCaptionPolicy.KEEP_MANUAL,
    ) -> CaptionPreview:
        with self.session_factory() as session:
            return self._build_preview(
                session, project_id, run_id, keep_states, trigger_words, policy
            )

    def apply_preview(self, preview: CaptionPreview) -> CaptionPreview:
        with self.session_factory() as session:
            current = self._build_preview(
                session,
                preview.project_id,
                preview.tagger_run_id,
                {name: action == "keep" for name, action, _ in preview.rules_snapshot},
                preview.trigger_words,
                preview.policy,
            )
            if current.token != preview.token:
                raise UserFacingError(
                    "プレビューの有効期限が切れています。再生成してください。"
                )
            repository = TaggingRepository(session)
            rules = tuple(
                ProjectTagRule(
                    normalized_tag_name=name,
                    action="keep" if keep == "keep" else "remove",
                    category=TagCategory(category),
                    updated_by="user",
                )
                for name, keep, category in preview.rules_snapshot
            )
            repository.upsert_rules(preview.project_id, rules)
            for change in preview.changes:
                if change.warning and change.before == change.after:
                    continue
                tags = self._tags_from_caption(
                    change.after,
                    change.image_id,
                    preview.trigger_words,
                )
                repository.save_caption(
                    change.image_id,
                    change.after,
                    tags,
                    CaptionEditSource.BULK_FILTER,
                    preview.tagger_run_id,
                    json.dumps(
                        {
                            "added": change.added_tags,
                            "removed": change.removed_tags,
                            "warning": change.warning,
                        },
                        ensure_ascii=False,
                    ),
                )
            session.commit()
            return current

    def save_image_caption(
        self, project_id: UUID, image_id: UUID, caption_text: str
    ) -> StoredCaption:
        tags = parse_caption_tags(caption_text)
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            current = TaggingRepository(session).get_current_caption(image_id)
            result = TaggingRepository(session).save_caption(
                image_id,
                format_caption(tags),
                tags,
                CaptionEditSource.MANUAL,
                current.source_tagger_run_id if current else None,
                json.dumps({"manual": True}, ensure_ascii=False),
            )
            session.commit()
            return result

    def get_caption(self, project_id: UUID, image_id: UUID) -> StoredCaption | None:
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            return TaggingRepository(session).get_current_caption(image_id)

    def source_tags_text(self, project_id: UUID, image_id: UUID) -> str:
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            result = TaggingRepository(
                session
            ).latest_completed_result_for_project_image(image_id, project_id)
            return format_caption(self._prediction_tags(result)) if result else ""

    def restore_from_source(self, project_id: UUID, image_id: UUID) -> StoredCaption:
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            repository = TaggingRepository(session)
            result = repository.latest_completed_result_for_project_image(
                image_id, project_id
            )
            if result is None:
                raise UserFacingError("復元できる元タグがありません。")
            tags = self._prediction_tags(result)
            caption = repository.save_caption(
                image_id,
                format_caption(tags),
                tags,
                CaptionEditSource.RESTORED,
                result.tagger_run_id,
                json.dumps({"restored": "source_tags"}, ensure_ascii=False),
            )
            session.commit()
            return caption

    def restore_previous(self, project_id: UUID, image_id: UUID) -> StoredCaption:
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            repository = TaggingRepository(session)
            current = repository.get_current_caption(image_id)
            if current is None or current.revision <= 1:
                raise UserFacingError("直前のキャプション履歴がありません。")
            history = repository.caption_history(image_id)
            previous = next(
                (item for item in history if item.new_revision == current.revision),
                None,
            )
            if previous is None:
                raise UserFacingError("直前のキャプション履歴がありません。")
            tags = parse_caption_tags(previous.before_text)
            restored = repository.save_caption(
                image_id,
                previous.before_text,
                tags,
                CaptionEditSource.RESTORED,
                current.source_tagger_run_id,
                json.dumps({"restored": current.revision - 1}, ensure_ascii=False),
            )
            session.commit()
            return restored

    def history(
        self, project_id: UUID, image_id: UUID
    ) -> list[CaptionEditHistoryRecord]:
        with self.session_factory() as session:
            self._ensure_image(session, project_id, image_id)
            return TaggingRepository(session).caption_history(image_id)

    def _build_preview(
        self,
        session: Session,
        project_id: UUID,
        run_id: UUID | None,
        keep_states: Mapping[str, bool] | None,
        trigger_words: Iterable[str] | str | None,
        policy: ManualCaptionPolicy,
    ) -> CaptionPreview:
        repository = TaggingRepository(session)
        run = TagFrequencyService(self.settings, self.projects)._select_run(
            repository, project_id, run_id
        )
        if run is None:
            raise UserFacingError("完了済みのTaggerRunがありません。")
        results = repository.results_for_run(project_id, run.id)
        rule_records = repository.get_rules(project_id)
        state = dict(keep_states or {})
        if not state:
            state = {
                rule.normalized_tag_name: rule.action == "keep" for rule in rule_records
            }
        tag_categories = {
            tag.tag_name_normalized: tag.category
            for result in results
            for tag in result.tags
        }
        tag_categories.update(
            {
                rule.normalized_tag_name: rule.category
                for rule in rule_records
                if rule.normalized_tag_name not in tag_categories
            }
        )
        rules_snapshot = tuple(
            (
                name,
                "keep" if keep else "remove",
                tag_categories.get(name, TagCategory.UNKNOWN).value,
            )
            for name, keep in sorted(state.items())
        )
        triggers = normalize_trigger_words(trigger_words)
        changes: list[CaptionChange] = []
        keep_names: set[str] = set()
        remove_names: set[str] = set()
        manual_count = 0
        for result in results:
            current = repository.get_current_caption(result.image_id)
            current_is_manual = bool(
                current
                and (
                    current.edit_source is CaptionEditSource.MANUAL
                    or any(
                        tag.manually_added or tag.manually_removed
                        for tag in current.tags
                    )
                )
            )
            if current_is_manual:
                manual_count += 1
            warning: str | None = None
            if current_is_manual and policy is ManualCaptionPolicy.EXCLUDE_MANUAL:
                after_tags = current.tags if current else ()
                warning = "手動編集済み画像を今回の一括適用から除外しました。"
            else:
                if current_is_manual and policy is ManualCaptionPolicy.KEEP_MANUAL:
                    base_tags = tuple(
                        tag
                        for tag in (current.tags if current else ())
                        if state.get(tag.normalized_name, True)
                    )
                else:
                    base_tags = self._prediction_tags(result)
                    base_tags = tuple(
                        tag for tag in base_tags if state.get(tag.normalized_name, True)
                    )
                after_tags = self._with_triggers(base_tags, triggers)
                warning = "空キャプションになります。" if not after_tags else None
            before = current.caption_text if current else ""
            after = format_caption(after_tags)
            before_names = {
                tag.normalized_name for tag in (current.tags if current else ())
            }
            after_names = {tag.normalized_name for tag in after_tags}
            added = tuple(sorted(after_names - before_names))
            removed = tuple(sorted(before_names - after_names))
            for name in after_names:
                keep_names.add(name)
            remove_names.update(removed)
            image = session.scalar(
                select(ImageAssetRecord).where(
                    ImageAssetRecord.id == str(result.image_id)
                )
            )
            filename = image.original_filename if image else str(result.image_id)
            changes.append(
                CaptionChange(
                    image_id=result.image_id,
                    filename=filename,
                    before=before,
                    after=after,
                    added_tags=added,
                    removed_tags=removed,
                    trigger_words=tuple(
                        trigger
                        for trigger in triggers
                        if normalize_tag_name(trigger) in after_names
                    ),
                    manual_policy=policy,
                    warning=warning,
                )
            )
        token_payload = {
            "project": str(project_id),
            "run": str(run.id),
            "policy": policy.value,
            "rules": rules_snapshot,
            "triggers": triggers,
            "changes": [
                (str(item.image_id), item.before, item.after) for item in changes
            ],
        }
        token = hashlib.sha256(
            json.dumps(token_payload, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        return CaptionPreview(
            token=token,
            project_id=project_id,
            tagger_run_id=run.id,
            changes=tuple(changes),
            target_image_count=len(changes),
            keep_tag_count=len(keep_names),
            remove_tag_count=len(remove_names),
            changed_image_count=sum(item.before != item.after for item in changes),
            empty_caption_count=sum(not item.after for item in changes),
            trigger_image_count=sum(bool(item.trigger_words) for item in changes),
            manual_image_count=manual_count,
            rules_snapshot=rules_snapshot,
            trigger_words=triggers,
            policy=policy,
        )

    @staticmethod
    def _prediction_tags(result: StoredTaggingResult) -> tuple[CaptionTagValue, ...]:
        return tuple(
            CaptionTagValue(
                tag_name=tag.tag_name_normalized,
                normalized_name=tag.tag_name_normalized,
                category=tag.category,
                source=tag.source,
                position=index,
                confidence=tag.confidence,
            )
            for index, tag in enumerate(result.tags)
        )

    @staticmethod
    def _with_triggers(
        tags: Iterable[CaptionTagValue], triggers: Sequence[str]
    ) -> tuple[CaptionTagValue, ...]:
        result: list[CaptionTagValue] = []
        seen: set[str] = set()
        for trigger in triggers:
            normalized = normalize_tag_name(trigger)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(
                CaptionTagValue(
                    tag_name=trigger,
                    normalized_name=normalized,
                    category=TagCategory.TRIGGER,
                    source=TagSource.TRIGGER_WORD,
                    position=len(result),
                )
            )
        for tag in tags:
            if tag.normalized_name in seen:
                continue
            seen.add(tag.normalized_name)
            result.append(tag)
        return tuple(
            CaptionTagValue(
                tag_name=tag.tag_name,
                normalized_name=tag.normalized_name,
                category=tag.category,
                source=tag.source,
                position=index,
                confidence=tag.confidence,
                manually_added=tag.manually_added,
                manually_removed=tag.manually_removed,
            )
            for index, tag in enumerate(result)
        )

    @staticmethod
    def _tags_from_caption(
        caption: str,
        image_id: UUID,
        trigger_words: Iterable[str] = (),
    ) -> tuple[CaptionTagValue, ...]:
        del image_id
        parsed = parse_caption_tags(caption)
        triggers = {normalize_tag_name(item) for item in trigger_words}
        return tuple(
            CaptionTagValue(
                tag_name=tag.tag_name,
                normalized_name=tag.normalized_name,
                category=(
                    TagCategory.TRIGGER
                    if tag.normalized_name in triggers
                    else TagCategory.GENERAL
                ),
                source=(
                    TagSource.TRIGGER_WORD
                    if tag.normalized_name in triggers
                    else TagSource.WD_TAGGER
                ),
                position=index,
                confidence=tag.confidence,
                manually_added=False,
                manually_removed=False,
            )
            for index, tag in enumerate(parsed)
        )

    @staticmethod
    def _ensure_image(session: Session, project_id: UUID, image_id: UUID) -> None:
        image = session.scalar(
            select(ImageAssetRecord).where(
                ImageAssetRecord.id == str(image_id),
                ImageAssetRecord.project_id == str(project_id),
            )
        )
        if image is None:
            raise UserFacingError("指定された画像が見つかりません。")


def normalize_trigger_words(words: Iterable[str] | str | None) -> tuple[str, ...]:
    if isinstance(words, str):
        values = words.replace("\r", "\n").splitlines()
        expanded: list[str] = []
        for value in values:
            expanded.extend(value.split(","))
        values = expanded
    else:
        values = list(words or [])
    result: list[str] = []
    for value in values:
        item = value.strip()
        if not item or "," in item or any(ord(char) < 32 for char in item):
            continue
        if len(item) > 512:
            raise UserFacingError("トリガーワードが長すぎます。")
        if normalize_tag_name(item) not in {
            normalize_tag_name(existing) for existing in result
        }:
            result.append(item)
    return tuple(result)


def parse_caption_tags(caption: str) -> tuple[CaptionTagValue, ...]:
    if any(ord(char) < 32 and char not in "\n\r\t" for char in caption):
        raise UserFacingError("タグに制御文字を含めることはできません。")
    values = caption.replace("\r", "\n").replace("\n", ",").split(",")
    result: list[CaptionTagValue] = []
    seen: set[str] = set()
    for value in values:
        tag = value.strip()
        if not tag:
            continue
        normalized = normalize_tag_name(tag)
        if normalized in seen:
            continue
        if len(tag) > 512:
            raise UserFacingError("タグが長すぎます。")
        seen.add(normalized)
        result.append(
            CaptionTagValue(
                tag_name=tag,
                normalized_name=normalized,
                category=TagCategory.MANUAL,
                source=TagSource.MANUAL,
                position=len(result),
                manually_added=True,
            )
        )
    caption_text = format_caption(result)
    if len(caption_text) > 16_384:
        raise UserFacingError("キャプションが長すぎます。")
    return tuple(result)


def format_caption(tags: Iterable[CaptionTagValue]) -> str:
    return ", ".join(tag.tag_name for tag in tags)
