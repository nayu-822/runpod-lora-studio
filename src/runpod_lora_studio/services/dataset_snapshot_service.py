from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import logging
import math
import shutil
import tomllib
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import UUID, uuid4

from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    DatasetIssueCategory,
    DatasetIssueSeverity,
    DatasetPreview,
    DatasetPreviewImage,
    DatasetPreviewSummary,
    DatasetReport,
    DatasetSettings,
    DatasetSimilarityGroupSummary,
    DatasetSnapshotItem,
    DatasetSnapshotStatus,
    DatasetSnapshotSummary,
    DatasetValidationIssue,
    ImageAsset,
    InspectionStatus,
    SelectionState,
    SimilarityReviewStatus,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.dataset_repository import DatasetRepository
from runpod_lora_studio.persistence.models import (
    SimilarityGroupMemberRecord,
    SimilarityGroupRecord,
    SimilarityPairReviewRecord,
)
from runpod_lora_studio.persistence.repositories import (
    ImageInspectionRepository,
    ImageRepository,
)
from runpod_lora_studio.persistence.tagging_repository import TaggingRepository
from runpod_lora_studio.services.caption_service import parse_caption_tags
from runpod_lora_studio.services.dataset_config_service import DatasetConfigService
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.dataset")


class DatasetSnapshotCanceled(Exception):
    """Internal signal used to stop materialization at a safe boundary."""


class DatasetSnapshotDbFinalizationPending(Exception):
    """The final files exist, but the DB completion transaction failed."""


@dataclass(frozen=True, slots=True)
class SimilarityGroupContext:
    member: SimilarityGroupMemberRecord
    review_status: SimilarityReviewStatus
    member_count: int
    rejected_pair_count: int
    representative_image_id: UUID | None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _settings_sha256(settings_snapshot: str) -> str:
    normalized = _json(json.loads(settings_snapshot))
    return _sha256_bytes(normalized.encode("utf-8"))


def _content_sha256(
    items: list[Any], dataset_toml_sha256: str, settings_snapshot: str
) -> str:
    payload = {
        "items": [
            {
                "sequence_number": int(item.sequence_number),
                "image_path": item.snapshot_image_relative_path,
                "caption_path": item.caption_relative_path,
                "image_sha256": item.snapshot_image_sha256,
                "caption_sha256": item.caption_sha256,
            }
            for item in sorted(items, key=lambda value: int(value.sequence_number))
        ],
        "dataset_toml_sha256": dataset_toml_sha256,
        "settings_sha256": _settings_sha256(settings_snapshot),
    }
    return _sha256_bytes(_json(payload).encode("utf-8"))


def _issue(
    code: str,
    severity: DatasetIssueSeverity,
    category: DatasetIssueCategory,
    message: str,
    image_id: UUID | None = None,
    measured: object = None,
) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        issue_code=code,
        severity=severity,
        category=category,
        message=message,
        image_id=image_id,
        measured_value=None if measured is None else str(measured),
    )


class DatasetSnapshotService:
    generator_version = "phase4-snapshot-v1"

    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)
        self.config = DatasetConfigService()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset")
        self._futures: dict[UUID, Future[Any]] = {}

    def default_settings(self) -> DatasetSettings:
        return DatasetSettings(
            resolution=self.settings.dataset_default_resolution,
            min_bucket_reso=self.settings.dataset_default_min_bucket_reso,
            max_bucket_reso=self.settings.dataset_default_max_bucket_reso,
            bucket_reso_steps=self.settings.dataset_default_bucket_reso_steps,
            num_repeats=self.settings.dataset_default_num_repeats,
            allow_empty_caption=self.settings.dataset_allow_empty_caption,
        )

    def preview(
        self, project_id: UUID, settings: DatasetSettings | None = None
    ) -> DatasetPreview:
        project = self.projects.get(project_id)
        selected = settings or self.default_settings()
        with self.session_factory() as session:
            image_repository = ImageRepository(session)
            images = image_repository.list_all_for_project(project_id)
            accepted = [
                image
                for image in images
                if image.selection_state is SelectionState.ACCEPTED
            ]
            config_issues = list(self.config.validate(selected, len(accepted)))
            repository = TaggingRepository(session)
            current_captions = {
                image.id: repository.get_current_caption(image.id) for image in accepted
            }
            inspection_repository = ImageInspectionRepository(session)
            inspections = {
                image.id: inspection_repository.list_for_image(image.id)
                for image in accepted
            }
            groups = self._similarity_members(session, accepted)
            previews: list[DatasetPreviewImage] = []
            all_issues: list[DatasetValidationIssue] = list(config_issues)
            for image in accepted:
                current = current_captions[image.id]
                image_preview = self._preview_image(
                    image,
                    current,
                    inspections[image.id],
                    groups.get(image.id),
                    project.trigger_words,
                    selected,
                )
                previews.append(image_preview)
                all_issues.extend(image_preview.warnings)
                all_issues.extend(image_preview.errors)
            previews.sort(key=lambda item: str(item.image_id))
            capacity_issues = self._storage_issues(
                project_root=self.projects.project_root(project_id),
                images=previews,
                settings=selected,
            )
            all_issues.extend(capacity_issues)
            summary = self._summary(previews, all_issues, groups, selected)
            source_runs = tuple(
                sorted(
                    {
                        current.source_tagger_run_id
                        for current in current_captions.values()
                        if current and current.source_tagger_run_id
                    },
                    key=str,
                )
            )
            preview = DatasetPreview(
                token="",
                project_id=project_id,
                project_updated_at=project.updated_at,
                project_name=project.name,
                trigger_words=project.trigger_words,
                settings=selected,
                images=tuple(previews),
                issues=tuple(all_issues),
                summary=summary,
                source_tagger_run_ids=source_runs,
                similarity_groups=self._similarity_group_summaries(groups),
            )
            return self._with_token(preview)

    def create_snapshot_sync(
        self,
        preview: DatasetPreview,
        *,
        name: str = "",
        description: str = "",
        confirm_warnings: bool = False,
        _snapshot_id: UUID | None = None,
    ) -> DatasetSnapshotSummary:
        name = self._validate_name(name)
        current = self.preview(preview.project_id, preview.settings)
        if current.token != preview.token:
            raise UserFacingError(
                "プレビューの有効期限が切れています。再生成してください。"
            )
        if any(issue.issue_code.startswith("disk_space_") for issue in current.issues):
            raise UserFacingError("保存先の空き容量が不足しているため作成できません。")
        if current.summary.error_count:
            raise UserFacingError("学習前検査にエラーがあるため作成できません。")
        if current.summary.warning_count and not confirm_warnings:
            raise UserFacingError("警告を確認してから作成してください。")
        snapshot_id = _snapshot_id or uuid4()
        project_root = self.projects.project_root(preview.project_id).resolve()
        snapshots_root = project_root / "dataset_snapshots"
        final_root = (snapshots_root / str(snapshot_id)).resolve()
        temp_root = (snapshots_root / f"{snapshot_id}.creating").resolve()
        self._ensure_inside(temp_root, project_root)
        self._validate_disk_capacity(project_root, preview)
        if final_root.exists() or temp_root.exists():
            raise UserFacingError("スナップショット保存先が既に存在します。")
        snapshots_root.mkdir(parents=True, exist_ok=True)
        source_run = (
            preview.source_tagger_run_ids[0]
            if len(preview.source_tagger_run_ids) == 1
            else None
        )
        with self.session_factory() as session:
            repository = DatasetRepository(session)
            if repository.active_snapshot(preview.project_id) is not None:
                raise UserFacingError(
                    "このプロジェクトでは既に作成中のスナップショットがあります。"
                )
            repository.create_snapshot(
                project_id=preview.project_id,
                snapshot_id=snapshot_id,
                name=name,
                description=description.strip(),
                snapshot_version="phase4-dataset-v1",
                generator_version=self.generator_version,
                source_project_version=preview.project_updated_at.isoformat(),
                source_tagger_run_id=source_run,
                target_image_count=preview.summary.target_image_count,
                warning_count=preview.summary.warning_count,
                total_size_bytes=preview.summary.estimated_size_bytes,
                snapshot_root=str(final_root),
                dataset_toml_path=str(final_root / "configs" / "dataset.toml"),
                manifest_path=str(final_root / "manifest.json"),
                report_path=str(final_root / "reports" / "dataset_report.json"),
                settings_snapshot=self.config.settings_snapshot(preview.settings),
                validation_summary=_json(self._summary_dict(preview.summary)),
            )
            session.commit()
        renamed = False
        try:
            items, manifest_hash, toml_hash, content_hash = self._materialize(
                preview, snapshot_id, temp_root
            )
            temp_root.replace(final_root)
            renamed = True
            try:
                with self.session_factory() as session:
                    repository = DatasetRepository(session)
                    for item in items:
                        repository.add_item(item)
                    for issue in preview.issues:
                        repository.add_issue(snapshot_id, issue)
                    repository.finish(
                        snapshot_id,
                        DatasetSnapshotStatus.COMPLETED,
                        copied_image_count=len(items),
                        manifest_sha256=manifest_hash,
                        dataset_toml_sha256=toml_hash,
                        content_sha256=content_hash,
                    )
                    session.commit()
            except Exception as exc:
                raise DatasetSnapshotDbFinalizationPending() from exc
        except DatasetSnapshotCanceled as exc:
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)
            with self.session_factory() as session:
                DatasetRepository(session).finish(
                    snapshot_id,
                    DatasetSnapshotStatus.CANCELED,
                    error_summary="ユーザー操作により作成をキャンセルしました。",
                )
                session.commit()
            raise UserFacingError("スナップショット作成をキャンセルしました。") from exc
        except DatasetSnapshotDbFinalizationPending as exc:
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)
            if renamed and final_root.is_dir():
                self._mark_db_finalization_pending(
                    snapshot_id,
                    "確定ファイルは存在しますが、DB確定に失敗しました。回復処理を実行してください。",
                )
            else:
                self._mark_failed(snapshot_id, "DB確定前に失敗しました。")
            raise UserFacingError(
                "確定ファイルは保存されましたが、DB確定に失敗しました。回復処理を実行してください。"
            ) from exc
        except Exception as exc:
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)
            if renamed and final_root.is_dir():
                self._mark_db_finalization_pending(
                    snapshot_id,
                    "確定ファイルは存在しますが、DB確定に失敗しました。回復処理を実行してください。",
                )
            else:
                self._mark_failed(snapshot_id, "スナップショット作成に失敗しました。")
            raise UserFacingError("スナップショット作成に失敗しました。") from exc
        with self.session_factory() as session:
            result = DatasetRepository(session).get(snapshot_id)
            if result is None:
                raise UserFacingError("作成済みスナップショットを読み込めません。")
            return DatasetRepository._summary(result)

    def start_snapshot(
        self,
        preview: DatasetPreview,
        *,
        name: str = "",
        description: str = "",
        confirm_warnings: bool = False,
    ) -> UUID:
        snapshot_id = uuid4()
        future = self._executor.submit(
            self.create_snapshot_sync,
            preview,
            name=name,
            description=description,
            confirm_warnings=confirm_warnings,
            _snapshot_id=snapshot_id,
        )
        self._futures[snapshot_id] = future
        return snapshot_id

    def cancel(self, snapshot_id: UUID) -> None:
        with self.session_factory() as session:
            repository = DatasetRepository(session)
            record = repository.get(snapshot_id)
            if record is not None:
                repository.request_cancel(snapshot_id)
                session.commit()

    def list_snapshots(self, project_id: UUID) -> list[DatasetSnapshotSummary]:
        with self.session_factory() as session:
            return DatasetRepository(session).list_snapshots(project_id)

    def recover_stale(self, project_id: UUID | None = None) -> int:
        try:
            with self.session_factory() as session:
                count = DatasetRepository(session).recover_stale(project_id)
                session.commit()
                return count
        except OperationalError:
            return 0

    def recover_finalized_snapshots(self, project_id: UUID | None = None) -> int:
        """Rebuild DB state for final directories left by a failed DB commit."""
        recovered = 0
        try:
            with self.session_factory() as session:
                records = DatasetRepository(session).list_records_for_recovery(
                    project_id
                )
        except OperationalError:
            return 0
        for record in records:
            root = Path(record.snapshot_root).resolve()
            if not root.is_dir():
                continue
            try:
                with self.session_factory() as session:
                    current = DatasetRepository(session).get(UUID(record.id))
                    if (
                        current is None
                        or current.status == DatasetSnapshotStatus.COMPLETED.value
                    ):
                        continue
                    items, issues, manifest_hash, toml_hash, content_hash = (
                        self._reconstruct_from_manifest(current, root)
                    )
                    repository = DatasetRepository(session)
                    for item in items:
                        repository.add_item_if_missing(item)
                    session.flush()
                    stored_items = repository.list_items(UUID(current.id))
                    self._ensure_db_items_match(items, stored_items)
                    for issue in issues:
                        repository.add_issue_if_missing(UUID(current.id), issue)
                    repository.finish(
                        UUID(current.id),
                        DatasetSnapshotStatus.COMPLETED,
                        copied_image_count=len(items),
                        manifest_sha256=manifest_hash,
                        dataset_toml_sha256=toml_hash,
                        content_sha256=content_hash,
                    )
                    session.commit()
                    recovered += 1
            except Exception:
                logger.warning("dataset snapshot recovery failed: %s", record.id)
                with self.session_factory() as session:
                    current = DatasetRepository(session).get(UUID(record.id))
                    if (
                        current is not None
                        and current.status != DatasetSnapshotStatus.COMPLETED.value
                    ):
                        DatasetRepository(session).finish(
                            UUID(record.id),
                            DatasetSnapshotStatus.CORRUPTED,
                            error_summary="確定ファイルの検証またはDB回復に失敗しました。",
                        )
                        session.commit()
        return recovered

    @staticmethod
    def _ensure_db_items_match(
        expected: list[DatasetSnapshotItem], stored: list[Any]
    ) -> None:
        by_sequence = {item.sequence_number: item for item in stored}
        if len(by_sequence) != len(expected):
            raise ValueError("database item count mismatch")
        for item in expected:
            actual = by_sequence.get(item.sequence_number)
            if actual is None or (
                actual.image_id != str(item.image_id)
                or actual.snapshot_image_relative_path
                != item.snapshot_image_relative_path
                or actual.caption_relative_path != item.caption_relative_path
                or actual.snapshot_image_sha256 != item.snapshot_image_sha256
                or actual.caption_sha256 != item.caption_sha256
            ):
                raise ValueError("database item mismatch")

    def revalidate(self, snapshot_id: UUID) -> DatasetSnapshotStatus:
        with self.session_factory() as session:
            record = DatasetRepository(session).get(snapshot_id)
            if record is None:
                raise UserFacingError("スナップショットが見つかりません。")
            root = Path(record.snapshot_root).resolve()
            try:
                self._verify_snapshot_files(record, root, session)
            except Exception:
                DatasetRepository(session).finish(
                    snapshot_id,
                    DatasetSnapshotStatus.CORRUPTED,
                    error_summary="スナップショットの整合性検証に失敗しました。",
                )
                session.commit()
                return DatasetSnapshotStatus.CORRUPTED
            session.commit()
            return DatasetSnapshotStatus(record.status)

    def _preview_image(
        self,
        image: ImageAsset,
        current: Any,
        inspections: list[Any],
        similarity: SimilarityGroupContext | None,
        trigger_words: tuple[str, ...],
        settings: DatasetSettings,
    ) -> DatasetPreviewImage:
        warnings: list[DatasetValidationIssue] = []
        errors: list[DatasetValidationIssue] = []
        if current is None:
            errors.append(
                _issue(
                    "caption_missing",
                    DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.CAPTION,
                    "currentキャプションがありません。",
                    image.id,
                )
            )
            caption_text = ""
        else:
            caption_text = current.caption_text
            if not caption_text.strip():
                target = (
                    "空キャプション許可"
                    if settings.allow_empty_caption
                    else "空キャプション"
                )
                issue = _issue(
                    "caption_empty",
                    DatasetIssueSeverity.WARNING
                    if settings.allow_empty_caption
                    else DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.CAPTION,
                    f"{target}です。",
                    image.id,
                )
                (warnings if settings.allow_empty_caption else errors).append(issue)
        if not image.original_path.is_file():
            errors.append(
                _issue(
                    "file_missing",
                    DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.FILE,
                    "原画像ファイルが見つかりません。",
                    image.id,
                )
            )
        actual_sha = image.sha256
        width, height = image.width, image.height
        file_size = image.file_size
        if image.original_path.is_file():
            try:
                actual_sha = _sha256_file(image.original_path)
                with Image.open(image.original_path) as source:
                    source.load()
                    width, height = source.size
                file_size = image.original_path.stat().st_size
            except Exception:
                errors.append(
                    _issue(
                        "file_corrupt",
                        DatasetIssueSeverity.ERROR,
                        DatasetIssueCategory.FILE,
                        "原画像を読み込めません。",
                        image.id,
                    )
                )
            if actual_sha != image.sha256:
                errors.append(
                    _issue(
                        "source_hash_changed",
                        DatasetIssueSeverity.ERROR,
                        DatasetIssueCategory.INTEGRITY,
                        "原画像のSHA-256がDB値と一致しません。",
                        image.id,
                    )
                )
            if file_size != image.file_size:
                errors.append(
                    _issue(
                        "source_size_changed",
                        DatasetIssueSeverity.ERROR,
                        DatasetIssueCategory.INTEGRITY,
                        "原画像のファイルサイズがDB値と一致しません。",
                        image.id,
                    )
                )
        quality_status = "unknown"
        for result in inspections:
            if result.status is InspectionStatus.FAILED:
                quality_status = "failed"
                warnings.append(
                    _issue(
                        "quality_failed",
                        DatasetIssueSeverity.WARNING,
                        DatasetIssueCategory.QUALITY,
                        result.reason,
                        image.id,
                    )
                )
            elif result.status is InspectionStatus.WARNING:
                if quality_status != "failed":
                    quality_status = "warning"
                warnings.append(
                    _issue(
                        f"quality_{result.rule.value}",
                        DatasetIssueSeverity.WARNING,
                        DatasetIssueCategory.QUALITY,
                        result.reason,
                        image.id,
                    )
                )
        if quality_status == "unknown" and inspections:
            quality_status = "pass"
        exact_status = "unknown"
        for result in inspections:
            if result.rule.value == "exact_duplicate":
                exact_status = (
                    "duplicate"
                    if result.status is InspectionStatus.WARNING
                    else "unique"
                )
                if result.status is InspectionStatus.WARNING:
                    warnings.append(
                        _issue(
                            "exact_duplicate",
                            DatasetIssueSeverity.WARNING,
                            DatasetIssueCategory.DUPLICATE,
                            result.reason,
                            image.id,
                        )
                    )
        tag_count = len(current.tags) if current else 0
        trigger_count = sum(
            1
            for tag in (current.tags if current else ())
            if tag.source.value == "trigger_word"
        )
        if trigger_words and trigger_count == 0:
            warnings.append(
                _issue(
                    "trigger_missing",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.TRIGGER,
                    "トリガーワードがキャプションにありません。",
                    image.id,
                )
            )
        if tag_count == 0:
            warnings.append(
                _issue(
                    "tag_empty",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.CAPTION,
                    "構造化タグがありません。",
                    image.id,
                )
            )
        member = similarity.member if similarity else None
        similarity_group_id = UUID(member.group_id) if member else None
        is_representative = bool(member.is_representative) if member else None
        if similarity and member is not None and not member.is_representative:
            warnings.append(
                _issue(
                    "similarity_nonrepresentative",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.DUPLICATE,
                    "近似重複グループの非代表画像です。",
                    image.id,
                )
            )
        if similarity and similarity.review_status is SimilarityReviewStatus.UNREVIEWED:
            warnings.append(
                _issue(
                    "similarity_group_unreviewed",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.DUPLICATE,
                    "近似重複グループが未確認です。",
                    image.id,
                )
            )
        if similarity and similarity.rejected_pair_count:
            warnings.append(
                _issue(
                    "similarity_group_rejected_pair",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.DUPLICATE,
                    "近似重複グループに否定済みペアがあります。",
                    image.id,
                    similarity.rejected_pair_count,
                )
            )
        aspect = max(width / height, height / width) if width and height else 0.0
        return DatasetPreviewImage(
            image_id=image.id,
            original_filename=image.original_filename,
            source_image_path=image.original_path,
            width=width,
            height=height,
            aspect_ratio=aspect,
            file_size=file_size,
            source_sha256=actual_sha,
            mime_type=image.mime_type,
            selection_state=image.selection_state,
            caption_id=current.id if current else None,
            caption_revision=current.revision if current else None,
            caption_text=caption_text,
            caption_sha256=_sha256_bytes(self._caption_bytes(caption_text)),
            tag_count=tag_count,
            trigger_word_count=trigger_count,
            quality_status=quality_status,
            exact_duplicate_status=exact_status,
            similarity_group_id=similarity_group_id,
            is_similarity_representative=is_representative,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def _materialize(
        self, preview: DatasetPreview, snapshot_id: UUID, temp_root: Path
    ) -> tuple[list[DatasetSnapshotItem], str, str, str]:
        temp_root.mkdir(parents=True, exist_ok=False)
        images_root = temp_root / "images"
        captions_root = temp_root / "captions"
        configs_root = temp_root / "configs"
        reports_root = temp_root / "reports"
        for directory in (images_root, captions_root, configs_root, reports_root):
            directory.mkdir(parents=True, exist_ok=True)
        items: list[DatasetSnapshotItem] = []
        for sequence, image in enumerate(preview.images, 1):
            self._raise_if_canceled(snapshot_id)
            if image.errors:
                raise UserFacingError("作成対象画像に必須エラーがあります。")
            if image.caption_id is None or image.caption_revision is None:
                raise UserFacingError("currentキャプションがありません。")
            suffix = image.source_image_path.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise UserFacingError("対応していない画像拡張子です。")
            # Sequence-only names are safe, deterministic, and keep Windows paths short.
            stem = f"{sequence:06d}"
            destination = images_root / f"{stem}{suffix}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            if not temporary.parent.is_dir():
                raise UserFacingError("一時保存先を作成できませんでした。")
            with (
                image.source_image_path.open("rb") as source,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(source, target, length=1024 * 1024)
            copied_sha = _sha256_file(temporary)
            if (
                copied_sha != image.source_sha256
                or temporary.stat().st_size != image.file_size
            ):
                raise UserFacingError("コピー後の画像検証に失敗しました。")
            temporary.replace(destination)
            caption_relative = f"captions/{stem}.txt"
            caption_path = captions_root / f"{stem}.txt"
            caption_path.write_bytes(self._caption_bytes(image.caption_text))
            caption_sha = _sha256_file(caption_path)
            items.append(
                DatasetSnapshotItem(
                    snapshot_id=snapshot_id,
                    image_id=image.image_id,
                    source_image_path=image.source_image_path,
                    snapshot_image_relative_path=f"images/{destination.name}",
                    caption_relative_path=caption_relative,
                    sequence_number=sequence,
                    source_image_sha256=image.source_sha256,
                    snapshot_image_sha256=copied_sha,
                    source_file_size=image.file_size,
                    snapshot_file_size=destination.stat().st_size,
                    width=image.width,
                    height=image.height,
                    aspect_ratio=image.aspect_ratio,
                    mime_type=image.mime_type,
                    caption_id=image.caption_id,
                    caption_revision=image.caption_revision,
                    caption_sha256=caption_sha,
                    caption_text=image.caption_text,
                    tag_count=image.tag_count,
                    trigger_word_count=image.trigger_word_count,
                    quality_status=image.quality_status,
                    exact_duplicate_status=image.exact_duplicate_status,
                    similarity_group_id=image.similarity_group_id,
                    is_similarity_representative=image.is_similarity_representative,
                    warnings=image.warnings,
                )
            )
            self._update_progress(snapshot_id, sequence, image.image_id)
        toml_text = self.config.to_toml(preview.settings)
        toml_path = configs_root / "dataset.toml"
        toml_path.write_text(toml_text, encoding="utf-8", newline="\n")
        toml_hash = _sha256_file(toml_path)
        report = self._report(items, preview)
        (reports_root / "dataset_report.json").write_text(
            _json(report.report_json) + "\n", encoding="utf-8", newline="\n"
        )
        (reports_root / "dataset_report.md").write_text(
            report.report_markdown, encoding="utf-8", newline="\n"
        )
        (reports_root / "tag_frequency.csv").write_text(
            report.tag_frequency_csv, encoding="utf-8", newline="\n"
        )
        (reports_root / "resolution_distribution.csv").write_text(
            report.resolution_csv, encoding="utf-8", newline="\n"
        )
        (reports_root / "aspect_ratio_distribution.csv").write_text(
            report.aspect_ratio_csv, encoding="utf-8", newline="\n"
        )
        (reports_root / "warnings.json").write_text(
            report.warnings_json, encoding="utf-8", newline="\n"
        )
        settings_snapshot = self.config.settings_snapshot(preview.settings)
        content_hash = _content_sha256(items, toml_hash, settings_snapshot)
        manifest = self._manifest(preview, snapshot_id, items, toml_hash, content_hash)
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8", newline="\n")
        manifest_hash = _sha256_file(manifest_path)
        snapshot_json = {
            "snapshot_id": str(snapshot_id),
            "status": "completed",
            "manifest_sha256": manifest_hash,
            "dataset_toml_sha256": toml_hash,
            "content_sha256": content_hash,
            "generator_version": self.generator_version,
        }
        (temp_root / "snapshot.json").write_text(
            _json(snapshot_json) + "\n", encoding="utf-8", newline="\n"
        )
        return items, manifest_hash, toml_hash, content_hash

    def _report(
        self, items: list[DatasetSnapshotItem], preview: DatasetPreview
    ) -> DatasetReport:
        tags: Counter[str] = Counter()
        tag_images: dict[str, set[str]] = {}
        resolutions: Counter[str] = Counter()
        aspects: Counter[str] = Counter()
        widths: list[int] = []
        heights: list[int] = []
        short_edges: list[int] = []
        long_edges: list[int] = []
        pixels: list[int] = []
        ratios: list[float] = []
        bucket_candidates: Counter[str] = Counter()
        for item in items:
            widths.append(item.width)
            heights.append(item.height)
            short_edges.append(min(item.width, item.height))
            long_edges.append(max(item.width, item.height))
            pixels.append(item.width * item.height)
            ratio = item.width / item.height if item.height else 0.0
            ratios.append(ratio)
            resolutions[f"{item.width}x{item.height}"] += 1
            bucket = (
                "square_near"
                if abs(ratio - 1.0) <= 0.05
                else ("landscape" if ratio > 1.0 else "portrait")
            )
            aspects[bucket] += 1
            bucket_candidates[self._bucket_candidate(item, preview.settings)] += 1
            seen: set[str] = set()
            for tag_value in parse_caption_tags(item.caption_text):
                if tag_value.normalized_name in seen:
                    continue
                seen.add(tag_value.normalized_name)
                tags[tag_value.normalized_name] += 1
                tag_images.setdefault(tag_value.normalized_name, set()).add(
                    str(item.image_id)
                )
        total = len(items)
        tag_rows = ["tag_name,image_count,occurrence_rate"]
        for tag in sorted(tags, key=lambda value: (-tags[value], value)):
            tag_rows.append(
                f"{self._csv(tag)},{tags[tag]},{tags[tag] / total if total else 0:.6f}"
            )
        resolution_rows = ["resolution,image_count"] + [
            f"{self._csv(key)},{value}" for key, value in sorted(resolutions.items())
        ]
        aspect_rows = ["aspect_class,image_count"] + [
            f"{self._csv(key)},{value}" for key, value in sorted(aspects.items())
        ]
        exact_count = sum(item.exact_duplicate_status == "duplicate" for item in items)
        approx_count = sum(item.similarity_group_id is not None for item in items)
        resolution_stats = {
            "width": self._numeric_stats(widths),
            "height": self._numeric_stats(heights),
            "short_edge": self._numeric_stats(short_edges),
            "long_edge": self._numeric_stats(long_edges),
            "total_pixels": self._numeric_stats(pixels),
            "bins": {
                "under_512": sum(value < 512 for value in short_edges),
                "512_767": sum(512 <= value <= 767 for value in short_edges),
                "768_1023": sum(768 <= value <= 1023 for value in short_edges),
                "1024_1279": sum(1024 <= value <= 1279 for value in short_edges),
                "1280_or_more": sum(value >= 1280 for value in short_edges),
            },
        }
        aspect_stats = {
            "portrait": aspects["portrait"],
            "landscape": aspects["landscape"],
            "square_near": aspects["square_near"],
            "square_near_definition": "abs(width / height - 1.0) <= 0.05",
            "ratio": self._numeric_stats(ratios),
            "bucket_candidates": dict(sorted(bucket_candidates.items())),
        }
        report_json: dict[str, object] = {
            "image_count": total,
            "total_size_bytes": sum(item.snapshot_file_size for item in items),
            "mean_width": mean([item.width for item in items]) if items else 0,
            "median_width": median([item.width for item in items]) if items else 0,
            "resolution_distribution": dict(sorted(resolutions.items())),
            "aspect_ratio_distribution": dict(sorted(aspects.items())),
            "resolution_stats": resolution_stats,
            "aspect_ratio_stats": aspect_stats,
            "tag_frequency": [
                {
                    "tag": tag,
                    "image_count": tags[tag],
                    "occurrence_rate": tags[tag] / total if total else 0,
                }
                for tag in sorted(tags, key=lambda value: (-tags[value], value))
            ],
            "exact_duplicate_rate": exact_count / total if total else 0,
            "approximate_duplicate_rate": approx_count / total if total else 0,
            "trigger_word_rate": sum(item.trigger_word_count > 0 for item in items)
            / total
            if total
            else 0,
            "empty_caption_count": sum(not item.caption_text.strip() for item in items),
            "warning_count": preview.summary.warning_count,
            "storage_estimate": {
                "image_size_bytes": preview.summary.image_size_bytes,
                "caption_size_bytes": preview.summary.caption_size_bytes,
                "metadata_size_bytes": preview.summary.metadata_size_bytes,
                "required_size_bytes": preview.summary.required_size_bytes,
                "safety_margin_bytes": preview.summary.safety_margin_bytes,
                "warning_margin_bytes": preview.summary.warning_margin_bytes,
                "available_disk_bytes": preview.summary.available_disk_bytes,
            },
            "unreviewed_group_count": preview.summary.unreviewed_group_count,
            "rejected_pair_group_count": preview.summary.rejected_pair_group_count,
            "unreviewed_group_image_count": preview.summary.unreviewed_group_image_count,
            "similarity_groups": [
                {
                    "group_id": str(group.group_id),
                    "review_status": group.review_status,
                    "member_count": group.member_count,
                    "target_image_count": group.target_image_count,
                    "representative_image_id": (
                        str(group.representative_image_id)
                        if group.representative_image_id
                        else None
                    ),
                    "rejected_pair_count": group.rejected_pair_count,
                }
                for group in preview.similarity_groups
            ],
        }
        markdown = (
            "# Dataset Report\n\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value in report_json.items()
                if not isinstance(value, (list, dict))
            )
            + "\n"
            + "## Resolution statistics\n\n"
            + _json(resolution_stats)
            + "\n\n## Aspect ratio statistics\n\n"
            + _json(aspect_stats)
            + "\n"
        )
        warnings = [
            _issue_dict(issue)
            for issue in preview.issues
            if issue.severity is not DatasetIssueSeverity.INFO
        ]
        return DatasetReport(
            report_json,
            markdown,
            "\n".join(tag_rows) + "\n",
            "\n".join(resolution_rows) + "\n",
            "\n".join(aspect_rows) + "\n",
            _json(warnings) + "\n",
        )

    @staticmethod
    def _numeric_stats(values: list[int] | list[float]) -> dict[str, float | int]:
        if not values:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p10": 0,
                "p25": 0,
                "p75": 0,
                "p90": 0,
            }
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": mean(ordered),
            "median": median(ordered),
            "p10": DatasetSnapshotService._percentile(ordered, 0.10),
            "p25": DatasetSnapshotService._percentile(ordered, 0.25),
            "p75": DatasetSnapshotService._percentile(ordered, 0.75),
            "p90": DatasetSnapshotService._percentile(ordered, 0.90),
        }

    @staticmethod
    def _percentile(values: list[int] | list[float], fraction: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(values[lower])
        weight = position - lower
        return float(values[lower]) * (1 - weight) + float(values[upper]) * weight

    @staticmethod
    def _bucket_candidate(item: DatasetSnapshotItem, settings: DatasetSettings) -> str:
        if not settings.enable_bucket or item.width <= 0 or item.height <= 0:
            return f"{settings.resolution}x{settings.resolution}"
        scale = math.sqrt((settings.resolution**2) / (item.width * item.height))
        width = max(1, round(item.width * scale / settings.bucket_reso_steps))
        height = max(1, round(item.height * scale / settings.bucket_reso_steps))
        width = min(
            settings.max_bucket_reso,
            max(settings.min_bucket_reso, width * settings.bucket_reso_steps),
        )
        height = min(
            settings.max_bucket_reso,
            max(settings.min_bucket_reso, height * settings.bucket_reso_steps),
        )
        return f"{width}x{height}"

    def _manifest(
        self,
        preview: DatasetPreview,
        snapshot_id: UUID,
        items: list[DatasetSnapshotItem],
        toml_hash: str,
        content_hash: str,
    ) -> dict[str, object]:
        return {
            "manifest_schema_version": "phase4-manifest-v1",
            "snapshot_id": str(snapshot_id),
            "project_id": str(preview.project_id),
            "project_name": preview.project_name,
            "created_at": datetime.now(UTC).isoformat(),
            "generator_version": self.generator_version,
            "source_tagger_run_id": str(preview.source_tagger_run_ids[0])
            if len(preview.source_tagger_run_ids) == 1
            else None,
            "image_count": len(items),
            "caption_count": len(items),
            "total_size_bytes": sum(item.snapshot_file_size for item in items),
            "trigger_words": list(preview.trigger_words),
            "dataset_settings": json.loads(
                self.config.settings_snapshot(preview.settings)
            ),
            "settings_sha256": _settings_sha256(
                self.config.settings_snapshot(preview.settings)
            ),
            "dataset_toml_relative_path": "configs/dataset.toml",
            "dataset_toml_sha256": toml_hash,
            "content_sha256": content_hash,
            "report_paths": {
                "json": "reports/dataset_report.json",
                "markdown": "reports/dataset_report.md",
                "warnings": "reports/warnings.json",
            },
            "items": [self._item_manifest(item) for item in items],
            "validation_issues": [_issue_dict(issue) for issue in preview.issues],
        }

    @staticmethod
    def _item_manifest(item: DatasetSnapshotItem) -> dict[str, object]:
        return {
            "sequence_number": item.sequence_number,
            "image_id": str(item.image_id),
            "source_image_path": str(item.source_image_path),
            "original_filename": Path(item.source_image_path).name,
            "snapshot_image_relative_path": item.snapshot_image_relative_path,
            "caption_relative_path": item.caption_relative_path,
            "source_image_sha256": item.source_image_sha256,
            "snapshot_image_sha256": item.snapshot_image_sha256,
            "source_file_size": item.source_file_size,
            "image_file_size": item.snapshot_file_size,
            "width": item.width,
            "height": item.height,
            "aspect_ratio": item.aspect_ratio,
            "mime_type": item.mime_type,
            "caption_revision": item.caption_revision,
            "caption_id": str(item.caption_id),
            "caption_sha256": item.caption_sha256,
            "caption_text": item.caption_text,
            "tag_count": item.tag_count,
            "trigger_word_count": item.trigger_word_count,
            "quality_status": item.quality_status,
            "warnings": [_issue_dict(issue) for issue in item.warnings],
            "exact_duplicate_status": item.exact_duplicate_status,
            "similarity_group_id": str(item.similarity_group_id)
            if item.similarity_group_id
            else None,
            "is_similarity_representative": item.is_similarity_representative,
        }

    def _similarity_members(
        self, session: Any, images: list[ImageAsset]
    ) -> dict[UUID, SimilarityGroupContext]:
        ids = [str(image.id) for image in images]
        if not ids:
            return {}
        records = session.scalars(
            select(SimilarityGroupMemberRecord)
            .join(SimilarityGroupRecord)
            .where(SimilarityGroupMemberRecord.image_id.in_(ids))
            .order_by(SimilarityGroupMemberRecord.updated_at.desc())
        ).all()
        group_ids = {record.group_id for record in records}
        all_members = (
            session.scalars(
                select(SimilarityGroupMemberRecord).where(
                    SimilarityGroupMemberRecord.group_id.in_(group_ids)
                )
            ).all()
            if group_ids
            else []
        )
        rejected_rows = (
            session.scalars(
                select(SimilarityPairReviewRecord).where(
                    SimilarityPairReviewRecord.project_id == str(images[0].project_id),
                    SimilarityPairReviewRecord.review_status
                    == SimilarityReviewStatus.REJECTED_SIMILARITY.value,
                )
            ).all()
            if records
            else []
        )
        result: dict[UUID, SimilarityGroupContext] = {}
        group_members: dict[str, set[str]] = {}
        for record in all_members:
            group_members.setdefault(record.group_id, set()).add(record.image_id)
        rejected_counts: Counter[str] = Counter()
        for row in rejected_rows:
            for group_id, members in group_members.items():
                if row.image_left_id in members and row.image_right_id in members:
                    rejected_counts[group_id] += 1
                    break
        for record in records:
            status = SimilarityReviewStatus(record.review_status)
            result.setdefault(
                UUID(record.image_id),
                SimilarityGroupContext(
                    member=record,
                    review_status=status,
                    member_count=len(group_members[record.group_id]),
                    rejected_pair_count=rejected_counts[record.group_id],
                    representative_image_id=(
                        UUID(record.group.representative_image_id)
                        if record.group.representative_image_id
                        else None
                    ),
                ),
            )
        return result

    @staticmethod
    def _similarity_group_summaries(
        groups: dict[UUID, SimilarityGroupContext],
    ) -> tuple[DatasetSimilarityGroupSummary, ...]:
        grouped: dict[str, list[SimilarityGroupContext]] = {}
        for context in groups.values():
            grouped.setdefault(context.member.group_id, []).append(context)
        return tuple(
            DatasetSimilarityGroupSummary(
                group_id=UUID(group_id),
                review_status=contexts[0].review_status.value,
                member_count=contexts[0].member_count,
                target_image_count=len(contexts),
                representative_image_id=contexts[0].representative_image_id,
                rejected_pair_count=contexts[0].rejected_pair_count,
            )
            for group_id, contexts in sorted(grouped.items())
        )

    def _summary(
        self,
        images: list[DatasetPreviewImage],
        issues: list[DatasetValidationIssue],
        groups: dict[UUID, SimilarityGroupContext],
        settings: DatasetSettings,
    ) -> DatasetPreviewSummary:
        quality_failed_images = {
            issue.image_id
            for issue in issues
            if issue.image_id
            and issue.category is DatasetIssueCategory.QUALITY
            and issue.issue_code == "quality_failed"
        }
        quality_warning_images = {
            issue.image_id
            for issue in issues
            if issue.image_id
            and issue.category is DatasetIssueCategory.QUALITY
            and issue.severity is DatasetIssueSeverity.WARNING
            and issue.image_id not in quality_failed_images
        }
        exact = [
            image for image in images if image.exact_duplicate_status == "duplicate"
        ]
        approx = [image for image in images if image.similarity_group_id is not None]
        image_size = sum(image.file_size for image in images if not image.errors)
        caption_size = sum(
            len(self._caption_bytes(image.caption_text))
            for image in images
            if not image.errors
        )
        metadata_size = self._metadata_estimate(len(images), settings)
        estimate = image_size + caption_size + metadata_size
        try:
            available = shutil.disk_usage(self.settings.projects_dir).free
        except OSError:
            available = -1
        required = estimate + self.settings.dataset_disk_safety_margin_bytes
        group_values = list(groups.values())
        unreviewed_ids = {
            context.member.group_id
            for context in group_values
            if context.review_status is SimilarityReviewStatus.UNREVIEWED
        }
        rejected_ids = {
            context.member.group_id
            for context in group_values
            if context.rejected_pair_count
        }
        return DatasetPreviewSummary(
            target_image_count=len(images),
            caption_present_count=sum(image.caption_id is not None for image in images),
            caption_missing_count=sum(image.caption_id is None for image in images),
            missing_file_count=sum(
                any(issue.issue_code == "file_missing" for issue in image.errors)
                for image in images
            ),
            corrupt_file_count=sum(
                any(issue.issue_code == "file_corrupt" for issue in image.errors)
                for image in images
            ),
            quality_warning_image_count=len(quality_warning_images),
            quality_failed_image_count=len(quality_failed_images),
            exact_duplicate_count=len(exact),
            exact_duplicate_nonrepresentative_count=sum(
                image.exact_duplicate_status == "duplicate" for image in images
            ),
            approximate_duplicate_count=len(approx),
            approximate_duplicate_nonrepresentative_count=sum(
                image.is_similarity_representative is False for image in approx
            ),
            unreviewed_group_count=len(unreviewed_ids),
            rejected_pair_group_count=len(rejected_ids),
            unreviewed_group_image_count=sum(
                context.member.group_id in unreviewed_ids for context in group_values
            ),
            empty_caption_count=sum(not image.caption_text.strip() for image in images),
            trigger_missing_count=sum(
                any(issue.issue_code == "trigger_missing" for issue in image.warnings)
                for image in images
            ),
            warning_count=sum(
                issue.severity is DatasetIssueSeverity.WARNING for issue in issues
            ),
            error_count=sum(
                issue.severity is DatasetIssueSeverity.ERROR for issue in issues
            ),
            estimated_size_bytes=estimate,
            image_size_bytes=image_size,
            caption_size_bytes=caption_size,
            metadata_size_bytes=metadata_size,
            required_size_bytes=required,
            safety_margin_bytes=self.settings.dataset_disk_safety_margin_bytes,
            warning_margin_bytes=self.settings.dataset_disk_warning_margin_bytes,
            available_disk_bytes=available,
            estimated_free_bytes=available - required if available >= 0 else -1,
        )

    @staticmethod
    def _metadata_estimate(
        image_count: int, settings: DatasetSettings | None = None
    ) -> int:
        # Conservative estimate for manifest, snapshot metadata, TOML, reports and CSVs.
        toml_size = (
            len(DatasetConfigService().to_toml(settings).encode("utf-8"))
            if settings is not None
            else 0
        )
        return max(16 * 1024, toml_size) + image_count * 4096

    def _storage_issues(
        self,
        *,
        project_root: Path,
        images: list[DatasetPreviewImage],
        settings: DatasetSettings,
    ) -> list[DatasetValidationIssue]:
        image_size = sum(image.file_size for image in images if not image.errors)
        caption_size = sum(
            len(self._caption_bytes(image.caption_text))
            for image in images
            if not image.errors
        )
        payload = (
            image_size + caption_size + self._metadata_estimate(len(images), settings)
        )
        required = payload + self.settings.dataset_disk_safety_margin_bytes
        try:
            available = shutil.disk_usage(project_root).free
        except OSError:
            return [
                _issue(
                    "disk_space_unavailable",
                    DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.STORAGE,
                    "保存先の空き容量を取得できません。",
                )
            ]
        if available < required:
            return [
                _issue(
                    "disk_space_insufficient",
                    DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.STORAGE,
                    "スナップショットに必要な空き容量が不足しています。",
                    measured=available,
                )
            ]
        if available - required < self.settings.dataset_disk_warning_margin_bytes:
            return [
                _issue(
                    "disk_space_low",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.STORAGE,
                    "スナップショット作成後の空き容量が少なくなります。",
                    measured=available - required,
                )
            ]
        return []

    def _validate_disk_capacity(
        self, project_root: Path, preview: DatasetPreview
    ) -> None:
        issues = self._storage_issues(
            project_root=project_root,
            images=list(preview.images),
            settings=preview.settings,
        )
        if any(issue.severity is DatasetIssueSeverity.ERROR for issue in issues):
            raise UserFacingError("保存先の空き容量が不足しているため作成できません。")

    def _mark_db_finalization_pending(self, snapshot_id: UUID, message: str) -> None:
        with self.session_factory() as session:
            DatasetRepository(session).mark_db_finalization_pending(
                snapshot_id, message
            )
            session.commit()

    def _mark_failed(self, snapshot_id: UUID, message: str) -> None:
        with self.session_factory() as session:
            DatasetRepository(session).finish(
                snapshot_id,
                DatasetSnapshotStatus.FAILED,
                error_summary=message,
            )
            session.commit()

    def _with_token(self, preview: DatasetPreview) -> DatasetPreview:
        payload = {
            "project_id": str(preview.project_id),
            "project_updated_at": preview.project_updated_at.isoformat(),
            "trigger_words": preview.trigger_words,
            "settings": json.loads(self.config.settings_snapshot(preview.settings)),
            "generator_version": self.generator_version,
            "images": [
                {
                    "id": str(image.image_id),
                    "state": image.selection_state.value,
                    "source_sha256": image.source_sha256,
                    "file_size": image.file_size,
                    "caption_id": str(image.caption_id) if image.caption_id else None,
                    "caption_revision": image.caption_revision,
                    "caption_sha256": image.caption_sha256,
                    "caption_text": image.caption_text,
                }
                for image in preview.images
            ],
            "source_tagger_run_ids": [
                str(value) for value in preview.source_tagger_run_ids
            ],
        }
        return DatasetPreview(
            token=_sha256_bytes(_json(payload).encode("utf-8")),
            project_id=preview.project_id,
            project_updated_at=preview.project_updated_at,
            project_name=preview.project_name,
            trigger_words=preview.trigger_words,
            settings=preview.settings,
            images=preview.images,
            issues=preview.issues,
            summary=preview.summary,
            source_tagger_run_ids=preview.source_tagger_run_ids,
            similarity_groups=preview.similarity_groups,
        )

    def _verify_snapshot_files(self, record: Any, root: Path, session: Any) -> None:
        legacy_manifest = root / "manifest.json"
        if legacy_manifest.is_file():
            legacy_data = json.loads(legacy_manifest.read_text(encoding="utf-8"))
            legacy_items = legacy_data.get("items", [])
            if not legacy_data.get("settings_sha256") or any(
                not isinstance(item, dict) or not item.get("caption_id")
                for item in legacy_items
            ):
                self._verify_legacy_snapshot_files(record, root, session)
                return
        manifest_items, _, manifest_hash, toml_hash, content_hash = (
            self._reconstruct_from_manifest(record, root)
        )
        if record.manifest_sha256 != manifest_hash:
            raise ValueError("manifest hash mismatch")
        if record.dataset_toml_sha256 != toml_hash:
            raise ValueError("dataset TOML hash mismatch")
        stored = DatasetRepository(session).list_items(UUID(record.id))
        if len(stored) != len(manifest_items):
            raise ValueError("database item count mismatch")
        for expected, actual in zip(manifest_items, stored, strict=True):
            if (
                str(expected.image_id) != actual.image_id
                or expected.snapshot_image_relative_path
                != actual.snapshot_image_relative_path
                or expected.caption_relative_path != actual.caption_relative_path
                or expected.snapshot_image_sha256 != actual.snapshot_image_sha256
                or expected.caption_sha256 != actual.caption_sha256
            ):
                raise ValueError("database item mismatch")
        if record.content_sha256 != content_hash:
            raise ValueError("snapshot content hash mismatch")

    def _verify_legacy_snapshot_files(
        self, record: Any, root: Path, session: Any
    ) -> None:
        if not root.is_dir():
            raise ValueError("snapshot root missing")
        manifest = root / "manifest.json"
        toml = root / "configs" / "dataset.toml"
        report = root / "reports" / "dataset_report.json"
        for path in (manifest, toml, report):
            self._ensure_inside(path, root)
            if not path.is_file():
                raise ValueError("required snapshot file missing")
        if record.manifest_sha256 != _sha256_file(manifest):
            raise ValueError("legacy manifest hash mismatch")
        if record.dataset_toml_sha256 != _sha256_file(toml):
            raise ValueError("legacy TOML hash mismatch")
        items = DatasetRepository(session).list_items(UUID(record.id))
        for item in items:
            image_path = root / item.snapshot_image_relative_path
            caption_path = root / item.caption_relative_path
            self._ensure_inside(image_path, root)
            self._ensure_inside(caption_path, root)
            if not image_path.is_file() or not caption_path.is_file():
                raise ValueError("legacy snapshot item missing")
            if _sha256_file(image_path) != item.snapshot_image_sha256:
                raise ValueError("legacy snapshot image hash mismatch")
            if _sha256_file(caption_path) != item.caption_sha256:
                raise ValueError("legacy snapshot caption hash mismatch")

    def _reconstruct_from_manifest(
        self, record: Any, root: Path
    ) -> tuple[list[DatasetSnapshotItem], list[DatasetValidationIssue], str, str, str]:
        if not root.is_dir():
            raise ValueError("snapshot root missing")
        manifest = root / "manifest.json"
        toml = root / "configs" / "dataset.toml"
        report = root / "reports" / "dataset_report.json"
        for path in (manifest, toml, report):
            self._ensure_inside(path, root)
            if not path.is_file():
                raise ValueError("required snapshot file missing")
        tomllib.loads(toml.read_text(encoding="utf-8"))
        json.loads(report.read_text(encoding="utf-8"))
        manifest_hash = _sha256_file(manifest)
        toml_hash = _sha256_file(toml)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        expected_settings = json.loads(record.settings_snapshot)
        if data.get("dataset_settings") != expected_settings:
            raise ValueError("dataset settings mismatch")
        if data.get("settings_sha256") != _settings_sha256(record.settings_snapshot):
            raise ValueError("settings hash mismatch")
        manifest_items = data.get("items")
        if not isinstance(manifest_items, list):
            raise ValueError("manifest item count mismatch")
        parsed_items: list[DatasetSnapshotItem] = []
        for raw in sorted(
            manifest_items,
            key=lambda value: (
                int(value["sequence_number"]) if isinstance(value, dict) else 0
            ),
        ):
            if not isinstance(raw, dict):
                raise ValueError("manifest item is invalid")
            try:
                sequence = int(raw["sequence_number"])
                image_id = UUID(str(raw["image_id"]))
                caption_id = UUID(str(raw["caption_id"]))
                image_relative = str(raw["snapshot_image_relative_path"])
                caption_relative = str(raw["caption_relative_path"])
                image_sha = str(raw["snapshot_image_sha256"])
                caption_sha = str(raw["caption_sha256"])
                image_size = int(raw["image_file_size"])
                width = int(raw["width"])
                height = int(raw["height"])
                aspect_ratio = float(raw["aspect_ratio"])
                caption_revision = int(raw["caption_revision"])
                caption_text = str(raw["caption_text"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("manifest item is incomplete") from exc
            image_path = root / image_relative
            caption_path = root / caption_relative
            self._ensure_inside(image_path, root)
            self._ensure_inside(caption_path, root)
            if not image_path.is_file() or not caption_path.is_file():
                raise ValueError("snapshot item missing")
            if _sha256_file(image_path) != image_sha:
                raise ValueError("snapshot image hash mismatch")
            if image_path.stat().st_size != image_size:
                raise ValueError("snapshot image size mismatch")
            if _sha256_file(caption_path) != caption_sha:
                raise ValueError("snapshot caption hash mismatch")
            if caption_path.read_bytes() != self._caption_bytes(caption_text):
                raise ValueError("snapshot caption content mismatch")
            warnings = tuple(
                _issue_from_dict(value)
                for value in raw.get("warnings", [])
                if isinstance(value, dict)
            )
            group_id = raw.get("similarity_group_id")
            parsed_items.append(
                DatasetSnapshotItem(
                    snapshot_id=UUID(record.id),
                    image_id=image_id,
                    source_image_path=Path(str(raw.get("source_image_path", ""))),
                    snapshot_image_relative_path=image_relative,
                    caption_relative_path=caption_relative,
                    sequence_number=sequence,
                    source_image_sha256=str(raw.get("source_image_sha256", image_sha)),
                    snapshot_image_sha256=image_sha,
                    source_file_size=int(raw.get("source_file_size", image_size)),
                    snapshot_file_size=image_size,
                    width=width,
                    height=height,
                    aspect_ratio=aspect_ratio,
                    mime_type=str(raw.get("mime_type", "application/octet-stream")),
                    caption_id=caption_id,
                    caption_revision=caption_revision,
                    caption_sha256=caption_sha,
                    caption_text=caption_text,
                    tag_count=int(raw.get("tag_count", 0)),
                    trigger_word_count=int(raw.get("trigger_word_count", 0)),
                    quality_status=str(raw.get("quality_status", "unknown")),
                    exact_duplicate_status=str(
                        raw.get("exact_duplicate_status", "unknown")
                    ),
                    similarity_group_id=UUID(str(group_id)) if group_id else None,
                    is_similarity_representative=raw.get(
                        "is_similarity_representative"
                    ),
                    warnings=warnings,
                )
            )
        if [item.sequence_number for item in parsed_items] != list(
            range(1, len(parsed_items) + 1)
        ):
            raise ValueError("manifest sequence is invalid")
        content_hash = _content_sha256(
            parsed_items, toml_hash, record.settings_snapshot
        )
        if data.get("content_sha256") != content_hash:
            raise ValueError("snapshot content hash mismatch")
        issues = tuple(
            _issue_from_dict(value)
            for value in data.get("validation_issues", [])
            if isinstance(value, dict)
        )
        return parsed_items, list(issues), manifest_hash, toml_hash, content_hash

    @staticmethod
    def _summary_dict(summary: DatasetPreviewSummary) -> dict[str, object]:
        return {key: getattr(summary, key) for key in summary.__dataclass_fields__}

    @staticmethod
    def _validate_name(name: str) -> str:
        value = name.strip() or f"dataset-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        if (
            len(value) > 200
            or any(ord(char) < 32 for char in value)
            or "/" in value
            or "\\" in value
            or ".." in value
        ):
            raise UserFacingError(
                "スナップショット名に使用できない文字が含まれています。"
            )
        return value

    @staticmethod
    def _ensure_inside(path: Path, root: Path) -> None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("snapshot path escapes root") from exc

    @staticmethod
    def _caption_bytes(text: str) -> bytes:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return (normalized + "\n").encode("utf-8")

    def _raise_if_canceled(self, snapshot_id: UUID) -> None:
        with self.session_factory() as session:
            if DatasetRepository(session).cancel_requested(snapshot_id):
                raise DatasetSnapshotCanceled()

    def _update_progress(
        self, snapshot_id: UUID, processed: int, image_id: UUID
    ) -> None:
        with self.session_factory() as session:
            DatasetRepository(session).update_progress(
                snapshot_id,
                processed_count=processed,
                current_step="copying",
                current_image_id=image_id,
            )
            session.commit()

    @staticmethod
    def _csv(value: str) -> str:
        safe = value.replace('"', '""')
        if safe.startswith(("=", "+", "-", "@")):
            safe = "'" + safe
        return f'"{safe}"'


def _issue_dict(issue: DatasetValidationIssue) -> dict[str, object]:
    return {
        "issue_code": issue.issue_code,
        "severity": issue.severity.value,
        "category": issue.category.value,
        "message": issue.message,
        "image_id": str(issue.image_id) if issue.image_id else None,
        "measured_value": issue.measured_value,
        "threshold_value": issue.threshold_value,
        "details": issue.details,
    }


def _issue_from_dict(value: dict[str, object]) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        issue_code=str(value.get("issue_code", "unknown")),
        severity=DatasetIssueSeverity(str(value.get("severity", "warning"))),
        category=DatasetIssueCategory(str(value.get("category", "integrity"))),
        message=str(value.get("message", "")),
        image_id=(UUID(str(value["image_id"])) if value.get("image_id") else None),
        measured_value=(
            str(value["measured_value"]) if value.get("measured_value") else None
        ),
        threshold_value=(
            str(value["threshold_value"]) if value.get("threshold_value") else None
        ),
        details=str(value.get("details", "")),
    )
