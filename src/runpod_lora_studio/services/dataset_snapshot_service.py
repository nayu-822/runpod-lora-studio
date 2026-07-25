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
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import UUID, uuid4

from PIL import Image
from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    DatasetIssueCategory,
    DatasetIssueSeverity,
    DatasetPreview,
    DatasetPreviewImage,
    DatasetPreviewSummary,
    DatasetReport,
    DatasetSettings,
    DatasetSnapshotItem,
    DatasetSnapshotStatus,
    DatasetSnapshotSummary,
    DatasetValidationIssue,
    ImageAsset,
    InspectionStatus,
    SelectionState,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.dataset_repository import DatasetRepository
from runpod_lora_studio.persistence.models import (
    SimilarityGroupMemberRecord,
    SimilarityGroupRecord,
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
            summary = self._summary(previews, all_issues)
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
        try:
            items, manifest_hash, toml_hash, content_hash = self._materialize(
                preview, snapshot_id, temp_root
            )
            temp_root.replace(final_root)
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
        except Exception as exc:
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)
            with self.session_factory() as session:
                DatasetRepository(session).finish(
                    snapshot_id,
                    DatasetSnapshotStatus.FAILED,
                    error_summary="スナップショット作成に失敗しました。",
                )
                session.commit()
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
        with self.session_factory() as session:
            count = DatasetRepository(session).recover_stale(project_id)
            session.commit()
            return count

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
        similarity: SimilarityGroupMemberRecord | None,
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
        similarity_group_id = UUID(similarity.group_id) if similarity else None
        is_representative = bool(similarity.is_representative) if similarity else None
        if similarity and not similarity.is_representative:
            warnings.append(
                _issue(
                    "similarity_nonrepresentative",
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.DUPLICATE,
                    "近似重複グループの非代表画像です。",
                    image.id,
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
            source_bytes = image.source_image_path.read_bytes()
            temporary.write_bytes(source_bytes)
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
        content_payload = [
            (item.sequence_number, item.snapshot_image_sha256, item.caption_sha256)
            for item in items
        ]
        content_hash = _sha256_bytes(_json(content_payload).encode("utf-8"))
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
        for item in items:
            resolutions[f"{item.width}x{item.height}"] += 1
            bucket = (
                "square"
                if math.isclose(item.width, item.height)
                else ("landscape" if item.width > item.height else "portrait")
            )
            aspects[bucket] += 1
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
        report_json: dict[str, object] = {
            "image_count": total,
            "total_size_bytes": sum(item.snapshot_file_size for item in items),
            "mean_width": mean([item.width for item in items]) if items else 0,
            "median_width": median([item.width for item in items]) if items else 0,
            "resolution_distribution": dict(sorted(resolutions.items())),
            "aspect_ratio_distribution": dict(sorted(aspects.items())),
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
        }
        markdown = (
            "# Dataset Report\n\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value in report_json.items()
                if not isinstance(value, (list, dict))
            )
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
            "dataset_toml_relative_path": "configs/dataset.toml",
            "dataset_toml_sha256": toml_hash,
            "content_sha256": content_hash,
            "report_paths": {
                "json": "reports/dataset_report.json",
                "markdown": "reports/dataset_report.md",
                "warnings": "reports/warnings.json",
            },
            "items": [self._item_manifest(item) for item in items],
        }

    @staticmethod
    def _item_manifest(item: DatasetSnapshotItem) -> dict[str, object]:
        return {
            "sequence_number": item.sequence_number,
            "image_id": str(item.image_id),
            "original_filename": Path(item.source_image_path).name,
            "snapshot_image_relative_path": item.snapshot_image_relative_path,
            "caption_relative_path": item.caption_relative_path,
            "source_image_sha256": item.source_image_sha256,
            "snapshot_image_sha256": item.snapshot_image_sha256,
            "image_file_size": item.snapshot_file_size,
            "width": item.width,
            "height": item.height,
            "mime_type": item.mime_type,
            "caption_revision": item.caption_revision,
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
    ) -> dict[UUID, SimilarityGroupMemberRecord]:
        ids = [str(image.id) for image in images]
        if not ids:
            return {}
        records = session.scalars(
            select(SimilarityGroupMemberRecord)
            .join(SimilarityGroupRecord)
            .where(SimilarityGroupMemberRecord.image_id.in_(ids))
            .order_by(SimilarityGroupMemberRecord.updated_at.desc())
        ).all()
        result: dict[UUID, SimilarityGroupMemberRecord] = {}
        for record in records:
            result.setdefault(UUID(record.image_id), record)
        return result

    def _summary(
        self, images: list[DatasetPreviewImage], issues: list[DatasetValidationIssue]
    ) -> DatasetPreviewSummary:
        warning_images = {
            issue.image_id
            for issue in issues
            if issue.severity is DatasetIssueSeverity.WARNING and issue.image_id
        }
        failed_images = {
            issue.image_id for issue in issues if issue.issue_code == "quality_failed"
        }
        exact = [
            image for image in images if image.exact_duplicate_status == "duplicate"
        ]
        approx = [image for image in images if image.similarity_group_id is not None]
        available = shutil.disk_usage(self.settings.projects_dir).free
        estimate = sum(
            image.file_size + len(self._caption_bytes(image.caption_text))
            for image in images
            if not image.errors
        )
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
            quality_warning_image_count=len(warning_images),
            quality_failed_image_count=len(failed_images),
            exact_duplicate_count=len(exact),
            exact_duplicate_nonrepresentative_count=len(exact),
            approximate_duplicate_count=len(approx),
            approximate_duplicate_nonrepresentative_count=sum(
                image.is_similarity_representative is False for image in approx
            ),
            unreviewed_group_count=0,
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
            available_disk_bytes=available,
            estimated_free_bytes=available - estimate,
        )

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
        )

    def _verify_snapshot_files(self, record: Any, root: Path, session: Any) -> None:
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
        if record.manifest_sha256 != _sha256_file(manifest):
            raise ValueError("manifest hash mismatch")
        if record.dataset_toml_sha256 != _sha256_file(toml):
            raise ValueError("dataset TOML hash mismatch")
        items = DatasetRepository(session).list_items(UUID(record.id))
        data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_items = data.get("items")
        if not isinstance(manifest_items, list) or len(manifest_items) != len(items):
            raise ValueError("manifest item count mismatch")
        content_payload: list[tuple[int, str, str]] = []
        for item in items:
            image_path = root / item.snapshot_image_relative_path
            caption_path = root / item.caption_relative_path
            self._ensure_inside(image_path, root)
            self._ensure_inside(caption_path, root)
            if not image_path.is_file() or not caption_path.is_file():
                raise ValueError("snapshot item missing")
            if _sha256_file(image_path) != item.snapshot_image_sha256:
                raise ValueError("snapshot image hash mismatch")
            if _sha256_file(caption_path) != item.caption_sha256:
                raise ValueError("snapshot caption hash mismatch")
            if caption_path.read_bytes() != self._caption_bytes(item.caption_text):
                raise ValueError("snapshot caption content mismatch")
            manifest_item = next(
                (
                    value
                    for value in manifest_items
                    if isinstance(value, dict)
                    and value.get("sequence_number") == item.sequence_number
                ),
                None,
            )
            if not isinstance(manifest_item, dict):
                raise ValueError("manifest item missing")
            if (
                manifest_item.get("snapshot_image_relative_path")
                != item.snapshot_image_relative_path
                or manifest_item.get("caption_relative_path")
                != item.caption_relative_path
                or manifest_item.get("snapshot_image_sha256")
                != item.snapshot_image_sha256
                or manifest_item.get("caption_sha256") != item.caption_sha256
            ):
                raise ValueError("manifest item mismatch")
            content_payload.append(
                (item.sequence_number, item.snapshot_image_sha256, item.caption_sha256)
            )
        content_hash = _sha256_bytes(_json(content_payload).encode("utf-8"))
        if (
            record.content_sha256 != content_hash
            or data.get("content_sha256") != content_hash
        ):
            raise ValueError("snapshot content hash mismatch")

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
    }
