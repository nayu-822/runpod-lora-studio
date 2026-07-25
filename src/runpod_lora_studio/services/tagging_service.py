from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    Project,
    SelectionState,
    TagCategory,
    TaggerInferenceSettings,
    TaggerModelIdentity,
    TaggerRunMode,
    TaggerRunStatus,
    TaggerRunSummary,
    TaggingResultStatus,
    TagPrediction,
)
from runpod_lora_studio.external.tagger import (
    TaggerAdapter,
    TaggerEnvironmentError,
    normalize_tag_name,
)
from runpod_lora_studio.external.wd_tagger import WDTaggerAdapter
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import ImageRepository
from runpod_lora_studio.persistence.tagging_repository import TaggingRepository
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.tagging")


AdapterFactory = Callable[[], TaggerAdapter]


class TaggingService:
    """Manage long-running tagger runs without changing final captions."""

    implementation_version = "phase3-v1"

    def __init__(
        self,
        settings: AppSettings,
        projects: ProjectService | None = None,
        adapter_factory: AdapterFactory | None = None,
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)
        self.adapter_factory = adapter_factory or (lambda: WDTaggerAdapter(settings))
        self._executor = ThreadPoolExecutor(
            max_workers=settings.tagger_max_workers, thread_name_prefix="tagger"
        )
        self._futures: dict[UUID, Future[None]] = {}

    def inference_settings(self, device: str | None = None) -> TaggerInferenceSettings:
        return TaggerInferenceSettings(
            device=device or self.settings.tagger_device,
            batch_size=self.settings.tagger_batch_size,
            general_threshold=self.settings.tagger_general_threshold,
            character_threshold=self.settings.tagger_character_threshold,
            save_rating=self.settings.tagger_save_rating,
            save_character=self.settings.tagger_save_character,
            save_general=self.settings.tagger_save_general,
            underscore_to_space=self.settings.tagger_underscore_to_space,
            escape_mode=self.settings.tagger_escape_mode,
            max_workers=self.settings.tagger_max_workers,
            allow_model_download=self.settings.tagger_allow_model_download,
        )

    def model_identity(self) -> TaggerModelIdentity:
        return self.adapter_factory().model_identity()

    def recover_stale_runs(self, project_id: UUID | None = None) -> int:
        with self.session_factory() as session:
            repository = TaggingRepository(session)
            count = repository.recover_stale_runs(project_id)
            session.commit()
            return count

    def list_runs(self, project_id: UUID) -> list[TaggerRunSummary]:
        with self.session_factory() as session:
            return TaggingRepository(session).list_runs(project_id)

    def get_run(self, run_id: UUID) -> TaggerRunSummary | None:
        with self.session_factory() as session:
            return TaggingRepository(session).get_run(run_id)

    def validate_environment(self) -> str:
        validation = self.adapter_factory().validate_environment()
        if validation.ok:
            return f"利用可能: {validation.resolved_device}"
        raise UserFacingError(validation.message)

    def start_run(
        self, project_id: UUID, mode: TaggerRunMode = TaggerRunMode.UNTAGGED_ONLY
    ) -> TaggerRunSummary:
        project = self.projects.get(project_id)
        with self.session_factory() as session:
            repository = TaggingRepository(session)
            repository.recover_stale_runs(project_id)
            active = repository.active_run(project_id)
            if active is not None:
                raise UserFacingError("このプロジェクトでは既にTaggerが実行中です。")
            identity = self.adapter_factory().model_identity()
            inference = self.inference_settings()
            snapshot = self._settings_snapshot(identity, inference)
            accepted = self._accepted_images(session, project)
            selected, skipped = self._select_images(
                session, project_id, accepted, mode, snapshot
            )
            record = repository.create_run(
                project_id,
                identity,
                inference,
                snapshot,
                len(selected),
            )
            record.skipped_image_count = skipped
            session.commit()
            run_id = UUID(record.id)
        future = self._executor.submit(self._execute, run_id, tuple(selected))
        self._futures[run_id] = future
        result = self.get_run(run_id)
        if result is None:
            raise RuntimeError("created tagger run could not be reloaded")
        return result

    def run_sync(
        self, project_id: UUID, mode: TaggerRunMode = TaggerRunMode.UNTAGGED_ONLY
    ) -> TaggerRunSummary:
        project = self.projects.get(project_id)
        with self.session_factory() as session:
            repository = TaggingRepository(session)
            repository.recover_stale_runs(project_id)
            if repository.active_run(project_id) is not None:
                raise UserFacingError("このプロジェクトでは既にTaggerが実行中です。")
            identity = self.adapter_factory().model_identity()
            inference = self.inference_settings()
            snapshot = self._settings_snapshot(identity, inference)
            accepted = self._accepted_images(session, project)
            selected, skipped = self._select_images(
                session, project_id, accepted, mode, snapshot
            )
            record = repository.create_run(
                project_id, identity, inference, snapshot, len(selected)
            )
            record.skipped_image_count = skipped
            session.commit()
            run_id = UUID(record.id)
        self._execute(run_id, tuple(selected))
        result = self.get_run(run_id)
        if result is None:
            raise RuntimeError("tagger run could not be reloaded")
        return result

    def cancel_run(self, run_id: UUID) -> None:
        with self.session_factory() as session:
            TaggingRepository(session).request_cancel(run_id)
            session.commit()

    def _execute(self, run_id: UUID, images: tuple[PathImage, ...]) -> None:
        adapter = self.adapter_factory()
        try:
            validation = adapter.validate_environment()
            if not validation.ok:
                self._finish(run_id, TaggerRunStatus.FAILED, validation.message)
                return
            with self.session_factory() as session:
                repository = TaggingRepository(session)
                repository.mark_running(run_id, validation.resolved_device)
                session.commit()
            adapter.load()
            self._process_images(run_id, images, adapter, validation.resolved_device)
        except TaggerEnvironmentError as exc:
            self._finish(run_id, TaggerRunStatus.FAILED, str(exc))
        except Exception:
            logger.exception("tagger_run_failed run_id=%s", run_id)
            self._finish(run_id, TaggerRunStatus.FAILED, "Tagger実行に失敗しました。")
        finally:
            try:
                adapter.unload()
            except Exception:
                logger.exception("tagger_unload_failed run_id=%s", run_id)
            self._futures.pop(run_id, None)

    def _process_images(
        self,
        run_id: UUID,
        images: tuple[PathImage, ...],
        adapter: TaggerAdapter,
        device: str,
    ) -> None:
        processed = succeeded = failed = 0
        with self.session_factory() as session:
            run = TaggingRepository(session).get_run_record(run_id)
            if run is None:
                raise ValueError("tagger run not found")
            inference = self.inference_settings(device)
            identity = adapter.model_identity()
        for batch in self._batches(images, inference.batch_size):
            for image in batch:
                with self.session_factory() as session:
                    repository = TaggingRepository(session)
                    if repository.cancel_requested(run_id):
                        repository.update_progress(
                            run_id,
                            processed=processed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=0,
                            current_image_id=None,
                        )
                        repository.finish_run(
                            run_id,
                            TaggerRunStatus.CANCELED,
                            "ユーザーによりキャンセルされました。",
                        )
                        session.commit()
                        return
                    repository.update_progress(
                        run_id,
                        processed=processed,
                        succeeded=succeeded,
                        failed=failed,
                        skipped=0,
                        current_image_id=image.image_id,
                    )
                    session.commit()
                try:
                    result = adapter.tag_image(image.path, inference)
                    predictions = self._sanitize_predictions(result.tags, inference)
                    with self.session_factory() as session:
                        repository = TaggingRepository(session)
                        repository.save_result(
                            run_id,
                            image.image_id,
                            TaggingResultStatus.COMPLETED,
                            predictions,
                            result.raw_output,
                            identity=identity,
                        )
                        processed += 1
                        succeeded += 1
                        repository.update_progress(
                            run_id,
                            processed=processed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=0,
                            current_image_id=None,
                        )
                        session.commit()
                except Exception as exc:
                    summary = self._inference_error_summary(exc)
                    logger.warning(
                        "tagging_image_failed image_id=%s reason=%s",
                        image.image_id,
                        summary,
                    )
                    with self.session_factory() as session:
                        repository = TaggingRepository(session)
                        repository.save_result(
                            run_id,
                            image.image_id,
                            TaggingResultStatus.FAILED,
                            error_summary=summary,
                            identity=identity,
                        )
                        processed += 1
                        failed += 1
                        repository.update_progress(
                            run_id,
                            processed=processed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=0,
                            current_image_id=None,
                        )
                        session.commit()
        status = (
            TaggerRunStatus.PARTIALLY_FAILED if failed else TaggerRunStatus.COMPLETED
        )
        self._finish(
            run_id, status, "一部画像のタグ付けに失敗しました。" if failed else None
        )

    def _finish(
        self, run_id: UUID, status: TaggerRunStatus, error_summary: str | None
    ) -> None:
        with self.session_factory() as session:
            repository = TaggingRepository(session)
            try:
                repository.finish_run(run_id, status, error_summary)
            except ValueError:
                return
            session.commit()

    @staticmethod
    def _accepted_images(session: Session, project: Project) -> list[PathImage]:
        repository = ImageRepository(session)
        images: list[PathImage] = []
        for batch in repository.iter_batches_for_project(project.id, 100):
            for image in batch:
                if image.selection_state is SelectionState.ACCEPTED:
                    images.append(PathImage(image.id, image.original_path))
        return images

    @staticmethod
    def _batches(
        images: tuple[PathImage, ...], batch_size: int
    ) -> list[tuple[PathImage, ...]]:
        size = max(batch_size, 1)
        return [images[start : start + size] for start in range(0, len(images), size)]

    @staticmethod
    def _inference_error_summary(error: Exception) -> str:
        message = str(error).casefold()
        error_name = type(error).__name__.casefold()
        if (
            "out of memory" in message
            or "cuda oom" in message
            or "outofmemory" in error_name
        ):
            return (
                "推論中にGPUメモリ不足が発生しました。"
                "バッチサイズを下げて再実行してください。"
            )
        return "画像のタグ付けに失敗しました。"

    def _select_images(
        self,
        session: Session,
        project_id: UUID,
        accepted: list[PathImage],
        mode: TaggerRunMode,
        snapshot: str,
    ) -> tuple[list[PathImage], int]:
        repository = TaggingRepository(session)
        if mode is TaggerRunMode.ALL_ACCEPTED:
            return accepted, 0
        selected: list[PathImage] = []
        skipped = 0
        for image in accepted:
            if mode is TaggerRunMode.UNTAGGED_ONLY:
                should_select = (
                    repository.completed_result_for_image(
                        image.image_id, project_id, snapshot
                    )
                    is None
                )
            else:
                failed = repository.latest_failed_result(image.image_id, project_id)
                should_select = failed is not None
            if should_select:
                selected.append(image)
            else:
                skipped += 1
        return selected, skipped

    def _settings_snapshot(
        self, identity: TaggerModelIdentity, settings: TaggerInferenceSettings
    ) -> str:
        return json.dumps(
            {
                "identity": {
                    "adapter": identity.adapter_name,
                    "model": identity.model_identifier,
                    "revision": identity.model_revision,
                    "implementation": identity.implementation_version,
                },
                "settings": {
                    "device": settings.device,
                    "batch_size": settings.batch_size,
                    "general_threshold": settings.general_threshold,
                    "character_threshold": settings.character_threshold,
                    "save_rating": settings.save_rating,
                    "save_character": settings.save_character,
                    "save_general": settings.save_general,
                    "underscore_to_space": settings.underscore_to_space,
                    "escape_mode": settings.escape_mode,
                },
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def _sanitize_predictions(
        predictions: tuple[TagPrediction, ...], settings: TaggerInferenceSettings
    ) -> tuple[TagPrediction, ...]:
        result: list[TagPrediction] = []
        seen: set[str] = set()
        for prediction in predictions:
            if prediction.category is TagCategory.RATING and not settings.save_rating:
                continue
            if (
                prediction.category is TagCategory.CHARACTER
                and not settings.save_character
            ):
                continue
            if prediction.category is TagCategory.GENERAL and not settings.save_general:
                continue
            if prediction.confidence is not None:
                threshold = (
                    settings.character_threshold
                    if prediction.category is TagCategory.CHARACTER
                    else settings.general_threshold
                )
                if (
                    prediction.category in {TagCategory.CHARACTER, TagCategory.GENERAL}
                    and prediction.confidence < threshold
                ):
                    continue
            normalized = normalize_tag_name(
                prediction.tag_name_raw, settings.underscore_to_space
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(
                TagPrediction(
                    tag_name_raw=prediction.tag_name_raw.strip(),
                    tag_name_normalized=normalized,
                    category=prediction.category,
                    confidence=prediction.confidence,
                    original_order=len(result),
                    source=prediction.source,
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class PathImage:
    image_id: UUID
    path: Path
