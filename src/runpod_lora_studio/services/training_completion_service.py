from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import DatasetSnapshotStatus
from runpod_lora_studio.domain.storage_models import (
    OverwritePolicy,
    RemoteSnapshotProvenance,
    StorageArtifactFile,
    StorageRemotePath,
    StorageTransferType,
    TransferStatus,
)
from runpod_lora_studio.domain.training_completion_models import (
    TrainingCompletionErrorCode,
    TrainingCompletionExport,
    TrainingCompletionFile,
    TrainingCompletionPreview,
    TrainingCompletionStatus,
)
from runpod_lora_studio.domain.training_models import (
    TRAINING_EXECUTION_SNAPSHOT_SCHEMA_VERSION,
)
from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifactValidationStatus,
)
from runpod_lora_studio.external.rclone import CancelToken, CopyOptions
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotRecord,
    ManagedModelRecord,
    StorageTransferJobRecord,
    TrainingArtifactRecord,
    TrainingCompletionExportRecord,
    TrainingConfigRecord,
    TrainingExecutionSummaryRecord,
    TrainingJobRecord,
)
from runpod_lora_studio.persistence.training_completion_repository import (
    TrainingCompletionRepository,
    export_from_record,
    utc_now,
)
from runpod_lora_studio.services.completion_filesystem import (
    CompletionFilesystemError,
    SafeExportDirectory,
    stable_file_hash,
)
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.services.training_artifact import TrainingArtifactScanner
from runpod_lora_studio.services.training_command import TrainingCommandValidationError
from runpod_lora_studio.services.training_service import TrainingService

logger = logging.getLogger("runpod_lora_studio.training_completion")

EXPORT_SCHEMA_VERSION = "phase9a-training-export-v1"
COMPLETION_MANIFEST_SCHEMA_VERSION = "phase9a-training-completion-v1"
COMPLETION_MANIFEST_NAME = "completion-manifest.json"
EXPORT_MANIFEST_NAME = "export-manifest.json"
MAX_COMPLETION_MANIFEST_BYTES = 1024 * 1024

_COMPLETION_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "project_id",
        "training_job_id",
        "training_config_id",
        "parent_training_job_id",
        "resume_artifact_id",
        "training_status",
        "exit_code",
        "started_at",
        "finished_at",
        "dataset_snapshot_id",
        "dataset_content_hash",
        "dataset_remote_provenance",
        "managed_model_id",
        "managed_model_sha256",
        "safe_config_fingerprint",
        "final_lora_relative_path",
        "final_lora_size",
        "final_lora_sha256",
        "export_files",
        "logs",
        "samples",
        "environment_provenance",
        "performance_provenance",
        "storage_transfer_job_id",
        "source_fingerprint",
        "export_fingerprint",
        "export_manifest_sha256",
        "remote_relative_path",
        "created_at",
    }
)
_DATASET_REMOTE_PROVENANCE_FIELDS = frozenset(
    {
        "snapshot_id",
        "remote_relative_path",
        "storage_transfer_job_id",
        "remote_manifest_sha256",
        "content_sha256",
        "verification_level",
    }
)
_EXECUTION_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "project_id",
        "dataset_snapshot_id",
        "managed_model_id",
        "name",
        "output_name",
        "output_directory",
        "sd_scripts_root",
        "trainer_script",
        "resolution",
        "batch_size",
        "epochs",
        "learning_rate",
        "optimizer",
        "scheduler",
        "network_module",
        "network_dim",
        "network_alpha",
        "mixed_precision",
        "save_every_n_epochs",
        "cache_latents",
        "gradient_checkpointing",
        "seed",
        "extra_options",
        "recommendation_id",
        "recommendation_engine_version",
        "recommendation_change_diff",
    }
)


class CompletionFailure(Exception):
    def __init__(self, code: TrainingCompletionErrorCode | str, summary: str) -> None:
        self.code = str(
            code.value if isinstance(code, TrainingCompletionErrorCode) else code
        )
        self.summary = summary
        super().__init__(summary)


@dataclass(frozen=True, slots=True)
class _ExecutionConfig:
    payload: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload["id"])

    @property
    def project_id(self) -> str:
        return str(self.payload["project_id"])

    @property
    def dataset_snapshot_id(self) -> str:
        return str(self.payload["dataset_snapshot_id"])

    @property
    def managed_model_id(self) -> str:
        return str(self.payload["managed_model_id"])

    @property
    def output_name(self) -> str:
        return str(self.payload["output_name"])


@dataclass(frozen=True, slots=True)
class _CompletionContext:
    job: TrainingJobRecord
    execution_config: _ExecutionConfig
    snapshot: DatasetSnapshotRecord
    model: ManagedModelRecord
    final_lora: TrainingArtifactRecord
    snapshot_remote: RemoteSnapshotProvenance
    target: StorageRemotePath
    files: tuple[TrainingCompletionFile, ...]
    source_fingerprint: str
    preview_fingerprint: str
    export_relative_path: str
    remote_relative_path: str


class TrainingCompletionService:
    """Durable Phase 9A boundary between a succeeded training job and Drive."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        storage_service: StorageService | None = None,
        training_service: TrainingService | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.storage = storage_service or StorageService(settings)
        self.training = training_service or TrainingService(settings)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="training-completion"
        )
        self._futures: dict[UUID, Future[Any]] = {}
        self._cancel_tokens: dict[UUID, CancelToken] = {}
        self._lock = threading.Lock()
        self._clock = clock or time.monotonic

    @property
    def export_root(self) -> Path:
        return Path(self.settings.projects_dir)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def get_export(self, export_id: UUID) -> TrainingCompletionExport:
        with self.session_factory() as session:
            record = TrainingCompletionRepository(session).get(export_id)
            if record is None:
                raise UserFacingError("成果物同期が見つかりません")
            return export_from_record(record)

    def get_export_for_training_job(
        self, training_job_id: UUID
    ) -> TrainingCompletionExport | None:
        with self.session_factory() as session:
            record = TrainingCompletionRepository(session).get_by_training_job(
                training_job_id
            )
            return export_from_record(record) if record else None

    def list_exports(
        self, project_id: UUID | None = None
    ) -> list[TrainingCompletionExport]:
        with self.session_factory() as session:
            return TrainingCompletionRepository(session).list_exports(project_id)

    def preview(self, training_job_id: UUID) -> TrainingCompletionPreview:
        context = self._build_context(training_job_id)
        settings = self.storage.get_project_storage_settings(context.job.project_id)
        return TrainingCompletionPreview(
            token=context.preview_fingerprint,
            source_fingerprint=context.source_fingerprint,
            training_job_id=training_job_id,
            project_id=UUID(context.job.project_id),
            training_config_id=UUID(context.execution_config.id),
            dataset_snapshot_id=UUID(context.snapshot.id),
            managed_model_id=UUID(context.model.id),
            final_lora_filename=context.final_lora.filename,
            final_lora_size_bytes=context.final_lora.file_size,
            final_lora_sha256=str(context.final_lora.sha256),
            local_export_relative_path=context.export_relative_path,
            remote_relative_path=context.remote_relative_path,
            files=context.files,
            dataset_remote_transfer_job_id=context.snapshot_remote.storage_transfer_job_id,
            dataset_remote_manifest_sha256=context.snapshot_remote.remote_manifest_sha256,
            overwrite_policy=settings.overwrite_policy.value,
            verification_policy=settings.verification_policy.value,
        )

    def start(self, training_job_id: UUID, *, preview_token: str | None = None) -> UUID:
        preview = self.preview(training_job_id)
        if preview_token is not None and preview.token != preview_token:
            raise UserFacingError(
                "プレビュー内容が変更されています。再度プレビューしてください"
            )
        with self.session_factory() as session:
            repository = TrainingCompletionRepository(session)
            existing = repository.get_by_training_job(training_job_id)
            if existing is not None:
                status = TrainingCompletionStatus(existing.status)
                if status is TrainingCompletionStatus.COMPLETED:
                    return UUID(existing.id)
                if status in {
                    TrainingCompletionStatus.FAILED,
                    TrainingCompletionStatus.STALE,
                    TrainingCompletionStatus.CANCELED,
                }:
                    repository.reset_for_retry(UUID(existing.id))
                    existing = repository.get(UUID(existing.id))
                if existing is None:
                    raise UserFacingError("成果物同期を再作成できません")
                existing.source_fingerprint = preview.source_fingerprint
                existing.preview_fingerprint = preview.token
                session.commit()
                export_id = UUID(existing.id)
            else:
                try:
                    record = repository.create(
                        training_job_id,
                        preview.project_id,
                        preview.source_fingerprint,
                        preview.token,
                    )
                    session.commit()
                except IntegrityError as exc:
                    session.rollback()
                    current = repository.get_by_training_job(training_job_id)
                    if current is None:
                        raise UserFacingError(
                            "同じ学習jobの成果物同期を作成できません"
                        ) from exc
                    return UUID(current.id)
                export_id = UUID(record.id)
        self._submit(export_id)
        return export_id

    def synchronize_sync(
        self, training_job_id: UUID, *, preview_token: str | None = None
    ) -> UUID:
        export_id = self._prepare_export(training_job_id, preview_token)
        self._run_export(export_id)
        return export_id

    def retry(self, export_id: UUID, *, preview_token: str | None = None) -> UUID:
        export = self.get_export(export_id)
        preview = self.preview(export.training_job_id)
        if preview_token is not None and preview.token != preview_token:
            raise UserFacingError(
                "プレビュー内容が変更されています。再度プレビューしてください"
            )
        with self.session_factory() as session:
            repository = TrainingCompletionRepository(session)
            if not repository.reset_for_retry(export_id):
                current = repository.get(export_id)
                if (
                    current is not None
                    and current.status == TrainingCompletionStatus.COMPLETED.value
                ):
                    return export_id
                raise UserFacingError("失敗・stale・cancel済みの同期だけ再試行できます")
            record = repository.get(export_id)
            if record is None:
                raise UserFacingError("成果物同期が見つかりません")
            record.source_fingerprint = preview.source_fingerprint
            record.preview_fingerprint = preview.token
            session.commit()
        self._submit(export_id)
        return export_id

    def cancel(self, export_id: UUID) -> None:
        with self.session_factory() as session:
            changed = TrainingCompletionRepository(session).request_cancel(export_id)
            session.commit()
        if changed:
            token = self._cancel_tokens.get(export_id)
            if token is not None:
                token.cancel()

    def recover_stale(self) -> int:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.settings.storage_job_stale_after_seconds)
        recovered = 0
        self._cleanup_temporary_exports()
        try:
            with self.session_factory() as session:
                records = [
                    record
                    for record in TrainingCompletionRepository(
                        session
                    ).list_recovery_records(limit=256)
                    if record.status
                    in {
                        TrainingCompletionStatus.PREPARING.value,
                        TrainingCompletionStatus.READY.value,
                        TrainingCompletionStatus.UPLOADING.value,
                        TrainingCompletionStatus.VERIFYING.value,
                    }
                ]
                repository = TrainingCompletionRepository(session)
                for record in records:
                    heartbeat = _utc(record.heartbeat_at)
                    if heartbeat is not None and heartbeat >= cutoff:
                        continue
                    future = self._futures.get(UUID(record.id))
                    if future is not None and not future.done():
                        continue
                    recovered += int(
                        repository.mark_stale_if_unchanged(
                            UUID(record.id),
                            expected_status=record.status,
                            expected_worker_id=record.worker_id,
                            expected_claim_token=record.claim_token,
                            expected_worker_generation=record.worker_generation,
                            heartbeat_cutoff=cutoff,
                            now=now,
                        )
                    )
                session.commit()
        except OperationalError:
            return 0
        return recovered

    def _cleanup_temporary_exports(self, *, max_items: int = 128) -> int:
        root = self.settings.projects_dir
        if root.is_symlink() or not root.is_dir():
            return 0
        removed = 0
        try:
            project_paths = sorted(root.iterdir(), key=lambda value: value.name)
        except OSError:
            return 0
        cutoff = datetime.now(UTC) - timedelta(
            seconds=self.settings.storage_job_stale_after_seconds
        )
        for project_path in project_paths:
            if removed >= max_items:
                break
            if project_path.is_symlink() or not project_path.is_dir():
                continue
            try:
                project_id = UUID(project_path.name)
            except ValueError:
                continue
            try:
                with self.session_factory() as session:
                    records = list(
                        session.scalars(
                            select(TrainingCompletionExportRecord).where(
                                TrainingCompletionExportRecord.project_id
                                == str(project_id)
                            )
                        ).all()
                    )
            except OperationalError:
                return removed
            active_project = False
            for record in records:
                future = self._futures.get(UUID(record.id))
                if future is not None and not future.done():
                    # A live in-process worker owns its temporary directory,
                    # even before it has persisted PREPARING.
                    active_project = True
                    break
                if record.status not in {
                    TrainingCompletionStatus.PREPARING.value,
                    TrainingCompletionStatus.READY.value,
                    TrainingCompletionStatus.UPLOADING.value,
                    TrainingCompletionStatus.VERIFYING.value,
                }:
                    continue
                heartbeat = _utc(record.heartbeat_at)
                if heartbeat is not None and heartbeat >= cutoff:
                    active_project = True
                    break
            if active_project:
                continue
            exports_path = project_path / "training" / "exports"
            if (
                exports_path.is_symlink()
                or not exports_path.exists()
                or not exports_path.is_dir()
            ):
                continue
            try:
                with SafeExportDirectory.open_root(
                    root,
                    (project_path.name, "training", "exports"),
                    create_missing=False,
                ) as exports_root:
                    for candidate in sorted(
                        exports_path.iterdir(), key=lambda value: value.name
                    ):
                        if removed >= max_items:
                            break
                        if not candidate.name.startswith(".creating-"):
                            continue
                        if exports_root.child_exists(candidate.name) != "directory":
                            continue
                        temporary = exports_root.open_child(candidate.name)
                        try:
                            exports_root.remove_child(temporary)
                        except (CompletionFilesystemError, OSError):
                            logger.warning(
                                "training_completion_temp_cleanup_failed path=%s",
                                candidate,
                                exc_info=True,
                            )
                            continue
                        finally:
                            temporary.close()
                        removed += 1
            except (CompletionFilesystemError, OSError, RuntimeError):
                logger.warning(
                    "training_completion_temp_cleanup_failed project=%s",
                    project_id,
                    exc_info=True,
                )
        return removed

    def reconcile_remote(self, *, time_budget_seconds: float = 5.0) -> int:
        """Enqueue bounded post-restart recovery without doing export I/O inline."""
        deadline = self._clock() + max(0.0, time_budget_seconds)
        enqueued = 0
        try:
            with self.session_factory() as session:
                ids = [
                    UUID(record.id)
                    for record in TrainingCompletionRepository(
                        session
                    ).list_recovery_records(limit=256)
                    if record.status == TrainingCompletionStatus.STALE.value
                ]
        except OperationalError:
            # The UI can be imported before the Phase 9A migration is applied.
            return 0
        for export_id in ids:
            if self._clock() >= deadline:
                break
            enqueued += int(self._submit(export_id))
        return enqueued

    def status_rows(self, project_id: UUID | None = None) -> list[list[str]]:
        try:
            exports = self.list_exports(project_id)
        except OperationalError:
            return []
        return [
            [
                str(export.id),
                str(export.training_job_id),
                export.status.value,
                export.current_stage,
                str(export.storage_transfer_job_id or ""),
                str(export.remote_relative_path or ""),
                str(export.export_manifest_sha256 or ""),
                str(export.completion_manifest_sha256 or ""),
                export.error_code or "",
                export.error_summary or "",
                export.completed_at.isoformat() if export.completed_at else "",
            ]
            for export in exports
        ]

    def _submit(self, export_id: UUID) -> bool:
        with self._lock:
            future = self._futures.get(export_id)
            if future is not None and not future.done():
                return False
            self._futures[export_id] = self._executor.submit(
                self._run_export, export_id
            )
            return True

    def _prepare_export(self, training_job_id: UUID, preview_token: str | None) -> UUID:
        preview = self.preview(training_job_id)
        if preview_token is not None and preview.token != preview_token:
            raise UserFacingError(
                "プレビュー内容が変更されています。再度プレビューしてください"
            )
        with self.session_factory() as session:
            repository = TrainingCompletionRepository(session)
            record = repository.get_by_training_job(training_job_id)
            if record is None:
                record = repository.create(
                    training_job_id,
                    preview.project_id,
                    preview.source_fingerprint,
                    preview.token,
                )
                session.commit()
            elif record.status == TrainingCompletionStatus.COMPLETED.value:
                return UUID(record.id)
            else:
                record.source_fingerprint = preview.source_fingerprint
                record.preview_fingerprint = preview.token
                session.commit()
            return UUID(record.id)

    def _run_export(self, export_id: UUID) -> None:
        worker_id = f"{os.getpid()}:{uuid4().hex}"
        claim_token = uuid4().hex
        worker_generation: int | None = None
        token: CancelToken | None = None
        try:
            with self.session_factory() as session:
                repository = TrainingCompletionRepository(session)
                if not repository.claim(
                    export_id,
                    worker_id=worker_id,
                    claim_token=claim_token,
                    stale_after_seconds=self.settings.storage_job_stale_after_seconds,
                ):
                    return
                session.commit()
                record = repository.get(export_id)
                if record is None or record.worker_id != worker_id:
                    return
                worker_generation = record.worker_generation
            token = CancelToken()
            self._cancel_tokens[export_id] = token
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(
                    export_id,
                    worker_id,
                    claim_token,
                    worker_generation,
                    token,
                    heartbeat_stop,
                ),
                name=f"training-completion-heartbeat-{export_id}",
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                self._run_claimed(
                    export_id,
                    worker_id,
                    claim_token,
                    worker_generation,
                    token,
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=2.0)
        except CompletionFailure as exc:
            self._finish_claimed(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                TrainingCompletionStatus.CANCELED
                if exc.code == TrainingCompletionErrorCode.CANCELED.value
                else TrainingCompletionStatus.FAILED,
                exc.code,
                exc.summary,
            )
        except UserFacingError as exc:
            canceled = self._is_cancel_requested(export_id) or bool(
                token is not None and token.cancelled
            )
            self._finish_claimed(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                TrainingCompletionStatus.CANCELED
                if canceled
                else TrainingCompletionStatus.FAILED,
                TrainingCompletionErrorCode.CANCELED.value
                if canceled
                else TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED.value,
                "completion exportのキャンセルです" if canceled else str(exc)[:500],
            )
        except Exception:
            logger.exception(
                "training_completion_worker_failed export_id=%s", export_id
            )
            self._finish_claimed(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                TrainingCompletionStatus.FAILED,
                TrainingCompletionErrorCode.UNKNOWN.value,
                "成果物同期workerでエラーが発生しました",
            )
        finally:
            self._cancel_tokens.pop(export_id, None)

    def _heartbeat_loop(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
        cancel_token: CancelToken,
        stop_event: threading.Event,
    ) -> None:
        interval = max(
            1.0,
            min(30.0, self.settings.storage_job_stale_after_seconds / 3.0),
        )
        while not stop_event.wait(interval):
            try:
                with self.session_factory() as session:
                    repository = TrainingCompletionRepository(session)
                    if not repository.update_claimed(
                        export_id,
                        worker_id=worker_id,
                        claim_token=claim_token,
                        worker_generation=worker_generation,
                        values={"heartbeat_at": utc_now()},
                    ):
                        cancel_token.cancel()
                        return
                    session.commit()
            except (OperationalError, OSError):
                logger.warning(
                    "training_completion_heartbeat_failed export_id=%s",
                    export_id,
                    exc_info=True,
                )

    def _run_claimed(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
        cancel_token: CancelToken,
    ) -> None:
        export = self.get_export(export_id)
        context = self._build_context(export.training_job_id)
        if export.source_fingerprint != context.source_fingerprint:
            raise CompletionFailure(
                TrainingCompletionErrorCode.PREVIEW_STALE,
                "学習入力またはremote状態がプレビュー後に変更されました",
            )
        self._check_cancel(export_id, cancel_token)
        final_root = self._build_local_export(context, cancel_token)
        export_manifest_path, export_manifest_sha256, export_fingerprint = (
            self._ensure_export_manifest(final_root, context)
        )
        self._claimed_update(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            status=TrainingCompletionStatus.READY.value,
            current_stage="local_export_ready",
            export_relative_path=context.export_relative_path,
            export_manifest_relative_path=f"{context.export_relative_path}/{EXPORT_MANIFEST_NAME}",
            export_manifest_sha256=export_manifest_sha256,
            export_fingerprint=export_fingerprint,
            remote_relative_path=context.remote_relative_path,
            heartbeat_at=utc_now(),
        )
        self._check_cancel(export_id, cancel_token)
        files = self._upload_files(final_root)
        self._check_claim(export_id, worker_id, claim_token, worker_generation)
        remote_marker, remote_marker_hash = self._read_remote_completion_with_hash(
            context.target
        )
        if remote_marker is not None:
            self._check_cancel(export_id, cancel_token)
            if (
                export.completion_manifest_sha256 is not None
                and remote_marker_hash != export.completion_manifest_sha256
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "remote completion manifestのhashが保存済み値と一致しません",
                )
            existing_storage_job_id = self._verify_existing_remote_completion(
                context,
                files,
                remote_marker,
                export_id,
                export_manifest_sha256=export_manifest_sha256,
                expected_export_fingerprint=export_fingerprint,
            )
            if remote_marker_hash is None:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "remote completion manifestのhashがありません",
                )
            self._complete_claimed(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                existing_storage_job_id,
                remote_marker_hash,
                context,
                export_fingerprint,
            )
            return
        settings = self.storage.get_project_storage_settings(
            UUID(context.job.project_id)
        )
        storage_job_id = self._find_matching_artifact_job_id(
            context,
            files,
            preferred_job_id=export.storage_transfer_job_id,
        )
        if storage_job_id is None:
            plan = self.storage.dry_run_artifact_upload(
                files, context.target, overwrite_policy=settings.overwrite_policy
            )
            if plan.errors:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "成果物remoteに衝突があります",
                )
            self._claimed_update(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                status=TrainingCompletionStatus.UPLOADING.value,
                current_stage="artifact_upload",
                heartbeat_at=utc_now(),
            )
            storage_job_id = self.storage.upload_artifact_files(
                project_id=UUID(context.job.project_id),
                training_run_id=UUID(context.job.id),
                files=files,
                target=context.target,
                plan_token=plan.token,
                overwrite_policy=settings.overwrite_policy,
                verification_policy=settings.verification_policy,
                cancel_token=cancel_token,
            )
        else:
            self._claimed_update(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                current_stage="artifact_upload_reused",
                heartbeat_at=utc_now(),
            )
        self._claimed_update(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            storage_transfer_job_id=str(storage_job_id),
            current_stage="artifact_upload_complete",
            heartbeat_at=utc_now(),
        )
        self._check_cancel(export_id, cancel_token)
        self._claimed_update(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            status=TrainingCompletionStatus.VERIFYING.value,
            current_stage="remote_artifact_verify",
            heartbeat_at=utc_now(),
        )
        self._check_claim(export_id, worker_id, claim_token, worker_generation)
        self.storage.verify_remote_artifact_files(
            context.target,
            files,
            settings.verification_policy,
            expected_transfer_job_id=storage_job_id,
            expected_project_id=UUID(context.job.project_id),
            expected_training_run_id=UUID(context.job.id),
        )
        self._check_cancel(export_id, cancel_token)
        current_context = self._build_context(context.job.id)
        if current_context.source_fingerprint != context.source_fingerprint:
            raise CompletionFailure(
                TrainingCompletionErrorCode.PREVIEW_STALE,
                "remote datasetまたは学習入力が検証中に変更されました",
            )
        marker_path, marker_hash = self._write_completion_manifest(
            final_root,
            current_context,
            export_id,
            storage_job_id,
            export_fingerprint,
            export_manifest_path,
        )
        self._check_cancel(export_id, cancel_token)
        self._claimed_update(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            remote_completion_manifest_relative_path="completion-manifest.json",
            completion_manifest_sha256=marker_hash,
            current_stage="completion_manifest_upload",
            heartbeat_at=utc_now(),
        )
        self._check_cancel(export_id, cancel_token)
        self._check_claim(export_id, worker_id, claim_token, worker_generation)
        result = self.storage.adapter.copy(
            marker_path,
            context.target.child(COMPLETION_MANIFEST_NAME),
            CopyOptions(
                overwrite_policy=OverwritePolicy.FAIL_IF_EXISTS,
                checksum=True,
            ),
            cancel_token=cancel_token,
        )
        if result.returncode != 0:
            remote_marker, remote_marker_hash = self._read_remote_completion_with_hash(
                context.target
            )
            if remote_marker is None:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                    "completion manifestのremote uploadに失敗しました",
                )
        else:
            remote_marker, remote_marker_hash = self._read_remote_completion_with_hash(
                context.target
            )
        self._check_cancel(export_id, cancel_token)
        if remote_marker_hash is None:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestがありません",
            )
        remote_hash = remote_marker_hash
        if remote_hash != marker_hash:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestのSHA-256が一致しません",
            )
        if remote_marker is None:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestの形式が不正です",
            )
        self._validate_completion_marker(
            remote_marker,
            current_context,
            export_fingerprint,
            files=files,
            export_manifest_sha256=export_manifest_sha256,
            expected_storage_job_id=storage_job_id,
            error_code=TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
        )
        verified_job_id = self._verify_existing_remote_completion(
            current_context,
            files,
            remote_marker,
            export_id,
            export_manifest_sha256=export_manifest_sha256,
            expected_export_fingerprint=export_fingerprint,
        )
        if verified_job_id != storage_job_id:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "completion manifestの転送jobが一致しません",
            )
        self._complete_claimed(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            storage_job_id,
            remote_hash,
            current_context,
            export_fingerprint,
        )

    def _build_context(self, training_job_id: UUID) -> _CompletionContext:
        with self.session_factory() as session:
            job = session.scalar(
                select(TrainingJobRecord).where(
                    TrainingJobRecord.id == str(training_job_id)
                )
            )
            if job is None:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                    "学習jobが見つかりません",
                )
            if job.status != "succeeded" or job.exit_code != 0:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.TRAINING_NOT_SUCCEEDED,
                    "exit code 0で正常終了した学習jobだけ同期できます",
                )
            if job.pid is not None and self.training.process_adapter.is_running(
                job.pid
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.TRAINING_PROCESS_ACTIVE,
                    "学習プロセスがまだ実行中です",
                )
            config = session.scalar(
                select(TrainingConfigRecord).where(
                    TrainingConfigRecord.id == job.training_config_id
                )
            )
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.id == job.dataset_snapshot_id
                )
            )
            model = session.scalar(
                select(ManagedModelRecord).where(
                    ManagedModelRecord.id == job.managed_model_id
                )
            )
            if config is None or snapshot is None or model is None:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                    "学習設定、dataset snapshot、modelを取得できません",
                )
            if (
                config.id != job.training_config_id
                or config.project_id != job.project_id
                or config.dataset_snapshot_id != job.dataset_snapshot_id
                or config.managed_model_id != job.managed_model_id
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                    "学習設定の関連先が学習jobと一致しません",
                )
        execution_config = self._parse_execution_config(job, config)
        self._validate_runtime_config_snapshot(job, config, execution_config)
        if snapshot.status != DatasetSnapshotStatus.COMPLETED.value:
            raise CompletionFailure(
                TrainingCompletionErrorCode.DATASET_NOT_COMPLETED,
                "completed dataset snapshotだけ同期できます",
            )
        try:
            snapshot_remote = self.storage.verify_remote_snapshot(UUID(snapshot.id))
        except (
            UserFacingError,
            OSError,
            OperationalError,
            RuntimeError,
            ValueError,
        ) as exc:
            del exc
            raise CompletionFailure(
                TrainingCompletionErrorCode.DATASET_REMOTE_NOT_VERIFIED,
                "remote dataset snapshotの再検証に失敗しました",
            ) from None
        if (
            snapshot_remote.snapshot_id != UUID(snapshot.id)
            or snapshot_remote.content_sha256 != snapshot.content_sha256
            or not snapshot_remote.remote_relative_path
            or snapshot_remote.verification_level
            in {"not_verified", "verification_failed"}
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.DATASET_REMOTE_NOT_VERIFIED,
                "remote dataset snapshot provenance is incomplete",
            )
        self._validate_model(model)
        final_lora = self._find_final_lora(job, execution_config)
        target = self.storage.training_remote_path(
            UUID(job.project_id), training_job_id
        )
        export_relative = f"projects/{job.project_id}/training/exports/{job.id}"
        files = self._preview_files(job, final_lora)
        storage_settings = self.storage.get_project_storage_settings(
            UUID(job.project_id)
        )
        remote_state = self._remote_state(target)
        source_fingerprint = _fingerprint(
            {
                "schema_version": COMPLETION_MANIFEST_SCHEMA_VERSION,
                "project_id": job.project_id,
                "training_job_id": job.id,
                "training_config_id": execution_config.id,
                "config_snapshot_sha256": _execution_config_fingerprint(
                    execution_config
                ),
                "training_status": job.status,
                "exit_code": job.exit_code,
                "final_lora_filename": final_lora.filename,
                "final_lora_size": final_lora.file_size,
                "final_lora_sha256": final_lora.sha256,
                "dataset_snapshot_id": snapshot.id,
                "dataset_content_sha256": snapshot.content_sha256,
                "dataset_manifest_sha256": snapshot.manifest_sha256,
                "dataset_remote_transfer_job_id": str(
                    snapshot_remote.storage_transfer_job_id
                ),
                "dataset_remote_manifest_sha256": (
                    snapshot_remote.remote_manifest_sha256
                ),
                "managed_model_id": model.id,
                "managed_model_sha256": model.local_sha256,
                "parent_training_job_id": job.parent_job_id,
                "resume_artifact_id": job.resume_artifact_id,
                "destination_remote_path": target.relative_path,
                "overwrite_policy": storage_settings.overwrite_policy.value,
                "verification_policy": storage_settings.verification_policy.value,
                "export_schema_version": EXPORT_SCHEMA_VERSION,
            }
        )
        preview_fingerprint = _fingerprint(
            {
                "source_fingerprint": source_fingerprint,
                "remote_state": remote_state,
            }
        )
        return _CompletionContext(
            job=job,
            execution_config=execution_config,
            snapshot=snapshot,
            model=model,
            final_lora=final_lora,
            snapshot_remote=snapshot_remote,
            target=target,
            files=files,
            source_fingerprint=source_fingerprint,
            preview_fingerprint=preview_fingerprint,
            export_relative_path=export_relative,
            remote_relative_path=target.relative_path,
        )

    def _parse_execution_config(
        self,
        job: TrainingJobRecord,
        config: TrainingConfigRecord,
        *,
        raw_json: str | None = None,
    ) -> _ExecutionConfig:
        raw = job.config_snapshot if raw_json is None else raw_json
        try:
            raw_size = len(raw.encode("utf-8")) if isinstance(raw, str) else -1
        except UnicodeError:
            raw_size = -1
        if raw_size < 0 or raw_size > MAX_COMPLETION_MANIFEST_BYTES:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "immutable execution config snapshotがありません",
            )
        try:
            decoded = json.loads(
                raw,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "immutable execution config snapshotのJSONが不正です",
            ) from None
        if not isinstance(decoded, dict) or set(decoded) != _EXECUTION_CONFIG_FIELDS:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "immutable execution config snapshotのfieldが不正です",
            )
        payload = cast(dict[str, Any], decoded)

        def fail() -> NoReturn:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "immutable execution config snapshotの値が不正です",
            )

        if payload["schema_version"] != TRAINING_EXECUTION_SNAPSHOT_SCHEMA_VERSION:
            fail()
        for key, expected in (
            ("id", config.id),
            ("project_id", job.project_id),
            ("dataset_snapshot_id", job.dataset_snapshot_id),
            ("managed_model_id", job.managed_model_id),
        ):
            if not isinstance(payload[key], str) or payload[key] != expected:
                fail()
            try:
                UUID(payload[key])
            except (TypeError, ValueError):
                fail()
        string_fields = (
            "name",
            "output_name",
            "output_directory",
            "sd_scripts_root",
            "trainer_script",
            "optimizer",
            "scheduler",
            "network_module",
            "mixed_precision",
        )
        if any(
            not isinstance(payload[key], str) or not payload[key].strip()
            for key in string_fields
        ):
            fail()
        if (
            not Path(payload["output_directory"]).is_absolute()
            or not Path(payload["sd_scripts_root"]).is_absolute()
        ):
            fail()
        if (
            Path(payload["output_name"]).name != payload["output_name"]
            or "/" in payload["output_name"]
            or "\\" in payload["output_name"]
        ):
            fail()
        integer_fields = (
            "resolution",
            "batch_size",
            "epochs",
            "network_dim",
            "network_alpha",
            "save_every_n_epochs",
            "seed",
        )
        if any(
            not isinstance(payload[key], int) or isinstance(payload[key], bool)
            for key in integer_fields
        ):
            fail()
        if (
            not isinstance(payload["learning_rate"], (int, float))
            or isinstance(payload["learning_rate"], bool)
            or not math.isfinite(float(payload["learning_rate"]))
        ):
            fail()
        if any(
            not isinstance(payload[key], bool)
            for key in ("cache_latents", "gradient_checkpointing")
        ):
            fail()
        if not isinstance(payload["extra_options"], dict) or any(
            not isinstance(key, str) for key in payload["extra_options"]
        ):
            fail()
        if not isinstance(payload["recommendation_change_diff"], dict):
            fail()
        if payload["recommendation_id"] is not None and not isinstance(
            payload["recommendation_id"], str
        ):
            fail()
        if payload["recommendation_id"] is not None:
            try:
                UUID(payload["recommendation_id"])
            except (TypeError, ValueError):
                fail()
        if payload["recommendation_engine_version"] is not None and not isinstance(
            payload["recommendation_engine_version"], str
        ):
            fail()
        if payload["recommendation_engine_version"] == "":
            fail()
        if not (
            64 <= payload["resolution"] <= 8192
            and payload["resolution"] % 8 == 0
            and 1 <= payload["batch_size"] <= 64
            and 1 <= payload["epochs"] <= 100000
            and 0 < payload["learning_rate"] <= 10
            and payload["network_dim"] > 0
            and payload["network_alpha"] > 0
            and payload["save_every_n_epochs"] > 0
        ):
            fail()
        command_builder = getattr(self.training, "command_builder", None)
        if command_builder is None:
            fail()
        try:
            if payload["trainer_script"] not in command_builder.allowed_trainer_scripts:
                fail()
            if payload["optimizer"] not in command_builder.allowed_optimizers:
                fail()
            if payload["scheduler"] not in command_builder.allowed_schedulers:
                fail()
            if payload["network_module"] not in command_builder.allowed_network_modules:
                fail()
            if (
                payload["mixed_precision"]
                not in command_builder.allowed_mixed_precision
            ):
                fail()
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in payload["extra_options"].values()
            ):
                fail()
            command_builder._extra_arguments(payload["extra_options"])
        except (AttributeError, TrainingCommandValidationError, TypeError, ValueError):
            fail()
        return _ExecutionConfig(payload)

    def _validate_runtime_config_snapshot(
        self,
        job: TrainingJobRecord,
        config: TrainingConfigRecord,
        execution: _ExecutionConfig,
    ) -> None:
        if not job.runtime_directory:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "training runtime directoryがありません",
            )
        path = Path(job.runtime_directory) / "config" / "training-config.json"
        if path.is_symlink() or not path.is_file():
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "runtime training config snapshotがありません",
            )
        try:
            if path.stat().st_size > MAX_COMPLETION_MANIFEST_BYTES:
                raise ValueError("runtime training config is too large")
            runtime_execution = self._parse_execution_config(
                job,
                config,
                raw_json=path.read_text(encoding="utf-8"),
            )
        except CompletionFailure:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "runtime training config snapshotを読み取れません",
            ) from exc
        if runtime_execution.payload != execution.payload:
            raise CompletionFailure(
                TrainingCompletionErrorCode.TRAINING_CONFIG_SNAPSHOT_MISMATCH,
                "runtime training config snapshotがimmutable job snapshotと"
                "一致しません",
            )

    def _find_final_lora(
        self, job: TrainingJobRecord, config: _ExecutionConfig
    ) -> TrainingArtifactRecord:
        if not job.runtime_directory:
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_MISSING,
                "学習runtime directoryがありません",
            )
        if (
            not config.output_name
            or Path(config.output_name).name != config.output_name
            or "/" in config.output_name
            or "\\" in config.output_name
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_INVALID,
                "output name must be a regular file name",
            )
        raw_runtime = Path(job.runtime_directory)
        if raw_runtime.is_symlink() or not raw_runtime.is_dir():
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_MISSING,
                "training runtime directory is missing",
            )
        runtime = raw_runtime.resolve()
        output = runtime / "output"
        if output.is_symlink() or not output.is_dir():
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_MISSING,
                "training output directory is missing",
            )
        exact_path = output / f"{config.output_name}.safetensors"
        if exact_path.is_symlink() or (
            exact_path.exists() and not exact_path.is_file()
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_INVALID,
                "final LoRAがregular fileではありません",
            )
        if not exact_path.is_file():
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_MISSING,
                "exact final LoRAがありません。checkpointを代用しません",
            )
        scanner = TrainingArtifactScanner(
            output,
            max_depth=self.settings.training_artifact_max_depth,
            max_count=self.settings.training_artifact_max_count,
            max_file_size=self.settings.training_artifact_max_file_size_bytes,
        )
        discovered = next(
            (
                item
                for item in scanner.scan(config.output_name)
                if item.filename == exact_path.name
            ),
            None,
        )
        if discovered is None or discovered.relative_path != Path(exact_path.name):
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_INVALID,
                "exact final LoRAを検証できません",
            )
        if discovered.validation_status is TrainingArtifactValidationStatus.CHANGING:
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_CHANGING,
                "final LoRAが検証中に変更されました",
            )
        if discovered.validation_status is not TrainingArtifactValidationStatus.VALID:
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_INVALID,
                "final LoRAのsafetensors検証に失敗しました",
            )
        if not discovered.sha256:
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_INVALID,
                "final LoRAのSHA-256を取得できません",
            )
        with self.session_factory() as session:
            record = cast(
                TrainingArtifactRecord | None,
                session.scalar(
                    select(TrainingArtifactRecord).where(
                        TrainingArtifactRecord.training_job_id == job.id,
                        TrainingArtifactRecord.relative_path
                        == str(discovered.relative_path),
                    )
                ),
            )
            if record is None:
                record = TrainingArtifactRecord(
                    id=str(uuid4()),
                    training_job_id=job.id,
                    artifact_type="lora_checkpoint",
                    relative_path=discovered.relative_path.as_posix(),
                    filename=discovered.filename,
                    epoch=discovered.epoch,
                    step=discovered.step,
                    file_size=discovered.file_size,
                    sha256=discovered.sha256,
                    modified_at=discovered.modified_at,
                    validation_status=discovered.validation_status.value,
                    validation_code=discovered.validation_code,
                    validation_message=discovered.validation_message,
                    metadata_json=json.dumps(discovered.metadata or {}, sort_keys=True),
                    discovered_at=utc_now(),
                    last_verified_at=utc_now(),
                )
                session.add(record)
                session.commit()
            else:
                record.file_size = discovered.file_size
                record.sha256 = discovered.sha256
                record.validation_status = discovered.validation_status.value
                record.validation_code = discovered.validation_code
                record.validation_message = discovered.validation_message
                record.last_verified_at = utc_now()
                session.commit()
            return record

    def _validate_model(self, model: ManagedModelRecord) -> None:
        if (
            model.status != "available"
            or not model.local_path
            or not model.local_sha256
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.MODEL_NOT_VERIFIED,
                "検証済みローカルmodelだけ同期できます",
            )
        raw_path = Path(model.local_path)
        if raw_path.is_symlink() or not raw_path.is_file():
            raise CompletionFailure(
                TrainingCompletionErrorCode.MODEL_NOT_VERIFIED,
                "local model is not a regular file",
            )
        path = raw_path.resolve()
        if path.stat().st_size <= 0:
            raise CompletionFailure(
                TrainingCompletionErrorCode.MODEL_NOT_VERIFIED,
                "検証済みローカルmodelがありません",
            )
        digest = _stable_file_hash(path)
        if digest != model.local_sha256:
            raise CompletionFailure(
                TrainingCompletionErrorCode.MODEL_NOT_VERIFIED,
                "model SHA-256が一致しません",
            )

    def _preview_files(
        self,
        job: TrainingJobRecord,
        final_lora: TrainingArtifactRecord,
    ) -> tuple[TrainingCompletionFile, ...]:
        if not job.runtime_directory:
            return ()
        raw_runtime = Path(job.runtime_directory)
        if raw_runtime.is_symlink() or not raw_runtime.is_dir():
            raise CompletionFailure(
                TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                "training runtime directory is missing",
            )
        runtime = raw_runtime.resolve()
        result: list[TrainingCompletionFile] = []
        final_source = runtime / "output" / final_lora.filename
        final_descriptor = self._file_descriptor(
            final_source, f"artifacts/{final_lora.filename}", "lora"
        )
        if (
            final_descriptor.size_bytes != final_lora.file_size
            or final_descriptor.sha256 != final_lora.sha256
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.FINAL_LORA_CHANGING,
                "final LoRA changed during validation",
            )
        result.append(final_descriptor)
        for source, relative in (
            (
                Path(job.stdout_log_path) if job.stdout_log_path else None,
                "logs/stdout.log",
            ),
            (
                Path(job.stderr_log_path) if job.stderr_log_path else None,
                "logs/stderr.log",
            ),
        ):
            if (
                source is None
                or source.is_symlink()
                or not source.is_file()
                or not _is_relative_to(source.resolve(), runtime)
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                    "学習stdout/stderr logがありません",
                )
            result.append(self._file_descriptor(source, relative, "log"))
        return tuple(result)

    def _build_local_export(
        self, context: _CompletionContext, cancel_token: CancelToken
    ) -> Path:
        projects_root = self.settings.projects_dir
        final_name = str(context.job.id)
        try:
            with SafeExportDirectory.open_root(
                projects_root,
                (str(context.job.project_id), "training", "exports"),
            ) as exports_root:
                final_state = exports_root.child_exists(final_name)
                if final_state != "missing":
                    if final_state != "directory":
                        raise CompletionFilesystemError(
                            "final export is not a directory"
                        )
                    with exports_root.open_child(final_name) as final_directory:
                        for marker_relative in (
                            EXPORT_MANIFEST_NAME,
                            "provenance/source-fingerprint.json",
                        ):
                            try:
                                payload = json.loads(
                                    final_directory.read_bytes(
                                        marker_relative,
                                        max_bytes=MAX_COMPLETION_MANIFEST_BYTES,
                                    )
                                )
                                if (
                                    isinstance(payload, dict)
                                    and payload.get("source_fingerprint")
                                    == context.source_fingerprint
                                ):
                                    final_directory.regular_files()
                                    return final_directory.path
                            except (
                                CompletionFilesystemError,
                                OSError,
                                TypeError,
                                ValueError,
                                json.JSONDecodeError,
                            ):
                                continue
                    raise CompletionFailure(
                        TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                        "同じ学習jobのlocal exportが異なる内容です",
                    )
                temporary: SafeExportDirectory | None = None
                renamed = False
                try:
                    temporary = exports_root.open_child(
                        f".creating-{uuid4().hex}", create=True
                    )
                    for item in context.files:
                        self._check_cancel_for_token(cancel_token)
                        temporary.copy_file(
                            item.source_path,
                            item.relative_path,
                            cancel=lambda: self._check_cancel_for_token(cancel_token),
                        )
                    self._write_generated_files(temporary, context)
                    temporary.fsync_tree()
                    final_directory = exports_root.rename_child(temporary, final_name)
                    renamed = True
                    try:
                        final_directory.regular_files()
                    finally:
                        final_directory.close()
                finally:
                    if temporary is not None and not renamed:
                        try:
                            exports_root.remove_child(temporary)
                        except (CompletionFilesystemError, OSError):
                            logger.warning(
                                "training_completion_temp_cleanup_failed path=%s",
                                temporary.path,
                                exc_info=True,
                            )
                        finally:
                            temporary.close()
        except CompletionFailure:
            raise
        except (CompletionFilesystemError, OSError, ValueError, RuntimeError) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local exportの構築に失敗しました",
            ) from exc
        return (
            Path(projects_root)
            / str(context.job.project_id)
            / "training"
            / "exports"
            / final_name
        )

    @staticmethod
    def _check_cancel_for_token(cancel_token: CancelToken) -> None:
        if cancel_token.cancelled:
            raise CompletionFailure(
                TrainingCompletionErrorCode.CANCELED,
                "completion exportをキャンセルしました",
            )

    def _write_generated_files(
        self, root: SafeExportDirectory, context: _CompletionContext
    ) -> None:
        root.write_json("provenance/dataset.json", self._dataset_provenance(context))
        root.write_json("provenance/model.json", self._model_provenance(context))
        root.write_json("provenance/runtime.json", self._runtime_provenance(context))
        root.write_json("provenance/resume.json", self._resume_provenance(context))
        performance = self._performance_provenance(context.job.id)
        if performance is not None:
            root.write_json("provenance/performance.json", performance)
        root.write_json(
            "config/training-config.json",
            self._safe_config(context.execution_config),
        )
        static_files = root.regular_files()
        root.write_json(
            "hashes/sha256.json",
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "files": [
                    {"relative_path": path, "size": size, "sha256": digest}
                    for path, size, digest in static_files
                ],
            },
        )
        root.write_json(
            "provenance/source-fingerprint.json",
            {
                "schema_version": "phase9a-source-fingerprint-v1",
                "source_fingerprint": context.source_fingerprint,
            },
        )

    def _ensure_export_manifest(
        self, root: Path, context: _CompletionContext
    ) -> tuple[Path, str, str]:
        path = root / EXPORT_MANIFEST_NAME
        try:
            with self._open_final_export_directory(context) as safe_root:
                manifest_state = safe_root.child_exists(EXPORT_MANIFEST_NAME)
                if manifest_state == "symlink" or manifest_state == "special":
                    raise CompletionFilesystemError("export manifest is not regular")
                if manifest_state == "file":
                    payload = json.loads(
                        safe_root.read_bytes(
                            EXPORT_MANIFEST_NAME,
                            max_bytes=MAX_COMPLETION_MANIFEST_BYTES,
                        )
                    )
                    if not isinstance(payload, dict):
                        raise ValueError("export manifest must be an object")
                    if payload.get("source_fingerprint") != context.source_fingerprint:
                        raise CompletionFailure(
                            TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                            "local export manifestの入力fingerprintが一致しません",
                        )
                    manifest_files = payload["files"]
                    actual_files = [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in safe_root.regular_files()
                        if relative
                        not in {EXPORT_MANIFEST_NAME, COMPLETION_MANIFEST_NAME}
                    ]
                    actual_by_path = {
                        str(item["relative_path"]): item for item in actual_files
                    }
                    if any(
                        actual_by_path.get(item.relative_path)
                        != {
                            "relative_path": item.relative_path,
                            "size": item.size_bytes,
                            "sha256": item.sha256,
                        }
                        for item in context.files
                    ):
                        raise ValueError("required export input is missing or changed")
                    if (
                        payload.get("schema_version") != EXPORT_SCHEMA_VERSION
                        or payload.get("export_relative_path")
                        != context.export_relative_path
                        or manifest_files != actual_files
                        or payload.get("export_fingerprint")
                        != _fingerprint(actual_files)
                    ):
                        raise ValueError("local export manifest does not match files")
                    size, digest = safe_root.hash_file(EXPORT_MANIFEST_NAME)
                    del size
                    return path, digest, str(payload.get("export_fingerprint", ""))
                if manifest_state != "missing":
                    raise CompletionFilesystemError("export manifest is not regular")
                files = [
                    item
                    for item in safe_root.regular_files()
                    if item[0] != COMPLETION_MANIFEST_NAME
                ]
                source_files_by_path: dict[str, tuple[int, str]] = {
                    relative: (size, digest) for relative, size, digest in files
                }
                if any(
                    source_files_by_path.get(item.relative_path)
                    != (item.size_bytes, item.sha256)
                    for item in context.files
                ):
                    raise CompletionFailure(
                        TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                        "required export input is missing or changed",
                    )
                export_fingerprint = _fingerprint(
                    [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in files
                    ]
                )
                payload = {
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "source_fingerprint": context.source_fingerprint,
                    "export_relative_path": context.export_relative_path,
                    "export_fingerprint": export_fingerprint,
                    "files": [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in files
                    ],
                    "created_at": utc_now().isoformat(),
                }
                safe_root.write_json(EXPORT_MANIFEST_NAME, payload)
                _size, digest = safe_root.hash_file(EXPORT_MANIFEST_NAME)
                return path, digest, export_fingerprint
        except CompletionFailure:
            raise
        except (
            CompletionFilesystemError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local export manifest is invalid",
            ) from exc

    def _open_final_export_directory(
        self, context: _CompletionContext
    ) -> SafeExportDirectory:
        return SafeExportDirectory.open_root(
            self.settings.projects_dir,
            (
                str(context.job.project_id),
                "training",
                "exports",
                str(context.job.id),
            ),
            create_missing=False,
        )

    def _upload_files(self, root: Path) -> tuple[StorageArtifactFile, ...]:
        files: list[StorageArtifactFile] = []
        try:
            relative_root = root.absolute().relative_to(
                self.settings.projects_dir.absolute()
            )
            components = tuple(relative_root.parts)
            if len(components) != 4 or components[1:] != (
                "training",
                "exports",
                components[3],
            ):
                raise ValueError("local export path is outside projects root")
            with SafeExportDirectory.open_root(
                self.settings.projects_dir,
                components,
                create_missing=False,
            ) as safe_root:
                for relative, size, digest in safe_root.regular_files():
                    if relative == COMPLETION_MANIFEST_NAME:
                        continue
                    files.append(
                        StorageArtifactFile(relative, root / relative, size, digest)
                    )
        except (CompletionFilesystemError, OSError, ValueError) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local exportを検証できません",
            ) from exc
        if not files:
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local exportに転送対象がありません",
            )
        return tuple(files)

    def _write_completion_manifest(
        self,
        root: Path,
        context: _CompletionContext,
        export_id: UUID,
        storage_job_id: UUID,
        export_fingerprint: str,
        export_manifest_path: Path,
    ) -> tuple[Path, str]:
        try:
            with self._open_final_export_directory(context) as safe_root:
                files = [
                    item
                    for item in safe_root.regular_files()
                    if item[0] != COMPLETION_MANIFEST_NAME
                ]
                logs = [item for item in files if item[0].startswith("logs/")]
                samples = [item for item in files if item[0].startswith("samples/")]
                payload = {
                    "schema_version": COMPLETION_MANIFEST_SCHEMA_VERSION,
                    "project_id": context.job.project_id,
                    "training_job_id": context.job.id,
                    "training_config_id": context.execution_config.id,
                    "parent_training_job_id": context.job.parent_job_id,
                    "resume_artifact_id": context.job.resume_artifact_id,
                    "training_status": context.job.status,
                    "exit_code": context.job.exit_code,
                    "started_at": _iso(context.job.started_at),
                    "finished_at": _iso(context.job.finished_at),
                    "dataset_snapshot_id": context.snapshot.id,
                    "dataset_content_hash": context.snapshot.content_sha256,
                    "dataset_remote_provenance": {
                        "snapshot_id": str(context.snapshot_remote.snapshot_id),
                        "remote_relative_path": (
                            context.snapshot_remote.remote_relative_path
                        ),
                        "storage_transfer_job_id": str(
                            context.snapshot_remote.storage_transfer_job_id
                        ),
                        "remote_manifest_sha256": (
                            context.snapshot_remote.remote_manifest_sha256
                        ),
                        "content_sha256": context.snapshot_remote.content_sha256,
                        "verification_level": (
                            context.snapshot_remote.verification_level
                        ),
                    },
                    "managed_model_id": context.model.id,
                    "managed_model_sha256": context.model.local_sha256,
                    "safe_config_fingerprint": _execution_config_fingerprint(
                        context.execution_config
                    ),
                    "final_lora_relative_path": (
                        f"artifacts/{context.final_lora.filename}"
                    ),
                    "final_lora_size": context.final_lora.file_size,
                    "final_lora_sha256": context.final_lora.sha256,
                    "export_files": [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in files
                    ],
                    "logs": [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in logs
                    ],
                    "samples": [
                        {"relative_path": relative, "size": size, "sha256": digest}
                        for relative, size, digest in samples
                    ],
                    "environment_provenance": self._runtime_provenance(context),
                    "performance_provenance": self._performance_provenance(
                        context.job.id
                    ),
                    "storage_transfer_job_id": str(storage_job_id),
                    "source_fingerprint": context.source_fingerprint,
                    "export_fingerprint": export_fingerprint,
                    "export_manifest_sha256": self._hash_export_manifest(
                        safe_root, export_manifest_path
                    ),
                    "remote_relative_path": context.remote_relative_path,
                    "created_at": utc_now().isoformat(),
                }
                safe_root.write_json(COMPLETION_MANIFEST_NAME, payload)
                path = root / COMPLETION_MANIFEST_NAME
                _size, digest = safe_root.hash_file(COMPLETION_MANIFEST_NAME)
                return path, digest
        except (CompletionFilesystemError, OSError, ValueError, RuntimeError) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "completion manifestの作成に失敗しました",
            ) from exc

    @staticmethod
    def _hash_export_manifest(
        safe_root: SafeExportDirectory, export_manifest_path: Path
    ) -> str:
        del export_manifest_path
        _size, digest = safe_root.hash_file(EXPORT_MANIFEST_NAME)
        return digest

    def _read_remote_completion_marker(
        self, target: StorageRemotePath
    ) -> dict[str, Any] | None:
        marker, _marker_hash = self._read_remote_completion_with_hash(target)
        return marker

    def _read_remote_completion_with_hash(
        self, target: StorageRemotePath
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            raw = self._read_remote_completion_bytes(target)
            if raw is None:
                return None, None
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("completion manifest must be an object")
            return value, hashlib.sha256(raw).hexdigest()
        except CompletionFailure:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                "remote completion manifestを読み取れません",
            ) from None

    def _read_remote_completion_bytes(self, target: StorageRemotePath) -> bytes | None:
        try:
            entries = self.storage._remote_entries(target, allow_missing=True)
            entry = entries.get(COMPLETION_MANIFEST_NAME)
            if entry is None:
                return None
            if (
                entry.is_directory
                or entry.size_bytes < 0
                or entry.size_bytes > MAX_COMPLETION_MANIFEST_BYTES
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "remote completion manifestが許容サイズを超えています",
                )
            raw = self.storage.adapter.read_remote_file(
                target.child(COMPLETION_MANIFEST_NAME),
                max_bytes=MAX_COMPLETION_MANIFEST_BYTES,
            )
            if len(raw) > MAX_COMPLETION_MANIFEST_BYTES:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "remote completion manifestが許容サイズを超えています",
                )
            if len(raw) != entry.size_bytes:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "remote completion manifestのサイズが一致しません",
                )
            return bytes(raw)
        except CompletionFailure:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, UserFacingError):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                "remote completion manifestを読み取れません",
            ) from None

    def _hash_remote_marker(self, target: StorageRemotePath) -> str:
        _marker, marker_hash = self._read_remote_completion_with_hash(target)
        if marker_hash is None:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestを検証できません",
            )
        return marker_hash

    def _verify_existing_remote_completion(
        self,
        context: _CompletionContext,
        files: tuple[StorageArtifactFile, ...],
        marker: dict[str, Any],
        export_id: UUID,
        *,
        export_manifest_sha256: str | None = None,
        expected_export_fingerprint: str | None = None,
    ) -> UUID:
        storage_job_id = self._validate_completion_marker(
            marker,
            context,
            expected_export_fingerprint or str(marker.get("export_fingerprint", "")),
            files=files,
            export_manifest_sha256=export_manifest_sha256,
            error_code=TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
        )
        self._verify_artifact_transfer_job(
            context,
            files,
            storage_job_id,
            error_code=TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
        )
        self.storage.verify_remote_artifact_files(
            context.target,
            files,
            self.storage.get_project_storage_settings(
                UUID(context.job.project_id)
            ).verification_policy,
            expected_transfer_job_id=storage_job_id,
            expected_project_id=UUID(context.job.project_id),
            expected_training_run_id=UUID(context.job.id),
        )
        del export_id
        return storage_job_id

    def _validate_completion_marker(
        self,
        marker: dict[str, Any],
        context: _CompletionContext,
        export_fingerprint: str,
        *,
        files: tuple[StorageArtifactFile, ...] | None = None,
        export_manifest_sha256: str | None = None,
        expected_storage_job_id: UUID | None = None,
        error_code: TrainingCompletionErrorCode,
    ) -> UUID:
        def fail(summary: str) -> None:
            raise CompletionFailure(error_code, summary)

        if not _COMPLETION_MANIFEST_REQUIRED_FIELDS.issubset(marker):
            fail("completion manifestの必須fieldが不足しています")
        if marker.get("schema_version") != COMPLETION_MANIFEST_SCHEMA_VERSION:
            fail("completion manifestのschema versionが一致しません")
        expected_values: dict[str, Any] = {
            "project_id": context.job.project_id,
            "training_job_id": context.job.id,
            "training_config_id": context.execution_config.id,
            "parent_training_job_id": context.job.parent_job_id,
            "resume_artifact_id": context.job.resume_artifact_id,
            "training_status": context.job.status,
            "exit_code": context.job.exit_code,
            "started_at": _iso(context.job.started_at),
            "finished_at": _iso(context.job.finished_at),
            "dataset_snapshot_id": context.snapshot.id,
            "dataset_content_hash": context.snapshot.content_sha256,
            "managed_model_id": context.model.id,
            "managed_model_sha256": context.model.local_sha256,
            "safe_config_fingerprint": _execution_config_fingerprint(
                context.execution_config
            ),
            "final_lora_relative_path": f"artifacts/{context.final_lora.filename}",
            "final_lora_size": context.final_lora.file_size,
            "final_lora_sha256": context.final_lora.sha256,
            "source_fingerprint": context.source_fingerprint,
            "export_fingerprint": export_fingerprint,
            "remote_relative_path": context.remote_relative_path,
        }
        if any(marker.get(key) != value for key, value in expected_values.items()):
            fail("completion manifestのprovenanceが一致しません")
        if not isinstance(marker.get("exit_code"), int) or isinstance(
            marker.get("exit_code"), bool
        ):
            fail("completion manifestのexit codeが不正です")
        expected_remote_provenance = {
            "snapshot_id": str(context.snapshot_remote.snapshot_id),
            "remote_relative_path": context.snapshot_remote.remote_relative_path,
            "storage_transfer_job_id": str(
                context.snapshot_remote.storage_transfer_job_id
            ),
            "remote_manifest_sha256": context.snapshot_remote.remote_manifest_sha256,
            "content_sha256": context.snapshot_remote.content_sha256,
            "verification_level": context.snapshot_remote.verification_level,
        }
        remote_provenance = marker.get("dataset_remote_provenance")
        if (
            not isinstance(remote_provenance, dict)
            or set(remote_provenance) != set(_DATASET_REMOTE_PROVENANCE_FIELDS)
            or remote_provenance != expected_remote_provenance
        ):
            fail("dataset remote provenanceが一致しません")
        if marker.get("environment_provenance") != self._runtime_provenance(context):
            fail("runtime provenanceが一致しません")
        if marker.get("performance_provenance") != self._performance_provenance(
            context.job.id
        ):
            fail("performance provenanceが一致しません")
        if not isinstance(marker.get("created_at"), str) or not marker["created_at"]:
            fail("completion manifestのcreated_atが不正です")
        completion_job_value = marker.get("storage_transfer_job_id")
        try:
            storage_job_id = UUID(str(completion_job_value))
        except (TypeError, ValueError):
            fail("completion manifestの転送job IDが不正です")
        if (
            expected_storage_job_id is not None
            and storage_job_id != expected_storage_job_id
        ):
            fail("completion manifestの転送job IDが一致しません")
        marker_export_manifest = marker.get("export_manifest_sha256")
        if not _is_sha256(marker_export_manifest):
            fail("completion manifestのexport manifest hashが不正です")
        if (
            export_manifest_sha256 is not None
            and marker_export_manifest != export_manifest_sha256
        ):
            fail("completion manifestのexport manifest hashが一致しません")
        if files is not None:
            expected_entries = _artifact_file_manifest_entries(files)
            actual_entries = _canonical_manifest_file_entries(
                marker.get("export_files"), fail
            )
            if actual_entries != expected_entries:
                fail("completion manifestのexport file集合が一致しません")
            expected_logs = tuple(
                entry
                for entry in expected_entries
                if str(entry["relative_path"]).startswith("logs/")
            )
            expected_samples = tuple(
                entry
                for entry in expected_entries
                if str(entry["relative_path"]).startswith("samples/")
            )
            if (
                _canonical_manifest_file_entries(marker.get("logs"), fail)
                != expected_logs
            ):
                fail("completion manifestのlogsが一致しません")
            if (
                _canonical_manifest_file_entries(marker.get("samples"), fail)
                != expected_samples
            ):
                fail("completion manifestのsamplesが一致しません")
        return storage_job_id

    def _marker_matches_context(
        self,
        marker: dict[str, Any],
        context: _CompletionContext,
        export_fingerprint: str,
        *,
        files: tuple[StorageArtifactFile, ...] | None = None,
        export_manifest_sha256: str | None = None,
        expected_storage_job_id: UUID | None = None,
    ) -> bool:
        try:
            self._validate_completion_marker(
                marker,
                context,
                export_fingerprint,
                files=files,
                export_manifest_sha256=export_manifest_sha256,
                expected_storage_job_id=expected_storage_job_id,
                error_code=TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
            )
        except CompletionFailure:
            return False
        return True

    def _find_matching_artifact_job_id(
        self,
        context: _CompletionContext,
        files: tuple[StorageArtifactFile, ...],
        *,
        preferred_job_id: UUID | None,
    ) -> UUID | None:
        with self.session_factory() as session:
            if preferred_job_id is not None:
                self._verify_artifact_transfer_job(
                    context,
                    files,
                    preferred_job_id,
                    error_code=TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    session=session,
                )
                return preferred_job_id
            records = session.scalars(
                select(StorageTransferJobRecord)
                .where(
                    StorageTransferJobRecord.project_id == context.job.project_id,
                    StorageTransferJobRecord.training_run_id == context.job.id,
                    StorageTransferJobRecord.transfer_type
                    == StorageTransferType.ARTIFACT_UPLOAD.value,
                    StorageTransferJobRecord.status == TransferStatus.COMPLETED.value,
                )
                .order_by(StorageTransferJobRecord.completed_at.desc())
            ).all()
            for record in records:
                job_id = UUID(record.id)
                try:
                    self._verify_artifact_transfer_job(
                        context,
                        files,
                        job_id,
                        error_code=TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                        session=session,
                    )
                except CompletionFailure:
                    continue
                return job_id
        return None

    def _verify_artifact_transfer_job(
        self,
        context: _CompletionContext,
        files: tuple[StorageArtifactFile, ...],
        storage_job_id: UUID,
        *,
        error_code: TrainingCompletionErrorCode,
        session: Any | None = None,
    ) -> None:
        owns_session = session is None
        db_session: Any = session
        if owns_session:
            session_context = self.session_factory()
            db_session = session_context.__enter__()
        try:
            record = db_session.scalar(
                select(StorageTransferJobRecord).where(
                    StorageTransferJobRecord.id == str(storage_job_id)
                )
            )
            if (
                record is None
                or record.project_id != context.job.project_id
                or record.training_run_id != context.job.id
                or record.transfer_type != StorageTransferType.ARTIFACT_UPLOAD.value
                or record.status != TransferStatus.COMPLETED.value
                or record.item_count != len(files)
            ):
                raise CompletionFailure(
                    error_code,
                    "completion manifestの転送jobが一致しません",
                )
            if not record.manifest_path:
                raise CompletionFailure(
                    error_code,
                    "成果物転送manifestがありません",
                )
            manifest_path = Path(record.manifest_path)
            transfer_root = (
                self.settings.transfer_temp_dir or self.settings.temp_dir / "transfers"
            )
            if (
                transfer_root.is_symlink()
                or manifest_path.is_symlink()
                or not manifest_path.is_file()
                or not _is_relative_to(manifest_path.resolve(), transfer_root.resolve())
                or manifest_path.stat().st_size > MAX_COMPLETION_MANIFEST_BYTES
            ):
                raise CompletionFailure(
                    error_code,
                    "成果物転送manifestの保存先が不正です",
                )
            try:
                payload = json.loads(manifest_path.read_bytes())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                raise CompletionFailure(
                    error_code,
                    "成果物転送manifestを読み取れません",
                ) from None
            if not isinstance(payload, dict):
                raise CompletionFailure(error_code, "成果物転送manifestが不正です")
            if any(
                payload.get(key) != value
                for key, value in {
                    "schema_version": "phase9a-artifact-transfer-v1",
                    "transfer_job_id": str(storage_job_id),
                    "transfer_type": StorageTransferType.ARTIFACT_UPLOAD.value,
                    "project_id": context.job.project_id,
                    "training_run_id": context.job.id,
                    "destination": context.target.rclone_value,
                    "status": TransferStatus.COMPLETED.value,
                    "item_count": len(files),
                }.items()
            ):
                raise CompletionFailure(
                    error_code,
                    "成果物転送manifestのprovenanceが一致しません",
                )
            actual_items = _canonical_transfer_manifest_items(
                payload.get("items"), error_code
            )
            expected_items = tuple(
                {
                    "relative_path": item.relative_path,
                    "size": item.size_bytes,
                    "local_sha256": item.sha256,
                }
                for item in files
            )
            if actual_items != expected_items:
                raise CompletionFailure(
                    error_code,
                    "成果物転送manifestのfile集合が一致しません",
                )
        finally:
            if owns_session:
                session_context.__exit__(None, None, None)

    def _complete_claimed(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
        storage_job_id: UUID,
        completion_hash: str,
        context: _CompletionContext,
        export_fingerprint: str,
    ) -> None:
        with self.session_factory() as session:
            repository = TrainingCompletionRepository(session)
            if not repository.update_claimed(
                export_id,
                worker_id=worker_id,
                claim_token=claim_token,
                worker_generation=worker_generation,
                values={
                    "status": TrainingCompletionStatus.COMPLETED.value,
                    "current_stage": "completed",
                    "storage_transfer_job_id": str(storage_job_id),
                    "completion_manifest_sha256": completion_hash,
                    "remote_completion_manifest_relative_path": (
                        COMPLETION_MANIFEST_NAME
                    ),
                    "export_fingerprint": export_fingerprint,
                    "remote_relative_path": context.remote_relative_path,
                    "completed_at": utc_now(),
                    "heartbeat_at": utc_now(),
                    "worker_id": None,
                    "claim_token": None,
                },
            ):
                session.rollback()
                raise CompletionFailure(
                    TrainingCompletionErrorCode.WORKER_CLAIM_LOST,
                    "completion workerのclaimが失われました",
                )
            session.commit()

    def _finish_claimed(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int | None,
        status: TrainingCompletionStatus,
        error_code: str,
        summary: str,
    ) -> None:
        if worker_generation is None:
            return
        try:
            with self.session_factory() as session:
                repository = TrainingCompletionRepository(session)
                repository.update_claimed(
                    export_id,
                    worker_id=worker_id,
                    claim_token=claim_token,
                    worker_generation=worker_generation,
                    values={
                        "status": status.value,
                        "current_stage": status.value,
                        "error_code": error_code,
                        "error_summary": summary,
                        "completed_at": utc_now(),
                        "heartbeat_at": utc_now(),
                        "worker_id": None,
                        "claim_token": None,
                    },
                )
                session.commit()
        except (OperationalError, OSError):
            logger.exception(
                "training_completion_failure_persist_failed export_id=%s", export_id
            )

    def _claimed_update(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
        **values: Any,
    ) -> None:
        with self.session_factory() as session:
            if not TrainingCompletionRepository(session).update_claimed(
                export_id,
                worker_id=worker_id,
                claim_token=claim_token,
                worker_generation=worker_generation,
                values=values,
            ):
                session.rollback()
                raise CompletionFailure(
                    TrainingCompletionErrorCode.WORKER_CLAIM_LOST,
                    "completion workerのclaimが失われました",
                )
            session.commit()

    def _check_cancel(self, export_id: UUID, token: CancelToken) -> None:
        if token.cancelled:
            raise CompletionFailure(
                TrainingCompletionErrorCode.CANCELED,
                "成果物同期をキャンセルしました",
            )
        with self.session_factory() as session:
            record = TrainingCompletionRepository(session).get(export_id)
            if record is not None and record.cancel_requested:
                token.cancel()
                raise CompletionFailure(
                    TrainingCompletionErrorCode.CANCELED,
                    "成果物同期をキャンセルしました",
                )

    def _check_claim(
        self,
        export_id: UUID,
        worker_id: str,
        claim_token: str,
        worker_generation: int,
    ) -> None:
        with self.session_factory() as session:
            if not TrainingCompletionRepository(session).update_claimed(
                export_id,
                worker_id=worker_id,
                claim_token=claim_token,
                worker_generation=worker_generation,
                values={"heartbeat_at": utc_now()},
            ):
                session.rollback()
                raise CompletionFailure(
                    TrainingCompletionErrorCode.WORKER_CLAIM_LOST,
                    "completion workerのclaimが失われました",
                )
            session.commit()

    def _is_cancel_requested(self, export_id: UUID) -> bool:
        with self.session_factory() as session:
            record = TrainingCompletionRepository(session).get(export_id)
            return bool(record is not None and record.cancel_requested)

    def _remote_state(self, target: StorageRemotePath) -> list[dict[str, Any]]:
        try:
            entries = self.storage._remote_entries(target, allow_missing=True)
        except UserFacingError as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote destination state cannot be verified",
            ) from exc
        return [
            {
                "relative_path": relative,
                "size": entry.size_bytes,
                "hash_type": entry.hash_type,
                "hash": entry.hash_value,
                "modified_at": entry.modified_at.isoformat()
                if entry.modified_at
                else None,
            }
            for relative, entry in sorted(entries.items())
        ]

    def _file_descriptor(
        self, source: Path, relative_path: str, category: str
    ) -> TrainingCompletionFile:
        if source.is_symlink() or not source.is_file():
            raise CompletionFailure(
                TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                "export対象ファイルがありません",
            )
        size, digest = _stable_file_copy_fingerprint(source)
        return TrainingCompletionFile(relative_path, source, size, digest, category)

    @staticmethod
    def _dataset_provenance(context: _CompletionContext) -> dict[str, Any]:
        return {
            "snapshot_id": context.snapshot.id,
            "content_sha256": context.snapshot.content_sha256,
            "manifest_sha256": context.snapshot.manifest_sha256,
            "remote_relative_path": _safe_manifest_text(
                context.snapshot_remote.remote_relative_path
            ),
            "storage_transfer_job_id": str(
                context.snapshot_remote.storage_transfer_job_id
            ),
            "remote_manifest_sha256": context.snapshot_remote.remote_manifest_sha256,
            "verification_level": context.snapshot_remote.verification_level,
        }

    @staticmethod
    def _model_provenance(context: _CompletionContext) -> dict[str, Any]:
        return {
            "managed_model_id": context.model.id,
            "display_name": _safe_manifest_text(context.model.display_name),
            "remote_file_name": _safe_manifest_text(context.model.remote_file_name),
            "size_bytes": context.model.local_size_bytes,
            "sha256": context.model.local_sha256,
        }

    @staticmethod
    def _runtime_provenance(context: _CompletionContext) -> dict[str, Any]:
        return {
            "schema_version": "phase9a-runtime-provenance-v1",
            "application_version": "0.1.0",
            "training_job_id": context.job.id,
            "trainer_script": _safe_manifest_text(
                str(context.execution_config.payload["trainer_script"])
            ),
            "resume_parent_job_id": context.job.parent_job_id,
            "resume_artifact_id": context.job.resume_artifact_id,
        }

    def _resume_provenance(self, context: _CompletionContext) -> dict[str, Any]:
        result: dict[str, Any] = {
            "parent_training_job_id": context.job.parent_job_id,
            "resume_artifact_id": context.job.resume_artifact_id,
            "resume_mode": context.job.resume_mode,
        }
        if context.job.resume_artifact_id:
            with self.session_factory() as session:
                artifact = session.scalar(
                    select(TrainingArtifactRecord).where(
                        TrainingArtifactRecord.id == context.job.resume_artifact_id
                    )
                )
                if artifact is not None:
                    result["resume_artifact_sha256"] = artifact.sha256
                    result["resume_artifact_relative_path"] = artifact.relative_path
        return result

    def _performance_provenance(self, training_job_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            summary = session.scalar(
                select(TrainingExecutionSummaryRecord).where(
                    TrainingExecutionSummaryRecord.training_job_id == training_job_id
                )
            )
            if summary is None:
                return None
            return {
                "job_result_status": summary.job_result_status,
                "measured_steps_per_second": summary.measured_steps_per_second,
                "elapsed_seconds": summary.elapsed_seconds,
                "peak_reserved_vram_bytes": summary.peak_reserved_vram_bytes,
                "memory_sample_count": summary.memory_sample_count,
                "oom_detected": summary.oom_detected,
                "exclusion_reasons": _json_list(summary.exclusion_reasons_json),
            }

    @staticmethod
    def _safe_config(config: _ExecutionConfig) -> dict[str, Any]:
        payload = config.payload
        return {
            "schema_version": "phase9a-safe-training-config-v1",
            "id": config.id,
            "project_id": config.project_id,
            "dataset_snapshot_id": config.dataset_snapshot_id,
            "managed_model_id": config.managed_model_id,
            "name": _safe_manifest_text(str(payload["name"])),
            "output_name": _safe_manifest_text(config.output_name),
            "output_role": "training-job-output",
            "trainer_script": _safe_manifest_text(str(payload["trainer_script"])),
            "trainer_root_role": "configured-sd-scripts",
            "resolution": payload["resolution"],
            "batch_size": payload["batch_size"],
            "epochs": payload["epochs"],
            "learning_rate": payload["learning_rate"],
            "optimizer": payload["optimizer"],
            "scheduler": payload["scheduler"],
            "network_module": payload["network_module"],
            "network_dim": payload["network_dim"],
            "network_alpha": payload["network_alpha"],
            "mixed_precision": payload["mixed_precision"],
            "save_every_n_epochs": payload["save_every_n_epochs"],
            "cache_latents": payload["cache_latents"],
            "gradient_checkpointing": payload["gradient_checkpointing"],
            "seed": payload["seed"],
            "extra_options": _safe_json_value(payload["extra_options"]),
            "recommendation_id": payload["recommendation_id"],
            "recommendation_engine_version": payload["recommendation_engine_version"],
            "recommendation_change_diff": _safe_json_value(
                payload["recommendation_change_diff"]
            ),
        }


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _iso(value: datetime | None) -> str | None:
    normalized = _utc(value)
    return normalized.isoformat() if normalized else None


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execution_config_fingerprint(config: _ExecutionConfig) -> str:
    safe = TrainingCompletionService._safe_config(config)
    return _fingerprint(safe)


def _stable_file_hash(path: Path) -> str:
    _size, digest = stable_file_hash(path)
    return digest


def _stable_file_copy_fingerprint(path: Path) -> tuple[int, str]:
    return stable_file_hash(path)


def _safe_json_value(value: object) -> object:
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized_key = key_text.casefold()
            if any(token in normalized_key for token in _SENSITIVE_JSON_KEY_TOKENS):
                continue
            if any(token in normalized_key for token in _LOCAL_PATH_KEY_TOKENS):
                sanitized[key_text] = "<local-path-redacted>"
                continue
            sanitized[key_text] = _safe_json_value(item)
        return sanitized
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, str):
        if _looks_like_local_path(value):
            return "<local-path-redacted>"
        return "".join(char for char in value if ord(char) >= 32 or char in "\r\n\t")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value}")


_SENSITIVE_JSON_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "raw_env",
    "environment",
    "rclone_config",
)
_LOCAL_PATH_KEY_TOKENS = (
    "path",
    "directory",
    "_dir",
    "root",
    "executable",
    "model",
    "dataset",
    "output",
    "python",
)


def _looks_like_local_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        os.path.isabs(value)
        or normalized.startswith("/")
        or normalized.startswith("//")
        or (
            len(normalized) >= 3
            and normalized[0].isalpha()
            and normalized[1] == ":"
            and normalized[2] == "/"
        )
    )


def _safe_manifest_text(value: str) -> str:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or ":" in normalized
        or any(part in {"", ".", ".."} for part in parts)
    ):
        return "<redacted>"
    return "".join(char for char in value if ord(char) >= 32 or char in "\r\n\t")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _valid_manifest_relative_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    return bool(
        normalized
        and not normalized.startswith("/")
        and not Path(normalized).is_absolute()
        and not any(part in {"", ".", ".."} for part in normalized.split("/"))
        and not any(ord(char) < 32 for char in normalized)
    )


def _artifact_file_manifest_entries(
    files: tuple[StorageArtifactFile, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "relative_path": item.relative_path,
            "size": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in sorted(files, key=lambda value: value.relative_path)
    )


def _canonical_manifest_file_entries(
    value: object,
    fail: Callable[[str], None],
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        fail("completion manifestのfile一覧が不正です")
        return ()
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "size",
            "sha256",
        }:
            fail("completion manifestのfile一覧が不正です")
        relative = item.get("relative_path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not _valid_manifest_relative_path(relative)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
        ):
            fail("completion manifestのfile一覧が不正です")
        relative_value = str(relative)
        if relative_value in seen:
            fail("completion manifestのfile一覧に重複があります")
        seen.add(relative_value)
        result.append(
            {
                "relative_path": relative_value,
                "size": size,
                "sha256": str(digest),
            }
        )
    return tuple(sorted(result, key=lambda item: str(item["relative_path"])))


def _canonical_transfer_manifest_items(
    value: object,
    error_code: TrainingCompletionErrorCode,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise CompletionFailure(error_code, "成果物転送manifestのitemsが不正です")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise CompletionFailure(error_code, "成果物転送manifestのitemsが不正です")
        relative = item.get("relative_path")
        size = item.get("size")
        digest = item.get("local_sha256")
        transfer_status = item.get("transfer_status")
        verification_status = item.get("verification_status")
        if (
            not _valid_manifest_relative_path(relative)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
            or transfer_status not in {"completed", "skipped"}
            or not isinstance(verification_status, str)
            or verification_status
            not in {
                "full_checksum",
                "remote_hash_and_size",
                "manifest_metadata_and_size",
                "existence_only",
            }
        ):
            raise CompletionFailure(error_code, "成果物転送manifestのitemsが不正です")
        relative_value = str(relative)
        if relative_value in seen:
            raise CompletionFailure(
                error_code, "成果物転送manifestのitemsに重複があります"
            )
        seen.add(relative_value)
        result.append(
            {
                "relative_path": relative_value,
                "size": size,
                "local_sha256": str(digest),
            }
        )
    return tuple(sorted(result, key=lambda item: str(item["relative_path"])))


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["TrainingCompletionService", "CompletionFailure"]
