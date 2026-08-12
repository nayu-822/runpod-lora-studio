from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select, update
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
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.services.training_artifact import TrainingArtifactScanner
from runpod_lora_studio.services.training_service import TrainingService

logger = logging.getLogger("runpod_lora_studio.training_completion")

EXPORT_SCHEMA_VERSION = "phase9a-training-export-v1"
COMPLETION_MANIFEST_SCHEMA_VERSION = "phase9a-training-completion-v1"
COMPLETION_MANIFEST_NAME = "completion-manifest.json"
EXPORT_MANIFEST_NAME = "export-manifest.json"


class CompletionFailure(Exception):
    def __init__(self, code: TrainingCompletionErrorCode | str, summary: str) -> None:
        self.code = str(
            code.value if isinstance(code, TrainingCompletionErrorCode) else code
        )
        self.summary = summary
        super().__init__(summary)


@dataclass(frozen=True, slots=True)
class _CompletionContext:
    job: TrainingJobRecord
    config: TrainingConfigRecord
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

    @property
    def export_root(self) -> Path:
        return self.settings.projects_dir

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
            training_config_id=UUID(context.config.id),
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
                records = TrainingCompletionRepository(session).list_recovery_records()
                for record in records:
                    heartbeat = _utc(record.heartbeat_at)
                    if heartbeat is not None and heartbeat >= cutoff:
                        continue
                    future = self._futures.get(UUID(record.id))
                    if future is not None and not future.done():
                        continue
                    result = session.execute(
                        update(TrainingCompletionExportRecord)
                        .where(
                            TrainingCompletionExportRecord.id == record.id,
                            TrainingCompletionExportRecord.status == record.status,
                        )
                        .values(
                            status=TrainingCompletionStatus.STALE.value,
                            current_stage=TrainingCompletionStatus.STALE.value,
                            worker_id=None,
                            claim_token=None,
                            heartbeat_at=now,
                            error_code="STALE_WORKER",
                            error_summary=(
                                "completion workerのheartbeatを確認できません"
                            ),
                            updated_at=now,
                        )
                    )
                    recovered += int(result.rowcount == 1)
                session.commit()
        except OperationalError:
            return 0
        return recovered

    def _cleanup_temporary_exports(self, *, max_items: int = 128) -> int:
        root = self.settings.projects_dir
        if root.is_symlink() or not root.is_dir():
            return 0
        removed = 0
        root_resolved = root.resolve()
        try:
            candidates = root.glob("*/training/exports/.creating-*")
            for candidate in candidates:
                if removed >= max_items:
                    break
                if (
                    candidate.is_symlink()
                    or not candidate.is_dir()
                    or not candidate.name.startswith(".creating-")
                ):
                    continue
                try:
                    if not _is_relative_to(candidate.resolve(), root_resolved):
                        continue
                    shutil.rmtree(candidate)
                except (OSError, RuntimeError):
                    logger.warning(
                        "training_completion_temp_cleanup_failed path=%s",
                        candidate,
                        exc_info=True,
                    )
                    continue
                removed += 1
        except OSError:
            return removed
        return removed

    def reconcile_remote(self, *, time_budget_seconds: float = 5.0) -> int:
        """Resume bounded marker reconciliation after a process restart."""
        deadline = time.monotonic() + max(0.1, time_budget_seconds)
        completed = 0
        try:
            with self.session_factory() as session:
                ids = [
                    UUID(record.id)
                    for record in TrainingCompletionRepository(
                        session
                    ).list_recovery_records()
                    if record.status == TrainingCompletionStatus.STALE.value
                ]
        except OperationalError:
            # The UI can be imported before the Phase 9A migration is applied.
            return 0
        for export_id in ids:
            if time.monotonic() >= deadline:
                break
            try:
                self._run_export(export_id)
            except (CompletionFailure, UserFacingError):
                continue
            current = self.get_export(export_id)
            completed += int(current.status is TrainingCompletionStatus.COMPLETED)
        return completed

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

    def _submit(self, export_id: UUID) -> None:
        with self._lock:
            future = self._futures.get(export_id)
            if future is not None and not future.done():
                return
            self._futures[export_id] = self._executor.submit(
                self._run_export, export_id
            )

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
        remote_marker = self._read_remote_completion_marker(context.target)
        if remote_marker is not None:
            if not self._marker_matches_context(
                remote_marker, context, export_fingerprint
            ):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "同一remote destinationに異なるcompletion manifestがあります",
                )
            self._verify_existing_remote_completion(
                context, files, remote_marker, export_id
            )
            storage_job_id = self._existing_artifact_job_id(context.job.id, export_id)
            if storage_job_id is None:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                    "既存completion manifestの転送履歴を確認できません",
                )
            completion_hash = self._hash_remote_marker(context.target)
            self._complete_claimed(
                export_id,
                worker_id,
                claim_token,
                worker_generation,
                storage_job_id,
                completion_hash,
                context,
                export_fingerprint,
            )
            return
        settings = self.storage.get_project_storage_settings(
            UUID(context.job.project_id)
        )
        storage_job_id = self._existing_artifact_job_id(context.job.id, export_id)
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
        self.storage.verify_remote_artifact_files(
            context.target, files, settings.verification_policy
        )
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
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "completion manifestのremote uploadに失敗しました",
            )
        remote_marker = self._read_remote_completion_marker(context.target)
        if remote_marker is None or not self._marker_matches_context(
            remote_marker, current_context, export_fingerprint
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestの再検証に失敗しました",
            )
        self._verify_existing_remote_completion(
            current_context, files, remote_marker, export_id
        )
        self._complete_claimed(
            export_id,
            worker_id,
            claim_token,
            worker_generation,
            storage_job_id,
            marker_hash,
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
        final_lora = self._find_final_lora(job, config)
        target = self.storage.training_remote_path(
            UUID(job.project_id), training_job_id
        )
        export_relative = f"projects/{job.project_id}/training/exports/{job.id}"
        files = self._preview_files(job, config, final_lora)
        storage_settings = self.storage.get_project_storage_settings(
            UUID(job.project_id)
        )
        remote_state = self._remote_state(target)
        source_fingerprint = _fingerprint(
            {
                "schema_version": COMPLETION_MANIFEST_SCHEMA_VERSION,
                "project_id": job.project_id,
                "training_job_id": job.id,
                "training_config_id": config.id,
                "config_snapshot_sha256": _config_fingerprint(config),
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
            config=config,
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

    def _find_final_lora(
        self, job: TrainingJobRecord, config: TrainingConfigRecord
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
        config: TrainingConfigRecord,
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
        config_path = runtime / "config" / "training-config.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise CompletionFailure(
                TrainingCompletionErrorCode.ELIGIBILITY_FAILED,
                "training config snapshot is missing",
            )
        result.append(
            self._file_descriptor(config_path, "config/training-config.json", "config")
        )
        del config
        return tuple(result)

    def _build_local_export(
        self, context: _CompletionContext, cancel_token: CancelToken
    ) -> Path:
        projects_root = self.settings.projects_dir
        if projects_root.is_symlink():
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "projects root must not be a symlink",
            )
        projects_root.mkdir(parents=True, exist_ok=True)
        project_root = projects_root / str(context.job.project_id)
        if project_root.is_symlink():
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "project export root must not be a symlink",
            )
        training_root = project_root / "training"
        exports_root = training_root / "exports"
        if training_root.is_symlink() or exports_root.is_symlink():
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "export parent must not be a symlink",
            )
        final_root = exports_root / str(context.job.id)
        exports_root.mkdir(parents=True, exist_ok=True)
        if exports_root.is_symlink():
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "export root must not be a symlink",
            )
        if final_root.is_symlink():
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local export繝ｭ繝ｼ繝医′symlink縺ｧ縺吶・",
            )
        if final_root.exists():
            for marker_path in (
                final_root / EXPORT_MANIFEST_NAME,
                final_root / "provenance" / "source-fingerprint.json",
            ):
                try:
                    payload = _read_json(marker_path)
                    if payload.get("source_fingerprint") == context.source_fingerprint:
                        return final_root
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "同じ学習jobのlocal exportが異なる内容です",
            )
        temporary = exports_root / f".creating-{uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            for item in context.files:
                self._check_cancel_for_token(cancel_token)
                destination = temporary / item.relative_path
                source = item.source_path
                _copy_stable(source, destination, cancel_token)
            self._write_generated_files(temporary, context)
            _fsync_tree(temporary)
            temporary.replace(final_root)
            _fsync_directory(exports_root)
        except CompletionFailure:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            raise CompletionFailure(
                TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                "local exportの構築に失敗しました",
            ) from exc
        return final_root

    @staticmethod
    def _check_cancel_for_token(cancel_token: CancelToken) -> None:
        if cancel_token.cancelled:
            raise CompletionFailure(
                TrainingCompletionErrorCode.CANCELED,
                "completion export縺ｮ繧ｭ繝｣繝ｳ繧ｻ繝ｫ縺ｧ縺吶・",
            )

    def _write_generated_files(self, root: Path, context: _CompletionContext) -> None:
        _write_json_durable(
            root / "provenance" / "dataset.json", self._dataset_provenance(context)
        )
        _write_json_durable(
            root / "provenance" / "model.json", self._model_provenance(context)
        )
        _write_json_durable(
            root / "provenance" / "runtime.json", self._runtime_provenance(context)
        )
        _write_json_durable(
            root / "provenance" / "resume.json", self._resume_provenance(context)
        )
        performance = self._performance_provenance(context.job.id)
        if performance is not None:
            _write_json_durable(root / "provenance" / "performance.json", performance)
        _write_json_durable(
            root / "config" / "safe-training-config.json",
            self._safe_config(context.config),
        )
        static_files = _list_export_files(root)
        _write_json_durable(
            root / "hashes" / "sha256.json",
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "files": [
                    {"relative_path": path, "size": size, "sha256": digest}
                    for path, size, digest in static_files
                ],
            },
        )
        _write_json_durable(
            root / "provenance" / "source-fingerprint.json",
            {
                "schema_version": "phase9a-source-fingerprint-v1",
                "source_fingerprint": context.source_fingerprint,
            },
        )

    def _ensure_export_manifest(
        self, root: Path, context: _CompletionContext
    ) -> tuple[Path, str, str]:
        path = root / EXPORT_MANIFEST_NAME
        if path.is_file():
            payload = _read_json(path)
            if payload.get("source_fingerprint") != context.source_fingerprint:
                raise CompletionFailure(
                    TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                    "local export manifestの入力fingerprintが一致しません",
                )
            try:
                manifest_files = payload["files"]
                actual_files = [
                    {"relative_path": relative, "size": size, "sha256": digest}
                    for relative, size, digest in _list_export_files(root)
                    if relative not in {EXPORT_MANIFEST_NAME, COMPLETION_MANIFEST_NAME}
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
                    or payload.get("export_fingerprint") != _fingerprint(actual_files)
                ):
                    raise ValueError("local export manifest does not match files")
            except (KeyError, TypeError, ValueError, OSError):
                raise CompletionFailure(
                    TrainingCompletionErrorCode.LOCAL_EXPORT_CONFLICT,
                    "local export manifest is invalid",
                ) from None
            return (
                path,
                _stable_file_hash(path),
                str(payload.get("export_fingerprint", "")),
            )
        files = [
            item
            for item in _list_export_files(root)
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
        _write_json_durable(path, payload)
        return path, _stable_file_hash(path), export_fingerprint

    def _upload_files(self, root: Path) -> tuple[StorageArtifactFile, ...]:
        files: list[StorageArtifactFile] = []
        for relative, size, digest in _list_export_files(root):
            if relative == COMPLETION_MANIFEST_NAME:
                continue
            files.append(StorageArtifactFile(relative, root / relative, size, digest))
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
        files = [
            item
            for item in _list_export_files(root)
            if item[0] != COMPLETION_MANIFEST_NAME
        ]
        logs = [item for item in files if item[0].startswith("logs/")]
        samples = [item for item in files if item[0].startswith("samples/")]
        payload = {
            "schema_version": COMPLETION_MANIFEST_SCHEMA_VERSION,
            "project_id": context.job.project_id,
            "training_job_id": context.job.id,
            "training_config_id": context.config.id,
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
                "remote_relative_path": context.snapshot_remote.remote_relative_path,
                "storage_transfer_job_id": str(
                    context.snapshot_remote.storage_transfer_job_id
                ),
                "remote_manifest_sha256": (
                    context.snapshot_remote.remote_manifest_sha256
                ),
                "content_sha256": context.snapshot_remote.content_sha256,
                "verification_level": context.snapshot_remote.verification_level,
            },
            "managed_model_id": context.model.id,
            "managed_model_sha256": context.model.local_sha256,
            "safe_config_fingerprint": _config_fingerprint(context.config),
            "final_lora_relative_path": f"artifacts/{context.final_lora.filename}",
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
            "performance_provenance": self._performance_provenance(context.job.id),
            "storage_transfer_job_id": str(storage_job_id),
            "source_fingerprint": context.source_fingerprint,
            "export_fingerprint": export_fingerprint,
            "export_manifest_sha256": _stable_file_hash(export_manifest_path),
            "remote_relative_path": context.remote_relative_path,
            "created_at": utc_now().isoformat(),
        }
        path = root / COMPLETION_MANIFEST_NAME
        _write_json_durable(path, payload)
        return path, _stable_file_hash(path)

    def _read_remote_completion_marker(
        self, target: StorageRemotePath
    ) -> dict[str, Any] | None:
        try:
            entries = self.storage._remote_entries(target, allow_missing=True)
            if COMPLETION_MANIFEST_NAME not in entries:
                return None
            raw = self.storage.adapter.read_remote_file(
                target.child(COMPLETION_MANIFEST_NAME)
            )
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_COMPLETION_CONFLICT,
                "remote completion manifestを読み取れません",
            ) from None

    def _hash_remote_marker(self, target: StorageRemotePath) -> str:
        try:
            content = self.storage.adapter.read_remote_file(
                target.child(COMPLETION_MANIFEST_NAME)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "remote completion manifestを検証できません",
            ) from exc
        return hashlib.sha256(content).hexdigest()

    def _verify_existing_remote_completion(
        self,
        context: _CompletionContext,
        files: tuple[StorageArtifactFile, ...],
        marker: dict[str, Any],
        export_id: UUID,
    ) -> None:
        try:
            marker_files = marker["export_files"]
            expected_files = {
                item.relative_path: (item.size_bytes, item.sha256) for item in files
            }
            actual_files = {
                str(item["relative_path"]): (int(item["size"]), str(item["sha256"]))
                for item in marker_files
                if isinstance(item, dict)
            }
        except (KeyError, TypeError, ValueError):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "completion manifestのfile一覧が不正です",
            ) from None
        if not expected_files.items() <= actual_files.items():
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "completion manifestの必須成果物が一致しません",
            )
        self.storage.verify_remote_artifact_files(
            context.target,
            files,
            self.storage.get_project_storage_settings(
                UUID(context.job.project_id)
            ).verification_policy,
        )
        if (
            marker.get("project_id") != context.job.project_id
            or marker.get("training_job_id") != context.job.id
        ):
            raise CompletionFailure(
                TrainingCompletionErrorCode.REMOTE_VERIFICATION_FAILED,
                "completion manifestのjob/projectが一致しません",
            )
        del export_id

    def _marker_matches_context(
        self,
        marker: dict[str, Any],
        context: _CompletionContext,
        export_fingerprint: str,
    ) -> bool:
        return bool(
            marker.get("schema_version") == COMPLETION_MANIFEST_SCHEMA_VERSION
            and marker.get("project_id") == context.job.project_id
            and marker.get("training_job_id") == context.job.id
            and marker.get("source_fingerprint") == context.source_fingerprint
            and marker.get("export_fingerprint") == export_fingerprint
            and marker.get("final_lora_sha256") == context.final_lora.sha256
        )

    def _existing_artifact_job_id(
        self, training_job_id: str, export_id: UUID
    ) -> UUID | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(StorageTransferJobRecord)
                .where(
                    StorageTransferJobRecord.training_run_id == training_job_id,
                    StorageTransferJobRecord.transfer_type
                    == StorageTransferType.ARTIFACT_UPLOAD.value,
                    StorageTransferJobRecord.status == TransferStatus.COMPLETED.value,
                )
                .order_by(StorageTransferJobRecord.completed_at.desc())
            )
            del export_id
            return UUID(record.id) if record else None

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
            "remote_relative_path": context.snapshot_remote.remote_relative_path,
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
            "display_name": context.model.display_name,
            "remote_file_name": context.model.remote_file_name,
            "size_bytes": context.model.local_size_bytes,
            "sha256": context.model.local_sha256,
        }

    @staticmethod
    def _runtime_provenance(context: _CompletionContext) -> dict[str, Any]:
        return {
            "schema_version": "phase9a-runtime-provenance-v1",
            "application_version": "0.1.0",
            "training_job_id": context.job.id,
            "trainer_script": _safe_manifest_text(context.config.trainer_script),
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
    def _safe_config(config: TrainingConfigRecord) -> dict[str, Any]:
        try:
            extra = json.loads(config.extra_options or "{}")
        except json.JSONDecodeError:
            extra = {}
        return {
            "schema_version": "phase9a-safe-training-config-v1",
            "id": config.id,
            "project_id": config.project_id,
            "dataset_snapshot_id": config.dataset_snapshot_id,
            "managed_model_id": config.managed_model_id,
            "name": config.name,
            "output_name": config.output_name,
            "trainer_script": _safe_manifest_text(config.trainer_script),
            "resolution": config.resolution,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "network_module": config.network_module,
            "network_dim": config.network_dim,
            "network_alpha": config.network_alpha,
            "mixed_precision": config.mixed_precision,
            "save_every_n_epochs": config.save_every_n_epochs,
            "cache_latents": bool(config.cache_latents),
            "gradient_checkpointing": bool(config.gradient_checkpointing),
            "seed": config.seed,
            "extra_options": _safe_json_value(extra),
            "recommendation_id": config.recommendation_id,
            "recommendation_engine_version": config.recommendation_engine_version,
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


def _config_fingerprint(config: TrainingConfigRecord) -> str:
    safe = TrainingCompletionService._safe_config(config)
    return _fingerprint(safe)


def _stable_file_hash(path: Path) -> str:
    before = path.stat()
    if not path.is_file() or path.is_symlink():
        raise OSError("not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OSError("file changed while hashing")
    return digest.hexdigest()


def _stable_file_copy_fingerprint(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = _stable_file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("file changed while hashing")
    return before.st_size, digest


def _copy_stable(
    source: Path,
    destination: Path,
    cancel_token: CancelToken | None = None,
) -> tuple[int, str]:
    if source.is_symlink() or not source.is_file():
        raise OSError("source is not a regular file")
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        source_handle = source.open("rb")
        target_handle = destination.open("xb")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    with source_handle, target_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            if cancel_token is not None and cancel_token.cancelled:
                destination.unlink(missing_ok=True)
                raise CompletionFailure(
                    TrainingCompletionErrorCode.CANCELED,
                    "completion export縺ｮ繧ｭ繝｣繝ｳ繧ｻ繝ｫ縺ｧ縺吶・",
                )
            target_handle.write(chunk)
            digest.update(chunk)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        destination.unlink(missing_ok=True)
        raise OSError("source changed while copying")
    return before.st_size, digest.hexdigest()


def _write_json_durable(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{uuid4().hex[:8]}.tmp"
    try:
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if path.is_file() and not path.is_symlink():
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir() and not path.is_symlink():
            _fsync_directory(path)


def _list_export_files(root: Path) -> list[tuple[str, int, str]]:
    if root.is_symlink() or not root.is_dir():
        raise OSError("export root is not a directory")
    result: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise OSError("export contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in {"", ".", ".."} for part in relative.split("/")):
            raise OSError("export path is invalid")
        size, digest = _stable_file_copy_fingerprint(path)
        result.append((relative, size, digest))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _safe_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
            if not any(
                token in str(key).casefold()
                for token in ("token", "secret", "password", "api_key", "authorization")
            )
        }
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, str):
        if os.path.isabs(value):
            return "<local-path-redacted>"
        return "".join(char for char in value if ord(char) >= 32 or char in "\r\n\t")
    return value


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
