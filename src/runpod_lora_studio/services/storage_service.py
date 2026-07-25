from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import DatasetSnapshotStatus
from runpod_lora_studio.domain.storage_models import (
    ManagedModel,
    ManagedModelStatus,
    ModelType,
    OverwritePolicy,
    ProjectStorageSettings,
    StorageEntry,
    StorageKind,
    StorageRemote,
    StorageRemotePath,
    StorageTransferJob,
    StorageTransferType,
    StorageValidationResult,
    TransferDirection,
    TransferItemPlan,
    TransferManifest,
    TransferPlan,
    TransferProgress,
    TransferStatus,
)
from runpod_lora_studio.external.rclone import (
    CancelToken,
    CopyOptions,
    ListOptions,
    RcloneAdapter,
    StorageTransferAdapter,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.dataset_repository import DatasetRepository
from runpod_lora_studio.persistence.models import (
    ModelTransferRecord,
    StorageTransferJobRecord,
    TransferItemRecord,
)
from runpod_lora_studio.persistence.storage_repository import StorageRepository
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.project_service import UserFacingError

logger = logging.getLogger("runpod_lora_studio.storage")


class StorageService:
    def __init__(
        self,
        settings: AppSettings,
        adapter: StorageTransferAdapter | None = None,
        datasets: DatasetSnapshotService | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter or RcloneAdapter(settings)
        self.datasets = datasets or DatasetSnapshotService(settings)
        self.session_factory = create_session_factory(settings)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="storage")
        self._futures: dict[UUID, Future[Any]] = {}
        self._cancel_tokens: dict[UUID, CancelToken] = {}

    @property
    def model_cache_root(self) -> Path:
        return self.settings.model_cache_dir or self.settings.models_dir / "base"

    @property
    def transfer_root(self) -> Path:
        return self.settings.transfer_temp_dir or self.settings.temp_dir / "transfers"

    def validate_environment(self) -> StorageValidationResult:
        return self.adapter.validate_environment()

    def list_remotes(self) -> tuple[StorageRemote, ...]:
        return self.adapter.list_remotes()

    def list_models(
        self,
        *,
        recursive: bool = True,
        query: str = "",
        extension: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> list[ManagedModel]:
        root = StorageRemotePath(
            self.settings.storage_remote_name,
            self.settings.storage_model_remote_root,
        )
        allowed = self._allowed_extensions()
        if extension and extension.casefold() not in allowed:
            return []
        try:
            entries = self.adapter.list_entries(
                root,
                ListOptions(
                    recursive=recursive,
                    max_entries=10_000,
                    page=page,
                    page_size=page_size,
                    extension=extension,
                    query=query,
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise UserFacingError("Google Driveのモデル一覧を取得できません") from exc
        version = self._rclone_version()
        models: list[ManagedModel] = []
        with self.session_factory() as session:
            repository = StorageRepository(session)
            for entry in entries:
                if entry.is_directory or not self._is_allowed_model(entry.name):
                    continue
                record = repository.upsert_model(
                    display_name=entry.name,
                    model_type=self._model_type(entry.name),
                    remote_name=entry.remote_path.remote_name,
                    remote_relative_path=entry.remote_path.relative_path,
                    remote_file_name=entry.name,
                    remote_size_bytes=entry.size_bytes,
                    remote_modified_at=entry.modified_at,
                    remote_hash_type=entry.hash_type,
                    remote_hash_value=entry.hash_value,
                    source="google_drive",
                    rclone_version=version,
                )
                models.append(repository.managed_model_from_record(record))
            session.commit()
        return models

    def get_model(self, model_id: UUID) -> ManagedModel:
        with self.session_factory() as session:
            record = StorageRepository(session).get_model(model_id)
            if record is None:
                raise UserFacingError("指定されたモデルが見つかりません")
            return StorageRepository.managed_model_from_record(record)

    def dry_run_model_download(self, model_id: UUID) -> TransferPlan:
        model = self.get_model(model_id)
        entry = self._refresh_model(model)
        destination = self._local_model_path(model_id, entry.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        plan = self.adapter.dry_run_copy(
            entry.remote_path,
            destination.with_suffix(destination.suffix + ".part"),
            CopyOptions(
                overwrite_policy=self.settings.storage_overwrite_policy,
                dry_run=True,
                checksum=self.settings.storage_use_checksum,
            ),
        )
        return replace(plan, token=self._model_plan_token(entry, destination))

    def download_model(
        self,
        model_id: UUID,
        *,
        plan_token: str | None = None,
        cancel_token: CancelToken | None = None,
        _job_id: UUID | None = None,
    ) -> UUID:
        model = self.get_model(model_id)
        entry = self._refresh_model(model)
        destination = self._local_model_path(model_id, entry.name)
        expected_token = self._model_plan_token(entry, destination)
        if plan_token is not None and plan_token != expected_token:
            raise UserFacingError(
                "モデル情報が変更されました。再度ドライランしてください"
            )
        if destination.is_file() and self._local_matches(destination, entry):
            with self.session_factory() as session:
                record = StorageRepository(session).get_model(model_id)
                if record is not None:
                    StorageRepository(session).update_model(
                        record,
                        local_path=str(destination),
                        local_size_bytes=destination.stat().st_size,
                        status=ManagedModelStatus.AVAILABLE.value,
                        verified_at=datetime.now(UTC),
                        downloaded_at=record.downloaded_at or datetime.now(UTC),
                        error_summary=None,
                    )
                    session.commit()
            if _job_id is not None:
                self._finish_job(_job_id, TransferStatus.COMPLETED, None)
            return model_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(destination.parent).free
        if available < entry.size_bytes + self.settings.model_disk_safety_margin_bytes:
            raise UserFacingError("モデル取得に必要なローカル空き容量が不足しています")
        with self.session_factory() as session:
            repository = StorageRepository(session)
            record = repository.get_model(model_id)
            if record is None:
                raise UserFacingError("指定されたモデルが見つかりません")
            repository.update_model(
                record, status=ManagedModelStatus.DOWNLOADING.value, error_summary=None
            )
            transfer = repository.create_transfer(
                model_id=model_id,
                direction=TransferDirection.DOWNLOAD,
                source_path=entry.remote_path.rclone_value,
                destination_path=str(destination),
                expected_size_bytes=entry.size_bytes,
                expected_hash=entry.hash_value if entry.hash_type == "sha256" else None,
                settings_snapshot=self._settings_snapshot(),
            )
            job = None
            if _job_id is None:
                job = repository.create_job(
                    project_id=None,
                    snapshot_id=None,
                    transfer_type=StorageTransferType.MODEL_DOWNLOAD,
                    source_kind=StorageKind.REMOTE,
                    destination_kind=StorageKind.LOCAL,
                    item_count=1,
                    total_bytes=entry.size_bytes,
                )
            session.commit()
            transfer_id = UUID(transfer.id)
            job_id = _job_id or UUID(job.id)  # type: ignore[union-attr]
        token = cancel_token or CancelToken()
        self._cancel_tokens[job_id] = token
        try:
            self._set_job_running(job_id)
            self._copy_model_with_retry(
                model_id, transfer_id, job_id, entry, destination, token
            )
            return model_id
        except Exception as exc:
            self._finish_model_failure(
                model_id, transfer_id, job_id, str(exc), token.cancelled
            )
            if isinstance(exc, UserFacingError):
                raise
            raise UserFacingError("モデル取得に失敗しました") from exc
        finally:
            self._cancel_tokens.pop(job_id, None)

    def start_model_download(
        self, model_id: UUID, plan_token: str | None = None
    ) -> UUID:
        model = self.get_model(model_id)
        entry = self._refresh_model(model)
        destination = self._local_model_path(model_id, entry.name)
        expected_token = self._model_plan_token(entry, destination)
        if plan_token is not None and plan_token != expected_token:
            raise UserFacingError(
                "繝｢繝・Ν諠・ｱ縺悟､画峩縺輔ｌ縺ｾ縺励◆縲ょ・蠎ｦ繝峨Λ繧､繝ｩ繝ｳ縺励※縺上□縺輔＞"
            )
        with self.session_factory() as session:
            job = StorageRepository(session).create_job(
                project_id=None,
                snapshot_id=None,
                transfer_type=StorageTransferType.MODEL_DOWNLOAD,
                source_kind=StorageKind.REMOTE,
                destination_kind=StorageKind.LOCAL,
                item_count=1,
                total_bytes=entry.size_bytes,
            )
            session.commit()
            job_id = UUID(job.id)
        future = self._executor.submit(
            self.download_model, model_id, plan_token=plan_token, _job_id=job_id
        )
        self._futures[job_id] = future
        return job_id

    def verify_model(self, model_id: UUID) -> bool:
        model = self.get_model(model_id)
        if model.local_path is None or not model.local_path.is_file():
            return False
        expected_hash = model.local_sha256
        actual_hash = _sha256_file(model.local_path)
        ok = model.local_path.stat().st_size == model.remote_size_bytes and (
            expected_hash is None or expected_hash == actual_hash
        )
        with self.session_factory() as session:
            record = StorageRepository(session).get_model(model_id)
            if record is not None:
                StorageRepository(session).update_model(
                    record,
                    local_size_bytes=model.local_path.stat().st_size,
                    local_sha256=actual_hash,
                    status=ManagedModelStatus.AVAILABLE.value
                    if ok
                    else ManagedModelStatus.VERIFICATION_FAILED.value,
                    verified_at=datetime.now(UTC) if ok else None,
                    error_summary=None
                    if ok
                    else "ローカルSHA-256またはサイズが一致しません",
                )
                session.commit()
        return ok

    def get_project_storage_settings(self, project_id: UUID) -> ProjectStorageSettings:
        with self.session_factory() as session:
            saved = StorageRepository(session).get_project_settings(project_id)
        if saved is not None:
            self._validate_project_roots(saved)
            return saved
        return ProjectStorageSettings(
            project_id=project_id,
            project_remote_root=self.settings.storage_project_remote_root,
            snapshot_remote_root=self.settings.storage_snapshot_remote_root,
            training_remote_root="training-runs",
            artifact_remote_root=self.settings.storage_artifact_remote_root,
            selected_managed_model_id=None,
            overwrite_policy=self.settings.storage_overwrite_policy,
            verification_policy=self.settings.storage_verification_policy,
        )

    def save_project_storage_settings(self, value: ProjectStorageSettings) -> None:
        self._validate_project_roots(value)
        with self.session_factory() as session:
            StorageRepository(session).save_project_settings(value)
            session.commit()

    def dry_run_snapshot_upload(
        self,
        snapshot_id: UUID,
        *,
        overwrite_policy: OverwritePolicy | None = None,
    ) -> TransferPlan:
        record, root = self._validated_snapshot(snapshot_id)
        project_settings = self.get_project_storage_settings(UUID(record.project_id))
        policy = overwrite_policy or project_settings.overwrite_policy
        target = self._snapshot_remote_path(
            UUID(record.project_id), snapshot_id, project_settings
        )
        files = self._snapshot_files(root)
        remote_entries = self._remote_entries(target)
        remote_content, has_remote_manifest = self._remote_snapshot_content(
            target, remote_entries
        )
        items: list[TransferItemPlan] = []
        for relative, _path, size, _digest in files:
            remote = remote_entries.get(relative)
            action = "copy"
            reason = ""
            if remote is not None:
                if policy is OverwritePolicy.FAIL_IF_EXISTS:
                    action, reason = "conflict", "remoteに同名ファイルがあります"
                elif (
                    policy is OverwritePolicy.SKIP_IDENTICAL
                    and remote.size_bytes == size
                ):
                    action, reason = "skip", "サイズ一致"
                elif policy is OverwritePolicy.COPY_MISSING:
                    action, reason = "conflict", "remoteに同名ファイルがあります"
                elif policy is OverwritePolicy.OVERWRITE_CHANGED:
                    action, reason = "copy", "変更されたremoteを明示的に上書き"
            items.append(TransferItemPlan(relative, size, action, reason))
        if remote_entries and (
            not has_remote_manifest or remote_content != record.content_sha256
        ):
            items = [
                TransferItemPlan(
                    item.relative_path,
                    item.size_bytes,
                    "conflict",
                    "remote内容ハッシュを確認できません",
                )
                for item in items
            ]
        fingerprint = self._snapshot_plan_token(
            record,
            files,
            target,
            policy,
            remote_entries,
            remote_content,
            has_remote_manifest,
        )
        self.transfer_root.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(self.transfer_root).free
        return TransferPlan(
            token=fingerprint,
            source=str(root),
            destination=target.rclone_value,
            items=tuple(items),
            total_bytes=sum(item.size_bytes for item in items if item.action == "copy"),
            available_bytes=available,
            errors=("remoteに衝突するファイルがあります",)
            if any(item.action == "conflict" for item in items)
            else (),
        )

    def upload_snapshot(
        self,
        snapshot_id: UUID,
        *,
        plan_token: str | None = None,
        overwrite_policy: OverwritePolicy | None = None,
        cancel_token: CancelToken | None = None,
        _job_id: UUID | None = None,
    ) -> UUID:
        plan = self.dry_run_snapshot_upload(
            snapshot_id, overwrite_policy=overwrite_policy
        )
        if plan_token is not None and plan.token != plan_token:
            raise UserFacingError(
                "スナップショットまたは保存先が変更されました。再度ドライランしてください"
            )
        if plan.errors:
            raise UserFacingError(
                "remoteに衝突があるためスナップショットをコピーできません"
            )
        record, root = self._validated_snapshot(snapshot_id)
        project_id = UUID(record.project_id)
        project_settings = self.get_project_storage_settings(project_id)
        policy = overwrite_policy or project_settings.overwrite_policy
        target = self._snapshot_remote_path(project_id, snapshot_id, project_settings)
        files = self._snapshot_files(root)
        with self.session_factory() as session:
            repository = StorageRepository(session)
            job = None
            if _job_id is None:
                job = repository.create_job(
                    project_id=project_id,
                    snapshot_id=snapshot_id,
                    transfer_type=StorageTransferType.SNAPSHOT_UPLOAD,
                    source_kind=StorageKind.LOCAL,
                    destination_kind=StorageKind.REMOTE,
                    item_count=len(files),
                    total_bytes=plan.total_bytes,
                )
            session.commit()
            job_id = _job_id or UUID(job.id)  # type: ignore[union-attr]
        token = cancel_token or CancelToken()
        self._cancel_tokens[job_id] = token
        manifest_items: list[dict[str, Any]] = []
        try:
            self._set_job_running(job_id)
            for index, (relative, path, size, digest) in enumerate(files, start=1):
                self._raise_if_canceled(job_id, token)
                action = next(
                    item.action for item in plan.items if item.relative_path == relative
                )
                with self.session_factory() as session:
                    repository = StorageRepository(session)
                    item_record = repository.add_item_if_missing(
                        job_id=job_id,
                        relative_path=relative,
                        item_type="snapshot_file",
                        direction=TransferDirection.UPLOAD,
                        expected_size=size,
                        source_sha256=digest,
                    )
                    if action == "skip":
                        item_record.status = TransferStatus.COMPLETED.value
                        item_record.verification_status = "size_and_manifest"
                        item_record.transferred_size = size
                        session.commit()
                        manifest_items.append(
                            self._manifest_item(
                                relative, size, digest, "skipped", "size_and_manifest"
                            )
                        )
                        self._update_job(
                            job_id, index, succeeded=0, skipped=1, transferred=size
                        )
                        continue
                result = self.adapter.copy(
                    path,
                    target.child(relative),
                    CopyOptions(
                        overwrite_policy=policy,
                        checksum=self.settings.storage_use_checksum,
                    ),
                    progress_callback=lambda progress: self._progress_job(
                        job_id, progress
                    ),
                    cancel_token=token,
                )
                if result.returncode != 0:
                    self._update_transfer_item(
                        job_id, relative, size, "failed", "not_verified"
                    )
                    manifest_items.append(
                        self._manifest_item(
                            relative,
                            size,
                            digest,
                            "failed",
                            "not_verified",
                        )
                    )
                    self._update_job(
                        job_id,
                        index,
                        succeeded=0,
                        failed=1,
                        skipped=0,
                        transferred=0,
                    )
                    raise UserFacingError("スナップショットの転送に失敗しました")
                self._update_transfer_item(
                    job_id, relative, size, "completed", "size_and_manifest"
                )
                manifest_items.append(
                    self._manifest_item(
                        relative, size, digest, "completed", "size_and_manifest"
                    )
                )
                self._update_job(
                    job_id, index, succeeded=1, skipped=0, transferred=size
                )
            manifest = self._write_transfer_manifest(
                job_id,
                StorageTransferType.SNAPSHOT_UPLOAD,
                project_id,
                snapshot_id,
                target,
                manifest_items,
                TransferStatus.COMPLETED,
                project_settings,
                record.content_sha256,
            )
            self.adapter.copy(
                manifest,
                target.child("transfer-manifest.json"),
                CopyOptions(overwrite_policy=policy, checksum=True),
                cancel_token=token,
            )
            self._verify_remote_snapshot(target, files)
            self._finish_job(job_id, TransferStatus.COMPLETED, manifest)
            return job_id
        except Exception as exc:
            status = (
                TransferStatus.CANCELED if token.cancelled else TransferStatus.FAILED
            )
            manifest = None
            if manifest_items:
                manifest = self._write_transfer_manifest(
                    job_id,
                    StorageTransferType.SNAPSHOT_UPLOAD,
                    project_id,
                    snapshot_id,
                    target,
                    manifest_items,
                    status,
                    project_settings,
                    record.content_sha256,
                )
            self._finish_job(job_id, status, manifest, str(exc))
            if isinstance(exc, UserFacingError):
                raise
            raise UserFacingError("スナップショット転送に失敗しました") from exc
        finally:
            self._cancel_tokens.pop(job_id, None)

    def start_snapshot_upload(
        self, snapshot_id: UUID, plan_token: str | None = None
    ) -> UUID:
        plan = self.dry_run_snapshot_upload(snapshot_id)
        if plan_token is not None and plan.token != plan_token:
            raise UserFacingError(
                "プレビュー内容が変更されています。再度プレビューしてください。"
            )
        if plan.errors:
            raise UserFacingError("ドライランにエラーがあるため転送できません")
        record, _root = self._validated_snapshot(snapshot_id)
        project_id = UUID(record.project_id)
        with self.session_factory() as session:
            job = StorageRepository(session).create_job(
                project_id=project_id,
                snapshot_id=snapshot_id,
                transfer_type=StorageTransferType.SNAPSHOT_UPLOAD,
                source_kind=StorageKind.LOCAL,
                destination_kind=StorageKind.REMOTE,
                item_count=len(plan.items),
                total_bytes=plan.total_bytes,
            )
            session.commit()
            job_id = UUID(job.id)
        future = self._executor.submit(
            self.upload_snapshot,
            snapshot_id,
            plan_token=plan_token,
            _job_id=job_id,
        )
        self._futures[job_id] = future
        return job_id

    def list_jobs(self, project_id: UUID | None = None) -> list[StorageTransferJob]:
        with self.session_factory() as session:
            return StorageRepository(session).list_jobs(project_id)

    def cancel(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            StorageRepository(session).request_cancel(job_id)
            session.commit()
        token = self._cancel_tokens.get(job_id)
        if token:
            token.cancel()

    def recover_stale_jobs(self) -> int:
        count = 0
        try:
            with self.session_factory() as session:
                records = session.scalars(
                    select(StorageTransferJobRecord).where(
                        StorageTransferJobRecord.status == TransferStatus.RUNNING.value
                    )
                ).all()
                for record in records:
                    if record.pid and _pid_exists(record.pid):
                        continue
                    record.status = TransferStatus.STALE.value
                    record.current_step = "stale"
                    record.error_summary = (
                        "アプリ再起動後に転送プロセスを確認できません"
                    )
                    record.completed_at = datetime.now(UTC)
                    record.updated_at = datetime.now(UTC)
                    count += 1
                session.commit()
        except OperationalError:
            return 0
        return count

    def _refresh_model(self, model: ManagedModel) -> StorageEntry:
        root = StorageRemotePath(
            model.remote_name, self.settings.storage_model_remote_root
        )
        entries = self.adapter.list_entries(
            root, ListOptions(recursive=True, page_size=500)
        )
        for entry in entries:
            if entry.remote_path.relative_path == model.remote_relative_path:
                if entry.size_bytes > self.settings.model_max_file_size_bytes:
                    raise UserFacingError("モデルファイルが最大サイズを超えています")
                return entry
        with self.session_factory() as session:
            record = StorageRepository(session).get_model(model.id)
            if record is not None:
                StorageRepository(session).update_model(
                    record, status=ManagedModelStatus.MISSING_REMOTE.value
                )
                session.commit()
        raise UserFacingError("Google Drive上のモデルが見つかりません")

    def _copy_model_with_retry(
        self,
        model_id: UUID,
        transfer_id: UUID,
        job_id: UUID,
        entry: StorageEntry,
        destination: Path,
        token: CancelToken,
    ) -> None:
        part_root = self.transfer_root / str(job_id)
        part_root.mkdir(parents=True, exist_ok=True)
        part = part_root / f"{destination.name}.part"
        attempts = max(1, self.settings.rclone_retries + 1)
        last_error = "モデル転送に失敗しました"
        for attempt in range(1, attempts + 1):
            if token.cancelled:
                raise UserFacingError("モデル転送をキャンセルしました")
            part.unlink(missing_ok=True)
            result = self.adapter.copy(
                entry.remote_path,
                part,
                CopyOptions(checksum=self.settings.storage_use_checksum),
                progress_callback=lambda progress: self._progress_job(job_id, progress),
                cancel_token=token,
            )
            if (
                result.returncode == 0
                and part.is_file()
                and part.stat().st_size == entry.size_bytes
            ):
                digest = _sha256_file(part)
                if (
                    entry.hash_type == "sha256"
                    and entry.hash_value
                    and digest != entry.hash_value
                ):
                    last_error = "remote SHA-256と一致しません"
                    break
                destination.parent.mkdir(parents=True, exist_ok=True)
                part.replace(destination)
                with self.session_factory() as session:
                    repository = StorageRepository(session)
                    record = repository.get_model(model_id)
                    if record is not None:
                        repository.update_model(
                            record,
                            local_path=str(destination),
                            local_size_bytes=destination.stat().st_size,
                            local_sha256=digest,
                            status=ManagedModelStatus.AVAILABLE.value,
                            downloaded_at=datetime.now(UTC),
                            verified_at=datetime.now(UTC),
                            error_summary=None,
                        )
                    transfer = session.scalar(
                        select(ModelTransferRecord).where(
                            ModelTransferRecord.id == str(transfer_id)
                        )
                    )
                    if transfer is not None:
                        transfer.status = TransferStatus.COMPLETED.value
                        transfer.transferred_size_bytes = entry.size_bytes
                        transfer.actual_hash = digest
                        transfer.attempt_count = attempt
                        transfer.completed_at = datetime.now(UTC)
                        transfer.updated_at = datetime.now(UTC)
                        transfer.rclone_exit_code = result.returncode
                    session.commit()
                self._update_job(
                    job_id, 1, succeeded=1, skipped=0, transferred=entry.size_bytes
                )
                self._finish_job(job_id, TransferStatus.COMPLETED, None)
                shutil.rmtree(part_root, ignore_errors=True)
                return
            last_error = result.stderr[-500:] or last_error
            if not _retryable(result.stderr) or attempt == attempts:
                break
        part.unlink(missing_ok=True)
        raise UserFacingError(last_error)

    def _finish_model_failure(
        self,
        model_id: UUID,
        transfer_id: UUID,
        job_id: UUID,
        message: str,
        canceled: bool,
    ) -> None:
        status = ManagedModelStatus.FAILED.value
        with self.session_factory() as session:
            repository = StorageRepository(session)
            record = repository.get_model(model_id)
            if record is not None:
                repository.update_model(
                    record, status=status, error_summary="モデル取得に失敗しました"
                )
            transfer = session.scalar(
                select(ModelTransferRecord).where(
                    ModelTransferRecord.id == str(transfer_id)
                )
            )
            if transfer is not None:
                transfer.status = (
                    TransferStatus.CANCELED.value
                    if canceled
                    else TransferStatus.FAILED.value
                )
                transfer.error_summary = message[:500]
                transfer.completed_at = datetime.now(UTC)
                transfer.updated_at = datetime.now(UTC)
            session.commit()
        self._finish_job(
            job_id,
            TransferStatus.CANCELED if canceled else TransferStatus.FAILED,
            None,
            message,
        )

    def _validated_snapshot(self, snapshot_id: UUID) -> tuple[Any, Path]:
        with self.session_factory() as session:
            record = DatasetRepository(session).get(snapshot_id)
            if record is None:
                raise UserFacingError("対象スナップショットが見つかりません")
            if record.status != DatasetSnapshotStatus.COMPLETED.value:
                raise UserFacingError("completedスナップショットだけを転送できます")
            status = self.datasets.revalidate(snapshot_id)
            if status is not DatasetSnapshotStatus.COMPLETED:
                raise UserFacingError("スナップショットの再検証に失敗しました")
            record = DatasetRepository(session).get(snapshot_id)
            if record is None:
                raise UserFacingError("対象スナップショットが見つかりません")
            return record, Path(record.snapshot_root).resolve()

    def _snapshot_files(self, root: Path) -> list[tuple[str, Path, int, str]]:
        if not root.is_dir():
            raise UserFacingError("スナップショットの保存先がありません")
        files: list[tuple[str, Path, int, str]] = []
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink():
                raise UserFacingError(
                    "スナップショットにシンボリックリンクを含められません"
                )
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise UserFacingError(
                    "スナップショットが保存先の外を参照しています"
                ) from exc
            relative = path.relative_to(root).as_posix()
            files.append((relative, path, path.stat().st_size, _sha256_file(path)))
        if not files:
            raise UserFacingError("スナップショットに転送対象ファイルがありません")
        return files

    def _remote_entries(self, target: StorageRemotePath) -> dict[str, StorageEntry]:
        try:
            return {
                entry.remote_path.relative_path[len(target.relative_path) :].strip(
                    "/"
                ): entry
                for entry in self.adapter.list_entries(
                    target, ListOptions(recursive=True, page_size=10000)
                )
            }
        except (OSError, RuntimeError, ValueError) as exc:
            raise UserFacingError("remoteのファイル一覧を取得できません") from exc

    def _remote_snapshot_content(
        self, target: StorageRemotePath, entries: dict[str, StorageEntry]
    ) -> tuple[str | None, bool]:
        manifest_entry = entries.get("transfer-manifest.json")
        if manifest_entry is None:
            return None, False
        reader = getattr(self.adapter, "read_remote_file", None)
        if not callable(reader):
            return None, True
        try:
            payload = json.loads(reader(target.child("transfer-manifest.json")))
            settings = payload.get("settings", {})
            value = settings.get("snapshot_content_sha256")
            return str(value) if value else None, True
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            return None, True

    def _verify_remote_snapshot(
        self,
        target: StorageRemotePath,
        files: list[tuple[str, Path, int, str]],
    ) -> None:
        entries = self._remote_entries(target)
        for relative, _path, size, _digest in files:
            entry = entries.get(relative)
            if entry is None or entry.size_bytes != size:
                raise UserFacingError("転送後のremoteファイル検証に失敗しました")
        if "transfer-manifest.json" not in entries:
            raise UserFacingError("転送結果マニフェストをremoteで確認できません")

    def _snapshot_remote_path(
        self, project_id: UUID, snapshot_id: UUID, value: ProjectStorageSettings
    ) -> StorageRemotePath:
        root = StorageRemotePath(
            self.settings.storage_remote_name, value.project_remote_root
        )
        return root.child(str(project_id), value.snapshot_remote_root, str(snapshot_id))

    def _validate_project_roots(self, value: ProjectStorageSettings) -> None:
        for root in (
            value.project_remote_root,
            value.snapshot_remote_root,
            value.training_remote_root,
            value.artifact_remote_root,
        ):
            StorageRemotePath(self.settings.storage_remote_name, root)

    def _allowed_extensions(self) -> set[str]:
        values = {
            extension.casefold() for extension in self.settings.model_allowed_extensions
        }
        if self.settings.model_allow_ckpt:
            values.add(".ckpt")
        return values

    def _is_allowed_model(self, name: str) -> bool:
        return Path(name).suffix.casefold() in self._allowed_extensions()

    @staticmethod
    def _model_type(name: str) -> ModelType:
        lowered = name.casefold()
        if "vae" in lowered:
            return ModelType.VAE
        if "checkpoint" in lowered or Path(name).suffix.casefold() == ".ckpt":
            return ModelType.CHECKPOINT
        if Path(name).suffix.casefold() == ".safetensors":
            return ModelType.BASE_MODEL
        return ModelType.UNKNOWN

    def _local_model_path(self, model_id: UUID, filename: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
        return (self.model_cache_root / str(model_id) / safe_name).resolve()

    def _local_matches(self, path: Path, entry: StorageEntry) -> bool:
        return path.stat().st_size == entry.size_bytes and (
            entry.hash_type != "sha256"
            or not entry.hash_value
            or _sha256_file(path) == entry.hash_value
        )

    def _model_plan_token(self, entry: StorageEntry, destination: Path) -> str:
        return _stable_hash(
            {
                "remote": entry.remote_path.rclone_value,
                "size": entry.size_bytes,
                "modified": entry.modified_at.isoformat()
                if entry.modified_at
                else None,
                "hash_type": entry.hash_type,
                "hash": entry.hash_value,
                "destination": str(destination),
                "storage_settings": self._storage_plan_settings(
                    self.settings.storage_overwrite_policy
                ),
            }
        )

    def _snapshot_plan_token(
        self,
        record: Any,
        files: list[tuple[str, Path, int, str]],
        target: StorageRemotePath,
        policy: OverwritePolicy,
        remote_entries: dict[str, StorageEntry],
        remote_content: str | None,
        has_remote_manifest: bool,
    ) -> str:
        return _stable_hash(
            {
                "snapshot_id": record.id,
                "content_sha256": record.content_sha256,
                "target": target.rclone_value,
                "policy": policy.value,
                "remote_content": remote_content,
                "has_remote_manifest": has_remote_manifest,
                "remote_entries": [
                    (
                        relative,
                        entry.size_bytes,
                        entry.modified_at.isoformat() if entry.modified_at else None,
                        entry.hash_type,
                        entry.hash_value,
                    )
                    for relative, entry in sorted(remote_entries.items())
                ],
                "storage_settings": self._storage_plan_settings(policy),
                "files": [
                    (relative, size, digest) for relative, _path, size, digest in files
                ],
            }
        )

    def _rclone_version(self) -> str | None:
        result = self.adapter.validate_environment()
        return result.rclone_version

    def _storage_plan_settings(self, policy: OverwritePolicy) -> dict[str, Any]:
        return {
            "rclone_executable": self.settings.rclone_executable,
            "rclone_config_path": str(self.settings.rclone_config_path)
            if self.settings.rclone_config_path
            else None,
            "rclone_transfers": self.settings.rclone_transfers,
            "rclone_checkers": self.settings.rclone_checkers,
            "rclone_retries": self.settings.rclone_retries,
            "rclone_low_level_retries": self.settings.rclone_low_level_retries,
            "rclone_use_checksum": self.settings.storage_use_checksum,
            "verification_policy": self.settings.storage_verification_policy.value,
            "overwrite_policy": policy.value,
            "remote_hash_fallback": self.settings.storage_remote_hash_fallback,
        }

    def _settings_snapshot(self) -> dict[str, Any]:
        return {
            "rclone_transfers": self.settings.rclone_transfers,
            "rclone_checkers": self.settings.rclone_checkers,
            "rclone_retries": self.settings.rclone_retries,
            "rclone_low_level_retries": self.settings.rclone_low_level_retries,
            "rclone_use_checksum": self.settings.storage_use_checksum,
        }

    def _set_job_running(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.status = TransferStatus.RUNNING.value
                record.current_step = "transferring"
                record.started_at = datetime.now(UTC)
                record.pid = os.getpid()
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _raise_if_canceled(self, job_id: UUID, token: CancelToken) -> None:
        if token.cancelled:
            raise UserFacingError("転送をキャンセルしました")
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record and record.cancel_requested:
                token.cancel()
                raise UserFacingError("転送をキャンセルしました")

    def _progress_job(self, job_id: UUID, progress: TransferProgress) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.transferred_bytes = progress.bytes_transferred
                record.current_step = progress.current_path or "transferring"
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _update_job(
        self,
        job_id: UUID,
        processed: int,
        *,
        succeeded: int,
        failed: int = 0,
        skipped: int,
        transferred: int,
    ) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.processed_item_count = processed
                record.succeeded_item_count += succeeded
                record.failed_item_count += failed
                record.skipped_item_count += skipped
                record.transferred_bytes += transferred
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _update_transfer_item(
        self,
        job_id: UUID,
        relative: str,
        size: int,
        status: str,
        verification: str,
    ) -> None:
        with self.session_factory() as session:
            item = session.scalar(
                select(TransferItemRecord).where(
                    TransferItemRecord.transfer_job_id == str(job_id),
                    TransferItemRecord.relative_path == relative,
                )
            )
            if item is not None:
                item.status = status
                item.transferred_size = size
                item.verification_status = verification
                session.commit()

    def _finish_job(
        self,
        job_id: UUID,
        status: TransferStatus,
        manifest: Path | None,
        error: str | None = None,
    ) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.status = status.value
                record.current_step = status.value
                record.completed_at = datetime.now(UTC)
                record.error_summary = error[:500] if error else None
                record.manifest_path = str(manifest) if manifest else None
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _write_transfer_manifest(
        self,
        job_id: UUID,
        transfer_type: StorageTransferType,
        project_id: UUID,
        snapshot_id: UUID,
        target: StorageRemotePath,
        items: list[dict[str, Any]],
        status: TransferStatus,
        settings: ProjectStorageSettings,
        snapshot_content_sha256: str | None = None,
    ) -> Path:
        path = self.transfer_root / "manifests" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = TransferManifest(
            "phase5-transfer-v1",
            job_id,
            transfer_type,
            project_id,
            snapshot_id,
            None,
            "local",
            target.rclone_value,
            None,
            datetime.now(UTC),
            self._rclone_version(),
            {
                "overwrite_policy": settings.overwrite_policy.value,
                "verification_policy": settings.verification_policy.value,
                "snapshot_content_sha256": snapshot_content_sha256,
            },
            len(items),
            sum(item["transfer_status"] == "completed" for item in items),
            sum(item["transfer_status"] == "failed" for item in items),
            sum(item["transfer_status"] == "skipped" for item in items),
            sum(int(item["size"]) for item in items),
            sum(
                int(item["size"])
                for item in items
                if item["transfer_status"] == "completed"
            ),
            settings.verification_policy,
            status,
            tuple(items),
        )
        path.write_text(
            json.dumps(
                _manifest_dict(payload), ensure_ascii=False, sort_keys=True, default=str
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _manifest_item(
        relative: str, size: int, digest: str, status: str, verification: str
    ) -> dict[str, Any]:
        return {
            "relative_path": relative,
            "size": size,
            "local_sha256": digest,
            "remote_hash_type": None,
            "remote_hash": None,
            "transfer_status": status,
            "verification_status": verification,
            "retry_count": 0,
            "error_summary": None,
        }


def _manifest_dict(manifest: TransferManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "transfer_job_id": str(manifest.transfer_job_id),
        "transfer_type": manifest.transfer_type.value,
        "project_id": str(manifest.project_id) if manifest.project_id else None,
        "snapshot_id": str(manifest.snapshot_id) if manifest.snapshot_id else None,
        "managed_model_id": str(manifest.managed_model_id)
        if manifest.managed_model_id
        else None,
        "source": "local",
        "destination": manifest.destination,
        "started_at": manifest.started_at.isoformat() if manifest.started_at else None,
        "completed_at": manifest.completed_at.isoformat()
        if manifest.completed_at
        else None,
        "rclone_version": manifest.rclone_version,
        "settings": manifest.settings,
        "item_count": manifest.item_count,
        "success_count": manifest.success_count,
        "failure_count": manifest.failure_count,
        "skipped_count": manifest.skipped_count,
        "total_bytes": manifest.total_bytes,
        "transferred_bytes": manifest.transferred_bytes,
        "verification_level": manifest.verification_level.value,
        "status": manifest.status.value,
        "items": list(manifest.items),
        "error_summary": manifest.error_summary,
        "application_version": "0.1.0",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _retryable(message: str) -> bool:
    lowered = message.casefold()
    permanent = (
        "auth",
        "permission",
        "forbidden",
        "not found",
        "no such",
        "checksum",
        "hash mismatch",
        "invalid",
    )
    transient = ("timeout", "temporar", "rate", "reset", "connection", "503", "429")
    return any(value in lowered for value in transient) and not any(
        value in lowered for value in permanent
    )


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True
