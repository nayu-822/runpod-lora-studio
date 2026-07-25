from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

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
    VerificationPolicy,
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


class RemoteModelChangedError(UserFacingError):
    """The remote model changed while a local copy was in progress."""


class StorageService:
    def __init__(
        self,
        settings: AppSettings,
        adapter: StorageTransferAdapter | None = None,
        datasets: DatasetSnapshotService | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter or RcloneAdapter(settings)
        self.datasets = datasets or DatasetSnapshotService(settings)
        self.session_factory = create_session_factory(settings)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="storage")
        self._futures: dict[UUID, Future[Any]] = {}
        self._cancel_tokens: dict[UUID, CancelToken] = {}
        self._worker_ids: dict[UUID, str] = {}
        self._heartbeat_stops: dict[UUID, threading.Event] = {}
        self._sleeper = sleeper

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
        preserve_existing = self._saved_local_matches(model, destination)
        expected_token = self._model_plan_token(entry, destination)
        if plan_token is not None and plan_token != expected_token:
            raise UserFacingError(
                "モデル情報が変更されました。再度ドライランしてください"
            )
        if destination.is_file() and self._local_matches(destination, entry, model):
            with self.session_factory() as session:
                record = StorageRepository(session).get_model(model_id)
                if record is not None:
                    StorageRepository(session).update_model(
                        record,
                        local_path=str(destination),
                        local_size_bytes=destination.stat().st_size,
                        local_sha256=_sha256_file(destination),
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
            if not preserve_existing:
                repository.update_model(
                    record,
                    status=ManagedModelStatus.DOWNLOADING.value,
                    error_summary=None,
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
                model_id,
                transfer_id,
                job_id,
                entry,
                destination,
                token,
            )
            return model_id
        except Exception as exc:
            self._finish_model_failure(
                model_id,
                transfer_id,
                job_id,
                str(exc),
                token.cancelled,
                preserve_existing and not isinstance(exc, RemoteModelChangedError),
                isinstance(exc, RemoteModelChangedError),
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
                "モデル情報が変更されました。再度ドライランしてください"
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
            with self.session_factory() as session:
                record = StorageRepository(session).get_model(model_id)
                if record is not None:
                    StorageRepository(session).update_model(
                        record,
                        status=ManagedModelStatus.VERIFICATION_FAILED.value,
                        verified_at=None,
                        error_summary="ローカルモデルが見つかりません",
                    )
                    session.commit()
            return False
        try:
            remote = self._refresh_model(model)
        except UserFacingError:
            remote = None
        actual_hash = _sha256_file(model.local_path)
        ok = (
            remote is not None
            and model.local_path.stat().st_size == remote.size_bytes
            and (
                not remote.hash_type
                or not remote.hash_value
                or _hash_matches(model.local_path, remote.hash_type, remote.hash_value)
            )
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
        remote_content, has_remote_manifest, remote_manifest_items = (
            self._remote_snapshot_content(target, remote_entries)
        )
        items: list[TransferItemPlan] = []
        for relative, path, size, _digest in files:
            remote = remote_entries.get(relative)
            action = "copy"
            reason = ""
            if remote is not None:
                identical = _remote_file_identical(remote, path)
                if not identical and not remote.hash_type and not remote.hash_value:
                    identical, _fallback_status = self._fallback_identity(
                        remote, size, remote_manifest_items.get(relative)
                    )
                if policy is OverwritePolicy.FAIL_IF_EXISTS:
                    action, reason = "conflict", "remoteに同名ファイルがあります"
                elif identical:
                    action, reason = "skip", "サイズとハッシュ一致"
                elif policy is OverwritePolicy.COPY_MISSING:
                    action, reason = "conflict", "remoteに同名ファイルがあります"
                elif policy is OverwritePolicy.OVERWRITE_CHANGED:
                    action, reason = "copy", "変更されたremoteを明示的に上書き"
                elif policy is OverwritePolicy.SKIP_IDENTICAL:
                    action, reason = "conflict", "remote内容を同一と確認できません"
            items.append(TransferItemPlan(relative, size, action, reason))
        if (
            remote_entries
            and has_remote_manifest
            and remote_content != record.content_sha256
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
        verification_phase = 0
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
                        item_record.verification_status = "remote_hash_and_size"
                        item_record.transferred_size = size
                        session.commit()
                        manifest_items.append(
                            self._manifest_item(
                                relative,
                                size,
                                digest,
                                "skipped",
                                "remote_hash_and_size",
                            )
                        )
                        self._update_job(
                            job_id, index, succeeded=0, skipped=1, transferred=0
                        )
                        continue
                self._set_current_file(job_id, size)
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
                    process_callback=lambda pid: self._set_rclone_pid(job_id, pid),
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
                    job_id, relative, size, "completed", "not_verified"
                )
                manifest_items.append(
                    self._manifest_item(
                        relative, size, digest, "completed", "not_verified"
                    )
                )
                self._complete_current_file(job_id, size)
                self._update_job(job_id, index, succeeded=1, skipped=0, transferred=0)
            if (
                project_settings.verification_policy
                is VerificationPolicy.SIZE_AND_MANIFEST
            ):
                verification_statuses = {
                    relative: "not_verified"
                    for relative, _path, _size, _digest in files
                }
            else:
                verification_phase = 1
                verification_statuses = self._remote_verification_statuses(
                    target, files, project_settings.verification_policy
                )
            verified_entries = self._remote_entries(target)
            for item in manifest_items:
                verification = verification_statuses[item["relative_path"]]
                item["verification_status"] = verification
                remote_entry = verified_entries.get(item["relative_path"])
                if remote_entry is not None:
                    item["remote_hash_type"] = remote_entry.hash_type
                    item["remote_hash"] = remote_entry.hash_value
                    item["remote_size"] = remote_entry.size_bytes
                    item["remote_modified_at"] = (
                        remote_entry.modified_at.isoformat()
                        if remote_entry.modified_at is not None
                        else None
                    )
                self._update_transfer_item(
                    job_id,
                    item["relative_path"],
                    int(item["size"]),
                    str(item["transfer_status"]),
                    verification,
                )
            verification_phase = 0
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
            manifest_result = self.adapter.copy(
                manifest,
                target.child("transfer-manifest.json"),
                CopyOptions(overwrite_policy=policy, checksum=True),
                cancel_token=token,
                process_callback=lambda pid: self._set_rclone_pid(job_id, pid),
            )
            if manifest_result.returncode != 0:
                raise UserFacingError("転送結果マニフェストの転送に失敗しました")
            verification_phase = 2
            verified_statuses = self._verify_remote_snapshot(
                target,
                files,
                record.content_sha256,
                project_settings.verification_policy,
            )
            if (
                project_settings.verification_policy
                is VerificationPolicy.SIZE_AND_MANIFEST
            ):
                for item in manifest_items:
                    verification = verified_statuses[item["relative_path"]]
                    item["verification_status"] = verification
                    self._update_transfer_item(
                        job_id,
                        item["relative_path"],
                        int(item["size"]),
                        str(item["transfer_status"]),
                        verification,
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
                final_manifest_result = self.adapter.copy(
                    manifest,
                    target.child("transfer-manifest.json"),
                    CopyOptions(overwrite_policy=policy, checksum=True),
                    cancel_token=token,
                    process_callback=lambda pid: self._set_rclone_pid(job_id, pid),
                )
                if final_manifest_result.returncode != 0:
                    raise UserFacingError("転送マニフェストの確定に失敗しました")
            self._validate_final_remote_manifest(
                target,
                manifest,
                job_id,
                project_id,
                snapshot_id,
                manifest_items,
                _manifest_verification_level(
                    manifest_items, project_settings.verification_policy
                ),
                record.content_sha256,
            )
            verification_phase = 0
            self._finish_job(job_id, TransferStatus.COMPLETED, manifest)
            return job_id
        except Exception as exc:
            status = (
                TransferStatus.CANCELED if token.cancelled else TransferStatus.FAILED
            )
            if verification_phase:
                for item in manifest_items:
                    item["verification_status"] = "verification_failed"
                    self._update_transfer_item(
                        job_id,
                        item["relative_path"],
                        int(item["size"]),
                        str(item["transfer_status"]),
                        "verification_failed",
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
                cutoff = datetime.now(UTC) - timedelta(
                    seconds=self.settings.storage_job_stale_after_seconds
                )
                for record in records:
                    job_id = UUID(record.id)
                    heartbeat = record.heartbeat_at
                    if heartbeat is not None and heartbeat.tzinfo is None:
                        heartbeat = heartbeat.replace(tzinfo=UTC)
                    if heartbeat is not None and heartbeat >= cutoff:
                        continue
                    future = self._futures.get(job_id)
                    worker_matches = (
                        record.worker_id is not None
                        and self._worker_ids.get(job_id) == record.worker_id
                    )
                    future_running = future is not None and not future.done()
                    process_running = (
                        record.pid is not None
                        and _pid_exists(record.pid)
                        and _is_expected_rclone_process(
                            record.pid, self.settings.rclone_executable
                        )
                    )
                    if process_running or (worker_matches and future_running):
                        continue
                    record.status = TransferStatus.STALE.value
                    record.current_step = "stale"
                    record.error_summary = (
                        "アプリ再起動後に転送プロセスを確認できません"
                    )
                    record.completed_at = datetime.now(UTC)
                    record.pid = None
                    stop = self._heartbeat_stops.pop(job_id, None)
                    if stop is not None:
                        stop.set()
                    record.heartbeat_at = datetime.now(UTC)
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

    def _fetch_remote_entry(self, remote_path: StorageRemotePath) -> StorageEntry:
        root = StorageRemotePath(
            remote_path.remote_name, self.settings.storage_model_remote_root
        )
        entries = self.adapter.list_entries(
            root, ListOptions(recursive=True, page_size=500)
        )
        for entry in entries:
            if entry.remote_path.relative_path == remote_path.relative_path:
                return entry
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
        self._set_current_file(job_id, entry.size_bytes)
        attempts = max(1, self.settings.rclone_retries + 1)
        last_error = "モデル転送に失敗しました"
        for attempt in range(1, attempts + 1):
            if token.cancelled:
                raise UserFacingError("モデル転送をキャンセルしました")
            part.unlink(missing_ok=True)
            self._update_transfer_attempt(job_id, transfer_id, attempt, 0, None)
            result = self.adapter.copy(
                entry.remote_path,
                part,
                CopyOptions(checksum=self.settings.storage_use_checksum),
                progress_callback=lambda progress: self._progress_job(job_id, progress),
                cancel_token=token,
                process_callback=lambda pid: self._set_rclone_pid(job_id, pid),
            )
            if (
                result.returncode == 0
                and part.is_file()
                and part.stat().st_size == entry.size_bytes
            ):
                digest = _sha256_file(part)
                if entry.hash_type and entry.hash_value:
                    if not _hash_matches(part, entry.hash_type, entry.hash_value):
                        last_error = "remoteハッシュと一致しません"
                        self._update_transfer_attempt(
                            job_id,
                            transfer_id,
                            attempt,
                            0,
                            "checksum",
                        )
                        break
                try:
                    current_entry = self._fetch_remote_entry(entry.remote_path)
                except UserFacingError as exc:
                    part.unlink(missing_ok=True)
                    raise RemoteModelChangedError(
                        "転送中にremoteモデルが変更されました。再度ドライランしてください"
                    ) from exc
                if _remote_signature(current_entry) != _remote_signature(entry):
                    part.unlink(missing_ok=True)
                    raise RemoteModelChangedError(
                        "転送中にremoteモデルが変更されました。再度ドライランしてください"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = destination.with_name(f".{destination.name}.{job_id}.backup")
                had_destination = destination.is_file()
                if had_destination:
                    backup.unlink(missing_ok=True)
                    destination.replace(backup)
                try:
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
                except Exception:
                    destination.unlink(missing_ok=True)
                    if backup.is_file():
                        backup.replace(destination)
                    raise
                backup.unlink(missing_ok=True)
                self._complete_current_file(job_id, entry.size_bytes)
                self._update_job(job_id, 1, succeeded=1, skipped=0, transferred=0)
                self._finish_job(job_id, TransferStatus.COMPLETED, None)
                shutil.rmtree(part_root, ignore_errors=True)
                return
            last_error = result.stderr[-500:] or last_error
            classification = _error_classification(
                result.stderr, timed_out=result.timed_out
            )
            delay = min(
                self.settings.storage_retry_max_backoff_seconds,
                self.settings.rclone_retry_interval_seconds * (2 ** (attempt - 1)),
            )
            self._update_transfer_attempt(
                job_id,
                transfer_id,
                attempt,
                max(0, attempt - 1),
                classification,
                delay if attempt < attempts else 0.0,
                result.returncode,
            )
            if (
                not _retryable(result.stderr, timed_out=result.timed_out)
                or attempt == attempts
            ):
                break
            self._sleep_before_retry(delay, token)
        part.unlink(missing_ok=True)
        raise UserFacingError(last_error)

    def _finish_model_failure(
        self,
        model_id: UUID,
        transfer_id: UUID,
        job_id: UUID,
        message: str,
        canceled: bool,
        preserve_existing: bool,
        verification_failed: bool,
    ) -> None:
        status = (
            ManagedModelStatus.VERIFICATION_FAILED.value
            if verification_failed
            else (
                ManagedModelStatus.AVAILABLE.value
                if preserve_existing
                else ManagedModelStatus.FAILED.value
            )
        )
        with self.session_factory() as session:
            repository = StorageRepository(session)
            record = repository.get_model(model_id)
            if record is not None:
                repository.update_model(
                    record,
                    status=status,
                    error_summary=(
                        message[:500]
                        if verification_failed
                        else (None if preserve_existing else "モデル取得に失敗しました")
                    ),
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
    ) -> tuple[str | None, bool, dict[str, dict[str, Any]]]:
        manifest_entry = entries.get("transfer-manifest.json")
        if manifest_entry is None:
            return None, False, {}
        reader = getattr(self.adapter, "read_remote_file", None)
        if not callable(reader):
            return None, True, {}
        try:
            payload = json.loads(reader(target.child("transfer-manifest.json")))
            settings = payload.get("settings", {})
            value = settings.get("snapshot_content_sha256")
            manifest_items = {
                str(item["relative_path"]): dict(item)
                for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("relative_path")
            }
            return str(value) if value else None, True, manifest_items
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            return None, True, {}

    def _fallback_identity(
        self,
        entry: StorageEntry,
        local_size: int,
        manifest_item: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        fallback = self.settings.storage_remote_hash_fallback
        if fallback == "size_and_manifest":
            return (
                manifest_item is not None
                and self._manifest_metadata_matches(entry, local_size, manifest_item),
                "manifest_metadata_and_size",
            )
        return False, "not_verified"

    def _fallback_verification(
        self,
        entry: StorageEntry,
        local_size: int,
        manifest_item: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if self.settings.storage_remote_hash_fallback == "existence_only":
            return True, "existence_only"
        return self._fallback_identity(entry, local_size, manifest_item)

    def _verify_remote_snapshot(
        self,
        target: StorageRemotePath,
        files: list[tuple[str, Path, int, str]],
        expected_content_sha256: str,
        verification_policy: VerificationPolicy,
    ) -> dict[str, str]:
        entries = self._remote_entries(target)
        content, has_manifest, _manifest_items = self._remote_snapshot_content(
            target, entries
        )
        if not has_manifest or content is None:
            raise UserFacingError("転送結果マニフェストをremoteで確認できません")
        statuses = self._remote_verification_statuses(
            target, files, verification_policy, expected_content_sha256
        )
        if content != expected_content_sha256:
            raise UserFacingError("snapshot content SHA-256が一致しません")

        return statuses

    def _validate_final_remote_manifest(
        self,
        target: StorageRemotePath,
        local_manifest: Path,
        job_id: UUID,
        project_id: UUID,
        snapshot_id: UUID,
        expected_items: list[dict[str, Any]],
        expected_verification_level: VerificationPolicy | str,
        expected_content_sha256: str,
    ) -> None:
        """Validate the final manifest after it has been copied to the remote."""
        try:
            local_payload = json.loads(local_manifest.read_text(encoding="utf-8"))
            if not isinstance(local_payload, dict):
                raise ValueError("local manifest is not an object")

            entries = self._remote_entries(target)
            remote_entry = entries.get("transfer-manifest.json")
            reader = getattr(self.adapter, "read_remote_file", None)
            if remote_entry is None or not callable(reader):
                raise ValueError("remote final manifest is missing")
            remote_payload = json.loads(reader(target.child("transfer-manifest.json")))
            if not isinstance(remote_payload, dict):
                raise ValueError("remote manifest is not an object")

            expected_level = (
                expected_verification_level.value
                if isinstance(expected_verification_level, VerificationPolicy)
                else expected_verification_level
            )
            if (
                remote_payload.get("schema_version") != "phase5-transfer-v1"
                or remote_payload.get("transfer_job_id") != str(job_id)
                or remote_payload.get("transfer_type")
                != StorageTransferType.SNAPSHOT_UPLOAD.value
                or remote_payload.get("project_id") != str(project_id)
                or remote_payload.get("snapshot_id") != str(snapshot_id)
                or remote_payload.get("status") != TransferStatus.COMPLETED.value
                or remote_payload.get("verification_level") != expected_level
            ):
                raise ValueError("remote manifest metadata does not match")

            settings = remote_payload.get("settings")
            if (
                not isinstance(settings, dict)
                or settings.get("snapshot_content_sha256") != expected_content_sha256
            ):
                raise ValueError("remote snapshot content does not match")

            remote_items = remote_payload.get("items")
            if not isinstance(remote_items, list):
                raise ValueError("remote manifest items are invalid")
            expected_by_path = {
                str(item["relative_path"]): item for item in expected_items
            }
            actual_by_path = {
                str(item["relative_path"]): item
                for item in remote_items
                if isinstance(item, dict) and item.get("relative_path")
            }
            if (
                len(expected_by_path) != len(expected_items)
                or len(actual_by_path) != len(remote_items)
                or set(expected_by_path) != set(actual_by_path)
                or remote_payload.get("item_count") != len(expected_items)
            ):
                raise ValueError("remote manifest item set does not match")
            for relative, expected in expected_by_path.items():
                actual = actual_by_path[relative]
                for field in (
                    "transfer_status",
                    "verification_status",
                    "size",
                    "local_sha256",
                ):
                    if actual.get(field) != expected.get(field):
                        raise ValueError("remote manifest item does not match")

            local_bytes = local_manifest.read_bytes()
            if remote_entry.size_bytes != len(local_bytes):
                raise ValueError("remote manifest size does not match")
            if remote_entry.hash_type and remote_entry.hash_value:
                if not _hash_matches(
                    local_manifest, remote_entry.hash_type, remote_entry.hash_value
                ):
                    raise ValueError("remote manifest hash does not match")
            if _canonical_json(local_payload) != _canonical_json(remote_payload):
                raise ValueError("remote manifest content does not match")
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            UserFacingError,
            json.JSONDecodeError,
        ) as exc:
            raise UserFacingError(
                "最終転送マニフェストを確認できません"
                "（仮manifestがremoteに残っている可能性があります）"
            ) from exc

    def _legacy_remote_verification_statuses(
        self,
        target: StorageRemotePath,
        files: list[tuple[str, Path, int, str]],
        verification_policy: VerificationPolicy,
    ) -> dict[str, str]:
        entries = self._remote_entries(target)
        statuses: dict[str, str] = {}
        for relative, path, size, _digest in files:
            entry = entries.get(relative)
            if entry is None:
                raise UserFacingError("転送後のremoteファイル検証に失敗しました")
            if verification_policy is VerificationPolicy.EXISTENCE_ONLY:
                statuses[relative] = "existence_only"
                continue
            if entry.size_bytes != size:
                raise UserFacingError("転送後のremoteファイル検証に失敗しました")
            if not entry.hash_type or not entry.hash_value:
                raise UserFacingError("remoteハッシュを取得できないため検証できません")
            if not _remote_file_identical(entry, path):
                raise UserFacingError("転送後のremoteハッシュ検証に失敗しました")
            statuses[relative] = "remote_hash_and_size"
        return statuses

    def _remote_verification_statuses(
        self,
        target: StorageRemotePath,
        files: list[tuple[str, Path, int, str]],
        verification_policy: VerificationPolicy,
        expected_content_sha256: str | None = None,
    ) -> dict[str, str]:
        entries = self._remote_entries(target)
        content, has_manifest, manifest_items = self._remote_snapshot_content(
            target, entries
        )
        statuses: dict[str, str] = {}
        for relative, path, size, digest in files:
            entry = entries.get(relative)
            if entry is None:
                raise UserFacingError("remoteファイル検証に失敗しました")
            if verification_policy is VerificationPolicy.EXISTENCE_ONLY:
                statuses[relative] = "existence_only"
                continue
            manifest_item = manifest_items.get(relative)
            if verification_policy is VerificationPolicy.SIZE_AND_MANIFEST:
                if (
                    not has_manifest
                    or content is None
                    or (
                        expected_content_sha256 is not None
                        and content != expected_content_sha256
                    )
                ):
                    raise UserFacingError("有効なtransfer manifestを確認できません")
                if not self._manifest_metadata_matches(entry, size, manifest_item):
                    raise UserFacingError("remoteメタデータ検証に失敗しました")
                statuses[relative] = "manifest_metadata_and_size"
                continue
            if entry.hash_type and entry.hash_value:
                if verification_policy is VerificationPolicy.FULL_CHECKSUM:
                    if _hash_algorithm(entry.hash_type) != "sha256":
                        raise UserFacingError(
                            "full checksumにはremote SHA-256が必要です"
                        )
                    if (
                        entry.size_bytes != size
                        or not _hash_matches(path, entry.hash_type, entry.hash_value)
                        or _sha256_file(path) != digest
                    ):
                        raise UserFacingError("remote SHA-256検証に失敗しました")
                    statuses[relative] = "full_checksum"
                    continue
                if entry.size_bytes != size or not _remote_file_identical(entry, path):
                    raise UserFacingError("remoteハッシュ検証に失敗しました")
                statuses[relative] = "remote_hash_and_size"
                continue
            identical, fallback_status = self._fallback_verification(
                entry, size, manifest_item
            )
            if not identical:
                raise UserFacingError("remote hashを取得できないため検証できません")
            statuses[relative] = fallback_status
        return statuses

    @staticmethod
    def _manifest_metadata_matches(
        entry: StorageEntry,
        local_size: int,
        manifest_item: dict[str, Any] | None,
    ) -> bool:
        if entry.size_bytes != local_size or entry.modified_at is None:
            return False
        if manifest_item is None:
            return False
        recorded_size = manifest_item.get("remote_size")
        recorded_modified = manifest_item.get("remote_modified_at")
        if recorded_size is None or not recorded_modified:
            return False
        try:
            expected_modified = datetime.fromisoformat(str(recorded_modified))
        except ValueError:
            return False
        if expected_modified.tzinfo is None:
            expected_modified = expected_modified.replace(tzinfo=UTC)
        actual_modified = entry.modified_at
        if actual_modified.tzinfo is None:
            actual_modified = actual_modified.replace(tzinfo=UTC)
        if (
            entry.size_bytes != int(recorded_size)
            or actual_modified != expected_modified
        ):
            return False
        recorded_hash_type = manifest_item.get("remote_hash_type")
        recorded_hash = manifest_item.get("remote_hash")
        current_hash_type = entry.hash_type
        current_hash = entry.hash_value
        if current_hash_type is not None or current_hash is not None:
            return (
                recorded_hash_type is not None
                and recorded_hash is not None
                and current_hash_type is not None
                and current_hash is not None
                and str(recorded_hash_type).casefold() == current_hash_type.casefold()
                and str(recorded_hash).casefold() == current_hash.casefold()
            )
        return True

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

    @staticmethod
    def _saved_local_matches(model: ManagedModel, destination: Path) -> bool:
        return (
            destination.is_file()
            and model.local_path == destination
            and model.local_sha256 is not None
            and _sha256_file(destination) == model.local_sha256
        )

    def _local_matches(
        self, path: Path, entry: StorageEntry, model: ManagedModel
    ) -> bool:
        if (
            model.remote_name != entry.remote_path.remote_name
            or model.remote_relative_path != entry.remote_path.relative_path
            or model.remote_size_bytes != entry.size_bytes
            or model.remote_modified_at != entry.modified_at
            or model.remote_hash_type != entry.hash_type
            or model.remote_hash_value != entry.hash_value
        ):
            return False
        if path.stat().st_size != entry.size_bytes:
            return False
        if model.local_sha256 is not None:
            if _sha256_file(path) != model.local_sha256:
                return False
            return (
                not entry.hash_type
                or not entry.hash_value
                or _hash_matches(path, entry.hash_type, entry.hash_value)
            )
        return bool(
            entry.hash_type
            and entry.hash_value
            and _hash_matches(path, entry.hash_type, entry.hash_value)
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
            "retry_max_backoff_seconds": (
                self.settings.storage_retry_max_backoff_seconds
            ),
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
        worker_id = str(uuid4())
        self._worker_ids[job_id] = worker_id
        stop = threading.Event()
        old_stop = self._heartbeat_stops.pop(job_id, None)
        if old_stop is not None:
            old_stop.set()
        self._heartbeat_stops[job_id] = stop
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.status = TransferStatus.RUNNING.value
                record.current_step = "transferring"
                record.started_at = datetime.now(UTC)
                record.worker_id = worker_id
                record.heartbeat_at = datetime.now(UTC)
                record.pid = None
                record.updated_at = datetime.now(UTC)
                session.commit()
        threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, worker_id, stop),
            name=f"storage-heartbeat-{job_id}",
            daemon=True,
        ).start()

    def _heartbeat_loop(
        self, job_id: UUID, worker_id: str, stop: threading.Event
    ) -> None:
        interval = min(
            self.settings.transfer_progress_interval_seconds,
            max(0.2, self.settings.storage_job_stale_after_seconds / 3),
        )
        while not stop.wait(interval):
            self._touch_heartbeat(job_id, worker_id)

    def _touch_heartbeat(self, job_id: UUID, worker_id: str) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if (
                record is None
                or record.status != TransferStatus.RUNNING.value
                or record.worker_id != worker_id
            ):
                stop = self._heartbeat_stops.pop(job_id, None)
                if stop is not None:
                    stop.set()
                return
            record.heartbeat_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)
            session.commit()

    def _set_rclone_pid(self, job_id: UUID, pid: int | None) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None and record.status == TransferStatus.RUNNING.value:
                record.pid = pid
                record.heartbeat_at = datetime.now(UTC)
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
                record.current_file_transferred_bytes = max(
                    record.current_file_transferred_bytes,
                    progress.bytes_transferred,
                )
                record.transferred_bytes = (
                    record.completed_transferred_bytes
                    + record.current_file_transferred_bytes
                )
                record.current_step = progress.current_path or "transferring"
                record.heartbeat_at = datetime.now(UTC)
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _set_current_file(self, job_id: UUID, total_bytes: int) -> None:
        del total_bytes
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.current_file_transferred_bytes = 0
                record.transferred_bytes = record.completed_transferred_bytes
                record.heartbeat_at = datetime.now(UTC)
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _complete_current_file(self, job_id: UUID, size: int) -> None:
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.completed_transferred_bytes += size
                record.current_file_transferred_bytes = 0
                record.transferred_bytes = record.completed_transferred_bytes
                record.heartbeat_at = datetime.now(UTC)
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
                record.transferred_bytes = (
                    record.completed_transferred_bytes
                    + record.current_file_transferred_bytes
                )
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _update_transfer_attempt(
        self,
        job_id: UUID,
        transfer_id: UUID,
        attempt: int,
        retry_count: int,
        classification: str | None,
        backoff_seconds: float = 0.0,
        exit_code: int | None = None,
    ) -> None:
        with self.session_factory() as session:
            transfer = session.scalar(
                select(ModelTransferRecord).where(
                    ModelTransferRecord.id == str(transfer_id)
                )
            )
            if transfer is not None:
                transfer.status = TransferStatus.RUNNING.value
                transfer.attempt_count = attempt
                transfer.retry_count = retry_count
                if exit_code is not None:
                    transfer.rclone_exit_code = exit_code
                transfer.started_at = transfer.started_at or datetime.now(UTC)
                if classification:
                    transfer.error_summary = (
                        f"attempt={attempt}; classification={classification}; "
                        f"backoff_seconds={backoff_seconds:.3f}"
                    )
                transfer.updated_at = datetime.now(UTC)
            job = StorageRepository(session).get_job(job_id)
            if job is not None:
                job.current_step = (
                    f"retry attempt={attempt} "
                    f"classification={classification or 'none'} "
                    f"backoff={backoff_seconds:.3f}"
                )
                job.heartbeat_at = datetime.now(UTC)
                job.updated_at = datetime.now(UTC)
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
        self._worker_ids.pop(job_id, None)
        stop = self._heartbeat_stops.pop(job_id, None)
        if stop is not None:
            stop.set()
        with self.session_factory() as session:
            record = StorageRepository(session).get_job(job_id)
            if record is not None:
                record.status = status.value
                record.current_step = status.value
                record.completed_at = datetime.now(UTC)
                record.error_summary = error[:500] if error else None
                record.manifest_path = str(manifest) if manifest else None
                record.pid = None
                record.heartbeat_at = datetime.now(UTC)
                record.updated_at = datetime.now(UTC)
                session.commit()

    def _sleep_before_retry(self, delay: float, token: CancelToken) -> None:
        if delay <= 0:
            return
        if self._sleeper is not None:
            if token.cancelled:
                raise UserFacingError("転送をキャンセルしました")
            self._sleeper(delay)
            if token.cancelled:
                raise UserFacingError("転送をキャンセルしました")
            return
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if token.cancelled:
                raise UserFacingError("転送をキャンセルしました")
            time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))

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
        verification_level = _manifest_verification_level(
            items, settings.verification_policy
        )
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
                "remote_hash_fallback": self.settings.storage_remote_hash_fallback,
                "snapshot_content_sha256": snapshot_content_sha256,
            },
            len(items),
            sum(item["transfer_status"] == "completed" for item in items),
            sum(item["transfer_status"] == "failed" for item in items),
            sum(item["transfer_status"] == "skipped" for item in items),
            sum(
                int(item["size"])
                for item in items
                if item["transfer_status"] != "skipped"
            ),
            sum(
                int(item["size"])
                for item in items
                if item["transfer_status"] == "completed"
            ),
            verification_level,
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
            "remote_size": None,
            "remote_modified_at": None,
            "transfer_status": status,
            "verification_status": verification,
            "retry_count": 0,
            "error_summary": None,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
        "verification_level": (
            manifest.verification_level.value
            if isinstance(manifest.verification_level, VerificationPolicy)
            else manifest.verification_level
        ),
        "status": manifest.status.value,
        "items": list(manifest.items),
        "error_summary": manifest.error_summary,
        "application_version": "0.1.0",
    }


def _manifest_verification_level(
    items: list[dict[str, Any]], requested: VerificationPolicy
) -> VerificationPolicy | str:
    statuses = {str(item.get("verification_status", "not_verified")) for item in items}
    if not statuses:
        return requested
    if "verification_failed" in statuses:
        return "verification_failed"
    if "not_verified" in statuses:
        return "not_verified"
    if "existence_only" in statuses:
        return VerificationPolicy.EXISTENCE_ONLY
    if "manifest_metadata_and_size" in statuses:
        return VerificationPolicy.SIZE_AND_MANIFEST
    if statuses == {"full_checksum"}:
        return VerificationPolicy.FULL_CHECKSUM
    if statuses.issubset({"full_checksum", "remote_hash_and_size"}):
        return VerificationPolicy.REMOTE_HASH_AND_SIZE
    return "not_verified"


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


def _hash_matches(path: Path, hash_type: str, expected: str) -> bool:
    algorithm = _hash_algorithm(hash_type)
    if algorithm not in hashlib.algorithms_available:
        return False
    try:
        return _hash_file(path, algorithm) == expected.casefold()
    except (ValueError, OSError):
        return False


def _hash_algorithm(hash_type: str) -> str:
    return hash_type.casefold().replace("-", "")


def _remote_signature(entry: StorageEntry) -> tuple[object, ...]:
    return (
        entry.remote_path.remote_name,
        entry.remote_path.relative_path,
        entry.size_bytes,
        entry.modified_at,
        entry.hash_type,
        entry.hash_value,
    )


def _remote_file_identical(
    entry: StorageEntry,
    local_path: Path,
) -> bool:
    if not entry.hash_type or not entry.hash_value:
        return False
    algorithm = entry.hash_type.casefold().replace("-", "")
    if algorithm not in hashlib.algorithms_available:
        return False
    return _hash_matches(local_path, entry.hash_type, entry.hash_value)


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


def _error_classification(message: str, *, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    lowered = message.casefold()
    if any(value in lowered for value in ("auth", "unauthorized", "401")):
        return "authentication"
    if any(value in lowered for value in ("permission", "forbidden", "403")):
        return "permission"
    if any(value in lowered for value in ("checksum", "hash mismatch")):
        return "checksum"
    if any(value in lowered for value in ("not found", "no such", "invalid path")):
        return "path"
    if any(value in lowered for value in ("quota", "no space", "capacity")):
        return "capacity"
    if any(value in lowered for value in ("rate", "429", "too many requests")):
        return "rate_limit"
    if any(value in lowered for value in ("temporar", "reset", "connection", "503")):
        return "network"
    return "unknown"


def _retryable(message: str, *, timed_out: bool = False) -> bool:
    return _error_classification(message, timed_out=timed_out) in {
        "timeout",
        "rate_limit",
        "network",
    }


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _is_expected_rclone_process(pid: int, executable: str) -> bool:
    """Return whether ``pid`` still belongs to the configured rclone binary."""
    if pid <= 0:
        return False
    expected = Path(executable).stem.casefold()
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode != 0:
                return False
            return any(
                line.split(",", 1)[0].strip('"').casefold() == f"{expected}.exe"
                or line.split(",", 1)[0].strip('"').casefold() == expected
                for line in result.stdout.splitlines()
            )
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        return any(
            Path(part).stem.casefold() == expected
            for part in command_line.split("\0")
            if part
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
