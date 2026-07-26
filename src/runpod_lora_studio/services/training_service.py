from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import DatasetSnapshotStatus
from runpod_lora_studio.domain.storage_models import (
    ManagedModelStatus,
    TransferDirection,
    TransferStatus,
)
from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
    TrainingJob,
    TrainingJobStateMachine,
    TrainingJobStatus,
)
from runpod_lora_studio.external.training_process import (
    StartedProcess,
    SubprocessTrainingAdapter,
    TrainingProcessAdapter,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotRecord,
    ManagedModelRecord,
    ModelTransferRecord,
    ProjectStorageSettingsRecord,
    TrainingJobRecord,
)
from runpod_lora_studio.persistence.training_repository import (
    TrainingRepository,
    utc_now,
)
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.training_command import (
    SdScriptsCommandBuilder,
    TrainingCommand,
    TrainingCommandValidationError,
)

logger = logging.getLogger("runpod_lora_studio.training")


class TrainingService:
    """Persist and execute Phase 6A training jobs outside Gradio callbacks."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        process_adapter: TrainingProcessAdapter | None = None,
        command_builder: SdScriptsCommandBuilder | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.process_adapter = process_adapter or SubprocessTrainingAdapter()
        self.command_builder = command_builder or SdScriptsCommandBuilder()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="training"
        )
        self._futures: dict[UUID, Future[None]] = {}
        self._lock = threading.Lock()

    @property
    def jobs_root(self) -> Path:
        return (
            self.settings.training_jobs_dir
            or self.settings.workspace_root / "training" / "jobs"
        ).resolve()

    def create_config(self, data: TrainingConfigInput) -> TrainingConfig:
        self._validate_config_input(data)
        with self.session_factory() as session:
            self._validate_references(session, data)
            record = TrainingRepository(session).create_config(data)
            session.commit()
            return (
                TrainingRepository(session).get_config(UUID(record.id))
                or self._missing()
            )

    def update_config(
        self, config_id: UUID, data: TrainingConfigInput
    ) -> TrainingConfig:
        self._validate_config_input(data)
        with self.session_factory() as session:
            repository = TrainingRepository(session)
            record = repository.get_config_record(config_id)
            if record is None:
                raise UserFacingError("指定された学習設定が見つかりません")
            self._validate_references(session, data)
            repository.update_config(record, data)
            session.commit()
            return repository.get_config(config_id) or self._missing()

    def get_config(self, config_id: UUID) -> TrainingConfig:
        with self.session_factory() as session:
            config = TrainingRepository(session).get_config(config_id)
            if config is None:
                raise UserFacingError("指定された学習設定が見つかりません")
            return config

    def list_configs(self, project_id: UUID) -> list[TrainingConfig]:
        with self.session_factory() as session:
            return TrainingRepository(session).list_configs(project_id)

    def list_completed_snapshots(self, project_id: UUID) -> list[tuple[UUID, str]]:
        with self.session_factory() as session:
            records = session.scalars(
                select(DatasetSnapshotRecord)
                .where(
                    DatasetSnapshotRecord.project_id == str(project_id),
                    DatasetSnapshotRecord.status
                    == DatasetSnapshotStatus.COMPLETED.value,
                )
                .order_by(DatasetSnapshotRecord.created_at.desc())
            ).all()
            return [(UUID(record.id), record.name) for record in records]

    def list_available_models(self, project_id: UUID) -> list[tuple[UUID, str]]:
        with self.session_factory() as session:
            selected = session.scalar(
                select(ProjectStorageSettingsRecord.selected_managed_model_id).where(
                    ProjectStorageSettingsRecord.project_id == str(project_id)
                )
            )
            query = select(ManagedModelRecord).where(
                ManagedModelRecord.status == ManagedModelStatus.AVAILABLE.value
            )
            records = session.scalars(
                query.order_by(ManagedModelRecord.display_name)
            ).all()
            if selected:
                records = [record for record in records if record.id == selected]
            return [(UUID(record.id), record.display_name) for record in records]

    def create_job(self, config_id: UUID) -> UUID:
        config = self.get_config(config_id)
        with self.session_factory() as session:
            self._validate_config_references(session, config)
            repository = TrainingRepository(session)
            active = session.scalar(
                select(TrainingJobRecord.id).where(
                    TrainingJobRecord.project_id == str(config.project_id),
                    TrainingJobRecord.status.in_(
                        [status.value for status in self._active_statuses()]
                    ),
                )
            )
            if active is not None:
                raise UserFacingError("同じプロジェクトで実行中の学習ジョブがあります")
            try:
                record = repository.create_job(config)
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise UserFacingError(
                    "同じプロジェクトで実行中の学習ジョブがあります"
                ) from exc
            return UUID(record.id)

    def start_job(self, job_id: UUID) -> UUID:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is None:
                raise UserFacingError("指定された学習ジョブが見つかりません")
            if TrainingJobStatus(record.status) is not TrainingJobStatus.QUEUED:
                raise UserFacingError("queued状態の学習ジョブだけ開始できます")
        with self._lock:
            future = self._futures.get(job_id)
            if future is not None and not future.done():
                return job_id
            try:
                self._futures[job_id] = self._executor.submit(self._run_job, job_id)
            except RuntimeError as exc:
                self._mark_failed(
                    job_id, "worker_start_failed", "学習workerを開始できません"
                )
                raise UserFacingError("学習workerを開始できません") from exc
        return job_id

    def request_cancel(self, job_id: UUID) -> str:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is None:
                raise UserFacingError("指定された学習ジョブが見つかりません")
            status = TrainingJobStatus(record.status)
            if status in {
                TrainingJobStatus.CANCELED,
                TrainingJobStatus.SUCCEEDED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.STALE,
            }:
                return "終了済みの学習ジョブです"
            if status is TrainingJobStatus.QUEUED:
                return "queued状態のため、まだプロセスは起動していません"
            if status is not TrainingJobStatus.CANCEL_REQUESTED:
                TrainingJobStateMachine.transition(
                    status, TrainingJobStatus.CANCEL_REQUESTED
                )
                record.status = TrainingJobStatus.CANCEL_REQUESTED.value
            record.cancel_requested = True
            record.updated_at = utc_now()
            session.commit()
            return "停止要求を受け付けました"

    cancel_job = request_cancel

    def get_job(self, job_id: UUID) -> TrainingJob:
        with self.session_factory() as session:
            job = TrainingRepository(session).get_job(job_id)
            if job is None:
                raise UserFacingError("指定された学習ジョブが見つかりません")
            return job

    def list_jobs(self, project_id: UUID | None = None) -> list[TrainingJob]:
        with self.session_factory() as session:
            return TrainingRepository(session).list_jobs(project_id)

    def active_job(self, project_id: UUID) -> TrainingJob | None:
        with self.session_factory() as session:
            jobs = TrainingRepository(session).list_jobs(project_id)
            return next(
                (job for job in jobs if job.status in self._active_statuses()), None
            )

    def reconcile_stale_jobs(self) -> int:
        try:
            return self._reconcile_stale_jobs()
        except OperationalError:
            # The UI can be imported before the new migration is applied.
            return 0

    def _reconcile_stale_jobs(self) -> int:
        now = datetime.now(UTC)
        reconciled = 0
        with self.session_factory() as session:
            repository = TrainingRepository(session)
            for record in repository.list_active_records():
                status = TrainingJobStatus(record.status)
                heartbeat = _utc(record.worker_heartbeat)
                reference = (
                    heartbeat or _utc(record.started_at) or _utc(record.created_at)
                )
                age = (now - reference).total_seconds() if reference else float("inf")
                if record.pid is not None and self.process_adapter.is_running(
                    record.pid
                ):
                    # An unknown but live process is intentionally preserved after
                    # application restart; stopping it would risk PID reuse.
                    continue
                if (
                    status is TrainingJobStatus.STARTING
                    and age < self.settings.training_starting_grace_seconds
                ):
                    continue
                if age < self.settings.training_job_stale_after_seconds:
                    continue
                TrainingJobStateMachine.transition(status, TrainingJobStatus.STALE)
                record.status = TrainingJobStatus.STALE.value
                record.failure_code = "stale_process_missing"
                record.failure_message = "workerと対応する学習プロセスを確認できません"
                record.finished_at = now
                record.updated_at = now
                reconciled += 1
            session.commit()
        return reconciled

    def tail_stdout(self, job_id: UUID, max_bytes: int | None = None) -> str:
        return self._tail_log(job_id, "stdout", max_bytes)

    def tail_stderr(self, job_id: UUID, max_bytes: int | None = None) -> str:
        return self._tail_log(job_id, "stderr", max_bytes)

    def _run_job(self, job_id: UUID) -> None:
        worker_id = f"{os.getpid()}:{uuid4().hex}"
        started: StartedProcess | None = None
        try:
            config, model_path, snapshot_toml, command, runtime = self._prepare_job(
                job_id, worker_id
            )
            environment = _safe_environment()
            started = self.process_adapter.start(
                command.arguments,
                cwd=runtime,
                stdout_path=runtime / "logs" / "stdout.log",
                stderr_path=runtime / "logs" / "stderr.log",
                env=environment,
            )
            self._mark_running(job_id, worker_id, started, command, runtime)
            self._monitor_job(job_id, started)
        except Exception:
            logger.exception("training_worker_failed job_id=%s", job_id)
            if started is not None and self._process_matches(job_id, started):
                try:
                    self.process_adapter.terminate(started.pid)
                except OSError:
                    logger.exception(
                        "training_worker_terminate_failed job_id=%s", job_id
                    )
            self._mark_failed(
                job_id, "worker_exception", "学習workerでエラーが発生しました"
            )

    def _prepare_job(
        self, job_id: UUID, worker_id: str
    ) -> tuple[TrainingConfig, Path, Path, TrainingCommand, Path]:
        with self.session_factory() as session:
            repository = TrainingRepository(session)
            record = repository.get_job_record(job_id)
            if record is None:
                raise UserFacingError("指定された学習ジョブが見つかりません")
            if TrainingJobStatus(record.status) is not TrainingJobStatus.QUEUED:
                raise UserFacingError("queued状態の学習ジョブだけ実行できます")
            record.status = TrainingJobStatus.STARTING.value
            record.worker_id = worker_id
            record.started_at = utc_now()
            record.worker_heartbeat = utc_now()
            record.updated_at = utc_now()
            session.commit()
            config = repository.get_config(UUID(record.training_config_id))
            if config is None:
                raise UserFacingError("学習設定が見つかりません")
            model_path, snapshot_toml = self._validate_config_references(
                session, config
            )

        runtime = self.jobs_root / str(job_id)
        (runtime / "config").mkdir(parents=True, exist_ok=True)
        (runtime / "logs").mkdir(parents=True, exist_ok=True)
        (runtime / "output").mkdir(parents=True, exist_ok=True)
        (runtime / "runtime").mkdir(parents=True, exist_ok=True)
        copied_toml = runtime / "config" / "dataset.toml"
        shutil.copy2(snapshot_toml, copied_toml)
        command = self.command_builder.build(
            config, model_path=model_path, dataset_config_path=copied_toml
        )
        (runtime / "config" / "training-config.json").write_text(
            config.snapshot_json(), encoding="utf-8"
        )
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is None:
                raise UserFacingError("指定された学習ジョブが見つかりません")
            record.runtime_directory = str(runtime)
            record.stdout_log_path = str(runtime / "logs" / "stdout.log")
            record.stderr_log_path = str(runtime / "logs" / "stderr.log")
            record.command_summary = command.summary
            record.config_snapshot = config.snapshot_json()
            record.updated_at = utc_now()
            session.commit()
        return config, model_path, copied_toml, command, runtime

    def _mark_running(
        self,
        job_id: UUID,
        worker_id: str,
        started: StartedProcess,
        command: TrainingCommand,
        runtime: Path,
    ) -> None:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is None:
                raise UserFacingError("学習ジョブが見つかりません")
            if record.status == TrainingJobStatus.CANCEL_REQUESTED.value:
                # A stop request may arrive in the start boundary. Keep the
                # state transition explicit before the worker sends SIGTERM.
                pass
            else:
                TrainingJobStateMachine.transition(
                    TrainingJobStatus(record.status), TrainingJobStatus.RUNNING
                )
                record.status = TrainingJobStatus.RUNNING.value
            record.pid = started.pid
            record.worker_id = worker_id
            record.process_start_time = started.process_start_time
            record.process_group_id = started.process_group_id
            record.process_identity = started.process_identity
            record.worker_heartbeat = utc_now()
            record.command_summary = command.summary
            record.runtime_directory = str(runtime)
            record.updated_at = utc_now()
            session.commit()

    def _monitor_job(self, job_id: UUID, started: StartedProcess) -> None:
        cancel_sent = False
        kill_deadline: float | None = None
        while True:
            exit_code = self.process_adapter.poll(started.pid)
            cancel_requested = self._cancel_requested(job_id)
            if cancel_requested and not cancel_sent:
                if self._process_matches(job_id, started):
                    self.process_adapter.terminate(started.pid)
                    cancel_sent = True
                    kill_deadline = (
                        time.monotonic() + self.settings.training_cancel_grace_seconds
                    )
                else:
                    cancel_sent = True
            if (
                cancel_sent
                and exit_code is None
                and kill_deadline is not None
                and time.monotonic() >= kill_deadline
                and self._process_matches(job_id, started)
            ):
                self.process_adapter.kill(started.pid)
                kill_deadline = None
            self._heartbeat(job_id)
            if exit_code is not None:
                self._finish_job(job_id, exit_code, cancel_requested)
                return
            time.sleep(self.settings.training_heartbeat_interval_seconds)

    def _finish_job(self, job_id: UUID, exit_code: int, cancel_requested: bool) -> None:
        status = (
            TrainingJobStatus.CANCELED
            if cancel_requested
            else TrainingJobStatus.SUCCEEDED
            if exit_code == 0
            else TrainingJobStatus.FAILED
        )
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is None:
                return
            current = TrainingJobStatus(record.status)
            if current is not status:
                TrainingJobStateMachine.transition(current, status)
                record.status = status.value
            record.exit_code = exit_code
            record.finished_at = utc_now()
            record.worker_heartbeat = utc_now()
            if status is TrainingJobStatus.FAILED:
                record.failure_code = "process_exit_nonzero"
                record.failure_message = (
                    f"学習プロセスが終了コード {exit_code} で終了しました"
                )
            record.updated_at = utc_now()
            session.commit()

    def _mark_failed(self, job_id: UUID, code: str, message: str) -> None:
        try:
            with self.session_factory() as session:
                record = TrainingRepository(session).get_job_record(job_id)
                if record is None:
                    return
                current = TrainingJobStatus(record.status)
                if current is not TrainingJobStatus.FAILED:
                    if current is TrainingJobStatus.QUEUED:
                        TrainingJobStateMachine.transition(
                            current, TrainingJobStatus.STARTING
                        )
                        current = TrainingJobStatus.STARTING
                    TrainingJobStateMachine.transition(
                        current, TrainingJobStatus.FAILED
                    )
                    record.status = TrainingJobStatus.FAILED.value
                record.failure_code = code
                record.failure_message = message
                record.finished_at = utc_now()
                record.updated_at = utc_now()
                session.commit()
        except Exception:
            logger.exception("training_job_failure_persist_failed job_id=%s", job_id)

    def _heartbeat(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            if record is not None:
                record.worker_heartbeat = utc_now()
                record.updated_at = utc_now()
                session.commit()

    def _cancel_requested(self, job_id: UUID) -> bool:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            return bool(record and record.cancel_requested)

    def _process_matches(self, job_id: UUID, started: StartedProcess) -> bool:
        with self.session_factory() as session:
            record = TrainingRepository(session).get_job_record(job_id)
            return bool(
                record
                and record.pid == started.pid
                and self.process_adapter.process_matches(
                    started.pid, record.process_group_id, record.process_identity
                )
            )

    def _validate_config_input(self, data: TrainingConfigInput) -> None:
        if not data.name.strip() or len(data.name.strip()) > 200:
            raise UserFacingError("学習設定名が不正です")
        if not _safe_filename(data.output_name):
            raise UserFacingError("output nameが不正です")
        if data.trainer_script not in self.command_builder.allowed_trainer_scripts:
            raise UserFacingError("許可されていないtrainer scriptです")
        if data.resolution < 64 or data.resolution > 8192 or data.resolution % 8:
            raise UserFacingError("resolutionが不正です")
        if not 1 <= data.batch_size <= 64:
            raise UserFacingError("batch sizeが不正です")
        if not 1 <= data.epochs <= 100000 or not 1 <= data.repeats <= 100000:
            raise UserFacingError("epochsまたはrepeatsが不正です")
        if not 0 < data.learning_rate <= 10:
            raise UserFacingError("learning rateが不正です")
        if data.network_dim <= 0 or data.network_alpha <= 0:
            raise UserFacingError("network dimまたはalphaが不正です")
        if data.save_every_n_epochs <= 0:
            raise UserFacingError("save every N epochsが不正です")
        if data.mixed_precision not in self.command_builder.allowed_mixed_precision:
            raise UserFacingError("mixed precisionが不正です")
        for value in (data.python_executable, data.trainer_script):
            if not value.strip() or any(ord(char) < 32 for char in value):
                raise UserFacingError("実行ファイル設定が不正です")
        try:
            self.command_builder._extra_arguments(data.extra_options)
        except TrainingCommandValidationError as exc:
            raise UserFacingError(str(exc)) from exc
        if not _is_under(
            data.output_directory.resolve(), self.settings.outputs_dir.resolve()
        ):
            raise UserFacingError("output directoryが許可領域外です")
        allowed_roots = [self.settings.training_sd_scripts_root.resolve()]
        if self.settings.workspace_root.exists():
            allowed_roots.append(self.settings.workspace_root.resolve())
        root = data.sd_scripts_root.resolve()
        if not any(_is_under(root, allowed) for allowed in allowed_roots):
            raise UserFacingError("sd-scripts rootが許可領域外です")

    def _validate_references(self, session: Any, data: TrainingConfigInput) -> None:
        model = session.scalar(
            select(ManagedModelRecord).where(
                ManagedModelRecord.id == str(data.managed_model_id)
            )
        )
        if model is None:
            raise UserFacingError("学習元モデルが見つかりません")
        snapshot = session.scalar(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.id == str(data.dataset_snapshot_id)
            )
        )
        if snapshot is None:
            raise UserFacingError("dataset snapshotが見つかりません")
        if snapshot.project_id != str(data.project_id):
            raise UserFacingError("snapshotが対象プロジェクトに属していません")
        project_settings = session.scalar(
            select(ProjectStorageSettingsRecord).where(
                ProjectStorageSettingsRecord.project_id == str(data.project_id)
            )
        )
        if (
            project_settings is not None
            and project_settings.selected_managed_model_id is not None
            and project_settings.selected_managed_model_id != str(data.managed_model_id)
        ):
            raise UserFacingError("学習元モデルが対象プロジェクトで選択されていません")

    def _validate_config_references(
        self, session: Any, config: TrainingConfig
    ) -> tuple[Path, Path]:
        snapshot = session.scalar(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.id == str(config.dataset_snapshot_id)
            )
        )
        model = session.scalar(
            select(ManagedModelRecord).where(
                ManagedModelRecord.id == str(config.managed_model_id)
            )
        )
        if snapshot is None or snapshot.project_id != str(config.project_id):
            raise UserFacingError("completed snapshotが対象プロジェクトにありません")
        if snapshot.status != DatasetSnapshotStatus.COMPLETED.value:
            raise UserFacingError("completed snapshotだけ学習に利用できます")
        snapshot_root = Path(snapshot.snapshot_root).resolve()
        dataset_toml = Path(snapshot.dataset_toml_path).resolve()
        manifest = Path(snapshot.manifest_path).resolve()
        if not (snapshot.manifest_sha256 or snapshot.content_sha256):
            raise UserFacingError("snapshotのmanifestまたはcontent hashがありません")
        if not dataset_toml.is_file() or not manifest.is_file():
            raise UserFacingError("snapshotのmanifestまたはdataset TOMLがありません")
        if not _is_under(dataset_toml, snapshot_root) or not _is_under(
            manifest, snapshot_root
        ):
            raise UserFacingError("snapshotファイルのパスが不正です")
        if model is None or model.status != ManagedModelStatus.AVAILABLE.value:
            raise UserFacingError("検証済みのローカルモデルだけ利用できます")
        if model.local_path is None:
            raise UserFacingError("ローカルモデルがありません")
        model_path = Path(model.local_path).resolve()
        allowed_model_roots = [self.settings.models_dir.resolve()]
        if self.settings.model_cache_dir is not None:
            allowed_model_roots.append(self.settings.model_cache_dir.resolve())
        if not any(_is_under(model_path, root) for root in allowed_model_roots):
            raise UserFacingError("ローカルモデルのパスが許可領域外です")
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise UserFacingError("ローカルモデルが存在しないか空です")
        if model.local_sha256 and _sha256_file(model_path) != model.local_sha256:
            raise UserFacingError("ローカルモデルのSHA-256が一致しません")
        completed_transfer = session.scalar(
            select(ModelTransferRecord.id).where(
                ModelTransferRecord.managed_model_id == str(config.managed_model_id),
                ModelTransferRecord.direction == TransferDirection.DOWNLOAD.value,
                ModelTransferRecord.status == TransferStatus.COMPLETED.value,
            )
        )
        if completed_transfer is None:
            raise UserFacingError("学習元モデルの転送が完了していません")
        self._validate_config_input(
            TrainingConfigInput(
                project_id=config.project_id,
                dataset_snapshot_id=config.dataset_snapshot_id,
                managed_model_id=config.managed_model_id,
                name=config.name,
                output_name=config.output_name,
                output_directory=config.output_directory,
                sd_scripts_root=config.sd_scripts_root,
                trainer_script=config.trainer_script,
                python_executable=config.python_executable,
                resolution=config.resolution,
                batch_size=config.batch_size,
                epochs=config.epochs,
                repeats=config.repeats,
                learning_rate=config.learning_rate,
                optimizer=config.optimizer,
                scheduler=config.scheduler,
                network_module=config.network_module,
                network_dim=config.network_dim,
                network_alpha=config.network_alpha,
                mixed_precision=config.mixed_precision,
                save_every_n_epochs=config.save_every_n_epochs,
                cache_latents=config.cache_latents,
                gradient_checkpointing=config.gradient_checkpointing,
                seed=config.seed,
                extra_options=config.extra_options,
            )
        )
        return model_path, dataset_toml

    def _tail_log(self, job_id: UUID, stream: str, max_bytes: int | None) -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("unknown log stream")
        limit = (
            self.settings.training_log_tail_bytes if max_bytes is None else max_bytes
        )
        if limit <= 0:
            raise ValueError("max_bytes must be positive")
        limit = min(limit, self.settings.training_log_tail_bytes)
        job = self.get_job(job_id)
        path = job.stdout_log_path if stream == "stdout" else job.stderr_log_path
        if path is None:
            return ""
        path = path.resolve()
        if job.runtime_directory is None or not _is_under(
            path, job.runtime_directory.resolve()
        ):
            raise UserFacingError("ログパスが学習ジョブ領域外です")
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            text = handle.read(limit).decode("utf-8", errors="replace")
            while len(text.encode("utf-8")) > limit:
                text = text[1:]
            return text

    def _missing(self) -> Any:
        raise UserFacingError("学習設定を取得できません")

    @staticmethod
    def _active_statuses() -> frozenset[TrainingJobStatus]:
        return frozenset(
            {
                TrainingJobStatus.QUEUED,
                TrainingJobStatus.STARTING,
                TrainingJobStatus.RUNNING,
                TrainingJobStatus.CANCEL_REQUESTED,
            }
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


def _safe_filename(value: str) -> bool:
    return bool(
        value
        and value not in {".", ".."}
        and Path(value).name == value
        and not any(char in value for char in ("/", "\\", "\x00", "\n", "\r"))
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value
