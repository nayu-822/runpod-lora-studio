from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.training_models import TrainingJob, TrainingJobStatus
from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifactType,
    TrainingArtifactValidationStatus,
)
from runpod_lora_studio.domain.training_resume_models import (
    ResumeCompatibility,
    ResumeMode,
    ResumeStateFile,
    ResumeValidationStatus,
    TrainingResumePreview,
    ValidatedResumeState,
)
from runpod_lora_studio.external.training_process import (
    SubprocessTrainingAdapter,
    TrainingProcessAdapter,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotRecord,
    ManagedModelRecord,
    TrainingArtifactRecord,
    TrainingConfigRecord,
    TrainingJobRecord,
    TrainingProgressRecord,
    TrainingResumeValidationRecord,
)
from runpod_lora_studio.persistence.training_repository import (
    TrainingRepository,
    _job_from_record,
    utc_now,
)
from runpod_lora_studio.services.project_service import UserFacingError

VALIDATOR_VERSION = "phase6c-v1"
RESUMABLE_STATUSES = frozenset(
    {TrainingJobStatus.FAILED, TrainingJobStatus.CANCELED, TrainingJobStatus.STALE}
)
ACTIVE_STATUSES = frozenset(
    {
        TrainingJobStatus.QUEUED,
        TrainingJobStatus.STARTING,
        TrainingJobStatus.RUNNING,
        TrainingJobStatus.CANCEL_REQUESTED,
    }
)


class TrainingResumeService:
    """Validate state artifacts and create immutable child training jobs."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        process_adapter: TrainingProcessAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.process_adapter = process_adapter or SubprocessTrainingAdapter()

    def list_resumable_jobs(self, project_id: UUID | None = None) -> list[TrainingJob]:
        with self.session_factory() as session:
            query = select(TrainingJobRecord).where(
                TrainingJobRecord.status.in_(
                    [status.value for status in RESUMABLE_STATUSES]
                )
            )
            if project_id is not None:
                query = query.where(TrainingJobRecord.project_id == str(project_id))
            records = session.scalars(
                query.order_by(TrainingJobRecord.updated_at.desc())
            ).all()
            return [
                _job_from_record(record)
                for record in records
                if self._source_is_safe(record)
            ]

    def list_state_artifacts(self, job_id: UUID) -> list[dict[str, str]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(TrainingArtifactRecord).where(
                    TrainingArtifactRecord.training_job_id == str(job_id),
                    TrainingArtifactRecord.artifact_type
                    == TrainingArtifactType.TRAINING_STATE.value,
                )
            ).all()
            return [
                {
                    "id": row.id,
                    "filename": row.filename,
                    "relative_path": row.relative_path,
                    "validation_status": row.validation_status,
                    "validation_code": row.validation_code or "",
                }
                for row in rows
            ]

    def refresh_artifact_fingerprints(self, job_id: UUID) -> None:
        """Persist the immutable fingerprint captured when a state is registered."""
        with self.session_factory() as session:
            parent = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            if parent is None:
                return
            rows = session.scalars(
                select(TrainingArtifactRecord).where(
                    TrainingArtifactRecord.training_job_id == str(job_id),
                    TrainingArtifactRecord.artifact_type
                    == TrainingArtifactType.TRAINING_STATE.value,
                    TrainingArtifactRecord.validation_status
                    == TrainingArtifactValidationStatus.VALID.value,
                )
            ).all()
            for artifact in rows:
                try:
                    validated = self._validate_state_path(parent, artifact)
                except UserFacingError:
                    continue
                metadata = _json_object(artifact.metadata_json)
                values = dict(metadata) if isinstance(metadata, dict) else {}
                values["resume_state_fingerprint"] = validated.fingerprint
                values["resume_validator_version"] = VALIDATOR_VERSION
                artifact.metadata_json = json.dumps(
                    values, ensure_ascii=False, sort_keys=True
                )
            session.commit()

    def preview(
        self,
        source_job_id: UUID,
        artifact_id: UUID,
        target_config_id: UUID | None = None,
    ) -> TrainingResumePreview:
        with self.session_factory() as session:
            parent, artifact, config, validated, compatibility = self._validate(
                session, source_job_id, artifact_id, target_config_id
            )
            progress = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == str(source_job_id)
                )
            )
            signature = self._signature(
                parent, artifact, config, validated, compatibility, session
            )
            output_name = config.output_name
            command_summary = (
                f"--resume <resume/{validated.source_relative_path.name}> "
                f"--output_name {output_name}"
            )
            state_epoch = validated.state_epoch
            state_step = validated.state_step
            position_warning = _state_position_warning(
                progress.current_epoch if progress else None,
                progress.current_step if progress else None,
                state_epoch,
                state_step,
            )
            return TrainingResumePreview(
                source_job_id=source_job_id,
                source_artifact_id=UUID(artifact.id),
                target_config_id=UUID(config.id),
                source_status=parent.status,
                source_state_name=artifact.filename,
                state_fingerprint=validated.fingerprint,
                current_epoch=progress.current_epoch if progress else None,
                current_step=progress.current_step if progress else None,
                source_total_epochs=_config_epochs(parent.config_snapshot),
                target_total_epochs=config.epochs,
                output_name=output_name,
                compatibility=compatibility,
                signature=signature,
                command_summary=command_summary,
                state_epoch=state_epoch,
                state_step=state_step,
                initial_epoch=state_epoch,
                initial_step=state_step,
                progress_epoch_offset=state_epoch,
                progress_step_offset=state_step,
                position_warning=position_warning,
            )

    def create_resume_job(
        self,
        source_job_id: UUID,
        artifact_id: UUID,
        *,
        target_config_id: UUID | None = None,
        preview_signature: str | None = None,
    ) -> UUID:
        with self.session_factory() as session:
            parent, artifact, config, validated, compatibility = self._validate(
                session, source_job_id, artifact_id, target_config_id
            )
            if compatibility.status is not ResumeValidationStatus.COMPATIBLE:
                raise UserFacingError(
                    "再開できないstateです: " + "; ".join(compatibility.issues)
                )
            request_fingerprint = _resume_request_fingerprint(
                parent.id,
                artifact.id,
                config.id,
                validated.fingerprint,
                compatibility.target_config_fingerprint,
            )
            existing = session.scalar(
                select(TrainingJobRecord).where(
                    TrainingJobRecord.resume_request_fingerprint == request_fingerprint,
                )
            )
            if existing is not None:
                return UUID(existing.id)
            if preview_signature is not None:
                expected = self._signature(
                    parent, artifact, config, validated, compatibility, session
                )
                if preview_signature != expected:
                    raise UserFacingError("再開プレビューが古くなっています")
            initial_epoch = validated.state_epoch
            initial_step = validated.state_step
            progress = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == str(source_job_id)
                )
            )
            position_warning = _state_position_warning(
                progress.current_epoch if progress else None,
                progress.current_step if progress else None,
                initial_epoch,
                initial_step,
            )
            record = TrainingRepository(session).create_job(config)
            record.parent_job_id = parent.id
            record.resume_artifact_id = artifact.id
            record.resume_mode = ResumeMode.COPY.value
            record.resume_requested_at = utc_now()
            record.resume_validation_status = compatibility.status.value
            record.resume_validation_code = "RESUME_STATE_VALID"
            record.resume_validation_message = (
                "; ".join(
                    compatibility.issues
                    + ((position_warning,) if position_warning else ())
                )
                or "compatible"
            )
            record.initial_epoch = initial_epoch
            record.initial_step = initial_step
            record.progress_step_offset = record.initial_step
            record.progress_epoch_offset = record.initial_epoch
            record.resume_request_fingerprint = request_fingerprint
            session.add(
                TrainingResumeValidationRecord(
                    id=str(uuid4()),
                    source_job_id=parent.id,
                    source_artifact_id=artifact.id,
                    target_training_config_id=config.id,
                    source_state_relative_path=artifact.relative_path,
                    source_state_fingerprint=validated.fingerprint,
                    source_job_config_fingerprint=compatibility.source_config_fingerprint,
                    target_config_fingerprint=compatibility.target_config_fingerprint,
                    compatibility_status=compatibility.status.value,
                    compatibility_issues=json.dumps(
                        compatibility.issues, ensure_ascii=False
                    ),
                    validated_at=utc_now(),
                    validator_version=VALIDATOR_VERSION,
                    created_at=utc_now(),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.scalar(
                    select(TrainingJobRecord).where(
                        TrainingJobRecord.resume_request_fingerprint
                        == request_fingerprint
                    )
                )
                if existing is None:
                    raise
                return UUID(existing.id)
            return UUID(record.id)

    def prepare_runtime(self, child_job_id: UUID, runtime: Path) -> tuple[Path, str]:
        """Revalidate, copy state, and write a safe manifest before process start."""
        with self.session_factory() as session:
            child = session.scalar(
                select(TrainingJobRecord).where(
                    TrainingJobRecord.id == str(child_job_id)
                )
            )
            if (
                child is None
                or child.parent_job_id is None
                or child.resume_artifact_id is None
            ):
                raise UserFacingError("再開情報がありません")
            parent, artifact, config, validated, compatibility = self._validate(
                session,
                UUID(child.parent_job_id),
                UUID(child.resume_artifact_id),
                UUID(child.training_config_id),
            )
            if compatibility.status is not ResumeValidationStatus.COMPATIBLE:
                self._mark_validation(child, compatibility, session)
                session.commit()
                raise UserFacingError("再開stateの再検証に失敗しました")
            destination = runtime / "runtime" / "resume" / "source-state"
            self._copy_state(validated, destination)
            copied = self._scan_state_path(
                destination, UUID(artifact.id), UUID(parent.id)
            )
            copied = replace(
                copied,
                state_epoch=validated.state_epoch,
                state_step=validated.state_step,
            )
            if copied.fingerprint != validated.fingerprint:
                raise UserFacingError("再開stateのコピー検証に失敗しました")
            source_progress = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == parent.id
                )
            )
            position_warning = _state_position_warning(
                source_progress.current_epoch if source_progress else None,
                source_progress.current_step if source_progress else None,
                copied.state_epoch,
                copied.state_step,
            )
            manifest = self._manifest(
                parent,
                artifact,
                config,
                copied,
                session,
                child.initial_epoch,
                child.initial_step,
                child.resume_request_fingerprint,
                compatibility.target_config_fingerprint,
                position_warning,
            )
            manifest_path = runtime / "config" / "resume-state-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            child.resume_validation_status = ResumeValidationStatus.COMPATIBLE.value
            child.resume_validation_code = "RESUME_STATE_COPIED"
            child.resume_validation_message = "resume state copied and verified"
            child.updated_at = utc_now()
            session.commit()
            return destination, _sha256_bytes(manifest_path.read_bytes())

    def series(self, job_id: UUID, max_depth: int = 20) -> list[UUID]:
        result: list[UUID] = []
        seen: set[UUID] = set()
        current = job_id
        with self.session_factory() as session:
            for _ in range(max_depth):
                if current in seen:
                    break
                seen.add(current)
                result.append(current)
                parent = session.scalar(
                    select(TrainingJobRecord.parent_job_id).where(
                        TrainingJobRecord.id == str(current)
                    )
                )
                if not parent:
                    break
                current = UUID(parent)
        return list(reversed(result))

    def _validate(
        self,
        session: Any,
        source_job_id: UUID,
        artifact_id: UUID,
        target_config_id: UUID | None,
    ) -> tuple[
        TrainingJobRecord,
        TrainingArtifactRecord,
        TrainingConfigRecord,
        ValidatedResumeState,
        ResumeCompatibility,
    ]:
        parent = session.scalar(
            select(TrainingJobRecord).where(TrainingJobRecord.id == str(source_job_id))
        )
        artifact = session.scalar(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.id == str(artifact_id)
            )
        )
        if parent is None or artifact is None:
            raise UserFacingError("再開元jobまたはstateが見つかりません")
        target_id = target_config_id or UUID(parent.training_config_id)
        config = session.scalar(
            select(TrainingConfigRecord).where(
                TrainingConfigRecord.id == str(target_id)
            )
        )
        if config is None:
            raise UserFacingError("再開先の学習設定が見つかりません")
        if parent.project_id != config.project_id:
            raise UserFacingError("parent jobと学習設定のprojectが一致しません")
        if parent.id == str(source_job_id) and parent.parent_job_id == parent.id:
            raise UserFacingError("再開parentの循環参照です")
        if parent.status not in {status.value for status in RESUMABLE_STATUSES}:
            raise UserFacingError("このstatusのjobからは再開できません")
        if not self._source_is_safe(parent):
            raise UserFacingError("再開元プロセスが終了したことを確認できません")
        if artifact.training_job_id != parent.id:
            raise UserFacingError("stateがparent jobの成果物ではありません")
        if artifact.artifact_type != TrainingArtifactType.TRAINING_STATE.value:
            raise UserFacingError("training state以外は再開に使用できません")
        validated = self._validate_state_path(parent, artifact)
        metadata = _json_object(artifact.metadata_json)
        if isinstance(metadata, dict):
            expected = metadata.get("resume_state_fingerprint")
            if expected is not None and expected != validated.fingerprint:
                raise UserFacingError("resume state changed after registration")
        if validated.state_epoch is None or validated.state_step is None:
            raise UserFacingError(
                "resume state epoch/step could not be determined safely"
            )
        compatibility = self._compatibility(session, parent, config, validated)
        return parent, artifact, config, validated, compatibility

    def _source_is_safe(self, parent: TrainingJobRecord) -> bool:
        if parent.status in {status.value for status in ACTIVE_STATUSES}:
            return False
        if parent.status not in {status.value for status in RESUMABLE_STATUSES}:
            return False

        process_running = False
        if parent.pid is not None:
            try:
                process_running = self.process_adapter.is_running(parent.pid)
                process_matches = self.process_adapter.process_matches(
                    parent.pid, parent.process_group_id, parent.process_identity
                )
            except (OSError, RuntimeError, ValueError):
                return False
            if process_running or process_matches:
                return False

        terminal_record = parent.finished_at is not None and (
            parent.exit_code is not None
            or parent.failure_code is not None
            or parent.status == TrainingJobStatus.CANCELED.value
        )
        if (
            parent.pid is None
            and any(
                value is not None
                for value in (
                    parent.worker_id,
                    parent.process_group_id,
                    parent.process_identity,
                    parent.process_start_time,
                )
            )
            and not terminal_record
        ):
            return False

        heartbeat = _as_utc(parent.worker_heartbeat)
        heartbeat_expired = (
            heartbeat is None
            or (datetime.now(UTC) - heartbeat).total_seconds()
            >= self.settings.training_job_stale_after_seconds
        )
        if parent.status == TrainingJobStatus.STALE.value and not heartbeat_expired:
            return False
        if not terminal_record and parent.status in {
            TrainingJobStatus.FAILED.value,
            TrainingJobStatus.CANCELED.value,
        }:
            if parent.pid is None and any(
                value is not None
                for value in (
                    parent.worker_id,
                    parent.process_group_id,
                    parent.process_identity,
                    parent.process_start_time,
                )
            ):
                return False
            if not heartbeat_expired:
                return False
        return True

    def _validate_state_path(
        self, parent: TrainingJobRecord, artifact: TrainingArtifactRecord
    ) -> ValidatedResumeState:
        if artifact.validation_status != TrainingArtifactValidationStatus.VALID.value:
            raise UserFacingError("stateのvalidation statusが再開可能ではありません")
        runtime = _trusted_runtime(parent.runtime_directory, self.settings)
        if runtime is None:
            raise UserFacingError("再開元runtimeが許可領域外です")
        relative = Path(artifact.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise UserFacingError("stateの相対パスが不正です")
        output = runtime / "output"
        if output.is_symlink() or not output.is_dir():
            raise UserFacingError("再開元outputがありません")
        state = output / relative
        if state.is_symlink() or not state.is_dir():
            raise UserFacingError("stateは通常ディレクトリである必要があります")
        try:
            state.resolve(strict=True).relative_to(output.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise UserFacingError("stateが許可領域外です") from exc
        validated = self._scan_state_path(
            state, UUID(artifact.id), UUID(parent.id), relative
        )
        state_epoch, state_step = _state_position(artifact)
        return replace(
            validated,
            state_epoch=state_epoch,
            state_step=state_step,
        )

    def _scan_state_path(
        self,
        state: Path,
        artifact_id: UUID,
        job_id: UUID,
        relative: Path | None = None,
    ) -> ValidatedResumeState:
        entries: list[ResumeStateFile] = []
        stat_map: dict[Path, tuple[int, int]] = {}
        queue: list[tuple[Path, int]] = [(state, 0)]
        total_size = 0
        while queue:
            directory, depth = queue.pop(0)
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise UserFacingError("stateを走査できません") from exc
            for child in children:
                if child.is_symlink():
                    raise UserFacingError("state配下のsymlinkは使用できません")
                child_relative = child.relative_to(state)
                if (
                    len(child_relative.parts)
                    > self.settings.training_artifact_max_depth
                ):
                    raise UserFacingError("stateの階層が深すぎます")
                if child.is_dir():
                    queue.append((child, depth + 1))
                    continue
                if not child.is_file():
                    raise UserFacingError("stateに通常ファイル以外があります")
                stat = child.stat()
                if stat.st_size > self.settings.training_artifact_max_file_size_bytes:
                    raise UserFacingError("state内ファイルが大きすぎます")
                total_size += stat.st_size
                if (
                    total_size
                    > self.settings.training_resume_state_max_total_size_bytes
                ):
                    raise UserFacingError("state全体の容量が大きすぎます")
                if len(entries) >= self.settings.training_artifact_max_count:
                    raise UserFacingError("state内ファイル数が多すぎます")
                digest = _sha256_file(child)
                after = child.stat()
                if (stat.st_size, stat.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise UserFacingError("stateが検証中に変更されました")
                entries.append(ResumeStateFile(child_relative, stat.st_size, digest))
                stat_map[child_relative] = (stat.st_size, stat.st_mtime_ns)
        if not entries:
            raise UserFacingError("空のstateは使用できません")
        manifest = {
            "validator_version": VALIDATOR_VERSION,
            "files": [
                {
                    "path": str(item.relative_path).replace("\\", "/"),
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in sorted(entries, key=lambda item: str(item.relative_path))
            ],
        }
        fingerprint = _sha256_bytes(_canonical_json(manifest))
        return ValidatedResumeState(
            job_id,
            artifact_id,
            relative or Path(state.name),
            state.resolve(),
            fingerprint,
            tuple(sorted(entries, key=lambda item: str(item.relative_path))),
            total_size,
            VALIDATOR_VERSION,
            datetime.now(UTC),
        )

    def _copy_state(self, validated: ValidatedResumeState, destination: Path) -> None:
        if destination.is_symlink() or destination.is_file():
            raise UserFacingError("resume state destination is not a directory")
        if destination.is_dir():
            shutil.rmtree(destination)
        temp = destination.with_name(f".{destination.name}.copying-{uuid4().hex}")
        temp.mkdir(parents=True, exist_ok=False)
        try:
            for item in validated.files:
                source = validated.source_path / item.relative_path
                target = temp / item.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_symlink() or target.exists():
                    raise UserFacingError("stateコピー中にsymlinkを検出しました")
                with (
                    source.open("rb") as source_handle,
                    target.open("wb") as target_handle,
                ):
                    shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                if (
                    target.stat().st_size != item.size
                    or _sha256_file(target) != item.sha256
                ):
                    raise UserFacingError("stateコピーのhash検証に失敗しました")
            os.replace(temp, destination)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def _compatibility(
        self,
        session: Any,
        parent: TrainingJobRecord,
        target: TrainingConfigRecord,
        validated: ValidatedResumeState,
    ) -> ResumeCompatibility:
        try:
            source_snapshot = json.loads(parent.config_snapshot or "")
        except json.JSONDecodeError:
            source_snapshot = None
        source_metadata = _runtime_metadata(parent.runtime_directory)
        snapshot = session.scalar(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.id == target.dataset_snapshot_id
            )
        )
        model = session.scalar(
            select(ManagedModelRecord).where(
                ManagedModelRecord.id == target.managed_model_id
            )
        )
        issues: list[str] = []
        if not isinstance(source_snapshot, dict) or not isinstance(
            source_metadata, dict
        ):
            issues.append("source job metadata is missing")
        if (
            snapshot is None
            or model is None
            or snapshot.content_sha256 is None
            or model.local_sha256 is None
        ):
            issues.append("dataset or model metadata is missing")
        source_values = _source_values(parent, source_snapshot, source_metadata)
        target_values = _target_values(target, snapshot, model, self.settings)
        for key in _REQUIRED_COMPATIBILITY_KEYS:
            if source_values.get(key) is None or target_values.get(key) is None:
                issues.append(f"{key} metadata is missing")
            elif source_values[key] != target_values[key]:
                issues.append(f"{key} is incompatible")
        source_epochs = _config_epochs(parent.config_snapshot)
        if source_epochs is not None and target.epochs < source_epochs:
            issues.append("epochs cannot be reduced")
        source_fingerprint = (
            _fingerprint_values(source_values) if source_values else None
        )
        target_fingerprint = (
            _fingerprint_values(target_values) if target_values else None
        )
        return ResumeCompatibility(
            (
                ResumeValidationStatus.COMPATIBLE
                if not issues
                else (
                    ResumeValidationStatus.MISSING_METADATA
                    if any("missing" in issue for issue in issues)
                    else ResumeValidationStatus.INCOMPATIBLE
                )
            ),
            tuple(dict.fromkeys(issues)),
            source_fingerprint,
            target_fingerprint,
        )

    def _signature(
        self,
        parent: TrainingJobRecord,
        artifact: TrainingArtifactRecord,
        target: TrainingConfigRecord,
        validated: ValidatedResumeState,
        compatibility: ResumeCompatibility,
        session: Any,
    ) -> str:
        active = session.scalar(
            select(TrainingJobRecord.id).where(
                TrainingJobRecord.project_id == parent.project_id,
                TrainingJobRecord.status.in_(
                    [status.value for status in ACTIVE_STATUSES]
                ),
            )
        )
        payload = {
            "parent": [
                parent.id,
                parent.status,
                (_as_utc(parent.updated_at) or datetime.now(UTC)).isoformat(),
            ],
            "artifact": [
                artifact.id,
                artifact.validation_status,
                validated.fingerprint,
            ],
            "target": [
                target.id,
                target.updated_at.isoformat(),
                target.epochs,
                compatibility.target_config_fingerprint,
            ],
            "active_job": active,
            "validator": VALIDATOR_VERSION,
            "compatibility": compatibility.status.value,
        }
        return _sha256_bytes(_canonical_json(payload))

    def _manifest(
        self,
        parent: TrainingJobRecord,
        artifact: TrainingArtifactRecord,
        config: TrainingConfigRecord,
        state: ValidatedResumeState,
        session: Any,
        initial_epoch: int | None,
        initial_step: int | None,
        request_fingerprint: str | None,
        target_config_fingerprint: str | None,
        position_warning: str | None,
    ) -> dict[str, Any]:
        snapshot = session.scalar(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.id == config.dataset_snapshot_id
            )
        )
        model = session.scalar(
            select(ManagedModelRecord).where(
                ManagedModelRecord.id == config.managed_model_id
            )
        )
        return {
            "manifest_version": "phase6c-v1",
            "source_job_id": parent.id,
            "source_artifact_id": artifact.id,
            "state_relative_path": artifact.relative_path,
            "state_fingerprint": state.fingerprint,
            "state_epoch": state.state_epoch,
            "state_step": state.state_step,
            "file_count": len(state.files),
            "total_size": state.total_size,
            "files": [
                {
                    "path": str(item.relative_path).replace("\\", "/"),
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in state.files
            ],
            "dataset_snapshot_id": config.dataset_snapshot_id,
            "dataset_content_sha256": snapshot.content_sha256 if snapshot else None,
            "model_id": config.managed_model_id,
            "model_sha256": model.local_sha256 if model else None,
            "trainer_script": config.trainer_script,
            "network_module": config.network_module,
            "network_dim": config.network_dim,
            "network_alpha": config.network_alpha,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "mixed_precision": config.mixed_precision,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "initial_epoch": initial_epoch,
            "initial_step": initial_step,
            "progress_epoch_offset": initial_epoch,
            "progress_step_offset": initial_step,
            "target_config_id": config.id,
            "target_config_fingerprint": target_config_fingerprint,
            "resume_request_fingerprint": request_fingerprint,
            "position_warning": position_warning,
            "validator_version": VALIDATOR_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _mark_validation(
        child: TrainingJobRecord,
        compatibility: ResumeCompatibility,
        session: Any,
    ) -> None:
        child.resume_validation_status = compatibility.status.value
        child.resume_validation_code = "RESUME_COMPATIBILITY_FAILED"
        child.resume_validation_message = "; ".join(compatibility.issues)


_REQUIRED_COMPATIBILITY_KEYS = (
    "dataset_snapshot_id",
    "dataset_content_sha256",
    "dataset_toml_sha256",
    "model_id",
    "model_sha256",
    "trainer_script",
    "network_module",
    "network_dim",
    "network_alpha",
    "optimizer",
    "scheduler",
    "mixed_precision",
    "cache_latents",
    "gradient_checkpointing",
    "resolution",
    "batch_size",
    "learning_rate",
    "extra_options",
    "dataset_repeats",
    "seed",
    "sd_scripts_root",
    "trusted_python_executable",
    "command_builder_version",
)


def _trusted_runtime(value: str | None, settings: AppSettings) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    root = (
        settings.training_jobs_dir or settings.workspace_root / "training" / "jobs"
    ).resolve()
    if raw.is_symlink():
        return None
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
        return resolved
    except (OSError, ValueError):
        return None


def _runtime_metadata(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        data = json.loads(
            (Path(value) / "runtime" / "metadata.json").read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _source_values(
    parent: TrainingJobRecord,
    snapshot: object,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    result = {key: snapshot.get(key) for key in _REQUIRED_COMPATIBILITY_KEYS}
    result["dataset_snapshot_id"] = parent.dataset_snapshot_id
    result["model_id"] = parent.managed_model_id
    if metadata:
        result["dataset_content_sha256"] = metadata.get("source_dataset_content_sha256")
        result["dataset_toml_sha256"] = metadata.get("source_dataset_toml_sha256")
        result["trusted_python_executable"] = metadata.get("trusted_python_executable")
        result["command_builder_version"] = metadata.get("command_builder_version")
        result["model_sha256"] = metadata.get("source_model_sha256")
        result["dataset_repeats"] = metadata.get("dataset_num_repeats")
    return result


def _target_values(
    config: TrainingConfigRecord,
    snapshot: DatasetSnapshotRecord | None,
    model: ManagedModelRecord | None,
    settings: AppSettings,
) -> dict[str, Any]:
    return {
        "dataset_snapshot_id": config.dataset_snapshot_id,
        "dataset_content_sha256": snapshot.content_sha256 if snapshot else None,
        "dataset_toml_sha256": snapshot.dataset_toml_sha256 if snapshot else None,
        "model_id": config.managed_model_id,
        "model_sha256": model.local_sha256 if model else None,
        "trainer_script": config.trainer_script,
        "network_module": config.network_module,
        "network_dim": config.network_dim,
        "network_alpha": config.network_alpha,
        "optimizer": config.optimizer,
        "scheduler": config.scheduler,
        "mixed_precision": config.mixed_precision,
        "cache_latents": bool(config.cache_latents),
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "resolution": config.resolution,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "extra_options": _json_object(config.extra_options),
        "dataset_repeats": _dataset_repeats(snapshot),
        "seed": config.seed,
        "sd_scripts_root": str(settings.training_sd_scripts_root.resolve()),
        "trusted_python_executable": str(
            Path(settings.training_python_executable).resolve()
        ),
        "command_builder_version": "phase6c-v1",
    }


def _config_epochs(snapshot: str | None) -> int | None:
    try:
        value = json.loads(snapshot or "{}").get("epochs")
        return int(value) if isinstance(value, (int, float)) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _dataset_repeats(snapshot: DatasetSnapshotRecord | None) -> list[int] | None:
    if snapshot is None:
        return None
    try:
        data = tomllib.loads(
            Path(snapshot.dataset_toml_path).read_text(encoding="utf-8")
        )
        datasets = data.get("datasets", [])
        result = [
            int(subset["num_repeats"])
            for dataset in datasets
            for subset in dataset.get("subsets", [])
        ]
        return result or None
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError):
        return None


def _json_object(value: str | None) -> object:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        return (
            parsed if isinstance(parsed, (dict, list, str, int, float, bool)) else None
        )
    except (TypeError, json.JSONDecodeError):
        return None


def _fingerprint_values(values: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(values))


def _resume_request_fingerprint(
    parent_job_id: str,
    artifact_id: str,
    target_config_id: str,
    state_fingerprint: str,
    target_config_fingerprint: str | None,
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "parent_job_id": parent_job_id,
                "resume_artifact_id": artifact_id,
                "target_training_config_id": target_config_id,
                "state_fingerprint": state_fingerprint,
                "target_config_fingerprint": target_config_fingerprint,
                "validator_version": VALIDATOR_VERSION,
            }
        )
    )


def _state_position(artifact: TrainingArtifactRecord) -> tuple[int | None, int | None]:
    metadata = _json_object(artifact.metadata_json)
    values = metadata if isinstance(metadata, dict) else {}
    epoch = artifact.epoch
    step = artifact.step
    if epoch is None:
        epoch = _metadata_int(values, ("state_epoch", "epoch", "current_epoch"))
    if step is None:
        step = _metadata_int(values, ("state_step", "step", "current_step"))
    return epoch, step


def _metadata_int(values: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = values.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _state_position_warning(
    parent_epoch: int | None,
    parent_step: int | None,
    state_epoch: int | None,
    state_step: int | None,
) -> str | None:
    differences: list[str] = []
    if state_epoch is None:
        differences.append("state epoch is unknown")
    if state_step is None:
        differences.append("state step is unknown")
    if (
        parent_epoch is not None
        and state_epoch is not None
        and parent_epoch != state_epoch
    ):
        differences.append(f"epoch parent={parent_epoch}, state={state_epoch}")
    if parent_step is not None and state_step is not None and parent_step != state_step:
        differences.append(f"step parent={parent_step}, state={state_step}")
    return (
        "selected state position differs from parent progress: "
        + ", ".join(differences)
        if differences
        else None
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value
