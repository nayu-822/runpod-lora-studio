from __future__ import annotations

import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.storage_models import RemoteSnapshotProvenance
from runpod_lora_studio.domain.training_completion_models import (
    TrainingCompletionErrorCode,
    TrainingCompletionStatus,
)
from runpod_lora_studio.domain.training_models import TrainingConfigInput
from runpod_lora_studio.external.fake_storage import FakeStorageTransferAdapter
from runpod_lora_studio.external.training_process import FakeTrainingProcessAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import (
    Base,
    DatasetSnapshotRecord,
    ManagedModelRecord,
    ModelTransferRecord,
    TrainingJobRecord,
)
from runpod_lora_studio.persistence.training_completion_repository import (
    TrainingCompletionRepository,
)
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.services.training_completion_service import (
    CompletionFailure,
    TrainingCompletionService,
)
from runpod_lora_studio.services.training_service import TrainingService


class CompletionStorageService(StorageService):
    def verify_remote_snapshot(self, snapshot_id: UUID) -> RemoteSnapshotProvenance:
        return RemoteSnapshotProvenance(
            snapshot_id=snapshot_id,
            remote_relative_path=f"snapshots/{snapshot_id}",
            storage_transfer_job_id=UUID(int=1),
            remote_manifest_sha256="b" * 64,
            content_sha256="a" * 64,
            verification_level="remote_hash_and_size",
        )


def _settings(root: Path) -> AppSettings:
    runtime = root / "runtime"
    settings = AppSettings(
        workspace_root=runtime,
        projects_dir=runtime / "projects",
        models_dir=runtime / "models",
        outputs_dir=runtime / "outputs",
        logs_dir=runtime / "logs",
        temp_dir=runtime / "tmp",
        database_path=runtime / "database" / "studio.sqlite3",
        training_jobs_dir=runtime / "training" / "jobs",
        training_sd_scripts_root=runtime / "sd-scripts",
        model_disk_safety_margin_bytes=0,
    )
    ensure_runtime_directories(settings)
    settings.training_sd_scripts_root.mkdir(parents=True, exist_ok=True)
    (settings.training_sd_scripts_root / "sdxl_train_network.py").write_text(
        "# fake trainer\n", encoding="utf-8"
    )
    Base.metadata.create_all(create_engine_for_settings(settings))
    return settings


def _fixture(root: Path) -> tuple[AppSettings, UUID, UUID, UUID]:
    settings = _settings(root)
    project = ProjectService(settings).create(ProjectInput("completion-test"))
    snapshot_id = uuid4()
    snapshot_root = settings.workspace_root / "snapshots" / str(snapshot_id)
    snapshot_root.mkdir(parents=True)
    dataset_toml = snapshot_root / "dataset.toml"
    manifest = snapshot_root / "manifest.json"
    report = snapshot_root / "report.json"
    dataset_toml.write_text(
        "[general]\nresolution = 1024\n\n[[datasets]]\n"
        "\n[[datasets.subsets]]\nimage_dir = 'images'\nnum_repeats = 1\n",
        encoding="utf-8",
    )
    manifest.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    model_id = uuid4()
    model_path = settings.models_dir / "base" / "model.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    model_sha256 = _sha256(model_path)
    now = datetime.now(UTC)
    with Session(create_engine_for_settings(settings)) as session:
        session.add(
            DatasetSnapshotRecord(
                id=str(snapshot_id),
                project_id=str(project.id),
                name="completed",
                description="",
                status="completed",
                snapshot_version="phase4-snapshot-v1",
                generator_version="test",
                source_project_version="1",
                source_tagger_run_id=None,
                source_created_at=now,
                target_image_count=1,
                copied_image_count=1,
                failed_image_count=0,
                warning_count=0,
                total_size_bytes=1,
                snapshot_root=str(snapshot_root),
                dataset_toml_path=str(dataset_toml),
                manifest_path=str(manifest),
                report_path=str(report),
                manifest_sha256=_sha256(manifest),
                dataset_toml_sha256=_sha256(dataset_toml),
                content_sha256="a" * 64,
                settings_snapshot="{}",
                validation_summary="{}",
                error_summary=None,
                started_at=now,
                completed_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ManagedModelRecord(
                id=str(model_id),
                display_name="test model",
                model_type="base_model",
                remote_name="gdrive",
                remote_relative_path="models/model.safetensors",
                remote_file_name="model.safetensors",
                remote_size_bytes=model_path.stat().st_size,
                remote_modified_at=now,
                remote_hash_type="sha256",
                remote_hash_value=model_sha256,
                local_path=str(model_path),
                local_size_bytes=model_path.stat().st_size,
                local_sha256=model_sha256,
                status="available",
                source="test",
                rclone_version="test",
                first_seen_at=now,
                last_seen_at=now,
                downloaded_at=now,
                verified_at=now,
                error_summary=None,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ModelTransferRecord(
                id=str(uuid4()),
                managed_model_id=str(model_id),
                direction="download",
                status="completed",
                source_path="gdrive:model.safetensors",
                destination_path=str(model_path),
                expected_size_bytes=model_path.stat().st_size,
                transferred_size_bytes=model_path.stat().st_size,
                expected_hash=model_sha256,
                actual_hash=model_sha256,
                attempt_count=1,
                retry_count=0,
                started_at=now,
                completed_at=now,
                error_summary=None,
                rclone_exit_code=0,
                rclone_version="test",
                settings_snapshot="{}",
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return settings, project.id, snapshot_id, model_id


def _config(
    service: TrainingService, project_id: UUID, snapshot_id: UUID, model_id: UUID
) -> UUID:
    return service.create_config(
        TrainingConfigInput(
            project_id=project_id,
            dataset_snapshot_id=snapshot_id,
            managed_model_id=model_id,
            name="test-config",
            output_name="test-lora",
            output_directory=service.settings.outputs_dir,
            sd_scripts_root=service.settings.training_sd_scripts_root,
            epochs=1,
        )
    ).id


def _write_valid_checkpoint(path: Path) -> None:
    header = {
        "__metadata__": {"epoch": "1", "step": "10"},
        "lora": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"data")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completion_fixture(
    test_workspace: Path,
) -> tuple[
    TrainingCompletionService,
    CompletionStorageService,
    TrainingService,
    UUID,
    UUID,
]:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    storage_adapter = FakeStorageTransferAdapter()
    storage = CompletionStorageService(settings, adapter=storage_adapter)
    training = TrainingService(
        settings,
        process_adapter=FakeTrainingProcessAdapter(running=False),
    )
    config = _config(training, project_id, snapshot_id, model_id)
    job_id = training.create_job(config)
    runtime = settings.training_jobs_dir / str(job_id)
    output = runtime / "output"
    output.mkdir(parents=True)
    _write_valid_checkpoint(output / "test-lora.safetensors")
    (runtime / "config").mkdir()
    (runtime / "config" / "training-config.json").write_text(
        '{"schema_version":"test"}\n', encoding="utf-8"
    )
    stdout = runtime / "stdout.log"
    stderr = runtime / "stderr.log"
    stdout.write_text("training complete\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    now = datetime.now(UTC)
    with Session(create_engine_for_settings(settings)) as session:
        record = session.scalar(
            select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
        )
        assert record is not None
        record.status = "succeeded"
        record.exit_code = 0
        record.pid = None
        record.runtime_directory = str(runtime)
        record.stdout_log_path = str(stdout)
        record.stderr_log_path = str(stderr)
        record.started_at = now - timedelta(minutes=1)
        record.finished_at = now
        record.updated_at = now
        session.commit()
    service = TrainingCompletionService(
        settings,
        storage_service=storage,
        training_service=training,
    )
    return service, storage, training, project_id, job_id


def test_completion_uploads_artifacts_verifies_and_writes_marker_last(
    test_workspace: Path,
) -> None:
    service, storage, training, project_id, job_id = _completion_fixture(test_workspace)
    try:
        preview = service.preview(job_id)
        export_id = service.synchronize_sync(job_id, preview_token=preview.token)
        export = service.get_export(export_id)
        assert export.status is TrainingCompletionStatus.COMPLETED
        assert export.completion_manifest_sha256

        target = storage.training_remote_path(project_id, job_id)
        marker_key = f"{target.relative_path}/completion-manifest.json"
        assert marker_key in storage.adapter.files
        assert storage.adapter.copy_calls[-1][1].endswith("completion-manifest.json")
        assert all(
            "dataset.toml" not in destination
            for _, destination in storage.adapter.copy_calls
        )

        copy_count = len(storage.adapter.copy_calls)
        service.synchronize_sync(
            job_id,
            preview_token=service.preview(job_id).token,
        )
        assert len(storage.adapter.copy_calls) == copy_count
        assert service.status_rows(project_id)[0][2] == "completed"
    finally:
        service.close()
        training.close()


def test_completion_requires_exact_final_lora_and_never_uses_checkpoint(
    test_workspace: Path,
) -> None:
    service, _storage, training, _project_id, job_id = _completion_fixture(
        test_workspace
    )
    try:
        with Session(create_engine_for_settings(service.settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            assert record is not None and record.runtime_directory is not None
            output = Path(record.runtime_directory) / "output"
            (output / "test-lora.safetensors").unlink()
            _write_valid_checkpoint(output / "test-lora-000001.safetensors")
        with pytest.raises(CompletionFailure) as raised:
            service.preview(job_id)
        assert raised.value.code == TrainingCompletionErrorCode.FINAL_LORA_MISSING.value
    finally:
        service.close()
        training.close()


def test_completion_rejects_training_job_that_is_not_succeeded(
    test_workspace: Path,
) -> None:
    service, _storage, training, _project_id, job_id = _completion_fixture(
        test_workspace
    )
    try:
        with Session(create_engine_for_settings(service.settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            assert record is not None
            record.status = "failed"
            record.exit_code = 1
            session.commit()
        with pytest.raises(CompletionFailure) as raised:
            service.preview(job_id)
        assert (
            raised.value.code
            == TrainingCompletionErrorCode.TRAINING_NOT_SUCCEEDED.value
        )
    finally:
        service.close()
        training.close()


def test_completion_claim_fencing_blocks_old_worker(
    test_workspace: Path,
) -> None:
    service, _storage, training, project_id, job_id = _completion_fixture(
        test_workspace
    )
    now = datetime.now(UTC)
    try:
        with Session(create_engine_for_settings(service.settings)) as session:
            repository = TrainingCompletionRepository(session)
            completion = repository.create(job_id, project_id, "a" * 64, "b" * 64)
            session.commit()
            export_id = UUID(completion.id)
            assert repository.claim(
                export_id,
                worker_id="worker-a",
                claim_token="claim-a",
                now=now,
                stale_after_seconds=120,
            )
            session.commit()
            assert repository.claim(
                export_id,
                worker_id="worker-b",
                claim_token="claim-b",
                now=now + timedelta(seconds=121),
                stale_after_seconds=120,
            )
            assert not repository.update_claimed(
                export_id,
                worker_id="worker-a",
                claim_token="claim-a",
                worker_generation=1,
                values={"current_stage": "old-worker-update"},
            )
            session.rollback()
    finally:
        service.close()
        training.close()
