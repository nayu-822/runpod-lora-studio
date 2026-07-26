from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories
from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
    TrainingJobStateMachine,
    TrainingJobStatus,
    TrainingJobTransitionError,
)
from runpod_lora_studio.external.training_process import FakeTrainingProcessAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import (
    Base,
    DatasetSnapshotRecord,
    ManagedModelRecord,
    ModelTransferRecord,
    TrainingJobRecord,
)
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)
from runpod_lora_studio.services.training_command import (
    SdScriptsCommandBuilder,
    TrainingCommandValidationError,
)
from runpod_lora_studio.services.training_service import TrainingService


def _settings(root: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=root / "runtime",
        projects_dir=root / "runtime" / "projects",
        models_dir=root / "runtime" / "models",
        outputs_dir=root / "runtime" / "outputs",
        logs_dir=root / "runtime" / "logs",
        temp_dir=root / "runtime" / "tmp",
        database_path=root / "runtime" / "database" / "studio.sqlite3",
        training_jobs_dir=root / "runtime" / "training" / "jobs",
        training_sd_scripts_root=root / "runtime" / "sd-scripts",
        training_heartbeat_interval_seconds=0.11,
        training_cancel_grace_seconds=0.2,
        training_job_stale_after_seconds=1.0,
        training_starting_grace_seconds=0.1,
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
    project = ProjectService(settings).create(ProjectInput("training-test"))
    snapshot_id = uuid4()
    snapshot_root = settings.workspace_root / "snapshots" / str(snapshot_id)
    snapshot_root.mkdir(parents=True)
    dataset_toml = snapshot_root / "dataset.toml"
    manifest = snapshot_root / "manifest.json"
    report = snapshot_root / "report.json"
    dataset_toml.write_text(
        "[general]\nresolution = 1024\n\n[[datasets]]\n\n[[datasets.subsets]]\n"
        "image_dir = 'images'\nnum_repeats = 2\n",
        encoding="utf-8",
    )
    manifest.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    model_id = uuid4()
    model_path = settings.models_dir / "base" / "model.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
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
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                dataset_toml_sha256=hashlib.sha256(
                    dataset_toml.read_bytes()
                ).hexdigest(),
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
    service: TrainingService,
    project_id: UUID,
    snapshot_id: UUID,
    model_id: UUID,
    *,
    epochs: int = 1,
):
    from runpod_lora_studio.domain.training_models import TrainingConfigInput

    return service.create_config(
        TrainingConfigInput(
            project_id=project_id,
            dataset_snapshot_id=snapshot_id,
            managed_model_id=model_id,
            name="test-config",
            output_name="test-lora",
            output_directory=service.settings.outputs_dir,
            sd_scripts_root=service.settings.training_sd_scripts_root,
            epochs=epochs,
        )
    )


def _write_valid_checkpoint(path: Path) -> None:
    header = {
        "__metadata__": {"epoch": "1", "step": "10"},
        "lora": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"data")


def _wait(service: TrainingService, job_id: UUID, expected: TrainingJobStatus) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if service.get_job(job_id).status is expected:
            return
        time.sleep(0.05)
    assert service.get_job(job_id).status is expected


def test_state_machine_rejects_terminal_restart() -> None:
    with pytest.raises(TrainingJobTransitionError):
        TrainingJobStateMachine.transition(
            TrainingJobStatus.SUCCEEDED, TrainingJobStatus.RUNNING
        )
    assert TrainingJobStateMachine.can_transition(
        TrainingJobStatus.RUNNING, TrainingJobStatus.CANCEL_REQUESTED
    )


def test_command_builder_uses_argument_array_and_redacts_paths(
    test_workspace: Path,
) -> None:
    root = test_workspace / "sd-scripts"
    root.mkdir()
    trainer = root / "sdxl_train_network.py"
    trainer.write_text("", encoding="utf-8")
    model = test_workspace / "model.safetensors"
    model.write_bytes(b"model")
    dataset = test_workspace / "dataset.toml"
    dataset.write_text("", encoding="utf-8")
    now = datetime.now(UTC)
    config = TrainingConfig(
        id=uuid4(),
        project_id=uuid4(),
        dataset_snapshot_id=uuid4(),
        managed_model_id=uuid4(),
        name="config",
        output_name="out",
        output_directory=test_workspace / "output",
        sd_scripts_root=root,
        trainer_script="sdxl_train_network.py",
        resolution=1024,
        batch_size=1,
        epochs=2,
        learning_rate=1e-4,
        optimizer="AdamW8bit",
        scheduler="cosine",
        network_module="networks.lora",
        network_dim=16,
        network_alpha=16,
        mixed_precision="fp16",
        save_every_n_epochs=1,
        cache_latents=True,
        gradient_checkpointing=True,
        seed=42,
        extra_options={"enable_bucket": True, "no_token_padding": False},
        created_at=now,
        updated_at=now,
    )
    command = SdScriptsCommandBuilder(trusted_trainer_root=root).build(
        config,
        model_path=model,
        dataset_config_path=dataset,
        allowed_model_roots=(test_workspace,),
        allowed_dataset_roots=(test_workspace,),
        allowed_output_roots=(test_workspace,),
    )
    assert "--cache_latents" in command.arguments
    assert "--gradient_checkpointing" in command.arguments
    assert "no_token_padding" not in command.arguments
    assert str(model) not in command.summary
    assert all(
        not (value.startswith("--") and " " in value) for value in command.arguments
    )
    for field_name, value in (
        ("network_module", "os"),
        ("optimizer", "UnknownOptimizer"),
        ("scheduler", "UnknownScheduler"),
    ):
        with pytest.raises(TrainingCommandValidationError):
            builder_config = replace(config, **{field_name: value})
            SdScriptsCommandBuilder(trusted_trainer_root=root).build(
                builder_config,
                model_path=model,
                dataset_config_path=dataset,
                allowed_model_roots=(test_workspace,),
                allowed_dataset_roots=(test_workspace,),
                allowed_output_roots=(test_workspace,),
            )
    for untrusted_root in (
        test_workspace / "other",
        test_workspace / "runtime" / "jobs",
        test_workspace / "runtime" / "outputs",
    ):
        with pytest.raises(TrainingCommandValidationError):
            SdScriptsCommandBuilder(trusted_trainer_root=root).build(
                replace(config, sd_scripts_root=untrusted_root),
                model_path=model,
                dataset_config_path=dataset,
                allowed_model_roots=(test_workspace,),
                allowed_dataset_roots=(test_workspace,),
                allowed_output_roots=(test_workspace,),
            )
    for executable in ("/bin/sh", "/bin/bash", "/bin/rm"):
        with pytest.raises(TrainingCommandValidationError):
            SdScriptsCommandBuilder(
                trusted_trainer_root=test_workspace, python_executable=executable
            ).validate_python_executable()


def test_python_symlink_cannot_escape_allowed_root(test_workspace: Path) -> None:
    outside = test_workspace / "outside" / "python"
    outside.parent.mkdir()
    outside.write_bytes(b"not executable")
    allowed_root = test_workspace / "allowed"
    allowed_root.mkdir()
    link = allowed_root / "python"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available")
    with pytest.raises(TrainingCommandValidationError):
        SdScriptsCommandBuilder(
            trusted_trainer_root=allowed_root, python_executable=link
        ).validate_python_executable()


def test_resolved_application_python_and_venv_symlink_are_trusted(
    test_workspace: Path,
) -> None:
    builder = SdScriptsCommandBuilder(
        trusted_trainer_root=test_workspace, python_executable=sys.executable
    )
    assert builder.validate_python_executable() == Path(sys.executable).resolve()
    link = test_workspace / "python"
    try:
        link.symlink_to(sys.executable)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available")
    linked_builder = SdScriptsCommandBuilder(
        trusted_trainer_root=test_workspace, python_executable=link
    )
    assert linked_builder.validate_python_executable() == Path(sys.executable).resolve()


def test_service_rejects_unapproved_training_values(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    service = TrainingService(settings, process_adapter=FakeTrainingProcessAdapter())

    def input_data(**overrides: object) -> TrainingConfigInput:
        values: dict[str, object] = {
            "project_id": project_id,
            "dataset_snapshot_id": snapshot_id,
            "managed_model_id": model_id,
            "name": "safe",
            "output_name": "safe-output",
            "output_directory": settings.outputs_dir,
            "sd_scripts_root": settings.training_sd_scripts_root,
        }
        values.update(overrides)
        return TrainingConfigInput(**values)  # type: ignore[arg-type]

    try:
        for field_name, value in (
            ("network_module", "os"),
            ("optimizer", "UnknownOptimizer"),
            ("scheduler", "UnknownScheduler"),
        ):
            with pytest.raises(UserFacingError):
                service.create_config(input_data(**{field_name: value}))
        with pytest.raises(UserFacingError):
            service.create_config(input_data(sd_scripts_root=test_workspace / "other"))
    finally:
        service.close()


def test_training_job_succeeds_with_fake_process(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=False, exit_code=0)
    service = TrainingService(settings, process_adapter=fake)
    try:
        config = _config(service, project_id, snapshot_id, model_id)
        job_id = service.create_job(config.id)
        service.start_job(job_id)
        _wait(service, job_id, TrainingJobStatus.SUCCEEDED)
        job = service.get_job(job_id)
        assert job.exit_code == 0
        assert job.pid == 41000
        assert job.stdout_log_path is not None
        assert job.stdout_log_path.is_relative_to(service.jobs_root)
        assert Path(fake.start_calls[0][0][0]).name.startswith("python")
        command = fake.start_calls[0][0]
        output_index = command.index("--output_dir") + 1
        assert Path(command[output_index]) == service.jobs_root / str(job_id) / "output"
        metadata = json.loads(
            (service.jobs_root / str(job_id) / "runtime" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["dataset_num_repeats"] == [2]
        assert (
            metadata["source_dataset_toml_sha256"]
            == metadata["job_dataset_toml_sha256"]
        )
        assert metadata["job_output_directory"] == "output"
        config_snapshot_path = (
            service.jobs_root / str(job_id) / "config" / "training-config.json"
        )
        snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        assert Path(snapshot["output_directory"]) == (
            service.jobs_root / str(job_id) / "output"
        )
    finally:
        service.close()


def test_artifacts_are_isolated_by_job_output_directory(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=False, exit_code=0)
    service = TrainingService(settings, process_adapter=fake)
    try:
        config = _config(service, project_id, snapshot_id, model_id)
        first = service.create_job(config.id)
        service.start_job(first)
        _wait(service, first, TrainingJobStatus.SUCCEEDED)
        first_output = service.jobs_root / str(first) / "output"
        _write_valid_checkpoint(first_output / "test-lora-000010.safetensors")
        _write_valid_checkpoint(settings.outputs_dir / "test-lora-000020.safetensors")
        service.rescan_artifacts(first)

        second = service.create_job(config.id)
        service.start_job(second)
        _wait(service, second, TrainingJobStatus.SUCCEEDED)
        second_output = service.jobs_root / str(second) / "output"
        _write_valid_checkpoint(second_output / "test-lora-000010.safetensors")
        service.rescan_artifacts(second)

        first_artifacts = service.list_artifacts(first)
        second_artifacts = service.list_artifacts(second)
        assert [item.relative_path.as_posix() for item in first_artifacts] == [
            "test-lora-000010.safetensors"
        ]
        assert [item.relative_path.as_posix() for item in second_artifacts] == [
            "test-lora-000010.safetensors"
        ]
    finally:
        service.close()


def test_nonzero_process_exit_is_failed_and_start_failure_does_not_stick(
    test_workspace: Path,
) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=False, exit_code=3)
    service = TrainingService(settings, process_adapter=fake)
    try:
        config = _config(service, project_id, snapshot_id, model_id)
        job_id = service.create_job(config.id)
        service.start_job(job_id)
        _wait(service, job_id, TrainingJobStatus.FAILED)
        assert service.get_job(job_id).failure_code == "process_exit_nonzero"
    finally:
        service.close()

    settings, project_id, snapshot_id, model_id = _fixture(
        test_workspace / "start-fail"
    )
    service = TrainingService(
        settings, process_adapter=FakeTrainingProcessAdapter(fail_start=True)
    )
    try:
        job_id = service.create_job(
            _config(service, project_id, snapshot_id, model_id).id
        )
        service.start_job(job_id)
        _wait(service, job_id, TrainingJobStatus.FAILED)
        assert service.get_job(job_id).failure_code == "worker_exception"
    finally:
        service.close()


def test_cancel_is_idempotent_and_uses_terminate(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=True, exit_code=None)
    service = TrainingService(settings, process_adapter=fake)
    try:
        job_id = service.create_job(
            _config(service, project_id, snapshot_id, model_id).id
        )
        service.start_job(job_id)
        _wait(service, job_id, TrainingJobStatus.RUNNING)
        assert "停止要求" in service.request_cancel(job_id)
        assert service.request_cancel(job_id) in {
            "停止要求を受け付けました",
            "終了済みの学習ジョブです",
        }
        _wait(service, job_id, TrainingJobStatus.CANCELED)
        assert fake.terminate_calls == [41000]
        assert fake.kill_calls == []
    finally:
        service.close()


def test_duplicate_active_job_and_stale_reconcile(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    service = TrainingService(settings, process_adapter=FakeTrainingProcessAdapter())
    try:
        config = _config(service, project_id, snapshot_id, model_id)
        first = service.create_job(config.id)
        with pytest.raises(UserFacingError):
            service.create_job(config.id)
        with Session(create_engine_for_settings(settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(first))
            )
            assert record is not None
            record.status = "running"
            record.pid = 99999
            record.worker_heartbeat = datetime.now(UTC) - timedelta(seconds=10)
            record.started_at = record.worker_heartbeat
            session.commit()
        assert service.reconcile_stale_jobs() == 1
        assert service.get_job(first).status is TrainingJobStatus.STALE
    finally:
        service.close()


def test_invalid_extra_option_and_log_tail_are_safe(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    service = TrainingService(settings, process_adapter=FakeTrainingProcessAdapter())
    try:
        from runpod_lora_studio.domain.training_models import TrainingConfigInput

        with pytest.raises(UserFacingError, match="unknown extra option"):
            service.create_config(
                TrainingConfigInput(
                    project_id=project_id,
                    dataset_snapshot_id=snapshot_id,
                    managed_model_id=model_id,
                    name="bad",
                    output_name="good",
                    output_directory=settings.outputs_dir,
                    sd_scripts_root=settings.training_sd_scripts_root,
                    extra_options={"--arbitrary": "command"},
                )
            )
        config = _config(service, project_id, snapshot_id, model_id)
        job_id = service.create_job(config.id)
        runtime = service.jobs_root / str(job_id)
        log = runtime / "logs"
        log.mkdir(parents=True)
        (log / "stdout.log").write_bytes(b"a" * 100 + b"\xfftail")
        with Session(create_engine_for_settings(settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            assert record is not None
            record.runtime_directory = str(runtime)
            record.stdout_log_path = str(log / "stdout.log")
            session.commit()
        output = service.tail_stdout(job_id, max_bytes=10)
        assert "tail" in output
        assert len(output.encode("utf-8")) <= 10
    finally:
        service.close()
