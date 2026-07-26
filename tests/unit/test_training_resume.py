from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_training import _config, _fixture, _wait

from runpod_lora_studio.domain.training_models import (
    TrainingConfig,
    TrainingConfigInput,
    TrainingJobStatus,
)
from runpod_lora_studio.domain.training_progress_models import (
    ParsedTrainingProgress,
    TrainingLogParseResult,
    TrainingLogParserState,
    TrainingMetricEvent,
    TrainingProgressSource,
)
from runpod_lora_studio.domain.training_resume_models import (
    parse_non_negative_integer,
)
from runpod_lora_studio.external.training_process import FakeTrainingProcessAdapter
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import (
    TrainingJobRecord,
    TrainingProgressRecord,
)
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.training_command import SdScriptsCommandBuilder
from runpod_lora_studio.services.training_progress_service import (
    _apply_resume_offsets,
)
from runpod_lora_studio.services.training_resume_service import (
    TrainingResumeService,
    validate_state_position,
)
from runpod_lora_studio.services.training_service import TrainingService


def test_resume_progress_offsets_local_steps_and_epochs() -> None:
    result = TrainingLogParseResult(
        ParsedTrainingProgress(
            current_epoch=2,
            total_epochs=5,
            current_step=30,
            total_steps=100,
            latest_loss=0.2,
            learning_rate=1e-4,
            speed=2.0,
            elapsed_seconds=10.0,
            estimated_remaining_seconds=20.0,
            progress_ratio=0.3,
            progress_source=TrainingProgressSource.LOG,
            metric_events=(TrainingMetricEvent("loss", 0.2, epoch=2, step=30),),
        ),
        TrainingLogParserState(),
    )
    shifted = _apply_resume_offsets(result, 100, 3)
    assert shifted.progress.current_step == 130
    assert shifted.progress.current_epoch == 5
    assert shifted.progress.total_steps == 200
    assert shifted.progress.total_epochs == 8
    assert shifted.progress.metric_events[0].step == 130
    assert shifted.progress.metric_events[0].epoch == 5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ("0", 0),
        ("500", 500),
        (True, None),
        (False, None),
        (-1, None),
        ("-1", None),
        (1.5, None),
        ("1.5", None),
        ("1e3", None),
        ("", None),
        ([], None),
        ({}, None),
        ("１２", None),
    ],
)
def test_parse_non_negative_integer_is_strict(
    value: object, expected: int | None
) -> None:
    assert parse_non_negative_integer(value) == expected


def _position_artifact(
    *, epoch: object = None, step: object = None, metadata: object = None
) -> SimpleNamespace:
    return SimpleNamespace(
        epoch=epoch,
        step=step,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
    )


def test_state_position_requires_consistent_sources_and_respects_bounds(
    test_workspace: Path,
) -> None:
    state = test_workspace / "test-lora-500-state"
    state.mkdir()
    (state / "training_state.json").write_text(
        json.dumps({"epoch": "5", "step": "500"}), encoding="utf-8"
    )
    artifact = _position_artifact(epoch=5, step=500)
    position = validate_state_position(
        artifact,
        state,
        target_total_epochs=10,
        estimated_total_steps=1000,
        max_epoch=100,
        max_step=10_000,
    )
    assert position.epoch == 5
    assert position.step == 500
    assert position.epoch_source == "artifact"
    assert position.step_source == "artifact"

    (state / "state.json").write_text(json.dumps({"state_step": 501}), encoding="utf-8")
    with pytest.raises(UserFacingError, match="STATE_POSITION_CONFLICT"):
        validate_state_position(
            artifact,
            state,
            target_total_epochs=10,
            estimated_total_steps=1000,
            max_epoch=100,
            max_step=10_000,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"epoch": True, "step": 1},
        {"epoch": 1.5, "step": 1},
        {"epoch": "1.5", "step": 1},
        {"epoch": "", "step": 1},
        {"epoch": "1e3", "step": 1},
        {"epoch": [], "step": 1},
        {"epoch": -1, "step": 1},
    ],
)
def test_state_position_rejects_invalid_metadata(
    test_workspace: Path, metadata: dict[str, object]
) -> None:
    state = test_workspace / "test-lora-1-state"
    state.mkdir()
    (state / "training_state.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UserFacingError, match="STATE_POSITION"):
        validate_state_position(
            _position_artifact(),
            state,
            target_total_epochs=10,
            estimated_total_steps=1000,
            max_epoch=100,
            max_step=10_000,
        )


def test_state_position_rejects_target_and_estimated_step_overflow(
    test_workspace: Path,
) -> None:
    state = test_workspace / "test-lora-500-state"
    state.mkdir()
    (state / "training_state.json").write_text(
        json.dumps({"epoch": 5, "step": 500}), encoding="utf-8"
    )
    with pytest.raises(UserFacingError, match="STATE_POSITION_OUT_OF_RANGE"):
        validate_state_position(
            _position_artifact(epoch=5, step=500),
            state,
            target_total_epochs=4,
            estimated_total_steps=1000,
            max_epoch=100,
            max_step=10_000,
        )
    with pytest.raises(UserFacingError, match="STATE_POSITION_OUT_OF_RANGE"):
        validate_state_position(
            _position_artifact(epoch=5, step=500),
            state,
            target_total_epochs=10,
            estimated_total_steps=499,
            max_epoch=100,
            max_step=10_000,
        )


def test_resume_command_is_fixed_and_redacts_state_path(test_workspace: Path) -> None:
    settings, _, _, _ = _fixture(test_workspace)
    builder = SdScriptsCommandBuilder(
        trusted_trainer_root=settings.training_sd_scripts_root
    )
    model = settings.models_dir / "base" / "model.safetensors"
    model.write_bytes(b"model")
    dataset = settings.workspace_root / "dataset.toml"
    resume = settings.workspace_root / "resume" / "source-state"
    dataset.write_text("", encoding="utf-8")
    resume.mkdir(parents=True)
    config = TrainingConfig(
        id=UUID(int=1),
        project_id=UUID(int=2),
        dataset_snapshot_id=UUID(int=3),
        managed_model_id=UUID(int=4),
        name="config",
        output_name="out",
        output_directory=settings.workspace_root / "output",
        sd_scripts_root=settings.training_sd_scripts_root,
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
        cache_latents=False,
        gradient_checkpointing=False,
        seed=42,
        extra_options={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    command = builder.build(
        config=config,
        model_path=model,
        dataset_config_path=dataset,
        allowed_model_roots=(settings.models_dir,),
        allowed_dataset_roots=(settings.workspace_root,),
        allowed_output_roots=(settings.workspace_root,),
        resume_path=resume,
        allowed_resume_roots=(settings.workspace_root / "resume",),
    )
    assert command.arguments[command.arguments.index("--resume") + 1] == str(resume)
    assert str(resume) not in command.summary


def test_resume_state_fingerprint_is_deterministic_and_detects_changes(
    test_workspace: Path,
) -> None:
    settings, _, _, _ = _fixture(test_workspace)
    service = TrainingResumeService(settings)
    state = test_workspace / "state"
    state.mkdir()
    (state / "optimizer.pt").write_bytes(b"optimizer")
    (state / "scheduler.pt").write_bytes(b"scheduler")

    first = service._scan_state_path(state, UUID(int=1), UUID(int=2))
    second = service._scan_state_path(state, UUID(int=1), UUID(int=2))
    assert first.fingerprint == second.fingerprint
    assert first.source_path == state.resolve()

    (state / "optimizer.pt").write_bytes(b"changed")
    changed = service._scan_state_path(state, UUID(int=1), UUID(int=2))
    assert changed.fingerprint != first.fingerprint

    empty = test_workspace / "empty-state"
    empty.mkdir()
    with pytest.raises(UserFacingError):
        service._scan_state_path(empty, UUID(int=1), UUID(int=2))


def test_resume_creates_child_copies_state_and_preserves_parent(
    test_workspace: Path,
) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=False, exit_code=0)
    service = TrainingService(settings, process_adapter=fake)
    try:
        config = _config(service, project_id, snapshot_id, model_id, epochs=10)
        parent_id = service.create_job(config.id)
        service.start_job(parent_id)
        _wait(service, parent_id, TrainingJobStatus.SUCCEEDED)

        parent_output = service.jobs_root / str(parent_id) / "output"
        source_state = parent_output / "test-lora-500-state"
        source_state.mkdir()
        (source_state / "optimizer.pt").write_bytes(b"optimizer-state")
        (source_state / "scheduler.pt").write_bytes(b"scheduler-state")
        (source_state / "training_state.json").write_text(
            json.dumps({"epoch": 5, "step": 500}), encoding="utf-8"
        )
        service.rescan_artifacts(parent_id)
        states = service.list_resume_states(parent_id)
        assert len(states) == 1
        artifact_id = UUID(states[0]["id"])

        with Session(create_engine_for_settings(settings)) as session:
            now = datetime.now(UTC)
            progress = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == str(parent_id)
                )
            )
            assert progress is not None
            progress.current_epoch = 10
            progress.total_epochs = 20
            progress.current_step = 1000
            progress.total_steps = 2000
            progress.progress_ratio = 0.5
            progress.latest_loss = 0.2
            progress.smoothed_loss = 0.2
            progress.learning_rate = 1e-4
            progress.steps_per_second = 1.0
            progress.samples_per_second = 1.0
            progress.elapsed_seconds = 10.0
            progress.estimated_remaining_seconds = 10.0
            progress.latest_log_at = now
            progress.updated_at = now
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(parent_id))
            )
            assert record is not None
            record.status = TrainingJobStatus.FAILED.value
            record.failure_code = "test_failure"
            record.updated_at = datetime.now(UTC)
            session.commit()

        preview = service.preview_resume(parent_id, artifact_id)
        assert preview.compatibility.status.value == "compatible"
        assert preview.state_epoch == 5
        assert preview.state_step == 500
        assert preview.initial_epoch == 5
        assert preview.initial_step == 500
        assert preview.state_epoch_source == "artifact"
        assert preview.state_step_source == "artifact"
        assert preview.position_warning is not None
        child_id = service.create_resume_job(
            parent_id, artifact_id, preview_signature=preview.signature
        )
        child = service.get_job(child_id)
        assert child.parent_job_id == parent_id
        assert child.resume_artifact_id == artifact_id
        assert child.initial_epoch == 5
        assert child.initial_step == 500
        assert child.initial_epoch_source == "artifact"
        assert child.initial_step_source == "artifact"
        assert child.resume_request_fingerprint is not None
        assert (
            service.create_resume_job(
                parent_id, artifact_id, preview_signature=preview.signature
            )
            == child_id
        )

        service.start_job(child_id)
        _wait(service, child_id, TrainingJobStatus.SUCCEEDED)
        command = fake.start_calls[-1][0]
        resume_index = command.index("--resume")
        copied_state = Path(command[resume_index + 1])
        assert copied_state == (
            service.jobs_root / str(child_id) / "runtime" / "resume" / "source-state"
        )
        assert (copied_state / "optimizer.pt").read_bytes() == b"optimizer-state"
        manifest_path = (
            service.jobs_root / str(child_id) / "config" / "resume-state-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_job_id"] == str(parent_id)
        assert manifest["state_epoch"] == 5
        assert manifest["state_step"] == 500
        assert manifest["initial_epoch"] == 5
        assert manifest["initial_step"] == 500
        assert manifest["state_epoch_source"] == "artifact"
        assert manifest["state_step_source"] == "artifact"
        assert manifest["progress_step_offset"] == 500
        assert (
            manifest["resume_request_fingerprint"] == child.resume_request_fingerprint
        )
        assert manifest["position_warning"] is not None

        parent = service.get_job(parent_id)
        assert parent.status is TrainingJobStatus.FAILED
        assert not (
            service.jobs_root / str(parent_id) / "config" / "resume-state-manifest.json"
        ).exists()

        extended_config = service.create_config(
            TrainingConfigInput(
                project_id=project_id,
                dataset_snapshot_id=snapshot_id,
                managed_model_id=model_id,
                name="extended-config",
                output_name="extended-lora",
                output_directory=settings.outputs_dir,
                sd_scripts_root=settings.training_sd_scripts_root,
                epochs=12,
            )
        )
        extended_preview = service.preview_resume(
            parent_id, artifact_id, extended_config.id
        )
        extended_child_id = service.create_resume_job(
            parent_id,
            artifact_id,
            target_config_id=extended_config.id,
            preview_signature=extended_preview.signature,
        )
        assert extended_child_id != child_id
        assert (
            service.get_job(extended_child_id).training_config_id == extended_config.id
        )
    finally:
        service.close()


def test_resume_rejects_active_parent_and_changed_state(test_workspace: Path) -> None:
    settings, project_id, snapshot_id, model_id = _fixture(test_workspace)
    service = TrainingService(
        settings, process_adapter=FakeTrainingProcessAdapter(running=False)
    )
    try:
        config = _config(service, project_id, snapshot_id, model_id)
        parent_id = service.create_job(config.id)
        output = service.jobs_root / str(parent_id) / "output"
        state = output / "test-lora-1-state"
        state.mkdir(parents=True)
        (state / "optimizer.pt").write_bytes(b"state")
        (state / "training_state.json").write_text(
            json.dumps({"epoch": 1, "step": 1}), encoding="utf-8"
        )
        with Session(create_engine_for_settings(settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(parent_id))
            )
            assert record is not None
            record.runtime_directory = str(service.jobs_root / str(parent_id))
            record.status = TrainingJobStatus.FAILED.value
            session.commit()
        service.rescan_artifacts(parent_id)
        artifact_id = UUID(service.list_resume_states(parent_id)[0]["id"])

        with Session(create_engine_for_settings(settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(parent_id))
            )
            assert record is not None
            record.status = TrainingJobStatus.RUNNING.value
            session.commit()
        with pytest.raises(UserFacingError):
            service.preview_resume(parent_id, artifact_id)

        with Session(create_engine_for_settings(settings)) as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(parent_id))
            )
            assert record is not None
            record.status = TrainingJobStatus.FAILED.value
            session.commit()
        (state / "optimizer.pt").write_bytes(b"changed-after-scan")
        with pytest.raises(UserFacingError):
            service.preview_resume(parent_id, artifact_id)
    finally:
        service.close()


def test_all_resumable_statuses_require_process_end_confirmation(
    test_workspace: Path,
) -> None:
    settings, _, _, _ = _fixture(test_workspace)
    fake = FakeTrainingProcessAdapter(running=True)
    fake._processes[41000] = (True, None, 41000, "identity")
    service = TrainingResumeService(settings, process_adapter=fake)
    try:
        for status in (
            TrainingJobStatus.FAILED,
            TrainingJobStatus.CANCELED,
            TrainingJobStatus.STALE,
        ):
            parent = SimpleNamespace(
                status=status.value,
                pid=41000,
                process_group_id=41000,
                process_identity="identity",
                process_start_time=None,
                worker_id="worker",
                finished_at=None,
                exit_code=None,
                failure_code=None,
                worker_heartbeat=datetime.now(UTC),
            )
            assert not service._source_is_safe(parent)

        finished = SimpleNamespace(
            status=TrainingJobStatus.FAILED.value,
            pid=41000,
            process_group_id=41000,
            process_identity="identity",
            process_start_time=None,
            worker_id="worker",
            finished_at=datetime.now(UTC),
            exit_code=3,
            failure_code="process_exit_nonzero",
            worker_heartbeat=datetime.now(UTC),
        )
        fake._processes[41000] = (False, 3, 41000, "identity")
        assert service._source_is_safe(finished)

        uncertain = SimpleNamespace(
            status=TrainingJobStatus.CANCELED.value,
            pid=None,
            process_group_id=None,
            process_identity=None,
            process_start_time=None,
            worker_id="worker",
            finished_at=None,
            exit_code=None,
            failure_code=None,
            worker_heartbeat=datetime.now(UTC),
        )
        assert not service._source_is_safe(uncertain)

        no_process_metadata = SimpleNamespace(
            status=TrainingJobStatus.FAILED.value,
            pid=None,
            process_group_id=None,
            process_identity=None,
            process_start_time=None,
            worker_id=None,
            finished_at=None,
            exit_code=None,
            failure_code=None,
            worker_heartbeat=None,
        )
        assert service._source_is_safe(no_process_metadata)
    finally:
        del service
