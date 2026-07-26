from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifactType,
    TrainingArtifactValidationStatus,
    TrainingParseStatus,
    TrainingProgressSnapshot,
    TrainingProgressSource,
)
from runpod_lora_studio.ui.training_controller import TrainingController


def test_training_controller_formats_progress_and_artifacts() -> None:
    job_id = uuid4()
    progress = TrainingProgressSnapshot(
        job_id,
        2,
        10,
        20,
        100,
        0.2,
        0.12,
        0.15,
        0.0001,
        2.0,
        2.0,
        30.0,
        120.0,
        datetime.now(UTC),
        "phase6b-v1",
        TrainingParseStatus.WARNING,
        "parser warning",
        TrainingProgressSource.LOG,
        datetime.now(UTC),
    )
    artifact = SimpleNamespace(
        filename="character-000010.safetensors",
        artifact_type=TrainingArtifactType.LORA_CHECKPOINT,
        epoch=2,
        step=10,
        file_size=123,
        sha256="a" * 64,
        validation_status=TrainingArtifactValidationStatus.VALID,
        validation_message="ok",
        modified_at=datetime.now(UTC),
    )
    service = SimpleNamespace(
        get_progress=lambda _: progress,
        list_metrics=lambda *_args, **_kwargs: [(20, 0.12, 2)],
        list_artifacts=lambda *_args, **_kwargs: [artifact],
    )
    controller = TrainingController(service)  # type: ignore[arg-type]

    assert controller.progress_row(str(job_id))[3] == "20.0%"
    assert controller.metric_rows(str(job_id)) == [[20, 0.12, 2]]
    row = controller.artifact_rows(str(job_id))[0]
    assert row[0] == "character-000010.safetensors"
    assert "C:" not in row[0]
