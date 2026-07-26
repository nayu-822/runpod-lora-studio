from __future__ import annotations

import json
import struct
from pathlib import Path

from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifactValidationStatus,
    TrainingProgressSource,
)
from runpod_lora_studio.services.training_artifact import TrainingArtifactScanner
from runpod_lora_studio.services.training_log_parser import (
    TrainingLogParser,
    TrainingStepEstimator,
)
from runpod_lora_studio.services.training_log_reader import IncrementalLogReader


def test_parser_handles_realistic_output_ansi_carriage_return_and_invalid_loss() -> (
    None
):
    parser = TrainingLogParser()
    result = parser.parse(
        b"\x1b[2K\repoch 1/3\rsteps: 12/30\rloss=0.25 lr: 1e-4 2.30it/s\n",
        b"warning: ignored 999/1000\n",
    )
    progress = result.progress
    assert progress.current_epoch == 1
    assert progress.total_epochs == 3
    assert progress.current_step == 12
    assert progress.total_steps == 30
    assert progress.latest_loss == 0.25
    assert progress.learning_rate == 0.0001
    assert progress.speed == 2.30
    assert progress.progress_source is TrainingProgressSource.LOG

    nan_result = parser.parse(b"loss=nan\nloss=inf\n", state=result.state)
    assert nan_result.progress.latest_loss == 0.25


def test_parser_preserves_incomplete_log_line_and_warns_on_backwards_step() -> None:
    parser = TrainingLogParser()
    first = parser.parse(b"steps: 10/20\nloss=0.2\npartial")
    assert first.state.remainder == "partial"
    second = parser.parse(b" loss=0.1\nsteps: 2/20\n", state=first.state)
    assert second.progress.latest_loss == 0.1
    assert any("step decreased" in warning for warning in second.progress.warnings)


def test_incremental_reader_reads_new_bytes_and_handles_truncate_and_invalid_utf8(
    test_workspace: Path,
) -> None:
    logs = test_workspace / "logs"
    logs.mkdir()
    path = logs / "stdout.log"
    path.write_bytes(b"epoch 1/2\n")
    reader = IncrementalLogReader(logs, max_bytes=64)
    first = reader.read(path)
    assert first.data == b"epoch 1/2\n"
    path.write_bytes(b"epoch 1/2\nsteps: 1/4\n\xff")
    second = reader.read(path, first.cursor)
    assert b"steps: 1/4" in second.data
    path.write_bytes(b"steps: 0/4\n")
    third = reader.read(path, second.cursor)
    assert third.reset is True
    assert third.warning is not None


def test_step_estimator_sums_repeats_and_rounds_up() -> None:
    plan = TrainingStepEstimator.estimate(
        subset_image_counts=(3, 4),
        num_repeats=(2, 3),
        batch_size=4,
        epochs=2,
    )
    assert plan.steps_per_epoch == 5
    assert plan.total_steps == 10
    assert "ceil" in plan.formula


def test_artifact_scanner_validates_safetensors_and_rejects_unrelated_files(
    test_workspace: Path,
) -> None:
    output = test_workspace / "output"
    output.mkdir()
    header = {
        "__metadata__": {"epoch": "2", "step": "10"},
        "lora": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (output / "character-000010.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"data"
    )
    (output / "other.safetensors").write_bytes(b"not a safetensors file")
    (output / ".hidden.safetensors").write_bytes(b"ignored")
    state = output / "character-000010-state"
    state.mkdir()
    (state / "optimizer.pt").write_bytes(b"state")

    artifacts = TrainingArtifactScanner(output).scan("character")
    assert {artifact.filename for artifact in artifacts} == {
        "character-000010.safetensors",
        "character-000010-state",
    }
    lora = next(item for item in artifacts if item.filename.endswith("safetensors"))
    assert lora.validation_status is TrainingArtifactValidationStatus.VALID
    assert lora.sha256 is not None
    assert lora.epoch == 2
    assert lora.step == 10
