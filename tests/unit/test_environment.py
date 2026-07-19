from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.environment import (
    GPUInfo,
    _query_nvidia_smi,
    _query_torch_gpus,
    collect_environment_report,
)


def test_collect_environment_report_returns_expected_shape(
    test_workspace: Path,
) -> None:
    report = collect_environment_report(AppSettings(workspace_root=test_workspace))

    payload = report.to_dict()

    assert payload["runtime_dir"] == str(test_workspace)
    assert "disk_free_bytes" in payload
    assert "disk_total_bytes" in payload
    assert "commands" in payload
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["errors"], list)


def test_runpod_environment_is_detected_from_settings(test_workspace: Path) -> None:
    settings = AppSettings(workspace_root=test_workspace, runpod_pod_id="pod-test")

    report = collect_environment_report(settings)

    assert report.is_runpod is True
    assert report.runpod_pod_id == "pod-test"


def test_nvidia_smi_empty_or_invalid_output_is_safe(monkeypatch) -> None:
    monkeypatch.setattr(
        "runpod_lora_studio.environment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="\ninvalid\n,\n"),
    )

    assert _query_nvidia_smi() == []


def test_multiple_gpu_information_is_collected() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: ["GPU A", "GPU B"][index],
        mem_get_info=lambda index: (0, (8 + index * 4) * 1024 * 1024),
    )

    gpus, available = _query_torch_gpus(SimpleNamespace(cuda=cuda))

    assert available is True
    assert gpus == [GPUInfo(0, "GPU A", 8), GPUInfo(1, "GPU B", 12)]


def test_missing_required_command_is_an_error(
    test_workspace: Path, monkeypatch
) -> None:
    def which(command: str) -> str | None:
        return None if command == "git" else f"/usr/bin/{command}"

    monkeypatch.setattr("runpod_lora_studio.environment.shutil.which", which)

    report = collect_environment_report(AppSettings(workspace_root=test_workspace))

    assert any("git" in error for error in report.errors)
