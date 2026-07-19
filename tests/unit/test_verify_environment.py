from __future__ import annotations

import runpy
from pathlib import Path

from runpod_lora_studio.environment import EnvironmentReport


def test_verify_environment_returns_nonzero_for_required_errors() -> None:
    module = runpy.run_path("scripts/verify_environment.py")
    report = EnvironmentReport(
        python_version="3.11.0",
        python_supported=True,
        platform="test",
        is_runpod=False,
        runpod_pod_id=None,
        torch_version=None,
        torch_cuda_version=None,
        torch_cuda_available=False,
        errors=["gitがありません"],
    )

    main = module["main"]

    assert main(["--json"], report) == 1


def test_verify_environment_returns_nonzero_for_unwritable_workspace() -> None:
    module = runpy.run_path("scripts/verify_environment.py")
    report = EnvironmentReport(
        python_version="3.11.0",
        python_supported=True,
        platform="test",
        is_runpod=False,
        runpod_pod_id=None,
        torch_version=None,
        torch_cuda_version=None,
        torch_cuda_available=False,
        runtime_dir=str(Path("/unwritable")),
        workspace_writable=False,
        errors=["作業ディレクトリへ書き込みできません"],
    )

    assert module["main"](["--json"], report) == 1


def test_verify_environment_formats_disk_as_gib(capsys) -> None:
    module = runpy.run_path("scripts/verify_environment.py")
    gib = 1024**3
    report = EnvironmentReport(
        python_version="3.11.0",
        python_supported=True,
        platform="test",
        is_runpod=False,
        runpod_pod_id=None,
        torch_version=None,
        torch_cuda_version=None,
        torch_cuda_available=False,
        disk_free_bytes=int(120.5 * gib),
        disk_total_bytes=250 * gib,
    )

    assert module["main"]([], report) == 0
    assert "ディスク: 空き 120.5 GiB / 総容量 250.0 GiB" in capsys.readouterr().out
