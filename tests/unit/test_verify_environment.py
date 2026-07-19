from __future__ import annotations

import runpy

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
