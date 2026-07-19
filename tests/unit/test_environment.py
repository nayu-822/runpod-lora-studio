from __future__ import annotations

from runpod_lora_studio.environment import collect_environment_report


def test_collect_environment_report_returns_expected_shape() -> None:
    report = collect_environment_report()

    payload = report.to_dict()

    assert "python_version" in payload
    assert "commands" in payload
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["errors"], list)
