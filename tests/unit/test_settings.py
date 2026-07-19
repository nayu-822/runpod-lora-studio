from __future__ import annotations

from pathlib import Path

from runpod_lora_studio.config.settings import AppSettings


def test_settings_defaults() -> None:
    settings = AppSettings()

    assert settings.app_title == "RunPod LoRA Studio"
    assert settings.gradio_server_name == "0.0.0.0"
    assert settings.gradio_server_port == 7860
    assert settings.workspace_root == Path("/workspace/ldts-runtime")


def test_settings_accept_runpod_aliases(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    settings = AppSettings()

    assert settings.runpod_pod_id == "pod-123"
