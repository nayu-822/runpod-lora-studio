from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from runpod_lora_studio.config.settings import AppSettings, ensure_runtime_directories


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RUNPOD_LORA_STUDIO_WORKSPACE_ROOT",
        "RUNPOD_LORA_STUDIO_PROJECTS_DIR",
        "RUNPOD_LORA_STUDIO_MODELS_DIR",
        "RUNPOD_LORA_STUDIO_OUTPUTS_DIR",
        "RUNPOD_LORA_STUDIO_LOGS_DIR",
        "RUNPOD_LORA_STUDIO_TEMP_DIR",
        "RUNPOD_LORA_STUDIO_DATABASE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings()

    assert settings.app_title == "RunPod LoRA Studio"
    assert settings.gradio_server_name == "0.0.0.0"
    assert settings.gradio_server_port == 7860
    assert settings.app_version
    assert settings.temp_dir == Path("/workspace/ldts-runtime/tmp")


def test_settings_accepts_prefixed_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_LORA_STUDIO_APP_TITLE", "Override Studio")
    monkeypatch.setenv("RUNPOD_LORA_STUDIO_GRADIO_SERVER_PORT", "9876")

    settings = AppSettings()

    assert settings.app_title == "Override Studio"
    assert settings.gradio_server_port == 9876


def test_settings_accept_runpod_aliases(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")

    settings = AppSettings()

    assert settings.runpod_pod_id == "pod-123"
    assert isinstance(settings.runpod_api_key, SecretStr)
    assert settings.runpod_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in str(settings)


@pytest.mark.parametrize("port", [0, 65536])
def test_settings_rejects_invalid_ports(port: int) -> None:
    with pytest.raises(ValidationError):
        AppSettings(gradio_server_port=port)


def test_settings_rejects_empty_server_name() -> None:
    with pytest.raises(ValidationError):
        AppSettings(gradio_server_name="  ")


def test_ensure_runtime_directories_creates_configured_paths(
    test_workspace: Path,
) -> None:
    settings = AppSettings(
        workspace_root=test_workspace / "workspace",
        projects_dir=test_workspace / "projects",
        models_dir=test_workspace / "models",
        outputs_dir=test_workspace / "outputs",
        logs_dir=test_workspace / "logs",
        temp_dir=test_workspace / "temp",
        database_path=test_workspace / "database" / "studio.sqlite3",
    )

    ensure_runtime_directories(settings)

    for directory in (
        settings.workspace_root,
        settings.projects_dir,
        settings.models_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.temp_dir,
        settings.database_path.parent,
    ):
        assert directory.is_dir()


def test_image_download_stale_threshold_exceeds_metadata_request_window() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            image_search_connect_timeout_seconds=10.0,
            image_search_read_timeout_seconds=30.0,
            image_search_min_interval_seconds=1.0,
            image_download_heartbeat_interval_seconds=2.0,
            image_download_stale_after_seconds=43.0,
        )

    settings = AppSettings(
        image_search_connect_timeout_seconds=10.0,
        image_search_read_timeout_seconds=30.0,
        image_search_min_interval_seconds=1.0,
        image_download_heartbeat_interval_seconds=2.0,
        image_download_stale_after_seconds=44.0,
    )
    assert settings.image_download_stale_after_seconds == 44.0
