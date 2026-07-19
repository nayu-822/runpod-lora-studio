from __future__ import annotations

from runpod_lora_studio.app import build_status_markdown, create_app
from runpod_lora_studio.config.settings import AppSettings


def test_build_status_markdown_contains_expected_values() -> None:
    settings = AppSettings(app_title="Test Studio", app_env="test")

    markdown = build_status_markdown(settings)

    assert "Environment: `test`" in markdown
    assert "Workspace root:" in markdown


def test_create_app_returns_blocks() -> None:
    settings = AppSettings()
    app = create_app(settings)

    assert app.title == settings.app_title
