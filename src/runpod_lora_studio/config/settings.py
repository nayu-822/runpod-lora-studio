from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _package_version() -> str:
    try:
        return version("runpod-lora-studio")
    except PackageNotFoundError:
        return "0.1.0"


class AppSettings(BaseSettings):  # type: ignore[misc]
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RUNPOD_LORA_STUDIO_",
        extra="ignore",
        populate_by_name=True,
    )

    app_title: str = "RunPod LoRA Studio"
    app_env: str = "development"
    app_version: str = Field(default_factory=_package_version)

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = Field(default=7860, ge=1, le=65535)
    gradio_share: bool = False

    log_level: LogLevel = "INFO"

    workspace_root: Path = Path("/workspace/ldts-runtime")
    projects_dir: Path = Path("/workspace/ldts-runtime/projects")
    models_dir: Path = Path("/workspace/ldts-runtime/models")
    outputs_dir: Path = Path("/workspace/ldts-runtime/outputs")
    logs_dir: Path = Path("/workspace/ldts-runtime/logs")
    temp_dir: Path = Path("/workspace/ldts-runtime/tmp")
    database_path: Path = Path("/workspace/ldts-runtime/database/ldts.sqlite3")
    max_upload_files: int = Field(default=100, ge=1, le=1000)
    max_upload_file_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=100_000_000, ge=1)
    thumbnail_size: int = Field(default=320, ge=32, le=2048)

    # Phase 2A image inspection defaults. These are warnings, not delete rules.
    inspection_min_width: int = Field(default=512, ge=1)
    inspection_min_height: int = Field(default=512, ge=1)
    inspection_max_aspect_ratio: float = Field(default=3.0, gt=1.0)
    inspection_low_information_stddev_threshold: float = Field(default=8.0, ge=0.0)
    inspection_blur_score_threshold: float = Field(default=50.0, ge=0.0)

    runpod_pod_id: str | None = Field(default=None, validation_alias="RUNPOD_POD_ID")
    # Tokens and credentials added later must use SecretStr and repr=False as well.
    runpod_api_key: SecretStr | None = Field(
        default=None, validation_alias="RUNPOD_API_KEY", repr=False
    )

    @field_validator("app_title", "gradio_server_name")  # type: ignore[untyped-decorator]
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("値を空にできません")
        return value

    @field_validator("log_level", mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return str(value).strip().upper()

    @field_validator(  # type: ignore[untyped-decorator]
        "workspace_root",
        "projects_dir",
        "models_dir",
        "outputs_dir",
        "logs_dir",
        "temp_dir",
        "database_path",
    )
    @classmethod
    def validate_paths(cls, value: Path) -> Path:
        if not str(value).strip() or str(value) == ".":
            raise ValueError("パスを空または現在のディレクトリにできません")
        return value


def ensure_runtime_directories(settings: AppSettings) -> None:
    directories = (
        settings.workspace_root,
        settings.projects_dir,
        settings.models_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.temp_dir,
        settings.database_path.parent,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    env_file = os.environ.get("RUNPOD_LORA_STUDIO_ENV_FILE", ".env")
    return AppSettings(_env_file=env_file)
