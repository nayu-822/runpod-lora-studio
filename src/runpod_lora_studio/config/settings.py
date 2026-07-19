from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RUNPOD_LORA_STUDIO_",
        extra="ignore",
    )

    app_title: str = "RunPod LoRA Studio"
    app_env: str = "development"

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = False

    log_level: str = "INFO"

    workspace_root: Path = Path("/workspace/ldts-runtime")
    projects_dir: Path = Path("/workspace/ldts-runtime/projects")
    models_dir: Path = Path("/workspace/ldts-runtime/models")
    outputs_dir: Path = Path("/workspace/ldts-runtime/outputs")
    logs_dir: Path = Path("/workspace/ldts-runtime/logs")
    database_path: Path = Path("/workspace/ldts-runtime/database/ldts.sqlite3")

    runpod_pod_id: str | None = Field(default=None, validation_alias="RUNPOD_POD_ID")
    runpod_api_key: str | None = Field(default=None, validation_alias="RUNPOD_API_KEY")

    def ensure_runtime_directories(self) -> None:
        directories = (
            self.workspace_root,
            self.projects_dir,
            self.models_dir,
            self.outputs_dir,
            self.logs_dir,
            self.database_path.parent,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
