from __future__ import annotations

import os
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from runpod_lora_studio.domain.storage_models import (
    OverwritePolicy,
    VerificationPolicy,
)

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

    # Phase 2B perceptual duplicate detection. Candidates are never deleted.
    phash_hash_size: int = Field(default=8, ge=4, le=32)
    phash_distance_threshold: int = Field(default=8, ge=0, le=1024)
    phash_batch_size: int = Field(default=100, ge=1, le=10_000)
    similarity_group_page_size: int = Field(default=20, ge=1, le=100)

    # Phase 3 tagging defaults. Model files are never downloaded by default.
    tagger_adapter_name: str = "wd14"
    tagger_model_identifier: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    tagger_model_revision: str = "main"
    tagger_model_dir: Path = Path("/workspace/ldts-runtime/models/taggers/wd14")
    tagger_device: Literal["auto", "cuda", "cpu"] = "auto"
    tagger_batch_size: int = Field(default=4, ge=1, le=128)
    tagger_general_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    tagger_character_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    tagger_save_rating: bool = False
    tagger_save_character: bool = True
    tagger_save_general: bool = True
    tagger_underscore_to_space: bool = False
    tagger_escape_mode: Literal["none", "backslash"] = "none"
    tagger_max_workers: int = Field(default=1, ge=1, le=8)
    tagger_allow_model_download: bool = False
    tagger_caption_page_size: int = Field(default=20, ge=1, le=100)

    # Phase 4 dataset snapshot defaults.
    dataset_snapshot_page_size: int = Field(default=20, ge=1, le=100)
    dataset_snapshot_copy_batch_size: int = Field(default=20, ge=1, le=500)
    dataset_snapshot_generator_version: str = "phase4-snapshot-v1"
    dataset_default_resolution: int = Field(default=1024, ge=64, le=8192)
    dataset_default_min_bucket_reso: int = Field(default=256, ge=1, le=8192)
    dataset_default_max_bucket_reso: int = Field(default=2048, ge=1, le=16384)
    dataset_default_bucket_reso_steps: int = Field(default=64, ge=1, le=2048)
    dataset_default_num_repeats: int = Field(default=1, ge=1, le=10000)
    dataset_allow_empty_caption: bool = False
    dataset_disk_safety_margin_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    dataset_disk_warning_margin_bytes: int = Field(default=1024 * 1024 * 1024, ge=0)

    # Phase 5 storage and rclone defaults. Credentials remain in the external
    # rclone config file and are never copied into the application database.
    rclone_executable: str = "rclone"
    rclone_config_path: Path | None = None
    storage_remote_name: str = "gdrive"
    storage_model_remote_root: str = "lora-studio/models"
    storage_project_remote_root: str = "lora-studio/projects"
    storage_snapshot_remote_root: str = "snapshots"
    storage_artifact_remote_root: str = "artifacts"
    model_cache_dir: Path | None = None
    transfer_temp_dir: Path | None = None
    rclone_transfers: int = Field(default=4, ge=1, le=64)
    rclone_checkers: int = Field(default=8, ge=1, le=128)
    rclone_retries: int = Field(default=3, ge=0, le=20)
    rclone_low_level_retries: int = Field(default=10, ge=0, le=100)
    rclone_retry_interval_seconds: float = Field(default=5.0, ge=0.0, le=3600.0)
    rclone_connect_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    rclone_transfer_timeout_seconds: float = Field(default=3600.0, gt=0.0)
    rclone_bandwidth_limit: str | None = None
    rclone_buffer_size: str = "16M"
    storage_use_checksum: bool = True
    storage_remote_hash_fallback: Literal[
        "size_and_manifest", "existence_only", "error"
    ] = "error"
    storage_dry_run_default: bool = True
    storage_overwrite_policy: OverwritePolicy = OverwritePolicy.SKIP_IDENTICAL
    storage_verification_policy: VerificationPolicy = (
        VerificationPolicy.REMOTE_HASH_AND_SIZE
    )
    model_allowed_extensions: tuple[str, ...] = (".safetensors", ".yaml", ".json")
    model_allow_ckpt: bool = False
    model_max_file_size_bytes: int = Field(default=30 * 1024**3, ge=1)
    model_disk_safety_margin_bytes: int = Field(default=256 * 1024**1024, ge=0)
    transfer_progress_interval_seconds: float = Field(default=2.0, gt=0.1, le=60.0)
    storage_retry_max_backoff_seconds: float = Field(default=300.0, ge=0.0)
    storage_job_stale_after_seconds: float = Field(default=120.0, gt=0.0)

    # Phase 6A training process defaults. Secrets are deliberately not passed
    # through the training job environment.
    training_jobs_dir: Path | None = None
    training_sd_scripts_root: Path = Path("/workspace/sd-scripts")
    training_python_executable: Path = Field(
        default_factory=lambda: Path(sys.executable)
    )
    training_heartbeat_interval_seconds: float = Field(default=2.0, gt=0.1, le=60.0)
    training_job_stale_after_seconds: float = Field(default=120.0, gt=0.0)
    training_starting_grace_seconds: float = Field(default=30.0, ge=0.0)
    training_cancel_grace_seconds: float = Field(default=10.0, gt=0.0)
    training_log_tail_bytes: int = Field(
        default=64 * 1024, ge=1024, le=10 * 1024 * 1024
    )
    training_progress_interval_seconds: float = Field(default=2.0, gt=0.1, le=60.0)
    training_progress_read_bytes: int = Field(
        default=256 * 1024, ge=1024, le=16 * 1024 * 1024
    )
    training_metric_max_points: int = Field(default=2000, ge=10, le=100000)
    training_artifact_scan_interval_seconds: float = Field(
        default=10.0, gt=0.1, le=300.0
    )
    training_artifact_max_depth: int = Field(default=3, ge=1, le=10)
    training_artifact_max_count: int = Field(default=500, ge=1, le=10000)
    training_artifact_max_file_size_bytes: int = Field(default=30 * 1024**3, ge=1)
    training_resume_state_max_total_size_bytes: int = Field(default=30 * 1024**3, ge=1)
    training_resume_max_epoch: int = Field(default=100_000, ge=1)
    training_resume_max_step: int = Field(default=1_000_000_000, ge=1)

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
        "tagger_model_dir",
        "training_sd_scripts_root",
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
        settings.training_jobs_dir or settings.workspace_root / "training" / "jobs",
    )
    for directory in (settings.model_cache_dir, settings.transfer_temp_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    env_file = os.environ.get("RUNPOD_LORA_STUDIO_ENV_FILE", ".env")
    return AppSettings(_env_file=env_file)
