from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class DiagnosticStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class QualityProfile(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    DETAIL_FOCUSED = "detail_focused"


class SpeedProfile(StrEnum):
    MEMORY_SAVER = "memory_saver"
    BALANCED = "balanced"
    SPEED_PRIORITY = "speed_priority"


class RecommendationStatus(StrEnum):
    COMPLETED = "completed"
    STALE = "stale"
    INVALID = "invalid"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True, slots=True)
class GPUDeviceInfo:
    index: int
    name: str
    uuid: str | None = None
    architecture: str | None = None
    compute_capability: str | None = None
    total_vram_bytes: int | None = None
    free_vram_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ComputeEnvironmentInfo:
    gpu_devices: tuple[GPUDeviceInfo, ...] = ()
    cuda_available: bool = False
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    bf16_supported: bool | None = None
    fp16_supported: bool | None = None
    xformers_available: bool | None = None
    bitsandbytes_available: bool | None = None
    operating_system: str = ""
    python_version: str = ""
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingEnvironmentInfo:
    sd_scripts_root: Path
    trainer_script: Path | None
    sd_scripts_version: str | None
    python_executable: Path
    safetensors_available: bool
    torch_available: bool
    xformers_available: bool
    bitsandbytes_available: bool
    bf16_supported: bool | None
    fp16_supported: bool | None
    cuda_available: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingDatasetStatistics:
    snapshot_id: UUID
    image_count: int
    effective_image_count: int
    subset_count: int
    subset_image_counts: tuple[int, ...]
    repeats: tuple[int, ...]
    caption_count: int
    empty_caption_count: int
    trigger_word_coverage: float | None
    duplicate_ratio: float | None
    similarity_group_count: int
    unreviewed_similarity_group_count: int
    min_width: int | None
    max_width: int | None
    min_height: int | None
    max_height: int | None
    mean_aspect_ratio: float | None
    min_aspect_ratio: float | None
    max_aspect_ratio: float | None
    bucket_count: int | None
    content_sha256: str | None
    dataset_toml_sha256: str | None
    analyzer_version: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecommendationWarning:
    code: str
    severity: WarningSeverity
    message: str
    parameter: str | None = None
    measured: str | None = None


@dataclass(frozen=True, slots=True)
class RecommendationInput:
    project_id: UUID
    dataset_snapshot_id: UUID
    model_id: UUID
    environment_snapshot_id: UUID
    environment: ComputeEnvironmentInfo
    training_environment: TrainingEnvironmentInfo
    dataset: TrainingDatasetStatistics
    concept_type: str
    quality_profile: QualityProfile
    speed_profile: SpeedProfile
    user_constraints: dict[str, object] = field(default_factory=dict)
    allowed_network_modules: tuple[str, ...] = ("networks.lora",)
    allowed_optimizers: tuple[str, ...] = ("AdamW", "AdamW8bit", "Lion", "Prodigy")
    allowed_schedulers: tuple[str, ...] = (
        "constant",
        "constant_with_warmup",
        "cosine",
        "cosine_with_restarts",
        "linear",
    )
    current_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingRecommendation:
    id: UUID
    request_id: UUID
    rank: int
    profile_name: str
    batch_size: int
    gradient_accumulation_steps: int
    network_module: str
    network_dim: int
    network_alpha: int
    epochs: int
    repeats: tuple[int, ...]
    learning_rate: float
    optimizer: str
    scheduler: str
    mixed_precision: str
    cache_latents: bool
    gradient_checkpointing: bool
    estimated_images_per_epoch: int | None
    estimated_steps_per_epoch: int | None
    estimated_total_steps: int | None
    estimated_vram_bytes: int | None
    estimated_duration_seconds: float | None
    confidence: str
    reasons: tuple[str, ...]
    warnings: tuple[RecommendationWarning, ...]
    settings_fingerprint: str
    engine_version: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    id: UUID
    project_id: UUID
    dataset_snapshot_id: UUID
    model_id: UUID
    environment_snapshot_id: UUID
    concept_type: str
    quality_profile: QualityProfile
    speed_profile: SpeedProfile
    user_constraints: dict[str, object]
    input_fingerprint: str
    engine_version: str
    status: RecommendationStatus
    warning_count: int
    created_at: datetime
    updated_at: datetime
