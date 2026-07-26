from __future__ import annotations

from dataclasses import dataclass

from runpod_lora_studio.domain.recommendation_models import GPUDeviceInfo

GIB = 1024**3
BASE_VRAM_BYTES = 2 * GIB
PER_PIXEL_BATCH_BYTES = 7
PER_NETWORK_DIM_BYTES = 64 * 1024 * 1024
GRADIENT_CHECKPOINTING_FACTOR = 0.72
CACHE_LATENTS_BYTES = 512 * 1024 * 1024
DEFAULT_SAFETY_MARGIN_BYTES = GIB
NEAR_LIMIT_RATIO = 0.90


@dataclass(frozen=True, slots=True)
class EstimatedMemoryUsage:
    required_bytes: int
    safe_available_bytes: int | None
    total_vram_bytes: int | None
    free_vram_bytes: int | None
    valid: bool
    confidence: str
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


class TrainingMemoryEstimator:
    """Conservative, deterministic estimate for an SDXL LoRA configuration."""

    def estimate(
        self,
        *,
        gpu: GPUDeviceInfo | None,
        resolution: int,
        batch_size: int,
        network_dim: int,
        mixed_precision: str,
        cache_latents: bool,
        gradient_checkpointing: bool,
        safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    ) -> EstimatedMemoryUsage:
        if resolution <= 0 or batch_size <= 0 or network_dim <= 0:
            return EstimatedMemoryUsage(
                required_bytes=0,
                safe_available_bytes=None,
                total_vram_bytes=gpu.total_vram_bytes if gpu else None,
                free_vram_bytes=gpu.free_vram_bytes if gpu else None,
                valid=False,
                confidence="low",
                assumptions=("invalid training dimensions",),
                warnings=("INVALID_TRAINING_DIMENSIONS",),
            )
        activation = resolution * resolution * batch_size * PER_PIXEL_BATCH_BYTES
        network = network_dim * PER_NETWORK_DIM_BYTES
        required = BASE_VRAM_BYTES + activation + network
        if mixed_precision == "fp16":
            required = int(required * 0.88)
        elif mixed_precision == "bf16":
            required = int(required * 0.92)
        if cache_latents:
            required += CACHE_LATENTS_BYTES
        if gradient_checkpointing:
            required = int(required * GRADIENT_CHECKPOINTING_FACTOR)
        warnings: list[str] = []
        if gpu is None or gpu.total_vram_bytes is None:
            warnings.append("GPU_NOT_AVAILABLE")
            return EstimatedMemoryUsage(
                required_bytes=required,
                safe_available_bytes=None,
                total_vram_bytes=None,
                free_vram_bytes=None,
                valid=False,
                confidence="low",
                assumptions=("GPU capacity is unknown",),
                warnings=tuple(warnings),
            )
        free = (
            gpu.free_vram_bytes
            if gpu.free_vram_bytes is not None
            else gpu.total_vram_bytes
        )
        safe = max(0, free - max(0, safety_margin_bytes))
        valid = required <= safe
        if not valid:
            warnings.append("INSUFFICIENT_FREE_VRAM")
        elif safe and required / safe >= NEAR_LIMIT_RATIO:
            warnings.append("VRAM_ESTIMATE_NEAR_LIMIT")
        return EstimatedMemoryUsage(
            required_bytes=required,
            safe_available_bytes=safe,
            total_vram_bytes=gpu.total_vram_bytes,
            free_vram_bytes=gpu.free_vram_bytes,
            valid=valid,
            confidence="medium",
            assumptions=(
                "SDXL activation and LoRA memory are estimated from named constants",
                "free VRAM is reduced by the safety margin",
            ),
            warnings=tuple(warnings),
        )
