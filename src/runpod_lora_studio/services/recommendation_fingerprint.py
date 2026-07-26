from __future__ import annotations

import hashlib
import json

from runpod_lora_studio.domain.recommendation_models import RecommendationInput

TRAINING_CONFIG_SCHEMA_VERSION = "phase7a-training-config-v1"
COMMAND_BUILDER_VERSION = "phase6c-v1"


def input_fingerprint(data: RecommendationInput, *, engine_version: str) -> str:
    """Fingerprint recommendation inputs while intentionally excluding free VRAM."""
    payload = {
        "project_id": str(data.project_id),
        "dataset_snapshot_id": str(data.dataset_snapshot_id),
        "dataset_content_sha256": data.dataset.content_sha256,
        "dataset_toml_sha256": data.dataset.dataset_toml_sha256,
        "model_id": str(data.model_id),
        "model_sha256": data.user_constraints.get("model_sha256"),
        "environment": _environment(data),
        "training_environment": _training_environment(data),
        "concept_type": data.concept_type,
        "quality_profile": data.quality_profile.value,
        "speed_profile": data.speed_profile.value,
        "user_constraints": data.user_constraints,
        "current_config": data.current_config,
        "engine_version": engine_version,
        "training_config_schema": TRAINING_CONFIG_SCHEMA_VERSION,
        "command_builder_version": COMMAND_BUILDER_VERSION,
    }
    serialized = json.dumps(
        payload, default=str, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _environment(data: RecommendationInput) -> dict[str, object]:
    return {
        "cuda_available": data.environment.cuda_available,
        "cuda_runtime_version": data.environment.cuda_runtime_version,
        "cuda_driver_version": data.environment.cuda_driver_version,
        "torch_version": data.environment.torch_version,
        "torch_cuda_version": data.environment.torch_cuda_version,
        "bf16_supported": data.environment.bf16_supported,
        "fp16_supported": data.environment.fp16_supported,
        "xformers_available": data.environment.xformers_available,
        "bitsandbytes_available": data.environment.bitsandbytes_available,
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "uuid": gpu.uuid,
                "architecture": gpu.architecture,
                "compute_capability": gpu.compute_capability,
                "total_vram_bytes": gpu.total_vram_bytes,
            }
            for gpu in data.environment.gpu_devices
        ],
    }


def _training_environment(data: RecommendationInput) -> dict[str, object]:
    environment = data.training_environment
    return {
        "sd_scripts_version": environment.sd_scripts_version,
        "trainer_script": str(environment.trainer_script)
        if environment.trainer_script
        else None,
        "safetensors_available": environment.safetensors_available,
        "torch_available": environment.torch_available,
        "xformers_available": environment.xformers_available,
        "bitsandbytes_available": environment.bitsandbytes_available,
        "bf16_supported": environment.bf16_supported,
        "fp16_supported": environment.fp16_supported,
        "cuda_available": environment.cuda_available,
    }
