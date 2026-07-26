from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
    QualityProfile,
    RecommendationInput,
    RecommendationWarning,
    SpeedProfile,
    TrainingRecommendation,
    WarningSeverity,
)
from runpod_lora_studio.services.recommendation_fingerprint import input_fingerprint
from runpod_lora_studio.services.training_memory_estimator import (
    DEFAULT_SAFETY_MARGIN_BYTES,
    TrainingMemoryEstimator,
)

ALLOWED_DIMS = (4, 8, 16, 32, 64, 128)
DEFAULT_TARGET_STEPS = {
    QualityProfile.CONSERVATIVE: 1000,
    QualityProfile.BALANCED: 1500,
    QualityProfile.DETAIL_FOCUSED: 2200,
}
PROFILE_DIM = {
    "character": 16,
    "style": 32,
    "outfit": 16,
    "object": 16,
    "pose": 16,
    "general_concept": 8,
}


class TrainingRecommendationEngine(Protocol):
    engine_version: str

    def recommend(
        self, data: RecommendationInput
    ) -> tuple[TrainingRecommendation, ...]: ...


class RuleBasedRecommendationEngine:
    engine_version = "phase7a-rules-v1"

    def __init__(self, memory_estimator: TrainingMemoryEstimator | None = None) -> None:
        self.memory_estimator = memory_estimator or TrainingMemoryEstimator()

    def recommend(
        self, data: RecommendationInput
    ) -> tuple[TrainingRecommendation, ...]:
        now = datetime.now(UTC)
        request_id = UUID(str(data.user_constraints.get("request_id", uuid4())))
        resolution = _positive_int(data.current_config.get("resolution"), 1024)
        target_steps = _positive_int(
            data.user_constraints.get("target_steps"),
            DEFAULT_TARGET_STEPS[data.quality_profile],
        )
        max_batch = _positive_int(data.user_constraints.get("max_batch_size"), 8)
        gpu = _primary_gpu(data.environment)
        warnings: list[RecommendationWarning] = []
        if not data.environment.cuda_available or gpu is None:
            warnings.append(
                _warning(
                    "GPU_NOT_AVAILABLE",
                    WarningSeverity.BLOCKING,
                    "CUDA GPUが利用できません。",
                )
            )
        if (
            data.environment.cuda_available
            and data.environment.bf16_supported is not True
        ):
            warnings.append(
                _warning(
                    "BF16_UNSUPPORTED",
                    WarningSeverity.WARNING,
                    "bf16非対応のためfp16またはnoへフォールバックします。",
                    "mixed_precision",
                )
            )
        dim = _select_dim(data)
        raw_alpha = data.user_constraints.get("network_alpha")
        alpha = _positive_int(raw_alpha, dim)
        invalid_alpha = False
        if raw_alpha is not None:
            if not isinstance(raw_alpha, (int, float, str)):
                invalid_alpha = True
            else:
                try:
                    raw_alpha_number = int(raw_alpha)
                except ValueError:
                    invalid_alpha = True
                else:
                    invalid_alpha = raw_alpha_number <= 0 or raw_alpha_number > dim
        if invalid_alpha:
            warnings.append(
                _warning(
                    "NETWORK_ALPHA_INVALID",
                    WarningSeverity.BLOCKING,
                    "network alphaはdim以下の正数が必要です。",
                )
            )
            alpha = dim
        alpha = min(dim, alpha)
        if dim >= 64 or (
            data.dataset.effective_image_count
            and data.dataset.effective_image_count < 20
            and dim >= 32
        ):
            warnings.append(
                _warning(
                    "NETWORK_DIM_HIGH_FOR_DATASET",
                    WarningSeverity.WARNING,
                    "データ規模に対してnetwork dimが大きい設定です。",
                    "network_dim",
                )
            )
        batch = 1
        for candidate in range(min(max_batch, 8), 0, -1):
            estimated = self.memory_estimator.estimate(
                gpu=gpu,
                resolution=resolution,
                batch_size=candidate,
                network_dim=dim,
                mixed_precision=_precision(data),
                cache_latents=_cache_latents(data),
                gradient_checkpointing=_checkpointing(data),
                safety_margin_bytes=_positive_int(
                    data.user_constraints.get("vram_safety_margin_bytes"),
                    DEFAULT_SAFETY_MARGIN_BYTES,
                ),
            )
            if estimated.valid:
                batch = candidate
                break
        estimated = self.memory_estimator.estimate(
            gpu=gpu,
            resolution=resolution,
            batch_size=batch,
            network_dim=dim,
            mixed_precision=_precision(data),
            cache_latents=_cache_latents(data),
            gradient_checkpointing=_checkpointing(data),
            safety_margin_bytes=_positive_int(
                data.user_constraints.get("vram_safety_margin_bytes"),
                DEFAULT_SAFETY_MARGIN_BYTES,
            ),
        )
        for code in estimated.warnings:
            severity = (
                WarningSeverity.BLOCKING
                if code == "INSUFFICIENT_FREE_VRAM"
                else WarningSeverity.WARNING
            )
            warnings.append(
                _warning(code, severity, _memory_message(code), "batch_size")
            )
        optimizer = _optimizer(data)
        if (
            optimizer == "AdamW"
            and "AdamW8bit" in data.allowed_optimizers
            and not _bitsandbytes(data)
        ):
            warnings.append(
                _warning(
                    "OPTIMIZER_DEPENDENCY_MISSING",
                    WarningSeverity.WARNING,
                    "bitsandbytesが利用できないためAdamWを使用します。",
                    "optimizer",
                )
            )
        if data.dataset.empty_caption_count:
            warnings.append(
                _warning(
                    "EMPTY_CAPTION_FOUND",
                    WarningSeverity.WARNING,
                    "空のキャプションが含まれています。",
                    "caption",
                )
            )
        if data.dataset.image_count < 10:
            warnings.append(
                _warning(
                    "DATASET_TOO_SMALL",
                    WarningSeverity.WARNING,
                    "学習画像数が少なく、過学習リスクがあります。",
                    "image_count",
                )
            )
        if data.dataset.image_count > 10_000:
            warnings.append(
                _warning(
                    "DATASET_TOO_LARGE",
                    WarningSeverity.INFO,
                    "学習画像数が多く、学習時間が長くなる可能性があります。",
                    "image_count",
                )
            )
        if (
            data.dataset.min_aspect_ratio is not None
            and data.dataset.max_aspect_ratio is not None
            and (
                data.dataset.min_aspect_ratio < 0.5
                or data.dataset.max_aspect_ratio > 2.0
            )
        ):
            warnings.append(
                _warning(
                    "EXTREME_ASPECT_RATIO_DISTRIBUTION",
                    WarningSeverity.WARNING,
                    "極端なアスペクト比の画像が含まれています。",
                    "aspect_ratio",
                )
            )
        if (
            data.dataset.trigger_word_coverage is not None
            and data.dataset.trigger_word_coverage < 0.8
        ):
            warnings.append(
                _warning(
                    "TRIGGER_WORD_COVERAGE_LOW",
                    WarningSeverity.WARNING,
                    "トリガーワードの付与率が低い状態です。",
                    "trigger_word_coverage",
                )
            )
        if (
            data.dataset.duplicate_ratio is not None
            and data.dataset.duplicate_ratio >= 0.2
        ):
            warnings.append(
                _warning(
                    "DUPLICATE_RATIO_HIGH",
                    WarningSeverity.WARNING,
                    "完全重複画像の割合が高い状態です。",
                    "duplicate_ratio",
                )
            )
        if data.dataset.unreviewed_similarity_group_count:
            warnings.append(
                _warning(
                    "SIMILARITY_GROUPS_UNREVIEWED",
                    WarningSeverity.WARNING,
                    "類似画像グループに未確認のものがあります。",
                    "similarity_groups",
                )
            )
        if data.user_constraints.get("model_hash_verified") is False:
            warnings.append(
                _warning(
                    "MODEL_HASH_UNVERIFIED",
                    WarningSeverity.BLOCKING,
                    "学習元モデルのSHA-256検証が完了していません。",
                    "model",
                )
            )
        steps_per_epoch = (
            math.ceil(data.dataset.effective_image_count / batch)
            if data.dataset.effective_image_count
            else None
        )
        epochs = (
            max(1, math.ceil(target_steps / steps_per_epoch)) if steps_per_epoch else 1
        )
        epochs = min(
            epochs, _positive_int(data.user_constraints.get("max_epochs"), 100)
        )
        total_steps = steps_per_epoch * epochs if steps_per_epoch else None
        if total_steps is not None and total_steps < 100:
            warnings.append(
                _warning(
                    "TOTAL_STEPS_TOO_LOW",
                    WarningSeverity.WARNING,
                    "推奨総step数が少なすぎます。",
                    "epochs",
                )
            )
        if total_steps is not None and total_steps > 100_000:
            warnings.append(
                _warning(
                    "TOTAL_STEPS_TOO_HIGH",
                    WarningSeverity.WARNING,
                    "推奨総step数が大きすぎます。",
                    "epochs",
                )
            )
        settings = {
            "resolution": resolution,
            "batch_size": batch,
            "epochs": epochs,
            "save_every_n_epochs": _positive_int(
                data.current_config.get("save_every_n_epochs"), 1
            ),
            "seed": _positive_int(data.current_config.get("seed"), 42),
            "network_dim": dim,
            "network_alpha": alpha,
            "optimizer": optimizer,
            "scheduler": _scheduler(data),
            "mixed_precision": _precision(data),
            "cache_latents": _cache_latents(data),
            "gradient_checkpointing": _checkpointing(data),
            "repeats": list(data.dataset.repeats),
            "dataset_snapshot_id": str(data.dataset.snapshot_id),
        }
        fingerprint = _fingerprint(data, settings)
        confidence = (
            estimated.confidence
            if "seconds_per_step" in data.user_constraints
            else "low"
        )
        return (
            TrainingRecommendation(
                id=uuid4(),
                request_id=request_id,
                rank=1,
                profile_name=f"{data.quality_profile.value}/{data.speed_profile.value}",
                batch_size=batch,
                gradient_accumulation_steps=1,
                network_module=_network_module(data),
                network_dim=dim,
                network_alpha=alpha,
                epochs=epochs,
                repeats=data.dataset.repeats,
                learning_rate=_learning_rate(data),
                optimizer=optimizer,
                scheduler=_scheduler(data),
                mixed_precision=_precision(data),
                cache_latents=_cache_latents(data),
                gradient_checkpointing=_checkpointing(data),
                estimated_images_per_epoch=data.dataset.effective_image_count,
                estimated_steps_per_epoch=steps_per_epoch,
                estimated_total_steps=total_steps,
                estimated_vram_bytes=estimated.required_bytes,
                estimated_duration_seconds=_duration(total_steps, data),
                confidence=confidence,
                reasons=_reasons(data, batch, dim),
                warnings=tuple(warnings),
                settings_fingerprint=fingerprint,
                engine_version=self.engine_version,
                created_at=now,
                resolution=resolution,
                save_every_n_epochs=_positive_int(
                    data.current_config.get("save_every_n_epochs"), 1
                ),
                seed=_positive_int(data.current_config.get("seed"), 42),
            ),
        )


def _primary_gpu(environment: ComputeEnvironmentInfo) -> GPUDeviceInfo | None:
    return environment.gpu_devices[0] if environment.gpu_devices else None


def _select_dim(data: RecommendationInput) -> int:
    requested = _positive_int(data.user_constraints.get("network_dim"), 0)
    baseline = requested or PROFILE_DIM.get(data.concept_type, 16)
    if data.quality_profile is QualityProfile.CONSERVATIVE:
        baseline = min(baseline, 16)
    elif data.quality_profile is QualityProfile.DETAIL_FOCUSED:
        baseline = max(baseline, 32)
    return min(ALLOWED_DIMS, key=lambda value: (abs(value - baseline), value))


def _precision(data: RecommendationInput) -> str:
    if not data.environment.cuda_available:
        return "no"
    if data.environment.bf16_supported:
        return "bf16"
    if data.environment.fp16_supported is not False:
        return "fp16"
    return "no"


def _optimizer(data: RecommendationInput) -> str:
    allowed = data.allowed_optimizers
    if data.environment.bitsandbytes_available and "AdamW8bit" in allowed:
        return "AdamW8bit"
    return "AdamW" if "AdamW" in allowed else (allowed[0] if allowed else "AdamW")


def _scheduler(data: RecommendationInput) -> str:
    return (
        "cosine"
        if "cosine" in data.allowed_schedulers
        else (data.allowed_schedulers[0] if data.allowed_schedulers else "constant")
    )


def _network_module(data: RecommendationInput) -> str:
    return (
        "networks.lora"
        if "networks.lora" in data.allowed_network_modules
        else data.allowed_network_modules[0]
    )


def _learning_rate(data: RecommendationInput) -> float:
    value = data.user_constraints.get("learning_rate", 1e-4)
    if not isinstance(value, (int, float, str)):
        return 1e-4
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1e-4
    return number if number > 0 and math.isfinite(number) else 1e-4


def _cache_latents(data: RecommendationInput) -> bool:
    return not bool(
        data.current_config.get("flip_aug")
        or data.current_config.get("random_crop")
        or data.current_config.get("color_aug")
    )


def _checkpointing(data: RecommendationInput) -> bool:
    return data.speed_profile is not SpeedProfile.SPEED_PRIORITY


def _bitsandbytes(data: RecommendationInput) -> bool:
    return bool(data.environment.bitsandbytes_available)


def _duration(total_steps: int | None, data: RecommendationInput) -> float | None:
    if total_steps is None:
        return None
    value = data.user_constraints.get("seconds_per_step", 2.0)
    if not isinstance(value, (int, float, str)):
        return None
    try:
        seconds_per_step = float(value)
    except ValueError:
        return None
    return total_steps * seconds_per_step if seconds_per_step > 0 else None


def _reasons(data: RecommendationInput, batch: int, dim: int) -> tuple[str, ...]:
    return (
        f"{data.concept_type}向けにnetwork dim={dim}を選択しました。",
        f"安全マージンを差し引いたVRAM見積もりでbatch size={batch}を選択しました。",
        "gradient accumulationはPhase 7Aのコマンド契約に合わせて1に固定します。",
    )


def _warning(
    code: str, severity: WarningSeverity, message: str, parameter: str | None = None
) -> RecommendationWarning:
    return RecommendationWarning(
        code=code, severity=severity, message=message, parameter=parameter
    )


def _memory_message(code: str) -> str:
    return {
        "INSUFFICIENT_FREE_VRAM": "安全マージンを含めたVRAM見積もりを満たしません。",
        "VRAM_ESTIMATE_NEAR_LIMIT": "VRAM使用量が安全上限に近い状態です。",
    }.get(code, "GPUメモリ診断に注意が必要です。")


def _positive_int(value: object, default: int) -> int:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _fingerprint(data: RecommendationInput, settings: dict[str, object]) -> str:
    import hashlib
    import json

    payload = {
        "input_fingerprint": input_fingerprint(
            data, engine_version=RuleBasedRecommendationEngine.engine_version
        ),
        "settings": settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
