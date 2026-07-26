from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
    QualityProfile,
    RecommendationInput,
    SpeedProfile,
    TrainingDatasetStatistics,
    TrainingEnvironmentInfo,
    WarningSeverity,
)
from runpod_lora_studio.domain.training_models import TrainingConfigInput
from runpod_lora_studio.services.recommendation_fingerprint import input_fingerprint
from runpod_lora_studio.services.training_memory_estimator import (
    TrainingMemoryEstimator,
)
from runpod_lora_studio.services.training_recommendation_engine import (
    RuleBasedRecommendationEngine,
)
from runpod_lora_studio.services.training_recommendation_service import (
    TrainingRecommendationService,
)
from runpod_lora_studio.services.training_risk_service import TrainingRiskService


def _input(
    *, gpu: GPUDeviceInfo | None = None, bf16: bool | None = False
) -> RecommendationInput:
    snapshot_id = uuid4()
    return RecommendationInput(
        project_id=uuid4(),
        dataset_snapshot_id=snapshot_id,
        model_id=uuid4(),
        environment_snapshot_id=uuid4(),
        environment=ComputeEnvironmentInfo(
            gpu_devices=(gpu,) if gpu else (),
            cuda_available=gpu is not None,
            bf16_supported=bf16,
            fp16_supported=True,
            bitsandbytes_available=True,
        ),
        training_environment=TrainingEnvironmentInfo(
            sd_scripts_root=Path("/workspace/sd-scripts"),
            trainer_script=Path("/workspace/sd-scripts/sdxl_train_network.py"),
            sd_scripts_version="1.0",
            python_executable=Path("/usr/bin/python"),
            safetensors_available=True,
            torch_available=True,
            xformers_available=True,
            bitsandbytes_available=True,
            bf16_supported=bf16,
            fp16_supported=True,
            cuda_available=gpu is not None,
        ),
        dataset=TrainingDatasetStatistics(
            snapshot_id=snapshot_id,
            image_count=20,
            effective_image_count=40,
            subset_count=1,
            subset_image_counts=(20,),
            repeats=(2,),
            caption_count=20,
            empty_caption_count=0,
            trigger_word_coverage=1.0,
            duplicate_ratio=0.0,
            similarity_group_count=0,
            unreviewed_similarity_group_count=0,
            min_width=1024,
            max_width=1024,
            min_height=1024,
            max_height=1024,
            mean_aspect_ratio=1.0,
            min_aspect_ratio=1.0,
            max_aspect_ratio=1.0,
            bucket_count=1,
            content_sha256="content",
            dataset_toml_sha256="toml",
            analyzer_version="test",
        ),
        concept_type="character",
        quality_profile=QualityProfile.BALANCED,
        speed_profile=SpeedProfile.BALANCED,
    )


def test_recommendation_uses_safe_batch_and_bf16() -> None:
    recommendation = RuleBasedRecommendationEngine().recommend(
        _input(
            gpu=GPUDeviceInfo(
                index=0,
                name="test",
                total_vram_bytes=24 * 1024**3,
                free_vram_bytes=20 * 1024**3,
            ),
            bf16=True,
        )
    )[0]

    assert recommendation.batch_size >= 1
    assert recommendation.gradient_accumulation_steps == 1
    assert recommendation.mixed_precision == "bf16"
    assert recommendation.estimated_total_steps == (
        recommendation.estimated_steps_per_epoch * recommendation.epochs
    )


def test_missing_gpu_is_blocking_and_does_not_claim_valid_memory() -> None:
    recommendation = RuleBasedRecommendationEngine().recommend(_input())[0]

    assert any(
        warning.code == "GPU_NOT_AVAILABLE"
        and warning.severity is WarningSeverity.BLOCKING
        for warning in recommendation.warnings
    )
    assert recommendation.batch_size == 1


def test_memory_estimator_never_uses_all_free_vram() -> None:
    result = TrainingMemoryEstimator().estimate(
        gpu=GPUDeviceInfo(
            index=0,
            name="test",
            total_vram_bytes=8 * 1024**3,
            free_vram_bytes=8 * 1024**3,
        ),
        resolution=1024,
        batch_size=1,
        network_dim=16,
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=True,
    )

    assert result.safe_available_bytes < result.free_vram_bytes
    assert result.required_bytes > 0


def test_apply_recommendation_preserves_snapshot_and_records_provenance() -> None:
    recommendation = RuleBasedRecommendationEngine().recommend(
        _input(
            gpu=GPUDeviceInfo(
                index=0,
                name="test",
                total_vram_bytes=24 * 1024**3,
                free_vram_bytes=20 * 1024**3,
            )
        )
    )[0]
    base = TrainingConfigInput(
        project_id=uuid4(),
        dataset_snapshot_id=uuid4(),
        managed_model_id=uuid4(),
        name="test",
        output_name="output",
        output_directory=Path("/workspace/outputs"),
        sd_scripts_root=Path("/workspace/sd-scripts"),
    )

    applied = TrainingRecommendationService.apply_to_config(base, recommendation)

    assert applied.dataset_snapshot_id == base.dataset_snapshot_id
    assert applied.recommendation_id == recommendation.id
    assert applied.recommendation_engine_version == recommendation.engine_version
    assert applied.recommendation_change_diff["batch_size"] == {
        "recommended": recommendation.batch_size,
        "applied": base.batch_size,
    }


def test_input_fingerprint_ignores_free_vram_but_tracks_stable_inputs() -> None:
    data = _input(
        gpu=GPUDeviceInfo(
            index=0,
            name="test",
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=20 * 1024**3,
        )
    )
    changed_free = replace(
        data,
        environment=replace(
            data.environment,
            gpu_devices=(
                replace(data.environment.gpu_devices[0], free_vram_bytes=8 * 1024**3),
            ),
        ),
    )
    changed_dataset = replace(
        data,
        dataset=replace(data.dataset, content_sha256="changed"),
    )

    assert input_fingerprint(data, engine_version="test") == input_fingerprint(
        changed_free, engine_version="test"
    )
    assert input_fingerprint(data, engine_version="test") != input_fingerprint(
        changed_dataset, engine_version="test"
    )


def test_training_risk_rechecks_user_edited_values() -> None:
    data = _input(
        gpu=GPUDeviceInfo(
            index=0,
            name="test",
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=20 * 1024**3,
        ),
        bf16=True,
    )
    recommendation = RuleBasedRecommendationEngine().recommend(data)[0]
    edited = TrainingConfigInput(
        project_id=data.project_id,
        dataset_snapshot_id=data.dataset_snapshot_id,
        managed_model_id=data.model_id,
        name="test",
        output_name="output",
        output_directory=Path("/workspace/outputs"),
        sd_scripts_root=Path("/workspace/sd-scripts"),
        network_dim=8,
        network_alpha=16,
        mixed_precision="bf16",
    )

    warnings = TrainingRiskService.evaluate(
        recommendation,
        edited,
        environment=data.environment,
        effective_image_count=data.dataset.effective_image_count,
        repeats=data.dataset.repeats,
        allowed_optimizers=frozenset({"AdamW8bit"}),
        allowed_schedulers=frozenset({"cosine"}),
    )

    assert any(warning.code == "NETWORK_ALPHA_INVALID" for warning in warnings)
