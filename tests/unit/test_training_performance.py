from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from runpod_lora_studio.domain.recommendation_models import TrainingRecommendation
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    GpuMemorySample,
    TrainingExecutionSummary,
    TrainingFailureCategory,
)
from runpod_lora_studio.services.gpu_memory_metrics import (
    StaticGpuMemoryMetricsAdapter,
    summarize_gpu_memory,
)
from runpod_lora_studio.services.training_calibration_service import (
    CalibratedRecommendationService,
    RecommendationCalibrationService,
    TrainingCalibrationMatcher,
    TrainingPerformanceOutlierFilter,
)
from runpod_lora_studio.services.training_failure_classifier import (
    TrainingFailureClassifier,
)


def _summary(
    *,
    speed: float | None = 2.0,
    step: int = 100,
    oom: bool = False,
    included: bool = True,
    gpu: str = "gpu-a",
    created_offset: int = 0,
) -> TrainingExecutionSummary:
    category = (
        TrainingFailureCategory.CUDA_OUT_OF_MEMORY
        if oom
        else TrainingFailureCategory.NONE
    )
    now = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=created_offset)
    summary = TrainingExecutionSummary(
        id=uuid4(),
        training_job_id=uuid4(),
        project_id=uuid4(),
        training_config_id=uuid4(),
        dataset_snapshot_id=uuid4(),
        managed_model_id=uuid4(),
        job_result_status="failed" if oom else "succeeded",
        gpu_identity_fingerprint=gpu,
        settings_fingerprint="settings-a",
        resolution=1024,
        batch_size=1,
        gradient_accumulation_steps=1,
        effective_batch_size=1,
        optimizer="AdamW8bit",
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=False,
        planned_total_steps=step,
        completed_steps=step,
        elapsed_seconds=step / speed if speed else 10.0,
        measured_steps_per_second=speed,
        peak_reserved_vram_bytes=8 * 1024**3,
        gpu_total_vram_bytes=16 * 1024**3,
        memory_sample_count=3,
        memory_confidence=CalibrationConfidence.HIGH,
        oom_detected=oom,
        failure_category=category,
        usable_for_speed_calibration=not oom and speed is not None,
        usable_for_memory_calibration=True,
        calibration_included=included,
        created_at=now,
        updated_at=now,
    )
    return summary


def test_failure_classifier_distinguishes_oom_cancel_and_kill() -> None:
    classifier = TrainingFailureClassifier()
    assert (
        classifier.classify(
            status="failed", exit_code=1, stderr="CUDA out of memory"
        ).category
        is TrainingFailureCategory.CUDA_OUT_OF_MEMORY
    )
    assert (
        classifier.classify(
            status="canceled", exit_code=-15, cancel_requested=True
        ).category
        is TrainingFailureCategory.USER_CANCELED
    )
    assert (
        classifier.classify(status="failed", exit_code=-9, stderr="").category
        is TrainingFailureCategory.PROCESS_KILLED
    )
    assert (
        classifier.classify(status="failed", exit_code=137, stderr="Killed\n").category
        is TrainingFailureCategory.SYSTEM_OUT_OF_MEMORY
    )


def test_gpu_memory_metrics_convert_mib_and_return_null_on_invalid_samples() -> None:
    timestamp = datetime.now(UTC)
    samples = (
        GpuMemorySample(timestamp, 0, 16 * 1024**3, 12 * 1024**3),
        GpuMemorySample(timestamp, 0, 16 * 1024**3, 10 * 1024**3),
    )
    adapter = StaticGpuMemoryMetricsAdapter(samples)
    assert adapter.collect(pid=100) == samples
    summary = summarize_gpu_memory(samples)
    assert summary.total_bytes == 16 * 1024**3
    assert summary.free_before_bytes == 12 * 1024**3
    assert summary.whole_gpu_min_free_bytes == 10 * 1024**3
    assert (
        summarize_gpu_memory((GpuMemorySample(timestamp, 0, 10, 11),)).total_bytes
        is None
    )


def test_calibration_is_deterministic_and_excludes_oom_from_speed() -> None:
    first = _summary(speed=2.0, created_offset=1)
    second = _summary(speed=4.0, created_offset=2)
    oom = _summary(oom=True, speed=None, created_offset=3)
    service = RecommendationCalibrationService()
    one = service.build(
        (first, second, oom),
        gpu_identity_fingerprint="gpu-a",
        gpu_total_vram_bytes=16 * 1024**3,
        resolution=1024,
        optimizer="AdamW8bit",
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=False,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    two = service.build(
        (oom, second, first),
        gpu_identity_fingerprint="gpu-a",
        gpu_total_vram_bytes=16 * 1024**3,
        resolution=1024,
        optimizer="AdamW8bit",
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=False,
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert one.calibration_fingerprint == two.calibration_fingerprint
    assert one.median_steps_per_second == 3.0
    assert one.oom_sample_count == 1
    assert one.id == two.id


def test_outlier_filter_and_matcher_preserve_manual_exclusion_and_gpu_boundary() -> (
    None
):
    good = _summary(speed=2.0)
    excluded = _summary(speed=3.0, included=False)
    other_gpu = _summary(speed=2.0, gpu="gpu-b")
    assert TrainingPerformanceOutlierFilter().filter_speed((good, excluded)) == (good,)
    snapshot = RecommendationCalibrationService().build(
        (good,), gpu_identity_fingerprint="gpu-a", resolution=1024
    )
    assert TrainingCalibrationMatcher().matches(good, snapshot)
    assert not TrainingCalibrationMatcher().matches(other_gpu, snapshot)


def test_oom_feedback_only_suggests_lower_batch_and_keeps_baseline_safety_floor() -> (
    None
):
    summary = _summary(oom=True, speed=None)
    snapshot = RecommendationCalibrationService().build(
        (summary,), gpu_identity_fingerprint="gpu-a", resolution=1024
    )
    recommendation = TrainingRecommendation(
        id=uuid4(),
        request_id=uuid4(),
        rank=1,
        profile_name="balanced",
        batch_size=2,
        gradient_accumulation_steps=1,
        network_module="networks.lora",
        network_dim=16,
        network_alpha=16,
        epochs=1,
        repeats=(1,),
        learning_rate=1e-4,
        optimizer="AdamW8bit",
        scheduler="cosine",
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=False,
        estimated_images_per_epoch=1,
        estimated_steps_per_epoch=100,
        estimated_total_steps=100,
        estimated_vram_bytes=8 * 1024**3,
        estimated_duration_seconds=50.0,
        confidence="high",
        reasons=(),
        warnings=(),
        settings_fingerprint="settings-a",
        engine_version="phase7a-rules-v1",
        created_at=datetime.now(UTC),
    )
    calibrated, result = CalibratedRecommendationService().apply(
        recommendation, snapshot
    )
    assert calibrated.batch_size == recommendation.batch_size
    assert result.suggested_batch_size == 1
    assert result.baseline_vram_bytes == recommendation.estimated_vram_bytes
