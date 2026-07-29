from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
    PhysicalGpuInfo,
    TrainingRecommendation,
)
from runpod_lora_studio.domain.training_environment_models import SelectedGpuStatus
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    GpuCalibrationExclusionReason,
    GpuMemorySample,
    GpuMemorySummary,
    TrainingExecutionSummary,
    TrainingFailureCategory,
)
from runpod_lora_studio.persistence.models import (
    TrainingExecutionSummaryRecord,
    TrainingJobSelectedGpuRecord,
    TrainingMemoryAggregateRecord,
)
from runpod_lora_studio.services.gpu_memory_metrics import (
    NvidiaSmiGpuMemoryAdapter,
    StaticGpuMemoryMetricsAdapter,
    gpu_uuid_fingerprint,
    summarize_gpu_memory,
)
from runpod_lora_studio.services.recommendation_calibration_service import (
    _refresh_summary_fingerprints,
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
from runpod_lora_studio.services.training_job_environment_service import (
    _record_selected_gpu_observation,
    _snapshot_from_info,
)
from runpod_lora_studio.services.training_memory_monitor import (
    _add_codes,
    _confidence,
    _merge_sample,
    _select_sample,
)
from runpod_lora_studio.services.training_performance_service import (
    _gpu_calibration_assessment,
    _gpu_calibration_is_usable,
    _mark_related_calibrations_stale,
)


def _summary(
    *,
    speed: float | None = 2.0,
    step: int = 100,
    oom: bool = False,
    included: bool = True,
    gpu: str = "gpu-a",
    created_offset: int = 0,
    batch_size: int = 1,
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
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        effective_batch_size=batch_size,
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


def test_execution_snapshot_maps_cuda_visible_devices_to_physical_gpu() -> None:
    info = ComputeEnvironmentInfo(
        gpu_devices=(
            GPUDeviceInfo(index=0, name="B", uuid="GPU-b", total_vram_bytes=20),
        ),
        cuda_available=True,
    )
    snapshot = _snapshot_from_info(
        uuid4(),
        info,
        "sd-scripts-test",
        True,
        {"CUDA_VISIBLE_DEVICES": "2"},
        datetime.now(UTC),
        "test-detector",
        physical_inventory=(
            PhysicalGpuInfo(
                index=2,
                uuid="GPU-b",
                name="B",
                architecture="B",
                total_vram_bytes=20,
            ),
        ),
    )

    assert snapshot.logical_gpu_index == 0
    assert snapshot.physical_gpu_index == 2
    assert snapshot.gpu_uuid_fingerprint == gpu_uuid_fingerprint("GPU-b")
    assert snapshot.visible_gpu_uuid_fingerprints == (gpu_uuid_fingerprint("GPU-b"),)


def test_gpu_sample_selection_never_falls_back_to_another_gpu() -> None:
    timestamp = datetime.now(UTC)
    samples = (
        GpuMemorySample(
            timestamp,
            0,
            16 * 1024**3,
            8 * 1024**3,
            gpu_uuid_fingerprint="a",
            gpu_identity_verified=True,
        ),
        GpuMemorySample(
            timestamp,
            1,
            16 * 1024**3,
            8 * 1024**3,
            gpu_uuid_fingerprint="b",
            gpu_identity_verified=True,
        ),
    )

    assert _select_sample(samples, ("b",)) is samples[1]
    assert _select_sample(samples, ()) is None
    assert _select_sample(samples, ("missing",)) is None


def test_selected_gpu_keeps_first_identity_and_records_later_change() -> None:
    first_at = datetime(2026, 1, 1, tzinfo=UTC)
    changed_at = datetime(2026, 1, 1, 1, tzinfo=UTC)
    record = TrainingJobSelectedGpuRecord(
        id=str(uuid4()),
        training_job_id=str(uuid4()),
        gpu_uuid_fingerprint="gpu-a",
        selected_at=first_at,
        selection_source="target_process",
        status="ok",
        warning_codes_json="[]",
        last_observed_gpu_uuid_fingerprint="gpu-a",
        gpu_change_count=0,
    )

    assert not _record_selected_gpu_observation(record, "gpu-a", changed_at)
    assert record.gpu_uuid_fingerprint == "gpu-a"
    assert record.selected_at == first_at
    assert _record_selected_gpu_observation(record, "gpu-b", changed_at)
    assert record.gpu_uuid_fingerprint == "gpu-a"
    assert record.last_observed_gpu_uuid_fingerprint == "gpu-b"
    assert record.gpu_change_detected_at == changed_at
    assert record.gpu_change_count == 1
    assert record.status == "changed"
    assert record.warning_codes_json == '["GPU_CHANGED_DURING_JOB"]'


def test_memory_gpu_change_does_not_merge_gpu_b_into_gpu_a() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = TrainingMemoryAggregateRecord(
        id=str(uuid4()),
        training_job_id=str(uuid4()),
        gpu_index=2,
        gpu_uuid_fingerprint="gpu-a",
        gpu_total_vram_bytes=16 * 1024**3,
        free_vram_before_bytes=12 * 1024**3,
        minimum_free_vram_bytes=10 * 1024**3,
        free_vram_after_bytes=10 * 1024**3,
        target_process_peak_used_bytes=6 * 1024**3,
        whole_gpu_peak_used_bytes=6 * 1024**3,
        other_process_peak_used_bytes=0,
        sample_count=2,
        failed_sample_count=0,
        process_identity_verified=True,
        gpu_identity_verified=True,
        confidence=CalibrationConfidence.HIGH.value,
        last_sample_fingerprint="old",
        measurement_version="test",
        warning_codes_json="[]",
        failure_codes_json="[]",
        updated_at=timestamp,
    )
    sample = GpuMemorySample(
        timestamp=timestamp + timedelta(seconds=1),
        gpu_index=0,
        total_bytes=24 * 1024**3,
        free_bytes=20 * 1024**3,
        process_used_bytes=8 * 1024**3,
        gpu_uuid_fingerprint="gpu-b",
        whole_gpu_used_bytes=4 * 1024**3,
        identity_verified=True,
        gpu_identity_verified=True,
    )

    code = _merge_sample(record, sample, "new")
    assert code == "GPU_CHANGED_DURING_JOB"
    _add_codes(record, "failure_codes_json", (code,))
    assert record.gpu_uuid_fingerprint == "gpu-a"
    assert record.gpu_total_vram_bytes == 16 * 1024**3
    assert record.target_process_peak_used_bytes == 6 * 1024**3
    assert record.minimum_free_vram_bytes == 10 * 1024**3
    assert not record.gpu_identity_verified
    assert not record.process_identity_verified
    assert _confidence(record) == CalibrationConfidence.NONE.value


def test_gpu_calibration_exclusion_reasons_are_classified() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)

    def selected(
        status: SelectedGpuStatus, warnings: tuple[str, ...] = ()
    ) -> TrainingJobSelectedGpuRecord:
        return TrainingJobSelectedGpuRecord(
            id=str(uuid4()),
            training_job_id=str(uuid4()),
            gpu_uuid_fingerprint="gpu-a",
            selected_at=timestamp,
            selection_source="target_process",
            status=status.value,
            warning_codes_json=json.dumps(warnings),
        )

    cases = (
        (
            selected(SelectedGpuStatus.CHANGED),
            GpuMemorySummary(),
            False,
            GpuCalibrationExclusionReason.GPU_CHANGED_DURING_JOB,
        ),
        (
            selected(SelectedGpuStatus.PHYSICAL_GPU_NOT_FOUND),
            GpuMemorySummary(),
            False,
            GpuCalibrationExclusionReason.PHYSICAL_GPU_NOT_FOUND,
        ),
        (
            selected(SelectedGpuStatus.IDENTITY_UNVERIFIED),
            GpuMemorySummary(),
            False,
            GpuCalibrationExclusionReason.GPU_IDENTITY_UNVERIFIED,
        ),
        (
            selected(SelectedGpuStatus.AMBIGUOUS_SELECTION),
            GpuMemorySummary(),
            False,
            GpuCalibrationExclusionReason.AMBIGUOUS_GPU_SELECTION,
        ),
        (
            selected(SelectedGpuStatus.OK),
            GpuMemorySummary(failure_codes=("TARGET_PID_NOT_FOUND",)),
            False,
            GpuCalibrationExclusionReason.TARGET_PROCESS_GPU_NOT_FOUND,
        ),
        (
            selected(SelectedGpuStatus.OK),
            GpuMemorySummary(),
            True,
            GpuCalibrationExclusionReason.SELECTED_GPU_MEMORY_MISMATCH,
        ),
    )

    for selected_gpu, memory, memory_mismatch, expected_reason in cases:
        _, _, reasons = _gpu_calibration_assessment(
            selected_gpu=selected_gpu,
            job_environment=None,
            memory=memory,
            memory_identity_mismatch=memory_mismatch,
        )
        assert expected_reason in reasons
        assert not _gpu_calibration_is_usable(reasons)


def test_gpu_calibration_same_gpu_has_no_exclusion_reason() -> None:
    _, _, reasons = _gpu_calibration_assessment(
        selected_gpu=TrainingJobSelectedGpuRecord(
            id=str(uuid4()),
            training_job_id=str(uuid4()),
            gpu_uuid_fingerprint="gpu-a",
            selected_at=datetime(2026, 1, 1, tzinfo=UTC),
            selection_source="target_process",
            status=SelectedGpuStatus.OK.value,
            warning_codes_json="[]",
        ),
        job_environment=None,
        memory=GpuMemorySummary(
            gpu_uuid_fingerprint="gpu-a", failure_codes=(), warning_codes=()
        ),
        memory_identity_mismatch=False,
    )
    assert reasons == ()
    assert _gpu_calibration_is_usable(reasons)


def test_gpu_reason_change_updates_calibration_fingerprint_and_stales_sources() -> None:
    record = TrainingExecutionSummaryRecord()
    record.selected_gpu_status = SelectedGpuStatus.OK.value
    record.selected_gpu_warning_codes_json = "[]"
    record.memory_failure_codes_json = "[]"
    record.exclusion_reasons_json = "[]"
    record.usable_for_speed_calibration = True
    record.usable_for_memory_calibration = True
    record.summary_content_fingerprint = ""
    _refresh_summary_fingerprints(record)
    previous_state = record.calibration_state_fingerprint

    record.exclusion_reasons_json = json.dumps(
        [GpuCalibrationExclusionReason.GPU_IDENTITY_UNVERIFIED.value]
    )
    record.usable_for_speed_calibration = False
    record.usable_for_memory_calibration = False
    _refresh_summary_fingerprints(record)
    assert record.calibration_state_fingerprint != previous_state

    session = Mock()
    session.scalars.return_value.all.return_value = ["calibration-a"]
    _mark_related_calibrations_stale(session, "summary-a")
    session.query.return_value.filter.return_value.update.assert_called_once_with(
        {"stale": True}, synchronize_session=False
    )


def test_nvidia_smi_attributes_target_pid_per_gpu_and_rejects_over_total(
    monkeypatch,
) -> None:
    adapter = NvidiaSmiGpuMemoryAdapter()
    global_uuid_a = "GPU-a"
    global_uuid_b = "GPU-b"
    queries: list[str] = []

    def fake_query(query: list[str]) -> list[str]:
        queries.append(query[0])
        if query[0].startswith("--query-gpu"):
            return [
                "0, GPU-a, 16000, 12000",
                "1, GPU-b, 16000, 11000",
            ]
        return [
            "42, GPU-a, 100",
            "99, GPU-a, 300",
            "42, GPU-b, 200",
            "99, GPU-b, 400",
            "42, GPU-a, 20000",
        ]

    monkeypatch.setattr(adapter, "_run_query", fake_query)
    samples = adapter.collect(
        pid=42,
        process_identity_verified=True,
        expected_gpu_uuid_fingerprints=(
            gpu_uuid_fingerprint(global_uuid_a),
            gpu_uuid_fingerprint(global_uuid_b),
        ),
    )

    assert queries == [
        "--query-gpu=index,uuid,memory.total,memory.free",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
    ]
    assert [(item.gpu_index, item.process_used_bytes) for item in samples] == [
        (0, 100 * 1024**2),
        (1, 200 * 1024**2),
    ]
    assert [item.other_process_used_bytes for item in samples] == [
        300 * 1024**2,
        400 * 1024**2,
    ]
    assert all(item.identity_verified for item in samples)


def test_nvidia_smi_process_query_can_select_one_runtime_gpu_without_expected_set(
    monkeypatch,
) -> None:
    adapter = NvidiaSmiGpuMemoryAdapter()

    def fake_query(query: list[str]) -> list[str]:
        if query[0].startswith("--query-gpu"):
            return [
                "2, GPU-b, 20000, 15000",
                "0, GPU-a, 16000, 12000",
            ]
        return ["42, GPU-b, 100"]

    monkeypatch.setattr(adapter, "_run_query", fake_query)

    samples = adapter.collect(
        pid=42,
        process_identity_verified=True,
        expected_gpu_uuid_fingerprints=(),
    )

    selected = [sample for sample in samples if sample.identity_verified]
    assert len(selected) == 1
    assert selected[0].gpu_index == 2
    assert selected[0].gpu_uuid_fingerprint == gpu_uuid_fingerprint("GPU-b")


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


def test_calibration_rejects_compute_capability_mismatch() -> None:
    summary = replace(_summary(), gpu_architecture="Arch-B", compute_capability="8.0")
    snapshot = RecommendationCalibrationService().build(
        (summary,),
        gpu_identity_fingerprint="gpu-a",
        gpu_architecture="Arch-B",
        compute_capability="8.0",
    )

    assert TrainingCalibrationMatcher().matches(summary, snapshot)
    assert not TrainingCalibrationMatcher().matches(
        replace(summary, compute_capability="7.5"), snapshot
    )


def test_oom_feedback_only_suggests_lower_batch_and_keeps_baseline_safety_floor() -> (
    None
):
    summary = _summary(oom=True, speed=None, batch_size=2)
    snapshot = RecommendationCalibrationService().build(
        (summary,),
        gpu_identity_fingerprint="gpu-a",
        resolution=1024,
        batch_size=2,
        network_module="networks.lora",
        network_dim=16,
        network_alpha=16,
        optimizer="AdamW8bit",
        mixed_precision="fp16",
        cache_latents=False,
        gradient_checkpointing=False,
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
    assert result.suggested_batch_size == 2
    assert "baseline_fallback" in result.reason_codes
    assert result.baseline_vram_bytes == recommendation.estimated_vram_bytes
