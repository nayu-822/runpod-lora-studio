from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    GpuMemoryAggregate,
    GpuMemorySample,
)
from runpod_lora_studio.external.training_process import StartedProcess
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    RecommendationCalibrationSnapshotRecord,
    RecommendationCalibrationSourceRecord,
    TrainingExecutionSummaryRecord,
    TrainingMemoryAggregateRecord,
)
from runpod_lora_studio.services.gpu_memory_metrics import (
    GpuMemoryMetricsAdapter,
    NvidiaSmiGpuMemoryAdapter,
)


class TrainingMemoryMonitor:
    """Persist bounded job-scoped GPU memory aggregates during execution."""

    version = "phase7b-memory-v2"

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: GpuMemoryMetricsAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.adapter = adapter or NvidiaSmiGpuMemoryAdapter()

    def capture_before_start(
        self,
        job_id: UUID,
        *,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> GpuMemoryAggregate:
        return self._capture(
            job_id,
            pid=None,
            process_identity=None,
            process_group_id=None,
            process_identity_verified=False,
            expected_gpu_uuid_fingerprints=expected_gpu_uuid_fingerprints,
        )

    def measure(
        self,
        job_id: UUID,
        started: StartedProcess,
        *,
        process_identity_verified: bool,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> GpuMemoryAggregate:
        return self._capture(
            job_id,
            pid=started.pid,
            process_identity=started.process_identity,
            process_group_id=started.process_group_id,
            process_identity_verified=process_identity_verified,
            expected_gpu_uuid_fingerprints=expected_gpu_uuid_fingerprints,
        )

    def capture_after_process(
        self,
        job_id: UUID,
        *,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> GpuMemoryAggregate:
        return self._capture(
            job_id,
            pid=None,
            process_identity=None,
            process_group_id=None,
            process_identity_verified=False,
            expected_gpu_uuid_fingerprints=expected_gpu_uuid_fingerprints,
        )

    def restore(self, job_id: UUID) -> GpuMemoryAggregate | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingMemoryAggregateRecord).where(
                    TrainingMemoryAggregateRecord.training_job_id == str(job_id)
                )
            )
            return _aggregate_from_record(record) if record else None

    def _capture(
        self,
        job_id: UUID,
        *,
        pid: int | None,
        process_identity: str | None,
        process_group_id: int | None,
        process_identity_verified: bool,
        expected_gpu_uuid_fingerprints: Sequence[str],
    ) -> GpuMemoryAggregate:
        samples: tuple[GpuMemorySample, ...]
        measurement_failed = False
        try:
            if pid is not None and pid <= 0:
                samples = ()
            else:
                samples = self.adapter.collect(
                    pid=pid,
                    process_identity=process_identity,
                    process_group_id=process_group_id,
                    process_identity_verified=process_identity_verified,
                    expected_gpu_uuid_fingerprints=expected_gpu_uuid_fingerprints,
                )
        except Exception:
            samples = ()
            measurement_failed = True
        sample = _select_sample(samples, expected_gpu_uuid_fingerprints)
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingMemoryAggregateRecord).where(
                    TrainingMemoryAggregateRecord.training_job_id == str(job_id)
                )
            )
            if record is None:
                record = TrainingMemoryAggregateRecord(
                    id=str(uuid4()),
                    training_job_id=str(job_id),
                    measurement_version=self.version,
                    updated_at=datetime.now(UTC),
                )
                session.add(record)
            aggregate_changed = False
            if measurement_failed:
                _add_codes(record, "failure_codes_json", ("NVIDIA_SMI_QUERY_FAILED",))
            if sample is None:
                record.failed_sample_count += 1
                aggregate_changed = True
                _add_codes(
                    record,
                    "failure_codes_json",
                    _selection_failure_codes(
                        samples,
                        pid=pid,
                        process_group_id=process_group_id,
                        process_identity_verified=process_identity_verified,
                        expected=expected_gpu_uuid_fingerprints,
                    ),
                )
            else:
                fingerprint = _sample_fingerprint(sample)
                if fingerprint != record.last_sample_fingerprint:
                    merge_code = _merge_sample(record, sample, fingerprint)
                    if merge_code:
                        _add_codes(record, "failure_codes_json", (merge_code,))
                    aggregate_changed = True
            if pid is not None and not process_identity_verified:
                _add_codes(record, "warning_codes_json", ("PROCESS_IDENTITY_MISMATCH",))
            if sample is not None and not sample.gpu_identity_verified:
                _add_codes(record, "warning_codes_json", ("GPU_IDENTITY_UNAVAILABLE",))
            if (
                sample is not None
                and pid is not None
                and sample.process_used_bytes is None
            ):
                _add_codes(record, "failure_codes_json", ("TARGET_PID_NOT_FOUND",))
            record.sample_count = min(
                record.sample_count, self.settings.training_memory_max_samples
            )
            record.failed_sample_count = min(
                record.failed_sample_count, self.settings.training_memory_max_samples
            )
            record.updated_at = datetime.now(UTC)
            if "GPU_CHANGED_DURING_JOB" in _codes_from_json(record.failure_codes_json):
                record.gpu_identity_verified = False
                record.process_identity_verified = False
            record.confidence = _confidence(record)
            if aggregate_changed:
                summary_ids = session.scalars(
                    select(TrainingExecutionSummaryRecord.id).where(
                        TrainingExecutionSummaryRecord.training_job_id == str(job_id)
                    )
                ).all()
                if summary_ids:
                    calibration_ids = session.scalars(
                        select(
                            RecommendationCalibrationSourceRecord.calibration_id
                        ).where(
                            RecommendationCalibrationSourceRecord.summary_id.in_(
                                summary_ids
                            )
                        )
                    ).all()
                    if calibration_ids:
                        session.query(RecommendationCalibrationSnapshotRecord).filter(
                            RecommendationCalibrationSnapshotRecord.id.in_(
                                calibration_ids
                            )
                        ).update({"stale": True}, synchronize_session=False)
            session.commit()
            return _aggregate_from_record(record)


def _select_sample(
    samples: Sequence[GpuMemorySample], expected: Sequence[str]
) -> GpuMemorySample | None:
    if not samples:
        return None
    target = [
        sample
        for sample in samples
        if sample.identity_verified and sample.gpu_identity_verified
    ]
    if len(target) == 1:
        return target[0]
    if len(target) > 1:
        return None
    expected_set = set(expected)
    matching = [
        sample for sample in samples if sample.gpu_uuid_fingerprint in expected_set
    ]
    if len(matching) == 1:
        return matching[0]
    if expected_set or len(samples) != 1:
        return None
    return samples[0]


def _merge_sample(
    record: TrainingMemoryAggregateRecord,
    sample: GpuMemorySample,
    fingerprint: str,
) -> str | None:
    if (
        record.sample_count > 0
        and record.gpu_uuid_fingerprint != sample.gpu_uuid_fingerprint
    ):
        record.failed_sample_count += 1
        record.gpu_identity_verified = False
        record.process_identity_verified = False
        return "GPU_CHANGED_DURING_JOB"
    if (
        record.gpu_uuid_fingerprint is None
        and record.sample_count > 0
        and sample.gpu_uuid_fingerprint is None
        and record.gpu_index != sample.gpu_index
    ):
        record.failed_sample_count += 1
        record.gpu_identity_verified = False
        record.process_identity_verified = False
        return "GPU_CHANGED_DURING_JOB"
    record.gpu_index = sample.gpu_index
    record.gpu_uuid_fingerprint = sample.gpu_uuid_fingerprint
    record.gpu_total_vram_bytes = _max_value(
        record.gpu_total_vram_bytes, sample.total_bytes
    )
    if sample.free_bytes is not None:
        if record.free_vram_before_bytes is None:
            record.free_vram_before_bytes = sample.free_bytes
        record.minimum_free_vram_bytes = _min_value(
            record.minimum_free_vram_bytes, sample.free_bytes
        )
        record.free_vram_after_bytes = sample.free_bytes
    record.target_process_peak_used_bytes = _max_value(
        record.target_process_peak_used_bytes,
        sample.process_used_bytes
        if sample.identity_verified and sample.gpu_identity_verified
        else None,
    )
    record.whole_gpu_peak_used_bytes = _max_value(
        record.whole_gpu_peak_used_bytes, sample.whole_gpu_used_bytes
    )
    record.other_process_peak_used_bytes = _max_value(
        record.other_process_peak_used_bytes, sample.other_process_used_bytes
    )
    if record.first_sampled_at is None or sample.timestamp < record.first_sampled_at:
        record.first_sampled_at = sample.timestamp
    if record.last_sampled_at is None or sample.timestamp > record.last_sampled_at:
        record.last_sampled_at = sample.timestamp
    record.sample_count += 1
    record.process_identity_verified = bool(
        record.process_identity_verified or sample.identity_verified
    )
    record.gpu_identity_verified = bool(
        record.gpu_identity_verified or sample.gpu_identity_verified
    )
    record.last_sample_fingerprint = fingerprint
    return None


def _aggregate_from_record(
    record: TrainingMemoryAggregateRecord,
) -> GpuMemoryAggregate:
    return GpuMemoryAggregate(
        job_id=UUID(record.training_job_id),
        gpu_index=record.gpu_index,
        gpu_uuid_fingerprint=record.gpu_uuid_fingerprint,
        gpu_total_vram_bytes=record.gpu_total_vram_bytes,
        free_vram_before_bytes=record.free_vram_before_bytes,
        minimum_free_vram_bytes=record.minimum_free_vram_bytes,
        free_vram_after_bytes=record.free_vram_after_bytes,
        target_process_peak_used_bytes=record.target_process_peak_used_bytes,
        whole_gpu_peak_used_bytes=record.whole_gpu_peak_used_bytes,
        other_process_peak_used_bytes=record.other_process_peak_used_bytes,
        sample_count=record.sample_count,
        failed_sample_count=record.failed_sample_count,
        first_sampled_at=record.first_sampled_at,
        last_sampled_at=record.last_sampled_at,
        process_identity_verified=bool(record.process_identity_verified),
        gpu_identity_verified=bool(record.gpu_identity_verified),
        confidence=CalibrationConfidence(record.confidence),
        last_sample_fingerprint=record.last_sample_fingerprint,
        measurement_version=record.measurement_version,
        warning_codes=_codes_from_json(record.warning_codes_json),
        failure_codes=_codes_from_json(record.failure_codes_json),
    )


def _selection_failure_codes(
    samples: Sequence[GpuMemorySample],
    *,
    pid: int | None,
    process_group_id: int | None,
    process_identity_verified: bool,
    expected: Sequence[str],
) -> tuple[str, ...]:
    process_codes: list[str] = []
    if pid is not None:
        if process_group_id is None:
            process_codes.append("PROCESS_GROUP_MISMATCH")
        if not process_identity_verified:
            process_codes.append("PROCESS_IDENTITY_MISMATCH")
    if not samples:
        if pid is not None:
            return tuple(process_codes + ["TARGET_PID_NOT_FOUND"])
        return ("GPU_IDENTITY_UNAVAILABLE",)
    target_count = sum(
        sample.identity_verified and sample.gpu_identity_verified for sample in samples
    )
    if target_count > 1:
        return tuple(process_codes + ["AMBIGUOUS_GPU_SELECTION"])
    expected_set = set(expected)
    selection_codes: list[str] = []
    if expected_set and not any(
        sample.gpu_uuid_fingerprint in expected_set for sample in samples
    ):
        selection_codes.append("EXPECTED_GPU_NOT_FOUND")
    if pid is not None and not any(
        sample.process_used_bytes is not None for sample in samples
    ):
        selection_codes.append("TARGET_PID_NOT_FOUND")
    return tuple(process_codes + selection_codes) or ("GPU_IDENTITY_UNAVAILABLE",)


def _add_codes(
    record: TrainingMemoryAggregateRecord, field: str, codes: Sequence[str]
) -> None:
    if not codes:
        return
    current = _codes_from_json(getattr(record, field, "[]"))
    setattr(record, field, json.dumps(sorted(set(current) | set(codes))))


def _codes_from_json(value: str | None) -> tuple[str, ...]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(sorted(str(item) for item in parsed if item))


def _sample_fingerprint(sample: GpuMemorySample) -> str:
    value = {
        "timestamp": sample.timestamp.isoformat(),
        "gpu_index": sample.gpu_index,
        "gpu_uuid": sample.gpu_uuid_fingerprint,
        "total": sample.total_bytes,
        "free": sample.free_bytes,
        "process": sample.process_used_bytes,
        "whole": sample.whole_gpu_used_bytes,
        "other": sample.other_process_used_bytes,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _max_value(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _min_value(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _confidence(record: TrainingMemoryAggregateRecord) -> str:
    if "GPU_CHANGED_DURING_JOB" in _codes_from_json(record.failure_codes_json):
        return CalibrationConfidence.NONE.value
    if record.sample_count == 0:
        return CalibrationConfidence.NONE.value
    if (
        record.sample_count >= 3
        and record.target_process_peak_used_bytes is not None
        and record.process_identity_verified
        and record.gpu_identity_verified
        and record.failed_sample_count <= record.sample_count // 10
    ):
        return CalibrationConfidence.HIGH.value
    if record.sample_count >= 2:
        return CalibrationConfidence.MEDIUM.value
    return CalibrationConfidence.LOW.value
