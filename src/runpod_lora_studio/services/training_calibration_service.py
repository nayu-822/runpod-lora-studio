from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from statistics import median
from typing import Any
from uuid import UUID, uuid5

from runpod_lora_studio.domain.recommendation_models import TrainingRecommendation
from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    CalibrationRecommendationResult,
    TrainingCalibrationSnapshot,
    TrainingExecutionSummary,
    TrainingFailureCategory,
)


class TrainingPerformanceOutlierFilter:
    """Apply deterministic, conservative filters to collected summaries."""

    def filter_speed(
        self, summaries: Iterable[TrainingExecutionSummary]
    ) -> tuple[TrainingExecutionSummary, ...]:
        values = tuple(summary for summary in summaries if self.speed_eligible(summary))
        speeds = [
            summary.measured_steps_per_second
            for summary in values
            if summary.measured_steps_per_second
        ]
        if len(speeds) < 3:
            return values
        center = median(speeds)
        deviations = [abs(value - center) for value in speeds]
        mad = median(deviations)
        limit = max(center * 3.0, center + 6.0 * mad)
        floor = max(0.000001, center / 3.0)
        return tuple(
            summary
            for summary in values
            if summary.measured_steps_per_second is not None
            and floor <= summary.measured_steps_per_second <= limit
        )

    def speed_eligible(self, summary: TrainingExecutionSummary) -> bool:
        return (
            summary.calibration_included
            and summary.usable_for_speed_calibration
            and summary.failure_category is TrainingFailureCategory.NONE
            and summary.planned_total_steps is not None
            and summary.completed_steps is not None
            and summary.completed_steps > 0
            and summary.elapsed_seconds is not None
            and summary.elapsed_seconds > 0
            and summary.measured_steps_per_second is not None
            and summary.measured_steps_per_second > 0
            and summary.gpu_identity_fingerprint is not None
        )

    def filter_memory(
        self, summaries: Iterable[TrainingExecutionSummary]
    ) -> tuple[TrainingExecutionSummary, ...]:
        return tuple(summary for summary in summaries if self.memory_eligible(summary))

    def memory_eligible(self, summary: TrainingExecutionSummary) -> bool:
        return (
            summary.calibration_included
            and summary.usable_for_memory_calibration
            and summary.gpu_identity_fingerprint is not None
            and summary.process_identity_verified
            and summary.gpu_identity_verified
            and summary.memory_sample_count >= 2
            and summary.memory_confidence
            in {CalibrationConfidence.MEDIUM, CalibrationConfidence.HIGH}
            and summary.memory_coverage_seconds is not None
            and summary.memory_coverage_seconds > 0
            and summary.peak_reserved_vram_bytes is not None
            and summary.gpu_total_vram_bytes is not None
            and summary.peak_reserved_vram_bytes <= summary.gpu_total_vram_bytes
            and summary.resolution is not None
            and summary.mixed_precision is not None
        )


class TrainingCalibrationMatcher:
    """Match only comparable GPU and training settings."""

    def matches(
        self, summary: TrainingExecutionSummary, snapshot: TrainingCalibrationSnapshot
    ) -> bool:
        if summary.gpu_identity_fingerprint != snapshot.gpu_identity_fingerprint:
            return False
        if (
            snapshot.gpu_total_vram_class is not None
            and _vram_class(summary.gpu_total_vram_bytes)
            != snapshot.gpu_total_vram_class
        ):
            return False
        for actual, expected in (
            (summary.gpu_architecture, snapshot.gpu_architecture),
            (summary.resolution, snapshot.resolution),
            (summary.batch_size, snapshot.batch_size),
            (
                summary.gradient_accumulation_steps,
                snapshot.gradient_accumulation_steps,
            ),
            (summary.effective_batch_size, snapshot.effective_batch_size),
            (summary.network_module, snapshot.network_module),
            (summary.network_dim, snapshot.network_dim),
            (summary.network_alpha, snapshot.network_alpha),
            (summary.optimizer, snapshot.optimizer),
            (summary.mixed_precision, snapshot.mixed_precision),
            (summary.cache_latents, snapshot.cache_latents),
            (summary.gradient_checkpointing, snapshot.gradient_checkpointing),
            (summary.world_size, snapshot.world_size),
            (summary.sd_scripts_version, snapshot.sd_scripts_version),
            (summary.xformers_available, snapshot.xformers_available),
        ):
            if expected is not None and actual != expected:
                return False
        return True

    def filter(
        self,
        summaries: Iterable[TrainingExecutionSummary],
        snapshot: TrainingCalibrationSnapshot,
    ) -> tuple[TrainingExecutionSummary, ...]:
        return tuple(
            summary for summary in summaries if self.matches(summary, snapshot)
        )


class RecommendationCalibrationService:
    """Build deterministic empirical calibration snapshots from eligible runs."""

    version = "phase7b-calibration-v1"

    def __init__(
        self, *, outlier_filter: TrainingPerformanceOutlierFilter | None = None
    ) -> None:
        self.outlier_filter = outlier_filter or TrainingPerformanceOutlierFilter()

    def build(
        self,
        summaries: Sequence[TrainingExecutionSummary],
        *,
        gpu_identity_fingerprint: str,
        gpu_total_vram_bytes: int | None = None,
        resolution: int | None = None,
        optimizer: str | None = None,
        mixed_precision: str | None = None,
        cache_latents: bool | None = None,
        gradient_checkpointing: bool | None = None,
        gpu_architecture: str | None = None,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        effective_batch_size: int | None = None,
        network_module: str | None = None,
        network_dim: int | None = None,
        network_alpha: int | None = None,
        world_size: int | None = None,
        sd_scripts_version: str | None = None,
        xformers_available: bool | None = None,
        scope_project_id: UUID | None = None,
        now: datetime | None = None,
    ) -> TrainingCalibrationSnapshot:
        matching = tuple(
            summary
            for summary in summaries
            if summary.calibration_included
            and summary.gpu_identity_fingerprint == gpu_identity_fingerprint
            and (
                gpu_total_vram_bytes is None
                or _vram_class(summary.gpu_total_vram_bytes)
                == _vram_class(gpu_total_vram_bytes)
            )
            and (resolution is None or summary.resolution == resolution)
            and (optimizer is None or summary.optimizer == optimizer)
            and (mixed_precision is None or summary.mixed_precision == mixed_precision)
            and (cache_latents is None or summary.cache_latents == cache_latents)
            and (
                gradient_checkpointing is None
                or summary.gradient_checkpointing == gradient_checkpointing
            )
            and (
                gpu_architecture is None or summary.gpu_architecture == gpu_architecture
            )
            and (batch_size is None or summary.batch_size == batch_size)
            and (
                gradient_accumulation_steps is None
                or summary.gradient_accumulation_steps == gradient_accumulation_steps
            )
            and (
                effective_batch_size is None
                or summary.effective_batch_size == effective_batch_size
            )
            and (network_module is None or summary.network_module == network_module)
            and (network_dim is None or summary.network_dim == network_dim)
            and (network_alpha is None or summary.network_alpha == network_alpha)
            and (world_size is None or summary.world_size == world_size)
            and (
                sd_scripts_version is None
                or summary.sd_scripts_version == sd_scripts_version
            )
            and (
                xformers_available is None
                or summary.xformers_available == xformers_available
            )
        )
        speed = self.outlier_filter.filter_speed(matching)
        memory = self.outlier_filter.filter_memory(matching)
        source_ids = tuple(sorted((summary.id for summary in matching), key=str))
        source_fingerprint = _hash(
            [
                _summary_source_token(summary)
                for summary in sorted(matching, key=_summary_source_token)
            ]
        )
        speeds = sorted(
            summary.measured_steps_per_second
            for summary in speed
            if summary.measured_steps_per_second is not None
        )
        peaks = sorted(
            summary.peak_reserved_vram_bytes
            for summary in memory
            if summary.peak_reserved_vram_bytes is not None
        )
        oom_count = sum(summary.oom_detected for summary in matching)
        confidence = _confidence(len(matching), len(speed), len(memory), oom_count)
        reasons: list[str] = []
        if not speed:
            reasons.append("insufficient_speed_history")
        if not memory:
            reasons.append("insufficient_memory_history")
        if oom_count:
            reasons.append("oom_history_present")
        fingerprint = _hash(
            {
                "scope_project_id": str(scope_project_id) if scope_project_id else None,
                "gpu": gpu_identity_fingerprint,
                "vram": gpu_total_vram_bytes,
                "gpu_architecture": gpu_architecture,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "network_module": network_module,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "world_size": world_size,
                "sd_scripts_version": sd_scripts_version,
                "xformers_available": xformers_available,
                "resolution": resolution,
                "optimizer": optimizer,
                "mixed_precision": mixed_precision,
                "cache_latents": cache_latents,
                "gradient_checkpointing": gradient_checkpointing,
                "source": source_fingerprint,
                "version": self.version,
            }
        )
        timestamp = now or datetime.now(UTC)
        upper_peak = _percentile(peaks, 0.75)
        return TrainingCalibrationSnapshot(
            id=uuid5(UUID("00000000-0000-0000-0000-0000000007b0"), fingerprint),
            scope_project_id=scope_project_id,
            gpu_identity_fingerprint=gpu_identity_fingerprint,
            gpu_total_vram_class=_vram_class(gpu_total_vram_bytes),
            resolution=resolution,
            optimizer=optimizer,
            mixed_precision=mixed_precision,
            cache_latents=cache_latents,
            gradient_checkpointing=gradient_checkpointing,
            sample_count=len(matching),
            successful_sample_count=len(speed),
            oom_sample_count=oom_count,
            median_steps_per_second=median(speeds) if speeds else None,
            lower_percentile_steps_per_second=_percentile(speeds, 0.25),
            median_peak_vram_bytes=int(median(peaks)) if peaks else None,
            upper_percentile_peak_vram_bytes=int(upper_peak)
            if upper_peak is not None
            else None,
            confidence=confidence,
            calibration_fingerprint=fingerprint,
            calibration_version=self.version,
            generated_at=timestamp,
            reason_codes=tuple(reasons),
            source_summary_ids=source_ids,
            source_summary_fingerprint=source_fingerprint,
            gpu_architecture=gpu_architecture,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            effective_batch_size=effective_batch_size,
            network_module=network_module,
            network_dim=network_dim,
            network_alpha=network_alpha,
            world_size=world_size,
            sd_scripts_version=sd_scripts_version,
            xformers_available=xformers_available,
        )

    def is_stale(
        self,
        snapshot: TrainingCalibrationSnapshot,
        summaries: Sequence[TrainingExecutionSummary],
    ) -> bool:
        current = _hash(
            [
                _summary_source_token(summary)
                for summary in sorted(
                    (
                        item
                        for item in summaries
                        if item.calibration_included
                        and self._matches_snapshot(item, snapshot)
                    ),
                    key=_summary_source_token,
                )
            ]
        )
        return snapshot.stale or current != snapshot.source_summary_fingerprint

    @staticmethod
    def _matches_snapshot(
        summary: TrainingExecutionSummary,
        snapshot: TrainingCalibrationSnapshot,
    ) -> bool:
        return (
            summary.gpu_identity_fingerprint == snapshot.gpu_identity_fingerprint
            and (
                snapshot.gpu_total_vram_class is None
                or _vram_class(summary.gpu_total_vram_bytes)
                == snapshot.gpu_total_vram_class
            )
            and (
                snapshot.resolution is None or summary.resolution == snapshot.resolution
            )
            and (snapshot.optimizer is None or summary.optimizer == snapshot.optimizer)
            and (
                snapshot.mixed_precision is None
                or summary.mixed_precision == snapshot.mixed_precision
            )
            and (
                snapshot.cache_latents is None
                or summary.cache_latents == snapshot.cache_latents
            )
            and (
                snapshot.gradient_checkpointing is None
                or summary.gradient_checkpointing == snapshot.gradient_checkpointing
            )
            and (
                snapshot.gpu_architecture is None
                or summary.gpu_architecture == snapshot.gpu_architecture
            )
            and (
                snapshot.batch_size is None or summary.batch_size == snapshot.batch_size
            )
            and (
                snapshot.gradient_accumulation_steps is None
                or summary.gradient_accumulation_steps
                == snapshot.gradient_accumulation_steps
            )
            and (
                snapshot.effective_batch_size is None
                or summary.effective_batch_size == snapshot.effective_batch_size
            )
            and (
                snapshot.network_module is None
                or summary.network_module == snapshot.network_module
            )
            and (
                snapshot.network_dim is None
                or summary.network_dim == snapshot.network_dim
            )
            and (
                snapshot.network_alpha is None
                or summary.network_alpha == snapshot.network_alpha
            )
            and (
                snapshot.world_size is None or summary.world_size == snapshot.world_size
            )
            and (
                snapshot.sd_scripts_version is None
                or summary.sd_scripts_version == snapshot.sd_scripts_version
            )
            and (
                snapshot.xformers_available is None
                or summary.xformers_available == snapshot.xformers_available
            )
        )


class CalibratedRecommendationService:
    """Apply empirical corrections without relaxing the Phase 7A safety floor."""

    def apply(
        self,
        recommendation: TrainingRecommendation,
        snapshot: TrainingCalibrationSnapshot | None,
        *,
        current_gpu_identity_fingerprint: str | None = None,
    ) -> tuple[TrainingRecommendation, CalibrationRecommendationResult]:
        baseline_duration = recommendation.estimated_duration_seconds
        baseline_vram = recommendation.estimated_vram_bytes
        baseline_batch = recommendation.batch_size
        if (
            snapshot is None
            or snapshot.stale
            or snapshot.confidence is CalibrationConfidence.NONE
        ):
            result = CalibrationRecommendationResult(
                baseline_duration,
                baseline_duration,
                baseline_vram,
                baseline_vram,
                baseline_batch,
                baseline_batch,
                CalibrationConfidence.NONE,
                0,
                ("baseline_fallback",),
                ("empirical_calibration_unavailable",),
            )
            return recommendation, result
        if (
            current_gpu_identity_fingerprint
            and current_gpu_identity_fingerprint != snapshot.gpu_identity_fingerprint
        ):
            result = CalibrationRecommendationResult(
                baseline_duration,
                baseline_duration,
                baseline_vram,
                baseline_vram,
                baseline_batch,
                baseline_batch,
                CalibrationConfidence.NONE,
                snapshot.sample_count,
                ("gpu_identity_changed", "baseline_fallback"),
                ("calibration_is_stale_for_current_gpu",),
                snapshot.calibration_fingerprint,
            )
            return recommendation, result
        duration = baseline_duration
        if (
            snapshot.lower_percentile_steps_per_second
            and snapshot.lower_percentile_steps_per_second > 0
            and recommendation.estimated_total_steps
        ):
            # Phase 7A estimates are not reinterpreted as exact measurements.
            duration = max(
                0.0,
                (recommendation.estimated_total_steps or 0)
                / snapshot.lower_percentile_steps_per_second,
            )
        vram = baseline_vram
        if snapshot.upper_percentile_peak_vram_bytes is not None:
            measured = int(snapshot.upper_percentile_peak_vram_bytes * 1.15)
            vram = (
                max(baseline_vram or 0, measured)
                if baseline_vram is not None
                else measured
            )
        suggested_batch = baseline_batch
        reasons = list(snapshot.reason_codes)
        warnings = ["empirical calibration is not a quality or safety guarantee"]
        oom_compatible = (
            snapshot.oom_sample_count > 0
            and snapshot.confidence
            in {
                CalibrationConfidence.MEDIUM,
                CalibrationConfidence.HIGH,
            }
            and snapshot.resolution == recommendation.resolution
            and snapshot.batch_size == recommendation.batch_size
            and snapshot.gradient_accumulation_steps
            == recommendation.gradient_accumulation_steps
            and snapshot.effective_batch_size
            == recommendation.batch_size * recommendation.gradient_accumulation_steps
            and snapshot.network_module == recommendation.network_module
            and snapshot.network_dim == recommendation.network_dim
            and snapshot.network_alpha == recommendation.network_alpha
            and snapshot.optimizer == recommendation.optimizer
            and snapshot.mixed_precision == recommendation.mixed_precision
            and snapshot.cache_latents == recommendation.cache_latents
            and snapshot.gradient_checkpointing == recommendation.gradient_checkpointing
        )
        if oom_compatible:
            suggested_batch = max(1, baseline_batch - 1)
            reasons.append("reduce_batch_after_oom")
            warnings.append(
                "OOM history requires user confirmation before applying a "
                "lower batch size"
            )
        elif snapshot.oom_sample_count:
            reasons.append("oom_history_low_similarity")
            warnings.append("OOM history did not match the current training settings")
        result = CalibrationRecommendationResult(
            baseline_duration,
            duration,
            baseline_vram,
            vram,
            baseline_batch,
            suggested_batch,
            snapshot.confidence,
            snapshot.sample_count,
            tuple(sorted(set(reasons))),
            tuple(warnings),
            snapshot.calibration_fingerprint,
        )
        calibrated = replace(
            recommendation,
            estimated_duration_seconds=duration,
            estimated_vram_bytes=vram,
            calibration_snapshot_id=snapshot.id,
            calibration_confidence=snapshot.confidence.value,
            calibration_reason_codes=result.reason_codes,
            baseline_duration_seconds=baseline_duration,
            calibrated_duration_seconds=duration,
            baseline_vram_bytes=baseline_vram,
            calibrated_vram_bytes=vram,
            baseline_batch_size=baseline_batch,
            calibrated_batch_size=suggested_batch,
            calibration_fingerprint=snapshot.calibration_fingerprint,
        )
        return calibrated, result


def _confidence(
    sample_count: int, speed_count: int, memory_count: int, oom_count: int
) -> CalibrationConfidence:
    if sample_count == 0 or (speed_count == 0 and memory_count == 0):
        return CalibrationConfidence.NONE
    if sample_count >= 5 and speed_count >= 3 and memory_count >= 3 and oom_count == 0:
        return CalibrationConfidence.HIGH
    if sample_count >= 2:
        return CalibrationConfidence.MEDIUM
    return CalibrationConfidence.LOW


def _percentile(values: Sequence[float | int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _vram_class(value: int | None) -> str | None:
    return f"{round(value / 1024**3)}GiB" if value and value > 0 else None


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True).encode()
    ).hexdigest()


def _summary_source_token(summary: TrainingExecutionSummary) -> str:
    content = summary.summary_content_fingerprint or summary.summary_fingerprint
    if not content:
        content = _hash(
            {
                key: value
                for key, value in asdict(summary).items()
                if key
                not in {
                    "id",
                    "training_job_id",
                    "created_at",
                    "updated_at",
                    "summary_fingerprint",
                    "summary_content_fingerprint",
                    "calibration_state_fingerprint",
                    "calibration_included",
                    "manual_exclusion_reason",
                    "failure_category",
                    "failure_evidence_codes",
                    "oom_detected",
                    "usable_for_speed_calibration",
                    "usable_for_memory_calibration",
                    "collector_version",
                    "classifier_version",
                }
            }
        )
    return _hash(
        {
            "summary": content,
            "summary_fingerprint": summary.summary_fingerprint,
            "calibration_state": summary.calibration_state_fingerprint,
            "included": summary.calibration_included,
            "manual_exclusion_reason": summary.manual_exclusion_reason,
            "failure_category": summary.failure_category.value,
            "oom_detected": summary.oom_detected,
            "usable_for_speed": summary.usable_for_speed_calibration,
            "usable_for_memory": summary.usable_for_memory_calibration,
            "collector_version": summary.collector_version,
            "classifier_version": summary.classifier_version,
        }
    )


class TrainingCalibrationService:
    """Compatibility facade for the persistence-backed calibration service."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from runpod_lora_studio.services.recommendation_calibration_service import (
            TrainingCalibrationService as PersistentTrainingCalibrationService,
        )

        self._service = PersistentTrainingCalibrationService(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)
