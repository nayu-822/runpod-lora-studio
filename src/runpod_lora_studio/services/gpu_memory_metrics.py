from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from runpod_lora_studio.domain.training_performance_models import (
    CalibrationConfidence,
    GpuMemorySample,
    GpuMemorySummary,
)


class GpuMemoryMetricsAdapter(Protocol):
    def collect(self, *, pid: int | None = None) -> tuple[GpuMemorySample, ...]: ...


class NvidiaSmiGpuMemoryAdapter:
    """Read bounded, fixed-format nvidia-smi output without shell execution."""

    def __init__(
        self,
        *,
        executable: str = "nvidia-smi",
        timeout_seconds: float = 2.0,
        max_output_bytes: int = 32 * 1024,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def collect(self, *, pid: int | None = None) -> tuple[GpuMemorySample, ...]:
        del pid  # Process attribution requires a validated monitor identity.
        executable = shutil.which(self.executable) or self.executable
        command = [
            executable,
            "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        output = result.stdout[: self.max_output_bytes]
        if result.returncode != 0:
            return ()
        now = datetime.now(UTC)
        samples: list[GpuMemorySample] = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                index = int(fields[0])
                total = int(float(fields[2]) * 1024**2)
                free = int(float(fields[3]) * 1024**2)
            except ValueError:
                continue
            if index < 0 or total <= 0 or free < 0 or free > total:
                continue
            samples.append(
                GpuMemorySample(
                    timestamp=now,
                    gpu_index=index,
                    total_bytes=total,
                    free_bytes=free,
                )
            )
        return tuple(samples)


class StaticGpuMemoryMetricsAdapter:
    """Small deterministic adapter for tests and offline collection."""

    def __init__(self, samples: Sequence[GpuMemorySample]) -> None:
        self.samples = tuple(samples)
        self.calls = 0

    def collect(self, *, pid: int | None = None) -> tuple[GpuMemorySample, ...]:
        del pid
        self.calls += 1
        return self.samples


def summarize_gpu_memory(samples: Sequence[GpuMemorySample]) -> GpuMemorySummary:
    valid = [
        sample
        for sample in samples
        if sample.total_bytes is not None
        and sample.total_bytes > 0
        and sample.free_bytes is not None
        and 0 <= sample.free_bytes <= sample.total_bytes
    ]
    if not valid:
        return GpuMemorySummary(missing_sample_count=len(samples))
    total = min(
        sample.total_bytes for sample in valid if sample.total_bytes is not None
    )
    frees = [sample.free_bytes for sample in valid if sample.free_bytes is not None]
    process_values = [
        sample.process_used_bytes
        for sample in valid
        if sample.process_used_bytes is not None
    ]
    other_values = [
        sample.other_process_used_bytes
        for sample in valid
        if sample.other_process_used_bytes is not None
    ]
    ratio = None
    if other_values:
        ratio = min(1.0, max(0.0, max(other_values) / total))
    confidence = CalibrationConfidence.HIGH
    if len(valid) < 3 or ratio is not None and ratio > 0.25:
        confidence = CalibrationConfidence.MEDIUM
    if len(valid) == 1:
        confidence = CalibrationConfidence.LOW
    return GpuMemorySummary(
        total_bytes=total,
        free_before_bytes=frees[0],
        free_after_bytes=frees[-1],
        target_peak_allocated_bytes=max(process_values) if process_values else None,
        target_peak_reserved_bytes=max(process_values) if process_values else None,
        whole_gpu_min_free_bytes=min(frees),
        sample_count=len(valid),
        missing_sample_count=len(samples) - len(valid),
        other_process_ratio=ratio,
        confidence=confidence,
    )
