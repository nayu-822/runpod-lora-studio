from __future__ import annotations

import hashlib
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
    def collect(
        self,
        *,
        pid: int | None = None,
        process_identity: str | None = None,
        process_group_id: int | None = None,
        process_identity_verified: bool = False,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> tuple[GpuMemorySample, ...]: ...


def gpu_uuid_fingerprint(value: str) -> str:
    return hashlib.sha256(f"gpu-uuid:{value.strip().lower()}".encode()).hexdigest()[:32]


class NvidiaSmiGpuMemoryAdapter:
    """Read bounded fixed-format nvidia-smi output without shell execution."""

    measurement_version = "phase7b-memory-v2"

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

    def collect(
        self,
        *,
        pid: int | None = None,
        process_identity: str | None = None,
        process_group_id: int | None = None,
        process_identity_verified: bool = False,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> tuple[GpuMemorySample, ...]:
        del process_identity, process_group_id
        now = datetime.now(UTC)
        valid_pid = isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
        global_rows = self._run_query(
            [
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        if not global_rows:
            return ()
        process_rows = (
            self._run_query(
                [
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            if valid_pid
            else []
        )
        parsed_processes = _parse_process_rows(process_rows)
        target_rows = [row for row in parsed_processes if row[0] == pid]
        target_gpu_uuids = {uuid.lower() for _, uuid, used in target_rows if used >= 0}
        target_by_uuid: dict[str, list[int]] = {}
        for target_pid, uuid, used in target_rows:
            if target_pid == pid:
                target_by_uuid.setdefault(uuid, []).append(used)
        expected = set(expected_gpu_uuid_fingerprints)
        samples: list[GpuMemorySample] = []
        for line in global_rows:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                index = int(fields[0])
                uuid = fields[1]
                total = _mib_to_bytes(fields[2])
                free = _mib_to_bytes(fields[3])
            except ValueError:
                continue
            if index < 0 or not uuid or total <= 0 or free < 0 or free > total:
                continue
            uuid_fp = gpu_uuid_fingerprint(uuid)
            runtime_gpu_verified = (
                valid_pid
                and process_identity_verified
                and len(target_gpu_uuids) == 1
                and uuid.lower() in target_gpu_uuids
            )
            gpu_verified = (
                bool(expected) and uuid_fp in expected
            ) or runtime_gpu_verified
            rows_for_gpu = [
                row for row in parsed_processes if row[1] == uuid and row[2] <= total
            ]
            target_values = target_by_uuid.get(uuid, [])
            target_value = next(
                (value for value in target_values if value <= total), None
            )
            target_used = None
            identity_verified = False
            if (
                valid_pid
                and target_value is not None
                and gpu_verified
                and process_identity_verified
            ):
                target_used = target_value
                identity_verified = True
            other_used = sum(
                row[2] for row in rows_for_gpu if not (valid_pid and row[0] == pid)
            )
            samples.append(
                GpuMemorySample(
                    timestamp=now,
                    gpu_index=index,
                    total_bytes=total,
                    free_bytes=free,
                    process_used_bytes=target_used,
                    other_process_used_bytes=other_used,
                    gpu_uuid_fingerprint=uuid_fp,
                    whole_gpu_used_bytes=total - free,
                    identity_verified=identity_verified,
                    gpu_identity_verified=gpu_verified,
                )
            )
        return tuple(samples)

    def _run_query(self, query: list[str]) -> list[str]:
        executable = shutil.which(self.executable) or self.executable
        try:
            result = subprocess.run(
                [executable, *query],
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0:
            return []
        return result.stdout[: self.max_output_bytes].splitlines()


class StaticGpuMemoryMetricsAdapter:
    """Deterministic adapter for tests and offline collection."""

    def __init__(self, samples: Sequence[GpuMemorySample]) -> None:
        self.samples = tuple(samples)
        self.calls = 0

    def collect(
        self,
        *,
        pid: int | None = None,
        process_identity: str | None = None,
        process_group_id: int | None = None,
        process_identity_verified: bool = False,
        expected_gpu_uuid_fingerprints: Sequence[str] = (),
    ) -> tuple[GpuMemorySample, ...]:
        del (
            pid,
            process_identity,
            process_group_id,
            process_identity_verified,
            expected_gpu_uuid_fingerprints,
        )
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
        if (
            sample.process_used_bytes is not None
            and sample.identity_verified
            and sample.gpu_identity_verified
        )
    ]
    whole_values = [
        sample.whole_gpu_used_bytes
        for sample in valid
        if sample.whole_gpu_used_bytes is not None
    ]
    other_values = [
        sample.other_process_used_bytes
        for sample in valid
        if sample.other_process_used_bytes is not None
    ]
    first = min(sample.timestamp for sample in valid)
    last = max(sample.timestamp for sample in valid)
    process_verified = bool(process_values) and all(
        sample.identity_verified and sample.gpu_identity_verified
        for sample in valid
        if sample.process_used_bytes is not None
    )
    gpu_verified = all(sample.gpu_identity_verified for sample in valid)
    ratio = None
    if other_values:
        ratio = min(1.0, max(0.0, max(other_values) / total))
    confidence = CalibrationConfidence.HIGH
    if len(valid) < 3 or not process_verified or not gpu_verified:
        confidence = CalibrationConfidence.MEDIUM
    if len(valid) == 1:
        confidence = CalibrationConfidence.LOW
    return GpuMemorySummary(
        gpu_index=valid[0].gpu_index,
        gpu_uuid_fingerprint=valid[0].gpu_uuid_fingerprint,
        total_bytes=total,
        free_before_bytes=frees[0],
        free_after_bytes=frees[-1],
        target_peak_allocated_bytes=max(process_values) if process_values else None,
        target_peak_reserved_bytes=max(process_values) if process_values else None,
        whole_gpu_min_free_bytes=min(frees),
        whole_gpu_peak_used_bytes=max(whole_values) if whole_values else None,
        other_process_peak_used_bytes=max(other_values) if other_values else None,
        sample_count=len(valid),
        missing_sample_count=len(samples) - len(valid),
        first_sampled_at=first,
        last_sampled_at=last,
        process_identity_verified=process_verified,
        gpu_identity_verified=gpu_verified,
        coverage_seconds=max(0.0, (last - first).total_seconds()),
        measurement_version=NvidiaSmiGpuMemoryAdapter.measurement_version,
        other_process_ratio=ratio,
        confidence=confidence,
    )


def _parse_process_rows(lines: Sequence[str]) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            used = _mib_to_bytes(fields[2])
        except ValueError:
            continue
        if pid <= 0 or not fields[1] or used < 0:
            continue
        rows.append((pid, fields[1], used))
    return rows


def _mib_to_bytes(value: str) -> int:
    parsed = float(value.strip())
    if parsed < 0 or parsed > 2**31:
        raise ValueError("invalid MiB value")
    return int(parsed * 1024**2)
