from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class SelectedGpuStatus(StrEnum):
    """Allowed persisted states for the GPU selected by a training process."""

    OK = "ok"
    CHANGED = "changed"
    IDENTITY_UNVERIFIED = "identity_unverified"
    PHYSICAL_GPU_NOT_FOUND = "physical_gpu_not_found"
    AMBIGUOUS_SELECTION = "ambiguous_selection"
    PROCESS_GPU_NOT_FOUND = "process_gpu_not_found"
    IDENTITY_MISMATCH = "identity_mismatch"


def normalize_selected_gpu_status(
    value: str | SelectedGpuStatus | None,
) -> SelectedGpuStatus:
    """Convert legacy/free-form values to one of the persisted status codes."""

    if isinstance(value, SelectedGpuStatus):
        return value
    try:
        return SelectedGpuStatus(value or "")
    except ValueError:
        return SelectedGpuStatus.IDENTITY_UNVERIFIED


@dataclass(frozen=True, slots=True)
class TrainingJobEnvironmentSnapshot:
    """Immutable compute environment captured for one training job."""

    id: UUID
    training_job_id: UUID
    logical_gpu_index: int | None
    physical_gpu_index: int | None
    gpu_uuid_fingerprint: str | None
    gpu_architecture: str | None
    compute_capability: str | None
    total_vram_bytes: int | None
    cuda_available: bool
    sd_scripts_version: str | None
    xformers_available: bool | None
    cuda_visible_devices: str
    visible_gpu_uuid_fingerprints: tuple[str, ...]
    detector_version: str
    detected_at: datetime
    status: str
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingJobSelectedGpu:
    """Immutable runtime GPU identity resolved from the target process."""

    id: UUID
    training_job_id: UUID
    logical_gpu_index: int | None
    physical_gpu_index: int | None
    gpu_uuid_fingerprint: str
    gpu_architecture: str | None
    compute_capability: str | None
    total_vram_bytes: int | None
    selected_at: datetime
    selection_source: str
    status: SelectedGpuStatus
    warning_codes: tuple[str, ...] = ()
    last_observed_gpu_uuid_fingerprint: str | None = None
    gpu_change_detected_at: datetime | None = None
    gpu_change_count: int = 0
