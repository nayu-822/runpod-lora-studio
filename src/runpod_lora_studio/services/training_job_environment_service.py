from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    PhysicalGpuInfo,
)
from runpod_lora_studio.domain.training_environment_models import (
    TrainingJobEnvironmentSnapshot,
    TrainingJobSelectedGpu,
)
from runpod_lora_studio.external.compute_environment import (
    ComputeEnvironmentAdapter,
    NvidiaSmiGpuInventoryAdapter,
    PhysicalGpuInventoryAdapter,
    TorchComputeEnvironmentAdapter,
)
from runpod_lora_studio.external.training_environment import (
    SdScriptsTrainingEnvironmentAdapter,
    TrainingEnvironmentAdapter,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    TrainingJobEnvironmentSnapshotRecord,
    TrainingJobSelectedGpuRecord,
)
from runpod_lora_studio.services.gpu_memory_metrics import (
    gpu_uuid_fingerprint as _gpu_uuid_fingerprint,
)


class TrainingJobEnvironmentService:
    """Capture the execution environment independently from recommendations."""

    detector_version = "phase7b-job-environment-v1"

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: ComputeEnvironmentAdapter | None = None,
        physical_inventory_adapter: PhysicalGpuInventoryAdapter | None = None,
        training_environment_adapter: TrainingEnvironmentAdapter | None = None,
    ) -> None:
        self.session_factory = create_session_factory(settings)
        self.adapter = adapter or TorchComputeEnvironmentAdapter()
        self.physical_inventory_adapter = physical_inventory_adapter or (
            NvidiaSmiGpuInventoryAdapter()
        )
        self.training_environment_adapter = training_environment_adapter or (
            SdScriptsTrainingEnvironmentAdapter(settings)
        )

    def capture(
        self,
        training_job_id: UUID,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> TrainingJobEnvironmentSnapshot:
        existing = self.get(training_job_id)
        if existing is not None:
            return existing
        detected_at = datetime.now(UTC)
        try:
            info = self.adapter.detect()
        except Exception:
            snapshot = TrainingJobEnvironmentSnapshot(
                id=uuid4(),
                training_job_id=training_job_id,
                logical_gpu_index=None,
                physical_gpu_index=None,
                gpu_uuid_fingerprint=None,
                gpu_architecture=None,
                compute_capability=None,
                total_vram_bytes=None,
                cuda_available=False,
                sd_scripts_version=None,
                xformers_available=None,
                cuda_visible_devices="",
                visible_gpu_uuid_fingerprints=(),
                detector_version=self.detector_version,
                detected_at=detected_at,
                status="failed",
                warning_codes=("ENVIRONMENT_DETECTION_FAILED",),
            )
        else:
            try:
                physical_inventory = self.physical_inventory_adapter.detect()
            except Exception:
                physical_inventory = ()
            training_version: str | None = None
            xformers_available: bool | None = None
            training_warning = False
            try:
                training_info = self.training_environment_adapter.detect()
                training_version = training_info.sd_scripts_version
                xformers_available = training_info.xformers_available
                training_warning = bool(training_info.warnings or training_info.errors)
            except Exception:
                training_warning = True
            snapshot = _snapshot_from_info(
                training_job_id,
                info,
                training_version,
                xformers_available,
                environment if environment is not None else os.environ,
                detected_at,
                self.detector_version,
                physical_inventory=physical_inventory,
                training_warning=training_warning,
            )
        with self.session_factory() as session:
            session.add(_record_from_snapshot(snapshot))
            session.commit()
        return snapshot

    def get(self, training_job_id: UUID) -> TrainingJobEnvironmentSnapshot | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingJobEnvironmentSnapshotRecord).where(
                    TrainingJobEnvironmentSnapshotRecord.training_job_id
                    == str(training_job_id)
                )
            )
            return _snapshot_from_record(record) if record is not None else None

    def record_runtime_gpu(
        self,
        training_job_id: UUID,
        gpu_uuid_fingerprint: str | None,
        *,
        selected_at: datetime | None = None,
        selection_source: str = "target_process",
    ) -> TrainingJobSelectedGpu | None:
        """Persist the unique GPU observed for the target process."""

        if not gpu_uuid_fingerprint:
            return None
        selected_at = selected_at or datetime.now(UTC)
        snapshot = self.get(training_job_id)
        try:
            info = self.adapter.detect()
        except Exception:
            info = ComputeEnvironmentInfo()
        try:
            inventory = self.physical_inventory_adapter.detect()
        except Exception:
            inventory = ()
        physical = next(
            (
                item
                for item in inventory
                if _gpu_uuid_fingerprint(item.uuid) == gpu_uuid_fingerprint
            ),
            None,
        )
        logical = next(
            (
                item.index
                for item in info.gpu_devices
                if item.uuid
                and _gpu_uuid_fingerprint(item.uuid) == gpu_uuid_fingerprint
            ),
            None,
        )
        if (
            logical is None
            and snapshot is not None
            and snapshot.gpu_uuid_fingerprint == gpu_uuid_fingerprint
        ):
            logical = snapshot.logical_gpu_index
        values = TrainingJobSelectedGpu(
            id=uuid4(),
            training_job_id=training_job_id,
            logical_gpu_index=logical,
            physical_gpu_index=physical.index if physical else None,
            gpu_uuid_fingerprint=gpu_uuid_fingerprint,
            gpu_architecture=physical.architecture if physical else None,
            compute_capability=physical.compute_capability if physical else None,
            total_vram_bytes=physical.total_vram_bytes if physical else None,
            selected_at=selected_at,
            selection_source=selection_source,
            status="ok" if physical else "warning",
            warning_codes=() if physical else ("PHYSICAL_GPU_NOT_FOUND",),
        )
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingJobSelectedGpuRecord).where(
                    TrainingJobSelectedGpuRecord.training_job_id == str(training_job_id)
                )
            )
            if record is None:
                session.add(_record_from_selected_gpu(values))
                session.commit()
                return values
            if record.gpu_uuid_fingerprint != values.gpu_uuid_fingerprint:
                warnings = _json_values(record.warning_codes_json)
                record.logical_gpu_index = values.logical_gpu_index
                record.physical_gpu_index = values.physical_gpu_index
                record.gpu_uuid_fingerprint = values.gpu_uuid_fingerprint
                record.gpu_architecture = values.gpu_architecture
                record.compute_capability = values.compute_capability
                record.total_vram_bytes = values.total_vram_bytes
                record.selected_at = values.selected_at
                record.selection_source = values.selection_source
                record.status = "changed"
                record.warning_codes_json = json.dumps(
                    sorted(set(warnings) | {"GPU_CHANGED_DURING_JOB"})
                )
                session.commit()
            return _selected_gpu_from_record(record)

    def get_runtime_gpu(self, training_job_id: UUID) -> TrainingJobSelectedGpu | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(TrainingJobSelectedGpuRecord).where(
                    TrainingJobSelectedGpuRecord.training_job_id == str(training_job_id)
                )
            )
            return _selected_gpu_from_record(record) if record is not None else None


@dataclass(frozen=True, slots=True)
class VisibleGpuMapping:
    """Mapping between Torch logical devices and host physical GPUs."""

    logical_index: int
    visible_token: str | None
    physical_index: int | None
    gpu_uuid: str | None
    gpu_uuid_fingerprint: str | None
    architecture: str | None
    compute_capability: str | None
    total_vram_bytes: int | None
    identity_verified: bool = False
    warning_codes: tuple[str, ...] = ()


def _snapshot_from_info(
    training_job_id: UUID,
    info: ComputeEnvironmentInfo,
    sd_scripts_version: str | None,
    xformers_available: bool | None,
    environment: Mapping[str, str],
    detected_at: datetime,
    detector_version: str,
    *,
    physical_inventory: Sequence[PhysicalGpuInfo] = (),
    training_warning: bool = False,
) -> TrainingJobEnvironmentSnapshot:
    visible_value = _normalize_visible_devices(environment.get("CUDA_VISIBLE_DEVICES"))
    inventory = tuple(physical_inventory)
    devices = _map_devices(info, visible_value, inventory)
    selected = (
        devices[0] if len(devices) == 1 and devices[0].identity_verified else None
    )
    warnings = list(info.warnings)
    warning_codes: list[str] = []
    warning_codes.extend(code for device in devices for code in device.warning_codes)
    if not inventory and info.gpu_devices:
        warning_codes.append("PHYSICAL_GPU_INVENTORY_UNAVAILABLE")
    if not devices or any(device.gpu_uuid_fingerprint is None for device in devices):
        warning_codes.append("GPU_IDENTITY_UNAVAILABLE")
    elif selected is None:
        warning_codes.append("AMBIGUOUS_GPU_SELECTION")
    if info.errors:
        warning_codes.append("ENVIRONMENT_DETECTION_ERROR")
    return TrainingJobEnvironmentSnapshot(
        id=uuid4(),
        training_job_id=training_job_id,
        logical_gpu_index=selected.logical_index if selected else None,
        physical_gpu_index=selected.physical_index if selected else None,
        gpu_uuid_fingerprint=selected.gpu_uuid_fingerprint if selected else None,
        gpu_architecture=selected.architecture if selected else None,
        compute_capability=selected.compute_capability if selected else None,
        total_vram_bytes=selected.total_vram_bytes if selected else None,
        cuda_available=info.cuda_available,
        sd_scripts_version=sd_scripts_version,
        xformers_available=xformers_available,
        cuda_visible_devices=visible_value,
        visible_gpu_uuid_fingerprints=tuple(
            device.gpu_uuid_fingerprint
            for device in devices
            if device.gpu_uuid_fingerprint is not None
        ),
        detector_version=detector_version,
        detected_at=detected_at,
        status="warning" if warnings or warning_codes or training_warning else "ok",
        warning_codes=tuple(sorted(set(warning_codes))),
    )


def _map_devices(
    info: ComputeEnvironmentInfo,
    visible_value: str,
    physical_inventory: Sequence[PhysicalGpuInfo] = (),
) -> list[VisibleGpuMapping]:
    tokens = (
        [token.strip().lower() for token in visible_value.split(",")]
        if visible_value
        else []
    )
    by_physical_index = {item.index: item for item in physical_inventory}
    by_uuid = {_normalize_uuid(item.uuid): item for item in physical_inventory}
    logical_devices = {device.index: device for device in info.gpu_devices}
    seen_tokens: set[str] = set()
    mapped: list[VisibleGpuMapping] = []
    logical_count = len(tokens) if tokens else len(info.gpu_devices)
    for logical_index in range(logical_count):
        token = tokens[logical_index] if tokens else None
        device = logical_devices.get(logical_index)
        physical: PhysicalGpuInfo | None = None
        codes: list[str] = []
        if token is not None:
            if token in seen_tokens:
                codes.append("DUPLICATE_VISIBLE_GPU_TOKEN")
            seen_tokens.add(token)
            if not token:
                codes.append("EMPTY_VISIBLE_GPU_TOKEN")
            elif token.isdigit():
                physical = by_physical_index.get(int(token))
                if physical is None:
                    codes.append("PHYSICAL_GPU_NOT_FOUND")
            elif _is_uuid_token(token):
                matches = [
                    item
                    for normalized, item in by_uuid.items()
                    if normalized.startswith(token)
                ]
                if len(matches) == 1:
                    physical = matches[0]
                elif len(matches) > 1:
                    codes.append("AMBIGUOUS_GPU_UUID_PREFIX")
                else:
                    codes.append("GPU_UUID_NOT_FOUND")
            else:
                codes.append("INVALID_VISIBLE_GPU_TOKEN")
        elif device is not None and physical_inventory:
            physical = next(
                (
                    item
                    for item in physical_inventory
                    if device.uuid
                    and _normalize_uuid(item.uuid) == _normalize_uuid(device.uuid)
                ),
                None,
            )

        torch_uuid = _normalize_uuid(device.uuid) if device and device.uuid else None
        physical_uuid = _normalize_uuid(physical.uuid) if physical else None
        if physical is not None and torch_uuid != physical_uuid:
            codes.append("GPU_UUID_MISMATCH")
        identity_verified = bool(
            torch_uuid and (physical is None or torch_uuid == physical_uuid)
        )
        if physical_inventory and physical is None:
            codes.append("PHYSICAL_GPU_MAPPING_UNVERIFIED")
            identity_verified = False
        uuid = (
            physical.uuid
            if physical and identity_verified
            else device.uuid
            if identity_verified and device
            else None
        )
        mapped.append(
            VisibleGpuMapping(
                logical_index=logical_index,
                visible_token=token,
                physical_index=physical.index
                if physical and identity_verified
                else None,
                gpu_uuid=uuid,
                gpu_uuid_fingerprint=(_gpu_uuid_fingerprint(uuid) if uuid else None),
                architecture=(
                    physical.architecture
                    if physical
                    else device.architecture
                    if device
                    else None
                ),
                compute_capability=(
                    physical.compute_capability
                    if physical
                    else device.compute_capability
                    if device
                    else None
                ),
                total_vram_bytes=(
                    physical.total_vram_bytes
                    if physical
                    else device.total_vram_bytes
                    if device
                    else None
                ),
                identity_verified=identity_verified,
                warning_codes=tuple(sorted(set(codes))),
            )
        )
    return mapped


def _normalize_visible_devices(value: str | None) -> str:
    if not value:
        return ""
    return ",".join(token.strip().lower() for token in value.split(","))


_GPU_UUID_RE = re.compile(r"^gpu-[0-9a-f-]+$")


def _normalize_uuid(value: str) -> str:
    return value.strip().lower()


def _is_uuid_token(value: str) -> bool:
    return bool(_GPU_UUID_RE.fullmatch(value))


def _record_from_snapshot(
    snapshot: TrainingJobEnvironmentSnapshot,
) -> TrainingJobEnvironmentSnapshotRecord:
    return TrainingJobEnvironmentSnapshotRecord(
        id=str(snapshot.id),
        training_job_id=str(snapshot.training_job_id),
        logical_gpu_index=snapshot.logical_gpu_index,
        physical_gpu_index=snapshot.physical_gpu_index,
        gpu_uuid_fingerprint=snapshot.gpu_uuid_fingerprint,
        gpu_architecture=snapshot.gpu_architecture,
        compute_capability=snapshot.compute_capability,
        total_vram_bytes=snapshot.total_vram_bytes,
        cuda_available=snapshot.cuda_available,
        sd_scripts_version=snapshot.sd_scripts_version,
        xformers_available=snapshot.xformers_available,
        cuda_visible_devices=snapshot.cuda_visible_devices,
        visible_gpu_uuids_json=json.dumps(
            snapshot.visible_gpu_uuid_fingerprints, sort_keys=True
        ),
        detector_version=snapshot.detector_version,
        detected_at=snapshot.detected_at,
        status=snapshot.status,
        warning_codes_json=json.dumps(snapshot.warning_codes, sort_keys=True),
    )


def _snapshot_from_record(
    record: TrainingJobEnvironmentSnapshotRecord,
) -> TrainingJobEnvironmentSnapshot:
    try:
        visible = tuple(json.loads(record.visible_gpu_uuids_json or "[]"))
    except (TypeError, ValueError):
        visible = ()
    try:
        warnings = tuple(json.loads(record.warning_codes_json or "[]"))
    except (TypeError, ValueError):
        warnings = ()
    return TrainingJobEnvironmentSnapshot(
        id=UUID(record.id),
        training_job_id=UUID(record.training_job_id),
        logical_gpu_index=record.logical_gpu_index,
        physical_gpu_index=record.physical_gpu_index,
        gpu_uuid_fingerprint=record.gpu_uuid_fingerprint,
        gpu_architecture=record.gpu_architecture,
        compute_capability=record.compute_capability,
        total_vram_bytes=record.total_vram_bytes,
        cuda_available=bool(record.cuda_available),
        sd_scripts_version=record.sd_scripts_version,
        xformers_available=(
            bool(record.xformers_available)
            if record.xformers_available is not None
            else None
        ),
        cuda_visible_devices=record.cuda_visible_devices,
        visible_gpu_uuid_fingerprints=tuple(str(value) for value in visible),
        detector_version=record.detector_version,
        detected_at=record.detected_at,
        status=record.status,
        warning_codes=tuple(str(value) for value in warnings),
    )


def _record_from_selected_gpu(
    selected: TrainingJobSelectedGpu,
) -> TrainingJobSelectedGpuRecord:
    return TrainingJobSelectedGpuRecord(
        id=str(selected.id),
        training_job_id=str(selected.training_job_id),
        logical_gpu_index=selected.logical_gpu_index,
        physical_gpu_index=selected.physical_gpu_index,
        gpu_uuid_fingerprint=selected.gpu_uuid_fingerprint,
        gpu_architecture=selected.gpu_architecture,
        compute_capability=selected.compute_capability,
        total_vram_bytes=selected.total_vram_bytes,
        selected_at=selected.selected_at,
        selection_source=selected.selection_source,
        status=selected.status,
        warning_codes_json=json.dumps(selected.warning_codes, sort_keys=True),
    )


def _selected_gpu_from_record(
    record: TrainingJobSelectedGpuRecord,
) -> TrainingJobSelectedGpu:
    return TrainingJobSelectedGpu(
        id=UUID(record.id),
        training_job_id=UUID(record.training_job_id),
        logical_gpu_index=record.logical_gpu_index,
        physical_gpu_index=record.physical_gpu_index,
        gpu_uuid_fingerprint=record.gpu_uuid_fingerprint,
        gpu_architecture=record.gpu_architecture,
        compute_capability=record.compute_capability,
        total_vram_bytes=record.total_vram_bytes,
        selected_at=record.selected_at,
        selection_source=record.selection_source,
        status=record.status,
        warning_codes=_json_values(record.warning_codes_json),
    )


def _json_values(value: str | None) -> tuple[str, ...]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return ()
    return (
        tuple(sorted(str(item) for item in parsed if item))
        if isinstance(parsed, list)
        else ()
    )
