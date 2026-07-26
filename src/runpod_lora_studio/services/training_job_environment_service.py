from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import ComputeEnvironmentInfo
from runpod_lora_studio.domain.training_environment_models import (
    TrainingJobEnvironmentSnapshot,
)
from runpod_lora_studio.external.compute_environment import (
    ComputeEnvironmentAdapter,
    TorchComputeEnvironmentAdapter,
)
from runpod_lora_studio.external.training_environment import (
    SdScriptsTrainingEnvironmentAdapter,
    TrainingEnvironmentAdapter,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import TrainingJobEnvironmentSnapshotRecord
from runpod_lora_studio.services.gpu_memory_metrics import gpu_uuid_fingerprint


class TrainingJobEnvironmentService:
    """Capture the execution environment independently from recommendations."""

    detector_version = "phase7b-job-environment-v1"

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: ComputeEnvironmentAdapter | None = None,
        training_environment_adapter: TrainingEnvironmentAdapter | None = None,
    ) -> None:
        self.session_factory = create_session_factory(settings)
        self.adapter = adapter or TorchComputeEnvironmentAdapter()
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


def _snapshot_from_info(
    training_job_id: UUID,
    info: ComputeEnvironmentInfo,
    sd_scripts_version: str | None,
    xformers_available: bool | None,
    environment: Mapping[str, str],
    detected_at: datetime,
    detector_version: str,
    *,
    training_warning: bool = False,
) -> TrainingJobEnvironmentSnapshot:
    visible_value = _normalize_visible_devices(environment.get("CUDA_VISIBLE_DEVICES"))
    devices = _map_devices(info, visible_value)
    selected = devices[0] if len(devices) == 1 else None
    warnings = list(info.warnings)
    warning_codes: list[str] = []
    if not devices or any(device["uuid_fingerprint"] is None for device in devices):
        warning_codes.append("GPU_IDENTITY_UNAVAILABLE")
    elif selected is None:
        warning_codes.append("AMBIGUOUS_GPU_SELECTION")
    if info.errors:
        warning_codes.append("ENVIRONMENT_DETECTION_ERROR")
    return TrainingJobEnvironmentSnapshot(
        id=uuid4(),
        training_job_id=training_job_id,
        logical_gpu_index=selected["logical_index"] if selected else None,
        physical_gpu_index=selected["physical_index"] if selected else None,
        gpu_uuid_fingerprint=selected["uuid_fingerprint"] if selected else None,
        gpu_architecture=selected["architecture"] if selected else None,
        compute_capability=selected["compute_capability"] if selected else None,
        total_vram_bytes=selected["total_vram_bytes"] if selected else None,
        cuda_available=info.cuda_available,
        sd_scripts_version=sd_scripts_version,
        xformers_available=xformers_available,
        cuda_visible_devices=visible_value,
        visible_gpu_uuid_fingerprints=tuple(
            device["uuid_fingerprint"]
            for device in devices
            if device["uuid_fingerprint"] is not None
        ),
        detector_version=detector_version,
        detected_at=detected_at,
        status="warning" if warnings or warning_codes or training_warning else "ok",
        warning_codes=tuple(sorted(set(warning_codes))),
    )


def _map_devices(
    info: ComputeEnvironmentInfo, visible_value: str
) -> list[dict[str, Any]]:
    tokens = visible_value.split(",") if visible_value else []
    mapped: list[dict[str, Any]] = []
    selected_devices = (
        info.gpu_devices
        if not tokens
        else tuple(
            next(
                (
                    candidate
                    for candidate in info.gpu_devices
                    if (token.isdigit() and candidate.index == int(token))
                    or (
                        not token.isdigit()
                        and candidate.uuid
                        and candidate.uuid.lower() == token.lower()
                    )
                ),
                None,
            )
            for token in tokens
        )
    )
    for logical_index, device in enumerate(selected_devices):
        token = tokens[logical_index] if tokens else None
        physical_index: int | None = device.index if device else None
        uuid = device.uuid if device else None
        architecture = device.architecture if device else None
        compute_capability = device.compute_capability if device else None
        total_vram_bytes = device.total_vram_bytes if device else None
        if token and token.isdigit():
            physical_index = int(token)
        elif token and not device:
            uuid = token
        mapped.append(
            {
                "logical_index": logical_index,
                "physical_index": physical_index,
                "uuid_fingerprint": gpu_uuid_fingerprint(uuid) if uuid else None,
                "architecture": architecture,
                "compute_capability": compute_capability,
                "total_vram_bytes": total_vram_bytes,
            }
        )
    return mapped


def _normalize_visible_devices(value: str | None) -> str:
    if not value:
        return ""
    return ",".join(
        token.strip().lower() for token in value.split(",") if token.strip()
    )


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
