from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import UUID, uuid4

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    DiagnosticStatus,
    TrainingEnvironmentInfo,
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
from runpod_lora_studio.persistence.models import (
    ComputeEnvironmentSnapshotRecord,
    TrainingEnvironmentSnapshotRecord,
)


class ComputeEnvironmentService:
    detector_version = "phase7a-compute-v1"

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: ComputeEnvironmentAdapter | None = None,
    ) -> None:
        self.session_factory = create_session_factory(settings)
        self.adapter = adapter or TorchComputeEnvironmentAdapter()

    def detect(self) -> ComputeEnvironmentInfo:
        return self.adapter.detect()

    def snapshot(self, project_id: UUID | None = None) -> UUID:
        info = self.detect()
        snapshot_id = uuid4()
        now = datetime.now(UTC)
        with self.session_factory() as session:
            session.add(
                ComputeEnvironmentSnapshotRecord(
                    id=str(snapshot_id),
                    project_id=str(project_id) if project_id else None,
                    status=_status(info.warnings, info.errors).value,
                    payload_json=json.dumps(asdict(info), default=str, sort_keys=True),
                    warning_json=json.dumps(info.warnings, ensure_ascii=False),
                    error_json=json.dumps(info.errors, ensure_ascii=False),
                    detector_version=self.detector_version,
                    detected_at=now,
                    created_at=now,
                )
            )
            session.commit()
        return snapshot_id


class TrainingEnvironmentService:
    detector_version = "phase7a-training-environment-v1"

    def __init__(
        self,
        settings: AppSettings,
        *,
        adapter: TrainingEnvironmentAdapter | None = None,
    ) -> None:
        self.session_factory = create_session_factory(settings)
        self.adapter = adapter or SdScriptsTrainingEnvironmentAdapter(settings)

    def detect(self) -> TrainingEnvironmentInfo:
        return self.adapter.detect()

    def snapshot(
        self, project_id: UUID | None = None, *, compute_snapshot_id: UUID | None = None
    ) -> UUID:
        info = self.detect()
        snapshot_id = uuid4()
        now = datetime.now(UTC)
        with self.session_factory() as session:
            session.add(
                TrainingEnvironmentSnapshotRecord(
                    id=str(snapshot_id),
                    project_id=str(project_id) if project_id else None,
                    compute_snapshot_id=(
                        str(compute_snapshot_id) if compute_snapshot_id else None
                    ),
                    status=_status(info.warnings, info.errors).value,
                    payload_json=json.dumps(asdict(info), default=str, sort_keys=True),
                    warning_json=json.dumps(info.warnings, ensure_ascii=False),
                    error_json=json.dumps(info.errors, ensure_ascii=False),
                    detector_version=self.detector_version,
                    detected_at=now,
                    created_at=now,
                )
            )
            session.commit()
        return snapshot_id


def _status(warnings: tuple[str, ...], errors: tuple[str, ...]) -> DiagnosticStatus:
    if errors:
        return DiagnosticStatus.ERROR
    if warnings:
        return DiagnosticStatus.WARNING
    return DiagnosticStatus.OK
