from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifact,
    TrainingArtifactType,
    TrainingArtifactValidationStatus,
    TrainingMetricEvent,
    TrainingParseStatus,
    TrainingProgressSnapshot,
    TrainingProgressSource,
)
from runpod_lora_studio.persistence.models import (
    TrainingArtifactRecord,
    TrainingMetricPointRecord,
    TrainingProgressRecord,
)
from runpod_lora_studio.services.training_artifact import DiscoveredArtifact


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class TrainingProgressRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_progress(
        self, job_id: UUID, values: dict[str, object]
    ) -> TrainingProgressRecord:
        record = self.session.scalar(
            select(TrainingProgressRecord).where(
                TrainingProgressRecord.training_job_id == str(job_id)
            )
        )
        now = datetime.now(UTC)
        if record is None:
            record = TrainingProgressRecord(
                id=str(uuid4()),
                training_job_id=str(job_id),
                parser_version=str(values.get("parser_version", "phase6b-v1")),
                parse_status=str(
                    values.get("parse_status", TrainingParseStatus.OK.value)
                ),
                progress_source=str(
                    values.get("progress_source", TrainingProgressSource.UNKNOWN.value)
                ),
                parser_state=str(values.get("parser_state", "{}")),
                updated_at=now,
            )
            self.session.add(record)
        for key, value in values.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = now
        self.session.flush()
        return cast(TrainingProgressRecord, record)

    def get_progress(self, job_id: UUID) -> TrainingProgressSnapshot | None:
        record = self.session.scalar(
            select(TrainingProgressRecord).where(
                TrainingProgressRecord.training_job_id == str(job_id)
            )
        )
        if record is None:
            return None
        return TrainingProgressSnapshot(
            job_id,
            record.current_epoch,
            record.total_epochs,
            record.current_step,
            record.total_steps,
            record.progress_ratio,
            record.latest_loss,
            record.smoothed_loss,
            record.learning_rate,
            record.steps_per_second,
            record.samples_per_second,
            record.elapsed_seconds,
            record.estimated_remaining_seconds,
            _utc(record.latest_log_at),
            record.parser_version,
            TrainingParseStatus(record.parse_status),
            record.parse_warning,
            TrainingProgressSource(record.progress_source),
            _utc(record.updated_at) or datetime.now(UTC),
        )

    def add_metrics(
        self, job_id: UUID, events: Iterable[TrainingMetricEvent], max_points: int
    ) -> None:
        for event in events:
            if event.name not in {
                "loss",
                "learning_rate",
                "average_loss",
                "steps_per_second",
            }:
                continue
            if event.step is None:
                continue
            existing = self.session.scalar(
                select(TrainingMetricPointRecord).where(
                    TrainingMetricPointRecord.training_job_id == str(job_id),
                    TrainingMetricPointRecord.metric_name == event.name,
                    TrainingMetricPointRecord.step == event.step,
                )
            )
            if existing is None:
                self.session.add(
                    TrainingMetricPointRecord(
                        id=str(uuid4()),
                        training_job_id=str(job_id),
                        metric_name=event.name,
                        epoch=event.epoch,
                        step=event.step,
                        value=event.value,
                        logged_at=event.logged_at,
                        source=event.source,
                        created_at=datetime.now(UTC),
                    )
                )
            else:
                existing.value = event.value
                existing.epoch = event.epoch
                existing.logged_at = event.logged_at
        self._trim_metrics(job_id, max_points)

    def _trim_metrics(self, job_id: UUID, max_points: int) -> None:
        limit = max(10, max_points)
        rows = list(
            self.session.scalars(
                select(TrainingMetricPointRecord)
                .where(TrainingMetricPointRecord.training_job_id == str(job_id))
                .order_by(
                    TrainingMetricPointRecord.step.asc(),
                    TrainingMetricPointRecord.internal_id.asc(),
                )
            ).all()
        )
        if len(rows) <= limit:
            return
        # Deterministic thinning: retain the first point, last point, and every
        # evenly spaced point in between.
        keep = {0, len(rows) - 1}
        for index in range(1, limit - 1):
            keep.add(round(index * (len(rows) - 1) / (limit - 1)))
        self.session.execute(
            delete(TrainingMetricPointRecord).where(
                TrainingMetricPointRecord.training_job_id == str(job_id),
                TrainingMetricPointRecord.internal_id.not_in(
                    [rows[index].internal_id for index in keep]
                ),
            )
        )

    def list_metrics(
        self, job_id: UUID, metric_name: str = "loss", limit: int = 500
    ) -> list[tuple[int, float, int | None]]:
        rows = self.session.scalars(
            select(TrainingMetricPointRecord)
            .where(
                TrainingMetricPointRecord.training_job_id == str(job_id),
                TrainingMetricPointRecord.metric_name == metric_name,
            )
            .order_by(TrainingMetricPointRecord.step.asc())
            .limit(max(1, min(limit, 5000)))
        ).all()
        return [(int(row.step or 0), row.value, row.epoch) for row in rows]

    def upsert_artifact(
        self, job_id: UUID, artifact: DiscoveredArtifact
    ) -> TrainingArtifactRecord:
        record = self.session.scalar(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.training_job_id == str(job_id),
                TrainingArtifactRecord.relative_path == str(artifact.relative_path),
            )
        )
        now = datetime.now(UTC)
        values = {
            "artifact_type": artifact.artifact_type.value,
            "filename": artifact.filename,
            "epoch": artifact.epoch,
            "step": artifact.step,
            "file_size": artifact.file_size,
            "sha256": artifact.sha256,
            "modified_at": artifact.modified_at,
            "validation_status": artifact.validation_status.value,
            "validation_code": artifact.validation_code,
            "validation_message": artifact.validation_message,
            "metadata_json": json.dumps(
                artifact.metadata, ensure_ascii=False, sort_keys=True
            )
            if artifact.metadata
            else None,
            "last_verified_at": now,
        }
        if record is None:
            candidate = TrainingArtifactRecord(
                id=str(uuid4()),
                training_job_id=str(job_id),
                relative_path=str(artifact.relative_path),
                discovered_at=now,
                **values,
            )
            self.session.add(candidate)
            try:
                self.session.flush()
                record = candidate
            except IntegrityError:
                # A terminal refresh and an explicit rescan can overlap.  The
                # artifact key is idempotent, so recover the concurrent row and
                # apply the newest validation values without changing job state.
                self.session.rollback()
                record = self.session.scalar(
                    select(TrainingArtifactRecord).where(
                        TrainingArtifactRecord.training_job_id == str(job_id),
                        TrainingArtifactRecord.relative_path
                        == str(artifact.relative_path),
                    )
                )
                if record is None:
                    raise
                for key, value in values.items():
                    setattr(record, key, value)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return cast(TrainingArtifactRecord, record)

    def list_artifacts(self, job_id: UUID, limit: int = 500) -> list[TrainingArtifact]:
        rows = self.session.scalars(
            select(TrainingArtifactRecord)
            .where(TrainingArtifactRecord.training_job_id == str(job_id))
            .order_by(
                TrainingArtifactRecord.modified_at.desc().nullslast(),
                TrainingArtifactRecord.relative_path.asc(),
            )
            .limit(max(1, min(limit, 1000)))
        ).all()
        result: list[TrainingArtifact] = []
        for row in rows:
            metadata = json.loads(row.metadata_json) if row.metadata_json else None
            result.append(
                TrainingArtifact(
                    UUID(row.id),
                    job_id,
                    TrainingArtifactType(row.artifact_type),
                    Path(row.relative_path),
                    row.filename,
                    row.epoch,
                    row.step,
                    row.file_size,
                    row.sha256,
                    _utc(row.modified_at),
                    TrainingArtifactValidationStatus(row.validation_status),
                    row.validation_code,
                    row.validation_message,
                    _utc(row.discovered_at) or datetime.now(UTC),
                    _utc(row.last_verified_at),
                    metadata,
                )
            )
        return result
