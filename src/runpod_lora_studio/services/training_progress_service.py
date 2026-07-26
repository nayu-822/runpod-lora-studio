from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.training_progress_models import (
    ParsedTrainingProgress,
    TrainingArtifact,
    TrainingLogParserState,
    TrainingParseStatus,
    TrainingProgressSnapshot,
    TrainingProgressSource,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    TrainingConfigRecord,
    TrainingJobRecord,
    TrainingProgressRecord,
)
from runpod_lora_studio.persistence.training_progress_repository import (
    TrainingProgressRepository,
)
from runpod_lora_studio.services.training_artifact import TrainingArtifactScanner
from runpod_lora_studio.services.training_log_parser import (
    TrainingLogParser,
    TrainingStepEstimator,
)
from runpod_lora_studio.services.training_log_reader import (
    IncrementalLogReader,
    TrainingLogCursor,
)

logger = logging.getLogger("runpod_lora_studio.training_progress")


class TrainingProgressService:
    """Persist progress independently from the process worker's memory state."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)
        self.parser = TrainingLogParser()

    def refresh_job(self, job_id: UUID) -> None:
        try:
            self._refresh_job(job_id)
        except Exception:
            # A parser or artifact failure must never terminate the trainer.
            logger.exception("training_progress_refresh_failed job_id=%s", job_id)
            self._store_warning(job_id, "progress refresh failed")

    def get_progress(self, job_id: UUID) -> TrainingProgressSnapshot | None:
        with self.session_factory() as session:
            return TrainingProgressRepository(session).get_progress(job_id)

    def list_metrics(
        self, job_id: UUID, metric_name: str = "loss", limit: int = 500
    ) -> list[tuple[int, float, int | None]]:
        with self.session_factory() as session:
            return TrainingProgressRepository(session).list_metrics(
                job_id, metric_name, limit
            )

    def list_artifacts(self, job_id: UUID, limit: int = 500) -> list[TrainingArtifact]:
        with self.session_factory() as session:
            return TrainingProgressRepository(session).list_artifacts(job_id, limit)

    def rescan_artifacts(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            config = (
                session.scalar(
                    select(TrainingConfigRecord).where(
                        TrainingConfigRecord.id == job.training_config_id
                    )
                )
                if job
                else None
            )
            if job is None or config is None:
                return
            output_root, warning = _job_output_directory(job, self.settings)
            if output_root is None:
                if warning:
                    logger.warning(
                        "artifact_scan_skipped job_id=%s: %s", job_id, warning
                    )
                return
            scanner = TrainingArtifactScanner(
                output_root,
                max_depth=self.settings.training_artifact_max_depth,
                max_count=self.settings.training_artifact_max_count,
                max_file_size=self.settings.training_artifact_max_file_size_bytes,
            )
            repository = TrainingProgressRepository(session)
            for artifact in scanner.scan(config.output_name):
                repository.upsert_artifact(job_id, artifact)
            session.commit()

    def _refresh_job(self, job_id: UUID) -> None:
        with self.session_factory() as session:
            job = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.id == str(job_id))
            )
            if job is None:
                return
            config = session.scalar(
                select(TrainingConfigRecord).where(
                    TrainingConfigRecord.id == job.training_config_id
                )
            )
            if config is None or job.runtime_directory is None:
                return
            runtime = Path(job.runtime_directory).resolve()
            jobs_root = (
                self.settings.training_jobs_dir
                or self.settings.workspace_root / "training" / "jobs"
            ).resolve()
            if not _is_under(runtime, jobs_root):
                self._store_warning(
                    job_id, "job runtime directory is outside the allowed root"
                )
                return
            logs_root = runtime / "logs"
            stdout_path = (
                Path(job.stdout_log_path)
                if job.stdout_log_path
                else logs_root / "stdout.log"
            )
            stderr_path = (
                Path(job.stderr_log_path)
                if job.stderr_log_path
                else logs_root / "stderr.log"
            )
            repository = TrainingProgressRepository(session)
            previous = session.scalar(
                select(TrainingProgressRecord).where(
                    TrainingProgressRecord.training_job_id == str(job_id)
                )
            )
            previous_values = (
                json.loads(previous.parser_state)
                if previous and previous.parser_state
                else {}
            )
            stdout_cursor = _cursor_from_dict(previous_values.get("stdout_cursor"))
            stderr_cursor = _cursor_from_dict(previous_values.get("stderr_cursor"))
            stdout = IncrementalLogReader(
                logs_root, self.settings.training_progress_read_bytes
            ).read(stdout_path, stdout_cursor)
            stderr = IncrementalLogReader(
                logs_root, self.settings.training_progress_read_bytes
            ).read(stderr_path, stderr_cursor)
            aggregate_state = _state_from_dict(
                previous_values.get("aggregate", previous_values.get("parser"))
            )
            stdout_state = _state_from_dict(previous_values.get("stdout_parser"))
            stderr_state = _state_from_dict(previous_values.get("stderr_parser"))
            stdout_remainder = "" if stdout.reset else stdout_state.remainder
            stderr_remainder = "" if stderr.reset else stderr_state.remainder
            stdout_result = self.parser.parse_stream(
                stdout.data,
                replace(aggregate_state, remainder=stdout_remainder),
                total_epochs=config.epochs,
                estimated_total_steps=_estimated_total_steps(runtime, config),
                started_at=job.started_at,
                now=datetime.now(UTC),
                source="stdout",
            )
            stderr_result = self.parser.parse_stream(
                stderr.data,
                replace(aggregate_state, remainder=stderr_remainder),
                total_epochs=config.epochs,
                estimated_total_steps=_estimated_total_steps(runtime, config),
                started_at=job.started_at,
                now=datetime.now(UTC),
                source="stderr",
            )
            result = self.parser.merge(
                aggregate_state,
                stdout_result,
                stderr_result,
                total_epochs=config.epochs,
                estimated_total_steps=_estimated_total_steps(runtime, config),
                started_at=job.started_at,
                now=datetime.now(UTC),
            )
            progress = result.progress
            warning_values = list(progress.warnings)
            current_epoch = progress.current_epoch
            total_epochs = progress.total_epochs
            current_step = progress.current_step
            total_steps = progress.total_steps
            progress_ratio = progress.progress_ratio
            latest_loss = progress.latest_loss
            if (
                previous is not None
                and previous.current_step is not None
                and current_step is not None
                and current_step < previous.current_step
            ):
                warning_values.append(
                    "progress moved backwards; previous persisted value retained"
                )
                current_epoch = previous.current_epoch
                total_epochs = previous.total_epochs
                current_step = previous.current_step
                total_steps = previous.total_steps
                progress_ratio = previous.progress_ratio
                latest_loss = previous.latest_loss
            old_loss = previous.smoothed_loss if previous else None
            smoothed = (
                latest_loss
                if old_loss is None
                else (
                    old_loss * 0.8 + latest_loss * 0.2
                    if latest_loss is not None
                    else old_loss
                )
            )
            for stream_name, warning in (
                ("stdout", stdout.warning),
                ("stderr", stderr.warning),
            ):
                if warning:
                    warning_values.append(f"{stream_name}: {warning}")
            if stdout.reset or stderr.reset:
                warning_values.append("log offset reset after truncate or rotation")
            output_root, output_warning = _job_output_directory(job, self.settings)
            if output_root is None:
                warning_values.append(
                    output_warning or "job output directory is unavailable"
                )
            smoothed_speed = _smoothed_speed(progress.speed, previous)
            state = {
                "aggregate": _state_to_dict(result.state),
                "stdout_parser": _state_to_dict(stdout_result.state),
                "stderr_parser": _state_to_dict(stderr_result.state),
                "stdout_cursor": _cursor_to_dict(stdout.cursor),
                "stderr_cursor": _cursor_to_dict(stderr.cursor),
            }
            latest_log_at = (
                datetime.now(UTC)
                if stdout.data or stderr.data
                else (previous.latest_log_at if previous else None)
            )
            parse_status = (
                TrainingParseStatus.WARNING.value
                if warning_values
                else TrainingParseStatus.OK.value
            )
            values = {
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress_ratio": progress_ratio,
                "latest_loss": latest_loss,
                "smoothed_loss": smoothed,
                "learning_rate": progress.learning_rate,
                "steps_per_second": smoothed_speed,
                "samples_per_second": smoothed_speed,
                "elapsed_seconds": progress.elapsed_seconds,
                "estimated_remaining_seconds": _eta(
                    progress, previous, job.status, smoothed_speed
                ),
                "latest_log_at": latest_log_at,
                "stdout_offset": stdout.cursor.offset,
                "stderr_offset": stderr.cursor.offset,
                "parser_version": self.parser.parser_version,
                "parse_status": parse_status,
                "parse_warning": "; ".join(dict.fromkeys(warning_values))[-4000:]
                or None,
                "progress_source": progress.progress_source.value,
                "parser_state": json.dumps(state, ensure_ascii=False, sort_keys=True),
            }
            repository.upsert_progress(job_id, values)
            repository.add_metrics(
                job_id, progress.metric_events, self.settings.training_metric_max_points
            )
            if output_root is not None:
                scanner = TrainingArtifactScanner(
                    output_root,
                    max_depth=self.settings.training_artifact_max_depth,
                    max_count=self.settings.training_artifact_max_count,
                    max_file_size=self.settings.training_artifact_max_file_size_bytes,
                )
                for artifact in scanner.scan(config.output_name):
                    repository.upsert_artifact(job_id, artifact)
            session.commit()

    def _store_warning(self, job_id: UUID, warning: str) -> None:
        try:
            with self.session_factory() as session:
                record = session.scalar(
                    select(TrainingProgressRecord).where(
                        TrainingProgressRecord.training_job_id == str(job_id)
                    )
                )
                if record is not None:
                    record.parse_status = TrainingParseStatus.WARNING.value
                    record.parse_warning = warning
                    record.updated_at = datetime.now(UTC)
                    session.commit()
        except Exception:
            logger.exception(
                "training_progress_warning_persist_failed job_id=%s", job_id
            )


def _eta(
    progress: ParsedTrainingProgress,
    previous: TrainingProgressRecord | None,
    status: str,
    smoothed_speed: float | None,
) -> float | None:
    if (
        status in {"canceled", "failed", "succeeded"}
        or progress.current_step is None
        or not progress.total_steps
    ):
        return None
    if progress.estimated_remaining_seconds is not None:
        return min(max(progress.estimated_remaining_seconds, 0.0), 30 * 24 * 3600)
    if (
        previous is None
        or previous.current_step is None
        or previous.updated_at >= datetime.now(UTC)
    ):
        return None
    elapsed = max((datetime.now(UTC) - previous.updated_at).total_seconds(), 0.001)
    delta = progress.current_step - int(previous.current_step)
    if delta <= 0:
        return (
            float(previous.estimated_remaining_seconds)
            if previous.estimated_remaining_seconds is not None
            else None
        )
    speed = smoothed_speed or delta / elapsed
    remaining = max(
        (int(progress.total_steps) - int(progress.current_step)) / speed,
        0.0,
    )
    return float(min(remaining, 30 * 24 * 3600))


def _smoothed_speed(
    current: float | None, previous: TrainingProgressRecord | None
) -> float | None:
    if current is None:
        return (
            float(previous.steps_per_second)
            if previous is not None and previous.steps_per_second is not None
            else None
        )
    if previous is None or previous.steps_per_second is None:
        return current
    return float(previous.steps_per_second) * 0.8 + current * 0.2


def _estimated_total_steps(runtime: Path, config: TrainingConfigRecord) -> int | None:
    metadata_path = runtime / "runtime" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        repeats = tuple(int(value) for value in metadata.get("dataset_num_repeats", []))
        counts = tuple(int(value) for value in metadata.get("dataset_image_counts", []))
        if not counts:
            return None
        plan = TrainingStepEstimator.estimate(
            subset_image_counts=tuple(counts),
            num_repeats=repeats,
            batch_size=config.batch_size,
            epochs=config.epochs,
        )
        return plan.total_steps
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return None


def _cursor_to_dict(cursor: TrainingLogCursor) -> dict[str, object]:
    return {
        "offset": cursor.offset,
        "file_key": list(cursor.file_key) if cursor.file_key else None,
        "pending": cursor.pending.hex(),
    }


def _cursor_from_dict(value: object) -> TrainingLogCursor:
    if not isinstance(value, dict):
        return TrainingLogCursor()
    key = value.get("file_key")
    return TrainingLogCursor(
        int(value.get("offset", 0)),
        tuple(key) if isinstance(key, list) and len(key) == 2 else None,
        bytes.fromhex(str(value.get("pending", ""))),
    )


def _state_to_dict(state: TrainingLogParserState) -> dict[str, object]:
    return {
        "remainder": state.remainder,
        "current_epoch": state.current_epoch,
        "total_epochs": state.total_epochs,
        "current_step": state.current_step,
        "total_steps": state.total_steps,
        "latest_loss": state.latest_loss,
        "learning_rate": state.learning_rate,
        "speed": state.speed,
        "elapsed_seconds": state.elapsed_seconds,
        "remaining_seconds": state.remaining_seconds,
        "total_steps_source": state.total_steps_source.value,
        "warnings": list(state.warnings),
    }


def _state_from_dict(value: object) -> TrainingLogParserState:
    if not isinstance(value, dict):
        return TrainingLogParserState()
    return TrainingLogParserState(
        remainder=str(value.get("remainder", "")),
        current_epoch=_int_or_none(value.get("current_epoch")),
        total_epochs=_int_or_none(value.get("total_epochs")),
        current_step=_int_or_none(value.get("current_step")),
        total_steps=_int_or_none(value.get("total_steps")),
        latest_loss=_float_or_none(value.get("latest_loss")),
        learning_rate=_float_or_none(value.get("learning_rate")),
        speed=_float_or_none(value.get("speed")),
        elapsed_seconds=_float_or_none(value.get("elapsed_seconds")),
        remaining_seconds=_float_or_none(value.get("remaining_seconds")),
        total_steps_source=_progress_source(value.get("total_steps_source")),
        warnings=tuple(value.get("warnings", ())),
    )


def _int_or_none(value: object) -> int | None:
    return (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _float_or_none(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _progress_source(value: object) -> TrainingProgressSource:
    try:
        return TrainingProgressSource(str(value))
    except ValueError:
        return TrainingProgressSource.UNKNOWN


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _job_output_directory(
    job: TrainingJobRecord, settings: AppSettings
) -> tuple[Path | None, str | None]:
    """Return only the output owned by this job, never the shared config root."""
    if not job.runtime_directory:
        return None, "legacy job format has no dedicated runtime directory"
    raw_runtime = Path(job.runtime_directory)
    if raw_runtime.is_symlink():
        return None, "job runtime directory is a symlink"
    jobs_root = (
        settings.training_jobs_dir or settings.workspace_root / "training" / "jobs"
    ).resolve()
    try:
        runtime = raw_runtime.resolve(strict=True)
    except OSError:
        return None, "job runtime directory is unavailable"
    if not _is_under(runtime, jobs_root):
        return None, "job runtime directory is outside the allowed root"
    raw_output = runtime / "output"
    if raw_output.is_symlink():
        return None, "job output directory is a symlink"
    try:
        output = raw_output.resolve(strict=True)
    except OSError:
        return None, "legacy job format has no dedicated output directory"
    if not output.is_dir():
        return None, "legacy job format has no dedicated output directory"
    if not _is_under(output, runtime) or not _is_under(output, jobs_root):
        return None, "job output directory is outside the allowed root"
    return output, None
