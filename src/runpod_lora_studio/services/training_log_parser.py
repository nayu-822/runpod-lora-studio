from __future__ import annotations

import math
import re
import time
from datetime import UTC, datetime

from runpod_lora_studio.domain.training_progress_models import (
    EstimatedTrainingPlan,
    ParsedTrainingProgress,
    TrainingLogParseResult,
    TrainingLogParserState,
    TrainingMetricEvent,
    TrainingProgressSource,
)

ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
EPOCH_RE = re.compile(r"\b(?:epoch|epochs?)\s*[:=]?\s*(\d+)\s*/\s*(\d+)\b", re.I)
STEP_RE = re.compile(r"\b(?:steps?|step)\s*[:=]?\s*(\d+)\s*/\s*(\d+)\b", re.I)
PAIR_RE = re.compile(r"(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])")
LOSS_RE = re.compile(r"\bloss\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
LR_RE = re.compile(
    r"\b(?:lr|learning[_ ]rate)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I
)
SPEED_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:it|step|steps|sample|samples)?/s\b", re.I
)
TIME_RE = re.compile(r"\b(?:elapsed|time)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b", re.I)
REMAINING_RE = re.compile(
    r"\b(?:remaining|eta)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b", re.I
)


class TrainingStepEstimator:
    """Estimate steps from the immutable dataset snapshot configuration."""

    @staticmethod
    def estimate(
        *,
        subset_image_counts: tuple[int, ...] | list[int] | None,
        num_repeats: tuple[int, ...] | list[int] | None,
        batch_size: int | None,
        epochs: int | None,
        gradient_accumulation_steps: int = 1,
        world_size: int = 1,
        max_train_steps: int | None = None,
    ) -> EstimatedTrainingPlan:
        if (
            not subset_image_counts
            or not num_repeats
            or len(subset_image_counts) != len(num_repeats)
        ):
            return EstimatedTrainingPlan(
                None, None, "unknown", "dataset counts are unknown"
            )
        counts = tuple(subset_image_counts)
        repeats = tuple(num_repeats)
        if any(value <= 0 for value in counts + repeats):
            return EstimatedTrainingPlan(
                None, None, "unknown", "dataset counts must be positive"
            )
        if not batch_size or batch_size <= 0 or not epochs or epochs <= 0:
            return EstimatedTrainingPlan(
                None, None, "unknown", "training dimensions are unknown"
            )
        if gradient_accumulation_steps <= 0 or world_size <= 0:
            return EstimatedTrainingPlan(
                None, None, "unknown", "invalid accumulation settings"
            )
        samples = sum(
            count * repeat for count, repeat in zip(counts, repeats, strict=True)
        )
        effective_batch = batch_size * gradient_accumulation_steps * world_size
        steps_per_epoch = math.ceil(samples / effective_batch)
        total = steps_per_epoch * epochs
        formula = (
            "ceil(sum(image_count * num_repeats) / "
            "(batch_size * gradient_accumulation_steps * world_size)) * epochs"
        )
        if max_train_steps is not None:
            if max_train_steps <= 0:
                return EstimatedTrainingPlan(
                    None, None, formula, "max_train_steps is invalid"
                )
            total = min(total, max_train_steps)
            formula += "; min(max_train_steps)"
        return EstimatedTrainingPlan(steps_per_epoch, total, formula)


class TrainingLogParser:
    parser_version = "phase6b-v1"

    def parse(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        state: TrainingLogParserState | None = None,
        *,
        total_epochs: int | None = None,
        estimated_total_steps: int | None = None,
        started_at: datetime | None = None,
        now: datetime | None = None,
    ) -> TrainingLogParseResult:
        old = state or TrainingLogParserState()
        warnings = list(old.warnings)
        combined = stdout + (b"\n" if stdout and stderr else b"") + stderr
        text = self._decode(combined)
        text = old.remainder + text
        lines = re.split(r"[\r\n]+", text)
        remainder = lines.pop() if lines and not text.endswith(("\n", "\r")) else ""
        epoch = old.current_epoch
        total_epoch_value = total_epochs or old.total_epochs
        step = old.current_step
        total_step = old.total_steps or estimated_total_steps
        total_step_from_log = old.total_steps_source is TrainingProgressSource.LOG
        loss = old.latest_loss
        learning_rate = old.learning_rate
        speed = old.speed
        elapsed = old.elapsed_seconds
        remaining = old.remaining_seconds
        events: list[TrainingMetricEvent] = []
        logged_at = now or datetime.now(UTC)
        for raw_line in lines:
            line = ANSI_RE.sub("", raw_line).strip()
            if not line:
                continue
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                new_epoch, reported_total = int(epoch_match[1]), int(epoch_match[2])
                if epoch is not None and new_epoch < epoch:
                    warnings.append(
                        "epoch decreased; log rotation or restart suspected"
                    )
                epoch, total_epoch_value = new_epoch, reported_total
            match = STEP_RE.search(line)
            if (
                match is None
                and not epoch_match
                and not re.search(r"traceback|warning|error", line, re.I)
            ):
                match = PAIR_RE.search(line)
            if match:
                new_step, reported_total = int(match[1]), int(match[2])
                if step is not None and new_step < step:
                    warnings.append("step decreased; log rotation or restart suspected")
                step, total_step = new_step, reported_total
                total_step_from_log = True
            match = LOSS_RE.search(line)
            if match:
                parsed = _finite_float(match[1])
                if parsed is not None:
                    loss = parsed
                    events.append(
                        TrainingMetricEvent("loss", parsed, epoch, step, logged_at)
                    )
            match = LR_RE.search(line)
            if match:
                parsed = _finite_float(match[1])
                if parsed is not None:
                    learning_rate = parsed
                    events.append(
                        TrainingMetricEvent(
                            "learning_rate", parsed, epoch, step, logged_at
                        )
                    )
            match = SPEED_RE.search(line)
            if match:
                speed = _finite_float(match[1])
            match = TIME_RE.search(line)
            if match:
                elapsed = _finite_float(match[1])
            match = REMAINING_RE.search(line)
            if match:
                remaining = _finite_float(match[1])
        current_time = now or datetime.now(UTC)
        if elapsed is None and started_at is not None:
            elapsed = max(0.0, (current_time - started_at).total_seconds())
        ratio: float | None = None
        source = TrainingProgressSource.UNKNOWN
        if step is not None and total_step and total_step > 0:
            ratio = _clamp(step / total_step)
            source = (
                TrainingProgressSource.LOG
                if total_step_from_log
                else TrainingProgressSource.ESTIMATED
            )
        elif epoch is not None and total_epoch_value and total_epoch_value > 0:
            ratio = _clamp(epoch / total_epoch_value)
            source = TrainingProgressSource.LOG
        next_state = TrainingLogParserState(
            remainder=remainder,
            current_epoch=epoch,
            total_epochs=total_epoch_value,
            current_step=step,
            total_steps=total_step,
            latest_loss=loss,
            learning_rate=learning_rate,
            speed=speed,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            total_steps_source=(
                TrainingProgressSource.LOG
                if total_step_from_log
                else TrainingProgressSource.ESTIMATED
                if total_step is not None
                else TrainingProgressSource.UNKNOWN
            ),
            warnings=tuple(dict.fromkeys(warnings[-20:])),
        )
        return TrainingLogParseResult(
            ParsedTrainingProgress(
                epoch,
                total_epoch_value,
                step,
                total_step,
                loss,
                learning_rate,
                speed,
                elapsed,
                remaining,
                ratio,
                source,
                tuple(events),
                next_state.warnings,
                next_state,
            ),
            next_state,
        )

    @staticmethod
    def _decode(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


SdScriptsLogParser = TrainingLogParser
TrainingProgressParser = TrainingLogParser


def _finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EtaSmoother:
    def __init__(
        self, window: int = 5, max_eta_seconds: float = 30 * 24 * 3600
    ) -> None:
        self.window = max(2, window)
        self.max_eta_seconds = max_eta_seconds
        self._samples: list[tuple[float, float]] = []

    def update(self, step: int | None, at: float | None = None) -> float | None:
        if step is None:
            return None
        stamp = at if at is not None else time.monotonic()
        if self._samples and step <= self._samples[-1][0]:
            return None
        self._samples.append((float(step), stamp))
        self._samples = self._samples[-self.window :]
        if len(self._samples) < 2:
            return None
        first_step, first_time = self._samples[0]
        speed = (step - first_step) / max(stamp - first_time, 1e-9)
        return speed if speed > 0 else None
