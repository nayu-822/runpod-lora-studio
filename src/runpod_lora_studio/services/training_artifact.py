from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runpod_lora_studio.domain.training_progress_models import (
    TrainingArtifactType,
    TrainingArtifactValidationStatus,
)
from runpod_lora_studio.domain.training_resume_models import parse_non_negative_integer


@dataclass(frozen=True, slots=True)
class DiscoveredArtifact:
    artifact_type: TrainingArtifactType
    relative_path: Path
    filename: str
    epoch: int | None
    step: int | None
    file_size: int
    sha256: str | None
    modified_at: datetime | None
    validation_status: TrainingArtifactValidationStatus
    validation_code: str | None
    validation_message: str | None
    metadata: dict[str, Any] | None = None


class TrainingArtifactScanner:
    def __init__(
        self,
        output_root: Path,
        *,
        max_depth: int = 3,
        max_count: int = 500,
        max_file_size: int = 30 * 1024**3,
        max_header_size: int = 16 * 1024 * 1024,
        max_metadata_size: int = 256 * 1024,
    ) -> None:
        self._input_is_symlink = output_root.is_symlink()
        self.output_root = output_root.resolve()
        self.max_depth = max(1, max_depth)
        self.max_count = max(1, max_count)
        self.max_file_size = max(1, max_file_size)
        self.max_header_size = max(1024, max_header_size)
        self.max_metadata_size = max(1024, max_metadata_size)

    def scan(self, output_name: str) -> tuple[DiscoveredArtifact, ...]:
        if self._input_is_symlink or not self.output_root.is_dir():
            return ()
        found: list[DiscoveredArtifact] = []
        for path in self._walk(self.output_root):
            if len(found) >= self.max_count:
                break
            relative = path.relative_to(self.output_root)
            if path.is_symlink() or _ignored(path.name):
                continue
            if path.is_dir():
                if (
                    path.name.endswith("-state") or path.name == "state"
                ) and _matches_state_name(path.name, output_name):
                    state = self._state(path, relative)
                    if state is not None:
                        found.append(state)
                continue
            if path.suffix.lower() != ".safetensors":
                continue
            if not _matches_output_name(path.name, output_name):
                continue
            found.append(self._safetensors(path, relative, output_name))
        return tuple(found)

    def _walk(self, root: Path) -> list[Path]:
        result: list[Path] = []
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue and len(result) < self.max_count:
            directory, depth = queue.pop(0)
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for child in children:
                if child.is_symlink() or _ignored(child.name):
                    continue
                try:
                    child.resolve().relative_to(self.output_root)
                except (OSError, ValueError):
                    continue
                result.append(child)
                if child.is_dir() and depth < self.max_depth:
                    queue.append((child, depth + 1))
        return result

    def _state(self, path: Path, relative: Path) -> DiscoveredArtifact | None:
        try:
            members = [
                item
                for item in path.iterdir()
                if item.is_file() and not item.is_symlink()
            ]
        except OSError:
            return None
        if not any(
            item.name
            in {
                "optimizer.pt",
                "scheduler.pt",
                "training_state.json",
                "random_states_0.pkl",
            }
            for item in members
        ):
            return None
        metadata_files, metadata_error = read_state_metadata_files(
            path, self.max_metadata_size
        )
        metadata = _state_metadata_snapshot(metadata_files)
        epoch = _number(path.name, "epoch")
        if epoch is None:
            epoch = _metadata_int(metadata, "epoch")
        step = _state_step(path.name)
        if step is None:
            step = _metadata_int(metadata, "step")
        stat = path.stat()
        validation_status = TrainingArtifactValidationStatus.VALID
        validation_code = "STATE_STRUCTURE_VALID"
        validation_message = "state directory structure recognized"
        position_error = _state_position_error(path.name, epoch, step, metadata)
        if metadata_error or position_error:
            validation_status = TrainingArtifactValidationStatus.INVALID
            validation_code = metadata_error or "STATE_POSITION_CONFLICT"
            validation_message = (
                position_error or metadata_error or "state metadata invalid"
            )
        return DiscoveredArtifact(
            TrainingArtifactType.TRAINING_STATE,
            relative,
            path.name,
            epoch,
            step,
            sum(item.stat().st_size for item in members),
            None,
            datetime.fromtimestamp(stat.st_mtime, UTC),
            validation_status,
            validation_code,
            validation_message,
            metadata,
        )

    def _safetensors(
        self, path: Path, relative: Path, output_name: str
    ) -> DiscoveredArtifact:
        try:
            before = path.stat()
            if before.st_size <= 0:
                return self._invalid(path, relative, "EMPTY_FILE", "file size is zero")
            if before.st_size > self.max_file_size:
                return self._invalid(
                    path,
                    relative,
                    "FILE_TOO_LARGE",
                    "file exceeds configured size limit",
                )
            metadata, header_error = _read_safetensors_header(
                path, self.max_header_size, self.max_metadata_size
            )
            if header_error:
                return self._invalid(path, relative, header_error[0], header_error[1])
            digest = _sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return DiscoveredArtifact(
                    TrainingArtifactType.LORA_CHECKPOINT,
                    relative,
                    path.name,
                    None,
                    None,
                    after.st_size,
                    None,
                    datetime.fromtimestamp(after.st_mtime, UTC),
                    TrainingArtifactValidationStatus.CHANGING,
                    "ARTIFACT_STILL_CHANGING",
                    "file changed while validating",
                )
            epoch = _metadata_int(metadata, "epoch") if metadata else None
            step = _metadata_int(metadata, "step") if metadata else None
            if epoch is None:
                epoch = _number(path.name, "epoch")
            if step is None:
                match = re.search(
                    re.escape(output_name) + r"-(\d{1,8})\.safetensors$", path.name
                )
                step = int(match[1]) if match else None
            stat = after
            return DiscoveredArtifact(
                TrainingArtifactType.LORA_CHECKPOINT,
                relative,
                path.name,
                epoch,
                step,
                after.st_size,
                digest,
                datetime.fromtimestamp(stat.st_mtime, UTC),
                TrainingArtifactValidationStatus.VALID,
                "SAFETENSORS_VALID",
                "safetensors header validated",
                metadata,
            )
        except OSError:
            return self._invalid(
                path, relative, "READ_FAILED", "artifact could not be read"
            )

    @staticmethod
    def _invalid(
        path: Path, relative: Path, code: str, message: str
    ) -> DiscoveredArtifact:
        try:
            stat = path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime, UTC)
        except OSError:
            size, modified = 0, None
        return DiscoveredArtifact(
            TrainingArtifactType.LORA_CHECKPOINT,
            relative,
            path.name,
            None,
            None,
            size,
            None,
            modified,
            TrainingArtifactValidationStatus.INVALID,
            code,
            message,
        )


def _read_safetensors_header(
    path: Path, max_header: int, max_metadata: int
) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            return None, ("INVALID_HEADER", "header length is missing")
        header_length = struct.unpack("<Q", raw_length)[0]
        if (
            header_length <= 0
            or header_length > max_header
            or header_length > path.stat().st_size - 8
        ):
            return None, ("INVALID_HEADER_LENGTH", "header length is invalid")
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ("INVALID_HEADER", "header is not valid JSON")
    if not isinstance(header, dict):
        return None, ("INVALID_HEADER", "header must be an object")
    tensors = [key for key in header if key != "__metadata__"]
    if not tensors:
        return None, ("NO_TENSORS", "no tensors found")
    file_size = path.stat().st_size
    for key in tensors:
        item = header[key]
        if (
            not isinstance(item, dict)
            or item.get("dtype") is None
            or not isinstance(item.get("data_offsets"), list)
            or len(item["data_offsets"]) != 2
        ):
            return None, ("INVALID_TENSOR", "tensor entry is invalid")
        start, end = item["data_offsets"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or 8 + header_length + end > file_size
        ):
            return None, ("INVALID_TENSOR", "tensor offsets are invalid")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        return None, ("INVALID_METADATA", "metadata is invalid")
    safe_metadata = _safe_metadata(metadata or {}, max_metadata)
    if safe_metadata is None:
        return None, ("METADATA_TOO_LARGE", "metadata exceeds configured limit")
    return safe_metadata, None


def _safe_metadata(metadata: dict[str, Any], limit: int) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        text = str(value).replace("\x00", "")
        result[key[:128]] = "".join(
            char for char in text if ord(char) >= 32 or char in "\r\n\t"
        )[:4096]
    try:
        if len(json.dumps(result, ensure_ascii=False)) > limit:
            return None
    except (TypeError, ValueError):
        return None
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignored(name: str) -> bool:
    return name.startswith(".") or any(
        name.endswith(suffix) for suffix in (".tmp", ".partial", ".creating")
    )


def _matches_output_name(filename: str, output_name: str) -> bool:
    return filename == f"{output_name}.safetensors" or bool(
        re.fullmatch(rf"{re.escape(output_name)}-\d{{1,8}}\.safetensors", filename)
    )


def _number(name: str, label: str) -> int | None:
    match = re.search(rf"(?:^|-){label}[-_ ]?(\d+)", name, re.I)
    return int(match[1]) if match else None


def _metadata_int(metadata: dict[str, Any] | None, key: str) -> int | None:
    if not metadata:
        return None
    return parse_non_negative_integer(metadata.get(key))


def read_state_metadata_files(
    path: Path, max_size: int
) -> tuple[dict[str, dict[str, Any]], str | None]:
    result: dict[str, dict[str, Any]] = {}
    for filename in ("training_state.json", "state.json", "resume-state-manifest.json"):
        metadata_path = path / filename
        try:
            if not metadata_path.is_file():
                continue
            if metadata_path.stat().st_size > max_size:
                return result, "STATE_METADATA_TOO_LARGE"
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return result, "STATE_METADATA_INVALID"
        if isinstance(value, dict):
            result[filename] = value
        else:
            return result, "STATE_METADATA_NOT_OBJECT"
    return result, None


def _state_metadata_snapshot(
    metadata_files: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not metadata_files:
        return None
    snapshot: dict[str, Any] = {"state_metadata_files": metadata_files}
    for filename in ("training_state.json", "state.json", "resume-state-manifest.json"):
        values = metadata_files.get(filename)
        if values is None:
            continue
        for key in (
            "state_epoch",
            "epoch",
            "current_epoch",
            "state_step",
            "step",
            "current_step",
        ):
            if key in values and key not in snapshot:
                snapshot[key] = values[key]
    return snapshot


def _state_position_error(
    name: str,
    epoch: int | None,
    step: int | None,
    metadata: dict[str, Any] | None,
) -> str | None:
    if not metadata:
        return None
    values = metadata.get("state_metadata_files")
    if not isinstance(values, dict):
        return None
    epochs: list[tuple[str, int]] = []
    steps: list[tuple[str, int]] = []
    for filename, raw in values.items():
        if not isinstance(raw, dict):
            continue
        for key in ("state_epoch", "epoch", "current_epoch"):
            if key in raw:
                parsed = parse_non_negative_integer(raw[key])
                if parsed is None:
                    return f"STATE_POSITION_INVALID: {filename}.{key}"
                epochs.append((f"{filename}.{key}", parsed))
        for key in ("state_step", "step", "current_step"):
            if key in raw:
                parsed = parse_non_negative_integer(raw[key])
                if parsed is None:
                    return f"STATE_POSITION_INVALID: {filename}.{key}"
                steps.append((f"{filename}.{key}", parsed))
    if epoch is not None:
        epochs.append(("state directory/artifact epoch", epoch))
    if step is not None:
        steps.append(("state directory/artifact step", step))
    for label, candidates in (("epoch", epochs), ("step", steps)):
        if len({value for _, value in candidates}) > 1:
            details = ", ".join(f"{source}={value}" for source, value in candidates)
            return f"STATE_POSITION_CONFLICT: {label}: {details}"
    return None


def _matches_state_name(name: str, output_name: str) -> bool:
    return name == "state" or bool(
        re.fullmatch(rf"{re.escape(output_name)}-\d{{1,8}}-state", name)
    )


def _state_step(name: str) -> int | None:
    match = re.search(r"-(\d{1,8})-state$", name)
    return int(match[1]) if match else None
