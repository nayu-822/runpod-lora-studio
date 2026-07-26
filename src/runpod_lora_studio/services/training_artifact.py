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
        self.output_root = output_root.resolve()
        self.max_depth = max(1, max_depth)
        self.max_count = max(1, max_count)
        self.max_file_size = max(1, max_file_size)
        self.max_header_size = max(1024, max_header_size)
        self.max_metadata_size = max(1024, max_metadata_size)

    def scan(self, output_name: str) -> tuple[DiscoveredArtifact, ...]:
        if not self.output_root.is_dir() or self.output_root.is_symlink():
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
        stat = path.stat()
        return DiscoveredArtifact(
            TrainingArtifactType.TRAINING_STATE,
            relative,
            path.name,
            _number(path.name, "epoch"),
            _state_step(path.name),
            sum(item.stat().st_size for item in members),
            None,
            datetime.fromtimestamp(stat.st_mtime, UTC),
            TrainingArtifactValidationStatus.VALID,
            "STATE_STRUCTURE_VALID",
            "state directory structure recognized",
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
    value = metadata.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _matches_state_name(name: str, output_name: str) -> bool:
    return name == "state" or bool(
        re.fullmatch(rf"{re.escape(output_name)}-\d{{1,8}}-state", name)
    )


def _state_step(name: str) -> int | None:
    match = re.search(r"-(\d{1,8})-state$", name)
    return int(match[1]) if match else None
