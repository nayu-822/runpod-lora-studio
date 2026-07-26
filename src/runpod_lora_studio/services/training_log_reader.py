from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingLogCursor:
    offset: int = 0
    file_key: tuple[int, int] | None = None
    pending: bytes = b""


@dataclass(frozen=True, slots=True)
class TrainingLogChunk:
    data: bytes
    cursor: TrainingLogCursor
    reset: bool = False
    warning: str | None = None


class IncrementalLogReader:
    """Read only new bytes from a job's trusted logs directory."""

    def __init__(self, logs_root: Path, max_bytes: int = 256 * 1024) -> None:
        self.logs_root = logs_root.resolve()
        self.max_bytes = max(1024, max_bytes)

    def read(
        self, path: Path, cursor: TrainingLogCursor | None = None
    ) -> TrainingLogChunk:
        current = cursor or TrainingLogCursor()
        try:
            if path.is_symlink():
                return TrainingLogChunk(
                    b"", TrainingLogCursor(), True, "log symlink rejected"
                )
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.logs_root)
            if not resolved.is_file():
                return TrainingLogChunk(
                    b"", TrainingLogCursor(), True, "log path is not a regular file"
                )
            stat = resolved.stat()
            key = (stat.st_dev, stat.st_ino)
            reset = current.file_key is not None and (
                key != current.file_key or stat.st_size < current.offset
            )
            offset = 0 if reset else current.offset
            with resolved.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(self.max_bytes)
            next_offset = offset + len(data)
            # A new inode or a shrink starts a new byte stream. Carrying the
            # previous file's incomplete UTF-8 suffix into it would join two
            # unrelated log files and can fabricate a metric line.
            prefix = b"" if reset else current.pending
            pending, complete = _split_incomplete_utf8(prefix + data)
            warning = "log was truncated or rotated; offset reset" if reset else None
            return TrainingLogChunk(
                complete,
                TrainingLogCursor(next_offset, key, pending),
                reset,
                warning,
            )
        except (OSError, ValueError):
            return TrainingLogChunk(
                b"", TrainingLogCursor(), True, "log path is unavailable"
            )


def _split_incomplete_utf8(data: bytes) -> tuple[bytes, bytes]:
    try:
        data.decode("utf-8")
        return b"", data
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.start >= len(data) - 3:
            return data[exc.start :], data[: exc.start]
        return b"", data
