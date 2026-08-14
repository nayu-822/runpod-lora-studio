from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4


class CompletionFilesystemError(OSError):
    """A local completion export path could not be proven safe."""


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int


def _identity_from_stat(value: os.stat_result) -> DirectoryIdentity:
    return DirectoryIdentity(value.st_dev, value.st_ino)


def _fd_traversal_supported() -> bool:
    supported: set[object] = getattr(os, "supports_dir_fd", set())
    return bool(
        os.name != "nt"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(
            operation in supported
            for operation in (
                os.open,
                os.mkdir,
                os.rename,
                os.stat,
                os.unlink,
                os.rmdir,
            )
        )
    )


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _check_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or os.path.isabs(component)
    ):
        raise CompletionFilesystemError("invalid directory component")


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    normalized = relative_path.replace("\\", "/")
    parts = tuple(normalized.split("/"))
    if (
        not normalized
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(char) < 32 for char in normalized)
    ):
        raise CompletionFilesystemError("invalid relative export path")
    return parts


@dataclass(slots=True)
class SafeExportDirectory:
    """A directory held by fd on POSIX and checked by identity on fallback OSes."""

    path: Path
    fd: int
    root: Path
    components: tuple[str, ...]
    identities: tuple[DirectoryIdentity, ...]
    parent: SafeExportDirectory | None = None
    name: str | None = None

    @classmethod
    def open_root(
        cls,
        root: Path,
        components: tuple[str, ...],
        *,
        create_missing: bool = True,
    ) -> SafeExportDirectory:
        if not components:
            raise CompletionFilesystemError("export directory components are empty")
        for component in components:
            _check_component(component)
        root_path = root.absolute()
        if create_missing:
            root_path.mkdir(parents=True, exist_ok=True)
        elif not root_path.exists():
            raise CompletionFilesystemError("export root is missing")
        if _fd_traversal_supported():
            current_fd = -1
            try:
                current_fd = os.open(root_path, _directory_flags())
                identities = [_identity_from_stat(os.fstat(current_fd))]
                current_path = root_path
                for component in components:
                    child_fd = _open_directory_at(
                        current_fd, component, create=create_missing
                    )
                    identities.append(_identity_from_stat(os.fstat(child_fd)))
                    os.close(current_fd)
                    current_fd = child_fd
                    current_path = current_path / component
                return cls(
                    current_path,
                    current_fd,
                    root_path,
                    components,
                    tuple(identities),
                )
            except Exception:
                if current_fd >= 0:
                    os.close(current_fd)
                raise
        if os.name != "nt":
            raise CompletionFilesystemError(
                "POSIX completion export requires fd traversal support"
            )
        current = root_path
        fallback_identities: list[DirectoryIdentity] = [
            _checked_directory_identity(current)
        ]
        for component in components:
            current = current / component
            if create_missing:
                current.mkdir(exist_ok=True)
            fallback_identities.append(_checked_directory_identity(current))
        return cls(current, -1, root_path, components, tuple(fallback_identities))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> SafeExportDirectory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def identity_matches(self) -> bool:
        if self.fd < 0:
            try:
                current = self.root
                identities = [_checked_directory_identity(current)]
                for component in self.components:
                    current = current / component
                    identities.append(_checked_directory_identity(current))
                return tuple(identities) == self.identities
            except (OSError, ValueError):
                return False
        try:
            with self._reopen_same_path() as reopened:
                return reopened.identities == self.identities
        except (OSError, ValueError):
            return False

    def _reopen_same_path(self) -> SafeExportDirectory:
        return SafeExportDirectory.open_root(
            self.root, self.components, create_missing=False
        )

    def child_exists(self, name: str) -> str:
        _check_component(name)
        if self.fd >= 0:
            try:
                value = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                return "missing"
            if stat.S_ISLNK(value.st_mode):
                return "symlink"
            if stat.S_ISDIR(value.st_mode):
                return "directory"
            if stat.S_ISREG(value.st_mode):
                return "file"
            return "special"
        path = self.path / name
        try:
            value = os.lstat(path)
        except FileNotFoundError:
            return "missing"
        if stat.S_ISLNK(value.st_mode):
            return "symlink"
        if stat.S_ISDIR(value.st_mode):
            return "directory"
        if stat.S_ISREG(value.st_mode):
            return "file"
        return "special"

    def open_child(self, name: str, *, create: bool = False) -> SafeExportDirectory:
        _check_component(name)
        if not self.identity_matches():
            raise CompletionFilesystemError("export parent identity changed")
        child_path = self.path / name
        if self.fd >= 0:
            child_fd = _open_directory_at(self.fd, name, create=create)
            return SafeExportDirectory(
                child_path,
                child_fd,
                self.root,
                (*self.components, name),
                (*self.identities, _identity_from_stat(os.fstat(child_fd))),
                self,
                name,
            )
        if create:
            try:
                child_path.mkdir()
            except FileExistsError:
                pass
        identity = _checked_directory_identity(child_path)
        return SafeExportDirectory(
            child_path,
            -1,
            self.root,
            (*self.components, name),
            (*self.identities, identity),
            self,
            name,
        )

    def write_json(self, relative_path: str, value: object) -> None:
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.write_bytes_atomic(relative_path, encoded)

    def read_bytes(self, relative_path: str, *, max_bytes: int | None = None) -> bytes:
        parts = _relative_parts(relative_path)
        parent_fd, parent_path = self._open_relative_parent(parts[:-1], create=False)
        descriptor = -1
        try:
            if self.fd >= 0:
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise CompletionFilesystemError("export marker is not regular")
                if max_bytes is not None and before.st_size > max_bytes:
                    raise CompletionFilesystemError("export marker is too large")
                data_buffer = bytearray()
                while True:
                    chunk = os.read(descriptor, min(1024 * 1024, before.st_size + 1))
                    if not chunk:
                        break
                    data_buffer.extend(chunk)
                    if max_bytes is not None and len(data_buffer) > max_bytes:
                        raise CompletionFilesystemError("export marker is too large")
                data = bytes(data_buffer)
                after = os.fstat(descriptor)
                if _identity_size_mtime(after) != _identity_size_mtime(before):
                    raise CompletionFilesystemError(
                        "export marker changed while reading"
                    )
            else:
                path = parent_path / parts[-1]
                before = _regular_file_stat(path)
                if max_bytes is not None and before.st_size > max_bytes:
                    raise CompletionFilesystemError("export marker is too large")
                data = path.read_bytes()
                after = _regular_file_stat(path)
                if _identity_size_mtime(after) != _identity_size_mtime(before):
                    raise CompletionFilesystemError(
                        "export marker changed while reading"
                    )
            if max_bytes is not None and len(data) > max_bytes:
                raise CompletionFilesystemError("export marker is too large")
            if not self.identity_matches():
                raise CompletionFilesystemError("export identity changed")
            return data
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_fd >= 0 and parent_fd != self.fd:
                os.close(parent_fd)

    def hash_file(self, relative_path: str) -> tuple[int, str]:
        parts = _relative_parts(relative_path)
        parent_fd, parent_path = self._open_relative_parent(parts[:-1], create=False)
        descriptor = -1
        digest = hashlib.sha256()
        try:
            if self.fd >= 0:
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise CompletionFilesystemError("export file is not regular")
                stream = os.fdopen(descriptor, "rb", closefd=True)
                descriptor = -1
            else:
                path = parent_path / parts[-1]
                before = _regular_file_stat(path)
                stream = path.open("rb")
            with stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if self.fd >= 0:
                after = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            else:
                after = _regular_file_stat(parent_path / parts[-1])
            if _identity_size_mtime(after) != _identity_size_mtime(before):
                raise CompletionFilesystemError("export file changed while hashing")
            if not self.identity_matches():
                raise CompletionFilesystemError("export identity changed")
            return before.st_size, digest.hexdigest()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_fd >= 0 and parent_fd != self.fd:
                os.close(parent_fd)

    def write_bytes_atomic(self, relative_path: str, encoded: bytes) -> None:
        parts = _relative_parts(relative_path)
        parent_fd, parent_path = self._open_relative_parent(parts[:-1], create=True)
        temporary_name = f".tmp-{uuid4().hex[:12]}"
        try:
            if self.fd >= 0:
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
                try:
                    with os.fdopen(descriptor, "wb", closefd=True) as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if not self.identity_matches():
                        raise CompletionFilesystemError("export identity changed")
                    os.rename(
                        temporary_name,
                        parts[-1],
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    os.fsync(parent_fd)
                except Exception:
                    try:
                        os.unlink(temporary_name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
                    raise
            else:
                temporary = parent_path / temporary_name
                try:
                    with _open_fallback_exclusive(temporary) as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if not self.identity_matches():
                        raise CompletionFilesystemError("export identity changed")
                    os.rename(temporary, parent_path / parts[-1])
                    _fsync_directory(parent_path)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            if parent_fd >= 0 and parent_fd != self.fd:
                os.close(parent_fd)

    def copy_file(
        self,
        source: Path,
        relative_path: str,
        *,
        cancel: Callable[[], None] | None = None,
    ) -> tuple[int, str]:
        parts = _relative_parts(relative_path)
        parent_fd, parent_path = self._open_relative_parent(parts[:-1], create=True)
        digest = hashlib.sha256()
        source_before = _regular_file_stat(source)
        source_fd = -1
        target_fd = -1
        temporary_name = f".tmp-{uuid4().hex[:12]}"
        try:
            if _fd_traversal_supported():
                source_fd = os.open(
                    source,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                source_opened = os.fstat(source_fd)
                if _identity_size_mtime(source_opened) != _identity_size_mtime(
                    source_before
                ):
                    raise CompletionFilesystemError("source changed before copying")
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                )
                target_fd = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
                with (
                    os.fdopen(source_fd, "rb", closefd=True) as source_handle,
                    os.fdopen(target_fd, "wb", closefd=True) as target_handle,
                ):
                    source_fd = -1
                    target_fd = -1
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        if cancel is not None:
                            cancel()
                        target_handle.write(chunk)
                        digest.update(chunk)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                source_after = os.stat(source, follow_symlinks=False)
                if _identity_size_mtime(source_after) != _identity_size_mtime(
                    source_before
                ):
                    raise CompletionFilesystemError("source changed while copying")
                if not self.identity_matches():
                    raise CompletionFilesystemError("export identity changed")
                os.rename(
                    temporary_name,
                    parts[-1],
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
            else:
                target_path = parent_path / temporary_name
                with (
                    source.open("rb") as source_handle,
                    _open_fallback_exclusive(target_path) as target_handle,
                ):
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        if cancel is not None:
                            cancel()
                        target_handle.write(chunk)
                        digest.update(chunk)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                source_after = _regular_file_stat(source)
                if _identity_size_mtime(source_after) != _identity_size_mtime(
                    source_before
                ):
                    raise CompletionFilesystemError("source changed while copying")
                if not self.identity_matches():
                    raise CompletionFilesystemError("export identity changed")
                os.rename(parent_path / temporary_name, parent_path / parts[-1])
                _fsync_directory(parent_path)
            return source_before.st_size, digest.hexdigest()
        except Exception:
            if self.fd >= 0:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except (FileNotFoundError, OSError):
                    pass
            else:
                (parent_path / temporary_name).unlink(missing_ok=True)
            raise
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)
            if parent_fd >= 0 and parent_fd != self.fd:
                os.close(parent_fd)

    def _open_relative_parent(
        self, parts: tuple[str, ...], *, create: bool
    ) -> tuple[int, Path]:
        current_path = self.path
        if self.fd < 0:
            for part in parts:
                _check_component(part)
                current_path = current_path / part
                if create:
                    current_path.mkdir(exist_ok=True)
                _checked_directory_identity(current_path)
            return -1, current_path
        current_fd = os.dup(self.fd)
        try:
            for part in parts:
                _check_component(part)
                child_fd = _open_directory_at(current_fd, part, create=create)
                os.close(current_fd)
                current_fd = child_fd
                current_path = current_path / part
            return current_fd, current_path
        except Exception:
            os.close(current_fd)
            raise

    def rename_child(
        self, child: SafeExportDirectory, final_name: str
    ) -> SafeExportDirectory:
        _check_component(final_name)
        if child.parent is not self or child.name is None:
            raise CompletionFilesystemError("temporary export has a different parent")
        if not self.identity_matches() or not child.identity_matches():
            raise CompletionFilesystemError("export directory identity changed")
        if self.child_exists(final_name) != "missing":
            raise CompletionFilesystemError("final export already exists")
        if self.fd >= 0:
            os.rename(child.name, final_name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        else:
            os.rename(child.path, self.path / final_name)
            _fsync_directory(self.path)
        child.close()
        return self.open_child(final_name, create=False)

    def remove_child(self, child: SafeExportDirectory) -> None:
        if child.parent is not self or child.name is None:
            raise CompletionFilesystemError("temporary export has a different parent")
        if self.fd >= 0:
            try:
                current = os.stat(child.name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                child.close()
                return
            if (
                not stat.S_ISDIR(current.st_mode)
                or _identity_from_stat(current) != child.identities[-1]
            ):
                raise CompletionFilesystemError("temporary export identity changed")
            _remove_tree_at(self.fd, child.name)
            os.fsync(self.fd)
        else:
            if not self.identity_matches() or not child.identity_matches():
                raise CompletionFilesystemError("export directory identity changed")
            if child.path.exists() or child.path.is_symlink():
                shutil.rmtree(child.path)
            _fsync_directory(self.path)
        child.close()

    def fsync_tree(self) -> None:
        if not self.identity_matches():
            raise CompletionFilesystemError("export directory identity changed")
        if self.fd >= 0:
            _fsync_tree_at(self.fd)
            os.fsync(self.fd)
        else:
            _fsync_tree_path(self.path)

    def regular_files(self) -> list[tuple[str, int, str]]:
        if not self.identity_matches():
            raise CompletionFilesystemError("export directory identity changed")
        if self.fd >= 0:
            return _regular_files_at(self.fd)
        result: list[tuple[str, int, str]] = []
        for path in sorted(self.path.rglob("*"), key=lambda value: value.as_posix()):
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                raise CompletionFilesystemError("export contains a symlink")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise CompletionFilesystemError("export contains a special file")
            relative = path.relative_to(self.path).as_posix()
            size, digest = stable_file_hash(path)
            result.append((relative, size, digest))
        if not self.identity_matches():
            raise CompletionFilesystemError("export directory identity changed")
        return result


def _open_directory_at(parent_fd: int, name: str, *, create: bool) -> int:
    flags = _directory_flags()
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)


def _open_fallback_exclusive(path: Path) -> BinaryIO:
    for attempt in range(3):
        try:
            return path.open("xb")
        except FileNotFoundError:
            if attempt == 2:
                raise
            path.parent.mkdir(parents=True, exist_ok=True)
    raise AssertionError("unreachable")


def _checked_directory_identity(path: Path) -> DirectoryIdentity:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise CompletionFilesystemError("export path component is not a directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute().resolve(strict=True):
        raise CompletionFilesystemError("export path component is a symlink")
    return _identity_from_stat(value)


def _regular_file_stat(path: Path) -> os.stat_result:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise CompletionFilesystemError("source is not a regular file")
    return value


def _identity_size_mtime(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def stable_file_hash(path: Path) -> tuple[int, str]:
    before = _regular_file_stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = _regular_file_stat(path)
    if _identity_size_mtime(after) != _identity_size_mtime(before):
        raise CompletionFilesystemError("file changed while hashing")
    return before.st_size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, _directory_flags())
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_at(directory_fd: int) -> None:
    for entry in os.scandir(directory_fd):
        value = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child_fd = os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
            try:
                _fsync_tree_at(child_fd)
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(value.st_mode):
            descriptor = os.open(
                entry.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise CompletionFilesystemError("export contains a special file")


def _regular_files_at(
    directory_fd: int, prefix: tuple[str, ...] = ()
) -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    with os.scandir(directory_fd) as entries:
        for entry in sorted(entries, key=lambda value: value.name):
            value = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
            relative = (*prefix, entry.name)
            if stat.S_ISDIR(value.st_mode):
                child_fd = os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
                try:
                    result.extend(_regular_files_at(child_fd, relative))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(value.st_mode):
                size, digest = _hash_file_at(directory_fd, entry.name)
                result.append(("/".join(relative), size, digest))
            else:
                raise CompletionFilesystemError("export contains a special file")
    return result


def _hash_file_at(directory_fd: int, name: str) -> tuple[int, str]:
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompletionFilesystemError("export file is not regular")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity_size_mtime(after) != _identity_size_mtime(before):
            raise CompletionFilesystemError("export file changed while hashing")
        return before.st_size, digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_tree_path(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        value = os.lstat(path)
        if stat.S_ISREG(value.st_mode):
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        elif stat.S_ISDIR(value.st_mode):
            _fsync_directory(path)
        elif stat.S_ISLNK(value.st_mode):
            raise CompletionFilesystemError("export contains a symlink")
        else:
            raise CompletionFilesystemError("export contains a special file")


def _remove_tree_at(parent_fd: int, name: str) -> None:
    child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    child_identity = _identity_from_stat(os.fstat(child_fd))
    try:
        for entry in list(os.scandir(child_fd)):
            value = os.stat(entry.name, dir_fd=child_fd, follow_symlinks=False)
            if stat.S_ISDIR(value.st_mode):
                _remove_tree_at(child_fd, entry.name)
            elif stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
                os.unlink(entry.name, dir_fd=child_fd)
            else:
                raise CompletionFilesystemError(
                    "temporary export contains a special file"
                )
    finally:
        os.close(child_fd)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity_from_stat(current) != child_identity:
        raise CompletionFilesystemError("temporary export identity changed")
    os.rmdir(name, dir_fd=parent_fd)


__all__ = [
    "CompletionFilesystemError",
    "DirectoryIdentity",
    "SafeExportDirectory",
    "stable_file_hash",
]
