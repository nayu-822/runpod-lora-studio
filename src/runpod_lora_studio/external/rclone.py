from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.storage_models import (
    OverwritePolicy,
    StorageEntry,
    StorageRemote,
    StorageRemotePath,
    StorageValidationCheck,
    StorageValidationResult,
    TransferItemPlan,
    TransferPlan,
    TransferProgress,
    VerificationPolicy,
    VerificationResult,
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...] = ()
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ListOptions:
    recursive: bool = False
    max_entries: int = 500
    page: int = 1
    page_size: int = 50
    extension: str | None = None
    query: str = ""


@dataclass(frozen=True, slots=True)
class CopyOptions:
    overwrite_policy: OverwritePolicy = OverwritePolicy.SKIP_IDENTICAL
    dry_run: bool = False
    checksum: bool = True
    timeout: float | None = None


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


ProgressCallback = Callable[[TransferProgress], None]
ProcessCallback = Callable[[int | None], None]


class StorageTransferAdapter(Protocol):
    def validate_environment(self) -> StorageValidationResult: ...

    def list_remotes(self) -> tuple[StorageRemote, ...]: ...

    def list_entries(
        self, remote_path: StorageRemotePath, options: ListOptions
    ) -> tuple[StorageEntry, ...]: ...

    def read_remote_file(
        self,
        remote_path: StorageRemotePath,
    ) -> bytes: ...

    def dry_run_copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
    ) -> TransferPlan: ...

    def copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
        process_callback: ProcessCallback | None = None,
    ) -> CommandResult: ...

    def verify(
        self,
        source: Path,
        expected_size: int,
        expected_hash: str | None,
        policy: VerificationPolicy,
    ) -> VerificationResult: ...


class RcloneRunner:
    """Small subprocess boundary retained for environment checks and tests."""

    def __init__(
        self,
        executable: str = "rclone",
        config_path: Path | None = None,
    ) -> None:
        self.executable = executable
        self.config_path = config_path

    def _command(self, arguments: Iterable[str]) -> list[str]:
        command = [self.executable]
        if self.config_path is not None:
            command.extend(["--config", str(self.config_path)])
        command.extend(arguments)
        return command

    def run(self, arguments: list[str], timeout: float = 10.0) -> CommandResult:
        command = self._command(arguments)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
            )
        except FileNotFoundError:
            return CommandResult(
                127, "", "rclone executable was not found", tuple(command)
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _safe_text(exc.stdout)
            stderr = _safe_text(exc.stderr)
            return CommandResult(124, stdout, stderr, tuple(command), timed_out=True)
        return CommandResult(
            result.returncode, result.stdout, result.stderr, tuple(command)
        )

    def version(self) -> CommandResult:
        return self.run(["version"])

    def list_remotes(self) -> CommandResult:
        return self.run(["listremotes"])

    def list_directory(self, remote_path: str) -> CommandResult:
        return self.run(["lsd", remote_path])


def _safe_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


class RcloneAdapter:
    def __init__(
        self, settings: AppSettings, runner: RcloneRunner | None = None
    ) -> None:
        self.settings = settings
        self.runner = runner or RcloneRunner(
            settings.rclone_executable, settings.rclone_config_path
        )

    def _common_args(self, *, timeout: float | None = None) -> list[str]:
        args = [
            "--transfers",
            str(self.settings.rclone_transfers),
            "--checkers",
            str(self.settings.rclone_checkers),
            "--retries",
            str(self.settings.rclone_retries),
            "--low-level-retries",
            str(self.settings.rclone_low_level_retries),
            "--retries-sleep",
            f"{self.settings.rclone_retry_interval_seconds}s",
            "--contimeout",
            f"{self.settings.rclone_connect_timeout_seconds}s",
            "--timeout",
            f"{timeout or self.settings.rclone_transfer_timeout_seconds}s",
            "--buffer-size",
            self.settings.rclone_buffer_size,
        ]
        if self.settings.rclone_bandwidth_limit:
            args.extend(["--bwlimit", self.settings.rclone_bandwidth_limit])
        return args

    def validate_environment(self) -> StorageValidationResult:
        checks: list[StorageValidationCheck] = []
        executable = self.settings.rclone_executable
        executable_exists = bool(shutil.which(executable) or Path(executable).is_file())
        checks.append(
            StorageValidationCheck(
                "rclone", executable_exists, "rcloneがインストールされていません"
            )
        )
        config = self.settings.rclone_config_path
        config_ok = (
            config is not None and config.is_file() and os.access(config, os.R_OK)
        )
        checks.append(
            StorageValidationCheck(
                "config",
                config_ok,
                "rclone設定ファイルがありません、または読み取れません",
            )
        )
        version_result = self.runner.version() if executable_exists else None
        version_ok = version_result is not None and version_result.returncode == 0
        checks.append(
            StorageValidationCheck(
                "version", version_ok, "rcloneのバージョンを取得できません"
            )
        )
        remotes = self.list_remotes() if version_ok and config_ok else ()
        remote_ok = any(
            remote.name == self.settings.storage_remote_name for remote in remotes
        )
        checks.append(
            StorageValidationCheck(
                "remote", remote_ok, "指定されたGoogle Drive remoteが登録されていません"
            )
        )
        remote_root = StorageRemotePath(
            self.settings.storage_remote_name, self.settings.storage_model_remote_root
        )
        connected = False
        if remote_ok:
            result = self.runner.run(
                [
                    *self._common_args(
                        timeout=self.settings.rclone_connect_timeout_seconds
                    ),
                    "lsd",
                    remote_root.rclone_value,
                ],
                timeout=self.settings.rclone_connect_timeout_seconds,
            )
            connected = result.returncode == 0
        checks.append(
            StorageValidationCheck(
                "connection", connected, "Google Driveへ接続できません"
            )
        )
        local_checks = (
            (
                "model_cache",
                self.settings.model_cache_dir or self.settings.models_dir / "base",
                "ローカルモデル保存先を利用できません",
            ),
            (
                "transfer_temp",
                self.settings.transfer_temp_dir or self.settings.temp_dir / "transfers",
                "転送一時領域を利用できません",
            ),
        )
        for name, path, message in local_checks:
            try:
                path.mkdir(parents=True, exist_ok=True)
                writable = os.access(path, os.W_OK)
            except OSError:
                writable = False
            checks.append(StorageValidationCheck(name, writable, message))
        version_text = _first_line(version_result.stdout) if version_result else None
        return StorageValidationResult(
            ok=all(check.ok for check in checks if check.required),
            checks=tuple(checks),
            rclone_version=version_text,
        )

    def list_remotes(self) -> tuple[StorageRemote, ...]:
        result = self.runner.list_remotes()
        if result.returncode != 0:
            return ()
        values: list[StorageRemote] = []
        for line in result.stdout.splitlines():
            name = line.strip().rstrip(":")
            if name and ":" not in name:
                values.append(StorageRemote(name, "unknown"))
        return tuple(values)

    def list_entries(
        self, remote_path: StorageRemotePath, options: ListOptions
    ) -> tuple[StorageEntry, ...]:
        args = [*self._common_args(), "lsjson", "--files-only", "--no-mimetype"]
        if options.recursive:
            args.append("--recursive")
        args.append(remote_path.rclone_value)
        result = self.runner.run(
            args, timeout=self.settings.rclone_transfer_timeout_seconds
        )
        if result.returncode != 0:
            raise RuntimeError("Google Driveの一覧を取得できません")
        entries: list[StorageEntry] = []
        raw_entries: list[object]
        try:
            decoded = json.loads(result.stdout)
            raw_entries = decoded if isinstance(decoded, list) else [decoded]
        except json.JSONDecodeError:
            raw_entries = []
            for line in result.stdout.splitlines():
                try:
                    raw_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        for value in raw_entries:
            if not isinstance(value, dict):
                continue
            raw = value
            relative = str(raw.get("Path") or raw.get("Name") or "").replace("\\", "/")
            if not relative or any(part == ".." for part in relative.split("/")):
                continue
            name = str(raw.get("Name") or Path(relative).name)
            if options.query and options.query.casefold() not in name.casefold():
                continue
            if options.extension and not name.casefold().endswith(
                options.extension.casefold()
            ):
                continue
            modified = None
            if raw.get("ModTime"):
                try:
                    modified = datetime.fromisoformat(
                        str(raw["ModTime"]).replace("Z", "+00:00")
                    )
                except ValueError:
                    modified = None
            hashes = raw.get("Hashes") or {}
            hash_type = next(iter(hashes), None)
            entries.append(
                StorageEntry(
                    remote_path=remote_path.child(relative),
                    name=name,
                    size_bytes=int(raw.get("Size") or 0),
                    modified_at=modified,
                    hash_type=hash_type,
                    hash_value=str(hashes[hash_type]) if hash_type else None,
                    is_directory=bool(raw.get("IsDir", False)),
                )
            )
        start = max(options.page - 1, 0) * options.page_size
        return tuple(
            entries[start : start + min(options.page_size, options.max_entries)]
        )

    def dry_run_copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
    ) -> TransferPlan:
        source_value = _path_value(source)
        destination_value = _path_value(destination)
        result = self.runner.run(
            [
                *self._common_args(timeout=options.timeout),
                "copyto" if isinstance(destination, Path) else "copy",
                source_value,
                destination_value,
                "--dry-run",
                "--use-json-log",
            ],
            timeout=options.timeout or self.settings.rclone_transfer_timeout_seconds,
        )
        errors = () if result.returncode == 0 else ("rcloneドライランに失敗しました",)
        size = (
            source.stat().st_size
            if isinstance(source, Path) and source.is_file()
            else 0
        )
        token = (
            f"{source_value}|{destination_value}|"
            f"{options.overwrite_policy.value}|{size}"
        )
        return TransferPlan(
            token=token,
            source=source_value,
            destination=destination_value,
            items=(TransferItemPlan(source_value, size, "copy"),),
            total_bytes=size,
            available_bytes=None,
            errors=errors,
        )

    def copy(
        self,
        source: str | Path | StorageRemotePath,
        destination: str | Path | StorageRemotePath,
        options: CopyOptions,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
        process_callback: ProcessCallback | None = None,
    ) -> CommandResult:
        command = [
            *self._common_args(timeout=options.timeout),
            "copyto"
            if _is_file_source(source) or isinstance(destination, Path)
            else "copy",
            _path_value(source),
            _path_value(destination),
            "--use-json-log",
            "--stats",
            "1s",
        ]
        if options.dry_run:
            command.append("--dry-run")
        if options.checksum and self.settings.storage_use_checksum:
            command.append("--checksum")
        if options.overwrite_policy is OverwritePolicy.FAIL_IF_EXISTS:
            command.append("--immutable")
        return self._run_streaming(
            command,
            options.timeout,
            progress_callback,
            cancel_token,
            process_callback,
        )

    def _run_streaming(
        self,
        command: list[str],
        timeout: float | None,
        progress_callback: ProgressCallback | None,
        cancel_token: CancelToken | None,
        process_callback: ProcessCallback | None,
    ) -> CommandResult:
        full_command = self.runner._command(command)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except FileNotFoundError:
            if process_callback:
                process_callback(None)
            return CommandResult(
                127, "", "rclone executable was not found", tuple(full_command)
            )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        if process_callback:
            process_callback(process.pid)

        def read_stream(stream: Any, target: list[str]) -> None:
            for line in iter(stream.readline, ""):
                if len(target) < 200:
                    target.append(line)
                progress = _progress_from_line(line, time.monotonic() - started)
                if progress_callback and progress is not None:
                    progress_callback(progress)
            stream.close()

        stdout_thread = threading.Thread(
            target=read_stream, args=(process.stdout, stdout_lines), daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(process.stderr, stderr_lines), daemon=True
        )
        try:
            stdout_thread.start()
            stderr_thread.start()
            timed_out = False
            while process.poll() is None:
                if cancel_token and cancel_token.cancelled:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if timeout is not None and time.monotonic() - started > timeout:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.05)
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            return CommandResult(
                process.returncode if process.returncode is not None else 1,
                "".join(stdout_lines),
                "".join(stderr_lines),
                tuple(full_command),
                timed_out=timed_out,
            )
        finally:
            if process_callback:
                process_callback(None)

    def verify(
        self,
        source: Path,
        expected_size: int,
        expected_hash: str | None,
        policy: VerificationPolicy,
    ) -> VerificationResult:
        if not source.is_file():
            return VerificationResult(
                False,
                policy,
                expected_size,
                None,
                expected_hash,
                None,
                "ローカルファイルがありません",
            )
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            return VerificationResult(
                False,
                policy,
                expected_size,
                actual_size,
                expected_hash,
                None,
                "ファイルサイズが一致しません",
            )
        actual_hash = None
        if expected_hash and policy is VerificationPolicy.FULL_CHECKSUM:
            import hashlib

            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                return VerificationResult(
                    False,
                    policy,
                    expected_size,
                    actual_size,
                    expected_hash,
                    actual_hash,
                    "SHA-256が一致しません",
                )
        return VerificationResult(
            True, policy, expected_size, actual_size, expected_hash, actual_hash
        )

    def read_remote_file(self, remote_path: StorageRemotePath) -> bytes:
        temp_root = (
            self.settings.transfer_temp_dir or self.settings.temp_dir / "transfers"
        )
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=temp_root, suffix=".part", delete=False
        ) as handle:
            path = Path(handle.name)
        try:
            result = self.copy(
                remote_path,
                path,
                CopyOptions(checksum=False),
            )
            if result.returncode != 0:
                raise RuntimeError("remoteファイルを読み取れません")
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)


def _first_line(value: str) -> str | None:
    line = value.splitlines()[0].strip() if value.splitlines() else ""
    return line or None


def _path_value(value: str | Path | StorageRemotePath) -> str:
    return value.rclone_value if isinstance(value, StorageRemotePath) else str(value)


def _is_file_source(value: str | Path | StorageRemotePath) -> bool:
    return isinstance(value, Path) and value.is_file()


def _progress_from_line(line: str, elapsed: float) -> TransferProgress | None:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    stats = raw.get("Stats") if isinstance(raw, dict) else None
    if not isinstance(stats, dict):
        stats = raw if isinstance(raw, dict) else {}
    if not any(key in stats for key in ("bytes", "totalBytes", "transfers", "errors")):
        return None
    total = int(stats.get("totalBytes") or 0)
    current = int(stats.get("bytes") or 0)
    speed = float(stats.get("speed") or 0.0)
    eta_value = stats.get("eta")
    eta = float(eta_value) if isinstance(eta_value, (int, float)) else None
    transferring = stats.get("transferring") or []
    checking = stats.get("checking") or []
    return TransferProgress(
        bytes_transferred=current,
        total_bytes=total,
        checks=int(stats.get("checks") or 0),
        transfers=int(stats.get("transfers") or 0),
        errors=int(stats.get("errors") or 0),
        elapsed_seconds=elapsed,
        speed_bytes_per_second=speed,
        eta_seconds=eta,
        current_path=(
            transferring[0].get("name")
            if transferring and isinstance(transferring[0], dict)
            else None
        ),
        checking=tuple(str(item) for item in checking if isinstance(item, str)),
        transferring=tuple(
            str(item.get("name"))
            for item in transferring
            if isinstance(item, dict) and item.get("name")
        ),
    )
