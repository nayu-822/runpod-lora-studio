from __future__ import annotations

import hashlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class StartedProcess:
    pid: int
    process_start_time: float | None
    process_group_id: int | None
    process_identity: str


class TrainingProcessAdapter(Protocol):
    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> StartedProcess: ...

    def is_running(self, pid: int) -> bool: ...

    def poll(self, pid: int) -> int | None: ...

    def process_matches(
        self, pid: int, process_group_id: int | None, process_identity: str | None
    ) -> bool: ...

    def terminate(self, pid: int) -> None: ...

    def kill(self, pid: int) -> None: ...


class SubprocessTrainingAdapter:
    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._identities: dict[int, tuple[int | None, str]] = {}

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> StartedProcess:
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"training cwd not found: {cwd}")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        identity = _command_identity(command)
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=dict(env),
                shell=False,
                start_new_session=True,
                close_fds=True,
                text=False,
            )
        process_group_id = _process_group_id(process.pid)
        self._processes[process.pid] = process
        self._identities[process.pid] = (process_group_id, identity)
        return StartedProcess(process.pid, None, process_group_id, identity)

    def is_running(self, pid: int) -> bool:
        process = self._processes.get(pid)
        if process is not None:
            return process.poll() is None
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def poll(self, pid: int) -> int | None:
        process = self._processes.get(pid)
        return process.poll() if process is not None else None

    def process_matches(
        self, pid: int, process_group_id: int | None, process_identity: str | None
    ) -> bool:
        expected = self._identities.get(pid)
        if expected is None or process_identity is None:
            return False
        return expected == (process_group_id, process_identity) and self.is_running(pid)

    def terminate(self, pid: int) -> None:
        process = self._processes.get(pid)
        if process is None or process.poll() is not None:
            return
        if os.name != "nt":
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # type: ignore[attr-defined]
        else:
            process.terminate()

    def kill(self, pid: int) -> None:
        process = self._processes.get(pid)
        if process is None or process.poll() is not None:
            return
        if os.name != "nt":
            os.killpg(os.getpgid(pid), _sigkill())  # type: ignore[attr-defined]
        else:
            process.kill()


@dataclass
class FakeTrainingProcessAdapter:
    next_pid: int = 41000
    running: bool = True
    exit_code: int | None = 0
    terminate_stops: bool = True
    kill_stops: bool = True
    fail_start: bool = False
    start_calls: list[tuple[tuple[str, ...], Path, Path, Path, dict[str, str]]] = field(
        default_factory=list
    )
    terminate_calls: list[int] = field(default_factory=list)
    kill_calls: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._processes: dict[int, tuple[bool, int | None, int, str]] = {}

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> StartedProcess:
        if self.fail_start:
            raise OSError("fake process start failed")
        pid = self.next_pid
        self.next_pid += 1
        identity = _command_identity(command)
        self.start_calls.append(
            (tuple(command), cwd, stdout_path, stderr_path, dict(env))
        )
        self._processes[pid] = (self.running, self.exit_code, pid, identity)
        return StartedProcess(pid, None, pid, identity)

    def is_running(self, pid: int) -> bool:
        state = self._processes.get(pid)
        return bool(state and state[0])

    def poll(self, pid: int) -> int | None:
        state = self._processes.get(pid)
        if state is None or state[0]:
            return None
        return state[1]

    def process_matches(
        self, pid: int, process_group_id: int | None, process_identity: str | None
    ) -> bool:
        state = self._processes.get(pid)
        return bool(
            state
            and state[0]
            and state[2] == process_group_id
            and state[3] == process_identity
        )

    def terminate(self, pid: int) -> None:
        self.terminate_calls.append(pid)
        state = self._processes.get(pid)
        if state and self.terminate_stops:
            self._processes[pid] = (False, -signal.SIGTERM, state[2], state[3])

    def kill(self, pid: int) -> None:
        self.kill_calls.append(pid)
        state = self._processes.get(pid)
        if state and self.kill_stops:
            self._processes[pid] = (False, -_sigkill(), state[2], state[3])


def _command_identity(command: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()


def _process_group_id(pid: int) -> int | None:
    if os.name == "nt":
        return pid
    try:
        return cast(int, os.getpgid(pid))  # type: ignore[attr-defined]
    except OSError:
        return None


def _sigkill() -> int:
    return int(getattr(signal, "SIGKILL", 9))
