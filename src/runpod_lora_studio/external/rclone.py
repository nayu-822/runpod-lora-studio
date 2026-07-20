from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class RcloneRunner:
    def __init__(self, executable: str = "rclone") -> None:
        self.executable = executable

    def run(self, arguments: list[str], timeout: float = 10.0) -> CommandResult:
        result = subprocess.run(
            [self.executable, *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def version(self) -> CommandResult:
        return self.run(["version"])

    def list_remotes(self) -> CommandResult:
        return self.run(["listremotes"])

    def list_directory(self, remote_path: str) -> CommandResult:
        return self.run(["lsd", remote_path])
