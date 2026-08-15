from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.storage_models import StorageRemotePath
from runpod_lora_studio.external.rclone import (
    CommandResult,
    CopyOptions,
    RcloneAdapter,
    RcloneRunner,
)


def test_rclone_runner_uses_argument_array_and_config_without_shell(
    monkeypatch,
    test_workspace: Path,
) -> None:
    config = test_workspace / "rclone.conf"
    config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("runpod_lora_studio.external.rclone.subprocess.run", fake_run)
    result = RcloneRunner("rclone", config).run(["lsd", "gdrive:models"])

    assert result.returncode == 0
    assert calls[0]["command"] == [
        "rclone",
        "--config",
        str(config),
        "lsd",
        "gdrive:models",
    ]
    assert calls[0]["shell"] is False


def test_rclone_dry_run_uses_copyto_for_a_local_file(
    monkeypatch, test_workspace: Path
) -> None:
    source = test_workspace / "export.bin"
    source.write_bytes(b"payload")
    calls: list[list[str]] = []

    def fake_run(arguments, timeout=10.0):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    settings = AppSettings(workspace_root=test_workspace)
    adapter = RcloneAdapter(settings)
    monkeypatch.setattr(adapter.runner, "run", fake_run)

    adapter.dry_run_copy(
        source,
        StorageRemotePath("gdrive", "training/job/export.bin"),
        CopyOptions(),
    )

    assert "copyto" in calls[0]


def test_rclone_remote_read_applies_transfer_level_max_size(
    monkeypatch, test_workspace: Path
) -> None:
    settings = AppSettings(workspace_root=test_workspace)
    adapter = RcloneAdapter(settings)
    commands: list[list[str]] = []

    def fake_streaming(
        command: list[str],
        timeout: float | None,
        progress_callback,
        cancel_token,
        process_callback,
    ) -> CommandResult:
        del timeout, progress_callback, cancel_token, process_callback
        commands.append(command)
        return CommandResult(0, "", "")

    monkeypatch.setattr(adapter, "_run_streaming", fake_streaming)
    adapter.copy(
        StorageRemotePath("gdrive", "bounded/manifest.json"),
        test_workspace / "manifest.json",
        CopyOptions(checksum=False, max_bytes=1024),
    )

    assert "--max-size" in commands[0]
    max_size_index = commands[0].index("--max-size")
    assert commands[0][max_size_index + 1] == "1025"
