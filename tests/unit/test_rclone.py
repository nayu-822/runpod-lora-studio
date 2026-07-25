from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runpod_lora_studio.external.rclone import RcloneRunner


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
