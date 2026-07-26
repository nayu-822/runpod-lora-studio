from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)

_spec = importlib.util.spec_from_file_location(
    "verify_local_environment",
    Path(__file__).parents[2] / "scripts" / "verify_local_environment.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_check_migration = _module._check_migration
_check_sqlite = _module._check_sqlite
collect_checks = _module.collect_checks


def local_settings(test_workspace: Path) -> AppSettings:
    runtime = test_workspace / "runtime"
    settings = AppSettings(
        workspace_root=runtime,
        projects_dir=runtime / "projects",
        models_dir=runtime / "models",
        outputs_dir=runtime / "outputs",
        logs_dir=runtime / "logs",
        temp_dir=runtime / "tmp",
        database_path=runtime / "database" / "studio.sqlite3",
    )
    ensure_runtime_directories(settings)
    return settings


def migrated_settings(test_workspace: Path, revision: str = "head") -> AppSettings:
    settings = local_settings(test_workspace)
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.upgrade(Config(str(Path("alembic.ini").resolve())), revision)
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    return settings


class FakeRclone:
    def __init__(self, returncode: int = 0, remotes: str = "") -> None:
        self.returncode = returncode
        self.remotes = remotes

    def version(self):
        from runpod_lora_studio.external.rclone import CommandResult

        return CommandResult(self.returncode, "rclone v1", "")

    def list_remotes(self):
        from runpod_lora_studio.external.rclone import CommandResult

        return CommandResult(0, self.remotes, "")


def test_sqlite_check_does_not_remove_existing_local_verify_table(
    test_workspace: Path,
) -> None:
    settings = migrated_settings(test_workspace)
    from runpod_lora_studio.persistence.database import create_engine_for_settings

    engine = create_engine_for_settings(settings)
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE _local_verify (id INTEGER)")
        connection.exec_driver_sql("INSERT INTO _local_verify VALUES (7)")

    checks = _check_sqlite(settings)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT id FROM _local_verify")) == 7
    assert all(check.ok for check in checks)
    assert list(settings.temp_dir.glob("verify-*.sqlite3")) == []


def test_sqlite_write_check_uses_temporary_database(test_workspace: Path) -> None:
    settings = migrated_settings(test_workspace)

    checks = _check_sqlite(settings)

    assert any(check.name == "SQLite一時書き込み" and check.ok for check in checks)
    assert list(settings.temp_dir.glob("verify-*.sqlite3")) == []


def test_sqlite_write_check_reports_temporary_database_creation_failure(
    test_workspace: Path, monkeypatch
) -> None:
    settings = migrated_settings(test_workspace)

    def fail_tempfile(**_kwargs):
        raise OSError("temporary database unavailable")

    monkeypatch.setattr(_module.tempfile, "NamedTemporaryFile", fail_tempfile)
    checks = _check_sqlite(settings)

    write_check = next(check for check in checks if check.name == "SQLite一時書き込み")
    assert not write_check.ok
    assert "書き込めません" in write_check.detail


def test_sqlite_cleanup_failure_does_not_hide_success(
    test_workspace: Path, monkeypatch
) -> None:
    settings = migrated_settings(test_workspace)
    original_unlink = Path.unlink

    def fail_unlink(_path: Path, missing_ok: bool = False) -> None:
        raise OSError("cleanup unavailable")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    checks = _check_sqlite(settings)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    for path in settings.temp_dir.glob("verify-*.sqlite3"):
        original_unlink(path, missing_ok=True)

    write_check = next(check for check in checks if check.name == "SQLite一時書き込み")
    assert write_check.ok


def test_migration_check_requires_current_to_equal_head(test_workspace: Path) -> None:
    settings = migrated_settings(test_workspace)

    check = _check_migration(settings)

    assert check.ok
    assert "current=0023_phase7b_memory_failure_codes" in check.detail


def test_migration_check_rejects_unmigrated_database(test_workspace: Path) -> None:
    settings = local_settings(test_workspace)
    create_engine(f"sqlite:///{settings.database_path.as_posix()}").dispose()
    check = _check_migration(settings)

    assert not check.ok
    assert "未適用" in check.detail


def test_migration_check_rejects_multiple_heads(
    test_workspace: Path, monkeypatch
) -> None:
    settings = migrated_settings(test_workspace)

    class MultipleHeads:
        def get_heads(self) -> list[str]:
            return ["head-a", "head-b"]

    monkeypatch.setattr(
        _module.ScriptDirectory,
        "from_config",
        lambda _config: MultipleHeads(),
    )
    check = _check_migration(settings)

    assert not check.ok
    assert "複数" in check.detail


def test_rclone_is_optional_and_remote_names_are_only_reported(
    test_workspace: Path,
) -> None:
    settings = migrated_settings(test_workspace)
    checks = collect_checks(settings, FakeRclone(remotes="drive:\n"))

    rclone = next(check for check in checks if check.name == "rclone")
    assert rclone.ok
    assert "drive:" in rclone.detail
    assert "secret" not in rclone.detail


def test_rclone_execution_failure_is_optional_warning(test_workspace: Path) -> None:
    settings = migrated_settings(test_workspace)
    checks = collect_checks(settings, FakeRclone(returncode=1))

    rclone = next(check for check in checks if check.name == "rclone")
    assert not rclone.ok
    assert all(check.required is False or check.ok for check in checks)


def test_git_check_handles_success_nonzero_and_missing_git(
    monkeypatch,
) -> None:
    success = SimpleNamespace(returncode=0, stdout="git version 2.0\n", stderr="")
    monkeypatch.setattr(_module.subprocess, "run", lambda *_args, **_kwargs: success)
    assert _module._check_git().ok

    failed = SimpleNamespace(returncode=1, stdout="", stderr="git failed\n")
    monkeypatch.setattr(_module.subprocess, "run", lambda *_args, **_kwargs: failed)
    failed_check = _module._check_git()
    assert not failed_check.ok
    assert "git failed" in failed_check.detail

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(_module.subprocess, "run", missing_git)
    missing_check = _module._check_git()
    assert not missing_check.ok
    assert "未導入" in missing_check.detail


def test_git_failure_does_not_stop_other_checks_and_is_required(
    test_workspace: Path, monkeypatch
) -> None:
    settings = migrated_settings(test_workspace)

    def fail_git(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(_module.subprocess, "run", fail_git)
    checks = collect_checks(settings, FakeRclone())

    names = {check.name for check in checks}
    git_check = next(check for check in checks if check.name == "Git")
    assert not git_check.ok and git_check.required
    assert {"SQLite", "Alembic", "rclone"}.issubset(names)
    assert any(check.required and not check.ok for check in checks)


def test_git_failure_makes_script_exit_nonzero(monkeypatch) -> None:
    settings = SimpleNamespace(app_env="local")
    checks = [
        _module.Check("Git", True, False, "Git未導入"),
        _module.Check("SQLite", True, True, "OK"),
    ]
    monkeypatch.setattr(_module, "get_settings", lambda: settings)
    monkeypatch.setattr(_module, "collect_checks", lambda _settings: checks)

    assert _module.main() == 1


def test_port_check_uses_configured_port_and_detects_in_use_port(
    test_workspace: Path,
) -> None:
    settings = migrated_settings(test_workspace).model_copy(
        update={"gradio_server_port": 18765}
    )
    available = collect_checks(settings, FakeRclone())
    port_check = next(check for check in available if check.name == "18765番ポート")
    assert port_check.ok

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((settings.gradio_server_name, settings.gradio_server_port))
        occupied = collect_checks(settings, FakeRclone())
    occupied_check = next(check for check in occupied if check.name == "18765番ポート")
    assert not occupied_check.ok
