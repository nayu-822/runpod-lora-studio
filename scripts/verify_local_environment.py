from __future__ import annotations

import importlib
import logging
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runpod_lora_studio.config.settings import (  # noqa: E402
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)
from runpod_lora_studio.external.rclone import RcloneRunner  # noqa: E402
from runpod_lora_studio.persistence.database import (  # noqa: E402
    create_engine_for_settings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    required: bool
    ok: bool
    detail: str


def _module_check(name: str) -> Check:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "利用可能")
        return Check(name, True, True, str(version))
    except ImportError:
        return Check(name, True, False, "未インストール")


def _port_available(settings: AppSettings) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((settings.gradio_server_name, settings.gradio_server_port))
        except OSError:
            return False
    return True


def _check_migration(settings: AppSettings) -> Check:
    try:
        config = Config(str(ROOT_DIR / "alembic.ini"))
        heads = ScriptDirectory.from_config(config).get_heads()
        if len(heads) != 1:
            return Check("Alembic", True, False, "複数のheadがあります")
        if not settings.database_path.is_file():
            return Check("Alembic", True, False, "DB未作成（マイグレーション未適用）")
        engine = create_engine_for_settings(settings)
        with engine.connect() as connection:
            current = MigrationContext.configure(connection).get_current_revision()
        engine.dispose()
        if current != heads[0]:
            detail = f"更新が必要（current={current or 'なし'}, head={heads[0]}）"
            return Check("Alembic", True, False, detail)
        return Check("Alembic", True, True, f"current={current}, head={heads[0]}")
    except Exception:
        return Check("Alembic", True, False, "current/headの確認に失敗")


def _check_sqlite(settings: AppSettings) -> list[Check]:
    checks: list[Check] = []
    if not settings.database_path.is_file():
        checks.append(
            Check("SQLite", True, False, "DB未作成（マイグレーション未適用）")
        )
    else:
        try:
            engine = create_engine_for_settings(settings)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            engine.dispose()
            checks.append(
                Check("SQLite", True, True, "接続可能（既存DBは変更していません）")
            )
        except Exception:
            checks.append(Check("SQLite", True, False, "既存DBへ接続できません"))

    temporary_path: Path | None = None
    temporary_engine = None
    try:
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=settings.temp_dir, prefix="verify-", suffix=".sqlite3", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        temporary_engine = create_engine(f"sqlite:///{temporary_path.as_posix()}")
        with temporary_engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE verify (id INTEGER)")
            connection.exec_driver_sql("INSERT INTO verify VALUES (1)")
            if connection.scalar(text("SELECT id FROM verify")) != 1:
                raise RuntimeError(
                    "temporary SQLite write check returned an unexpected value"
                )
        checks.append(Check("SQLite一時書き込み", True, True, "一時DBで確認済み"))
    except Exception:
        checks.append(
            Check("SQLite一時書き込み", True, False, "一時DBへ書き込めません")
        )
    finally:
        if temporary_engine is not None:
            temporary_engine.dispose()
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "temporary SQLite cleanup failed: %s", temporary_path.name
                )
    return checks


def collect_checks(
    settings: AppSettings, rclone_runner: RcloneRunner | None = None
) -> list[Check]:
    checks = [
        Check("Python", True, sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("設定", True, bool(settings.app_env), settings.app_env),
    ]
    try:
        ensure_runtime_directories(settings)
        checks.extend(_check_sqlite(settings))
    except OSError:
        checks.append(Check("SQLite", True, False, "作業ディレクトリを準備できません"))
    checks.append(_check_migration(settings))

    checks.extend([_module_check("PIL"), _module_check("gradio")])
    checks.append(
        Check(
            "Git",
            True,
            subprocess.run(
                ["git", "--version"], check=False, capture_output=True
            ).returncode
            == 0,
            "確認",
        )
    )
    checks.append(Check("7860番ポート", True, _port_available(settings), "使用可能"))

    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        checks.append(Check("PyTorch", False, True, str(torch.__version__)))
        checks.append(Check("CUDA", False, True, "利用可能" if cuda else "CPU環境"))
    except ImportError:
        checks.extend(
            [
                Check("PyTorch", False, True, "未導入（任意）"),
                Check("CUDA", False, True, "PyTorch未導入"),
            ]
        )

    runner = rclone_runner or RcloneRunner()
    try:
        result = runner.version()
        if result.returncode != 0:
            checks.append(Check("rclone", False, False, "導入済みですが実行に失敗"))
        else:
            remotes = runner.list_remotes()
            remote_names = [
                line for line in remotes.stdout.splitlines() if line.strip()
            ]
            detail = (
                "設定リモート: " + ", ".join(remote_names)
                if remote_names
                else "未設定（任意）"
            )
            checks.append(Check("rclone", False, True, detail))
    except (OSError, subprocess.SubprocessError):
        checks.append(Check("rclone", False, True, "未導入（任意）"))
    return checks


def main() -> int:
    settings = get_settings()
    checks = collect_checks(settings)
    print(f"ローカル環境: {settings.app_env}")
    for check in checks:
        category = "必須" if check.required else "任意"
        status = "OK" if check.ok else "NG"
        print(f"[{category}] {status} {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks if check.required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
