from __future__ import annotations

import importlib
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


def collect_checks(settings: AppSettings) -> list[Check]:
    checks = [
        Check("Python", True, sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("設定", True, bool(settings.app_env), settings.app_env),
    ]
    try:
        ensure_runtime_directories(settings)
        engine = create_engine_for_settings(settings)
        with engine.connect() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS _local_verify (id INTEGER)"
            )
            connection.exec_driver_sql("DROP TABLE _local_verify")
            connection.commit()
        engine.dispose()
        checks.append(Check("SQLite", True, True, "接続・書き込み可能"))
    except Exception:
        checks.append(Check("SQLite", True, False, "接続または書き込みに失敗"))

    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "current"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        checks.append(Check("Alembic", True, result.returncode == 0, "current確認"))
    except (OSError, subprocess.SubprocessError):
        checks.append(Check("Alembic", True, False, "実行できません"))

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

    runner = RcloneRunner()
    try:
        result = runner.version()
        detail = result.stdout.splitlines()[0] if result.stdout else "実行失敗"
        checks.append(Check("rclone", False, result.returncode == 0, detail))
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
