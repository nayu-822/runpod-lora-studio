from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runpod_lora_studio.environment import EnvironmentReport

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main(
    argv: list[str] | None = None,
    report: EnvironmentReport | None = None,
) -> int:
    parser = ArgumentParser(description="RunPod LoRA Studioの実行環境を確認します。")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力します。")
    args = parser.parse_args(argv)

    if report is None:
        from runpod_lora_studio.config.settings import get_settings
        from runpod_lora_studio.environment import collect_environment_report

        report = collect_environment_report(get_settings())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"RunPod: {'はい' if report.is_runpod else 'いいえ'}")
        support = "対応" if report.python_supported else "非対応"
        print(f"Python: {report.python_version} ({support})")
        print(f"作業ディレクトリ: {report.runtime_dir}")
        print(f"GPU: {', '.join(gpu.name for gpu in report.gpus) or '未認識'}")
        free = report.disk_free_bytes or "不明"
        total = report.disk_total_bytes or "不明"
        print(f"ディスク: {free} bytes free / {total} bytes total")
        for warning in report.warnings:
            print(f"警告: {warning}")
        for error in report.errors:
            print(f"エラー: {error}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
