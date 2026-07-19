from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    from runpod_lora_studio.environment import collect_environment_report

    report = collect_environment_report()
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
