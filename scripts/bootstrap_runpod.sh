#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python executable not found." >&2
    exit 1
  fi
fi

echo "Using Python executable: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e ".[dev]"
"$PYTHON_BIN" scripts/verify_environment.py

cat <<'EOF'

Bootstrap complete.
Start the app with:
  bash scripts/start.sh
EOF
