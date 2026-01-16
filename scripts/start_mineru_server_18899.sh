#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${MINERU_HOST:-0.0.0.0}"
PORT="${MINERU_PORT:-18899}"
WORKERS="${MINERU_WORKERS:-1}"
MAX_JOBS="${MINERU_MAX_JOBS:-1}"

OUTPUT_DIR="${MINERU_OUTPUT_DIR:-$ROOT_DIR/outputs/mineru_outputs}"
TEMP_DIR="${MINERU_TEMP_DIR:-$ROOT_DIR/outputs/mineru_temp}"

mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

python3 - <<'PY' >/dev/null
import importlib
import sys

mods = ["fastapi", "uvicorn", "pydantic", "loguru", "requests", "multipart"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    sys.stderr.write(
        "Missing MinerU server Python deps: %s\n"
        "Install (example): python3 -m pip install fastapi uvicorn python-multipart pydantic loguru requests\n"
        % ", ".join(missing)
    )
    raise SystemExit(2)
PY

echo "Starting MinerU server on http://${HOST}:${PORT}"
echo "output_dir=${OUTPUT_DIR}"
echo "temp_dir=${TEMP_DIR}"
echo "Health check: curl http://127.0.0.1:${PORT}/health"

# caption_mode is off by default for cost/stability; images are still extracted.
# caption-max-images <= 0 means "no limit" (when captioning is enabled).
exec python3 "$ROOT_DIR/mineru/mineru_main.py" server \
  --host "$HOST" --port "$PORT" \
  --workers "$WORKERS" --max-jobs "$MAX_JOBS" \
  --output-dir "$OUTPUT_DIR" --temp-dir "$TEMP_DIR" \
  --caption-mode off \
  --caption-max-images 0

