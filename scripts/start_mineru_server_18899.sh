#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

HOST="${MINERU_HOST:-0.0.0.0}"
PORT="${MINERU_PORT:-18899}"
WORKERS="${MINERU_WORKERS:-1}"
MAX_JOBS="${MINERU_MAX_JOBS:-1}"

OUTPUT_DIR="${MINERU_OUTPUT_DIR:-$ROOT_DIR/outputs/mineru_outputs}"
TEMP_DIR="${MINERU_TEMP_DIR:-$ROOT_DIR/outputs/mineru_temp}"

# Best-effort load .env if present. tmux sessions often don't inherit your shell env.
# This keeps secrets local: we only export to this process environment.
if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_CHAT_MODEL:-}" ]]; then
  for env_file in "$ROOT_DIR/.env" "$REPO_ROOT/.env"; do
    if [[ -f "$env_file" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" != *"="* ]] && continue
        key="${line%%=*}"
        val="${line#*=}"
        key="$(echo "$key" | tr -d '[:space:]')"
        val="$(echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
        [[ -z "$key" ]] && continue
        if [[ -z "${!key:-}" ]]; then
          export "$key=$val"
        fi
      done <"$env_file"
      break
    fi
  done
fi

# Captioning config (OpenAI-compatible multimodal endpoint).
# Defaults:
# - caption_mode: content_list_then_llm
# - caption_max_images: 0 (no limit)
# Note: LLM captions require a multimodal chat endpoint (base_url/api_key/model).
CAPTION_MODE="${MINERU_CAPTION_MODE:-content_list_then_llm}" # off|content_list|llm|content_list_then_llm
# Allow both "MINERU_CHAT_*" (recommended) and "CHAT_*" (server-native) and fall back to OPENAI_*.
CHAT_API_BASE_URL="${MINERU_CHAT_API_BASE_URL:-${CHAT_API_BASE_URL:-${OPENAI_BASE_URL:-}}}"
CHAT_API_KEY="${MINERU_CHAT_API_KEY:-${CHAT_API_KEY:-${OPENAI_API_KEY:-}}}"
CHAT_API_KEY_FILE="${MINERU_CHAT_API_KEY_FILE:-}"
CHAT_MODEL="${MINERU_CHAT_MODEL:-${CHAT_MODEL:-${OPENAI_CHAT_MODEL:-}}}"
CHAT_TIMEOUT_S="${MINERU_CHAT_TIMEOUT_S:-60}"
# <=0 means "no limit" on images to caption with LLM per task.
CAPTION_MAX_IMAGES="${MINERU_CAPTION_MAX_IMAGES:-0}"

# Page screenshots + block crops (used by dataset_gen --read-with-images).
# Defaults: enabled.
DUMP_PAGE_SCREENSHOTS="${MINERU_DUMP_PAGE_SCREENSHOTS:-1}"
DUMP_BLOCK_CROPS="${MINERU_DUMP_BLOCK_CROPS:-1}"
# Also crop image blocks (not only tables/figures). Default: enabled.
CROP_IMAGES="${MINERU_CROP_IMAGES:-1}"

mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

# Optional autoscaling of uvicorn workers based on free GPU memory.
# Note: each worker is a separate process and may load its own GPU-resident models.
# Tune MINERU_WORKER_VRAM_GB / MINERU_WORKERS_MAX for your environment.
if [[ "${WORKERS}" == "auto" || "${WORKERS}" == "0" ]]; then
  WORKERS="$(python3 - <<'PY'
import os
import math

workers_max = int(os.environ.get("MINERU_WORKERS_MAX", "4"))
per_worker_gb = float(os.environ.get("MINERU_WORKER_VRAM_GB", "18"))
per_worker_gb = max(1.0, per_worker_gb)

def guess() -> int:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return 1
        free_b, _total_b = torch.cuda.mem_get_info()
        free_gb = float(free_b) / (1024**3)
        n = int(math.floor(free_gb / per_worker_gb))
        return max(1, min(workers_max, n))
    except Exception:
        return 1

print(guess())
PY
)"
fi

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
        "Install: python3 -m pip install -r mineru/requirements.txt\n"
        % ", ".join(missing)
    )
    raise SystemExit(2)
PY

# If screenshots/crops are enabled, make sure optional deps exist (fail fast with a clear message).
python3 - <<PY >/dev/null
import importlib
import os
import sys

need = []
if os.environ.get("MINERU_DUMP_PAGE_SCREENSHOTS", "1") == "1":
    need.append(("fitz", "PyMuPDF"))
if os.environ.get("MINERU_DUMP_BLOCK_CROPS", "1") == "1" or os.environ.get("MINERU_CROP_IMAGES", "1") == "1":
    need.append(("PIL", "Pillow"))

missing = []
for mod, pkg in need:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    sys.stderr.write(
        "Missing optional MinerU deps for screenshots/crops: %s\n"
        "Install: python3 -m pip install -r mineru/requirements.txt\n"
        % ", ".join(sorted(set(missing)))
    )
    raise SystemExit(2)
PY

echo "Starting MinerU server on http://${HOST}:${PORT}"
echo "output_dir=${OUTPUT_DIR}"
echo "temp_dir=${TEMP_DIR}"
echo "caption_mode=${CAPTION_MODE}"
echo "Health check: curl http://127.0.0.1:${PORT}/health"

# caption-max-images <= 0 means "no limit" (when captioning is enabled).
exec python3 "$ROOT_DIR/mineru/mineru_main.py" server \
  --host "$HOST" --port "$PORT" \
  --workers "$WORKERS" --max-jobs "$MAX_JOBS" \
  --output-dir "$OUTPUT_DIR" --temp-dir "$TEMP_DIR" \
  --caption-mode "$CAPTION_MODE" \
  --caption-max-images "$CAPTION_MAX_IMAGES" \
  ${CHAT_API_BASE_URL:+--chat-api-base-url "$CHAT_API_BASE_URL"} \
  ${CHAT_API_KEY:+--chat-api-key "$CHAT_API_KEY"} \
  ${CHAT_API_KEY_FILE:+--chat-api-key-file "$CHAT_API_KEY_FILE"} \
  ${CHAT_MODEL:+--chat-model "$CHAT_MODEL"} \
  --chat-timeout-s "$CHAT_TIMEOUT_S" \
  $([ "$DUMP_PAGE_SCREENSHOTS" = "1" ] && echo --dump-page-screenshots) \
  $([ "$DUMP_BLOCK_CROPS" = "1" ] && echo --dump-block-crops) \
  $([ "$CROP_IMAGES" = "1" ] && echo --crop-images)
