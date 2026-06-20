#!/usr/bin/env bash
# LOCAL DEVELOPMENT — Start the pipeline without Docker
#
# This runs the API server with LOCAL processing (no RabbitMQ/Celery needed).
# Inference runs directly in the API process.
#
# Prerequisites:
#   - Python venv with deps: pip install -r requirements.txt
#   - FFmpeg installed: brew install ffmpeg
#   - YOLO model at ./models/plate.pt
#   - PostgreSQL running: brew services start postgresql
#     (creates database 'plate_pipeline' with user 'postgres' password 'god123great')
#
# Optional (for caching/dedup):
#   - Redis running: brew services start redis
#
# Usage:
#   ./scripts/local_simple.sh [<mode>]           # e.g. direct_llm, yolo_ocr, yolo_only, ocr_only, yolo_ocr_llm
#   ./scripts/local_simple.sh [<mode>] --reload  # Start with auto-reload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Allow selecting a specific configuration mode
CONFIG_MODE="${1:-local}"
if [[ "$CONFIG_MODE" =~ ^(yolo_only|ocr_only|yolo_ocr|yolo_ocr_llm|direct_llm|local)$ ]]; then
    # Shift so subsequent args (e.g., --reload) are passed directly to uvicorn
    shift
else
    # Default back to standard config.local.yaml if no valid mode specified
    CONFIG_MODE="local"
fi

export PLATE_PIPELINE_CONFIG="$PROJECT_DIR/config/config.${CONFIG_MODE}.yaml"

echo "============================================"
echo "  PLATE PIPELINE — LOCAL MODE"
echo "============================================"
echo ""
echo "  Config  : $PLATE_PIPELINE_CONFIG"
echo "  API     : http://localhost:8000"
echo "  Docs    : http://localhost:8000/docs"
echo "  Metrics : http://localhost:8000/metrics"
echo ""

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

# Check PostgreSQL (for database persistence)
if ! command -v psql &> /dev/null; then
    echo "WARNING: PostgreSQL client not found. Install with: brew install postgresql"
    echo "         Database persistence will not work without it."
    echo ""
fi

# Check YOLO model
if [ ! -f "$PROJECT_DIR/models/plate.pt" ]; then
    echo "WARNING: No YOLO model at ./models/plate.pt"
    echo "         Detection will be skipped until you provide one."
    echo ""
fi

# Check virtual environment
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Using venv: $PROJECT_DIR/.venv"
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# Validate config loads
echo "Validating config..."
$PYTHON -c "
import os
os.environ['PLATE_PIPELINE_CONFIG'] = '$PLATE_PIPELINE_CONFIG'
from services.config import load_config
c = load_config()
print(f'  Mode     : {c.system.mode.value}')
print(f'  Inference: {c.inference.mode.value}')
print(f'  OCR      : {c.ocr.engine.value}')
print(f'  YOLO     : {c.model_registry.yolo.path}')
print(f'  Redis    : {c.redis.url}')
print()
"

# Pass extra args (like --reload) to uvicorn
EXTRA_ARGS=("$@")

echo "Starting API server..."
echo ""

if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    $PYTHON -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info
else
    $PYTHON -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        "${EXTRA_ARGS[@]}"
fi