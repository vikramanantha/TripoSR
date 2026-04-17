#!/usr/bin/env bash
# train_sdf.sh  —  Run train_sdf_head.py
#
#   ./train_sdf.sh                  # precompute then train (default)
#   ./train_sdf.sh --precompute    # precompute only
#   ./train_sdf.sh --train         # train only (dataset must exist)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
PY_SCRIPT="$SCRIPT_DIR/train_sdf_head.py"

# ── Parse optional flag ──────────────────────────────────────────────────────
MODE=""
for arg in "$@"; do
    case "$arg" in
        --precompute) MODE="precompute" ;;
        --train)      MODE="train" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [[ -n "$MODE" ]]; then
    echo "Overriding COMMAND -> $MODE"
    sed -i "s/^COMMAND = .*/COMMAND = \"$MODE\"/" "$PY_SCRIPT"
fi

COMMAND=$(grep -oP '^COMMAND\s*=\s*"\K[^"]+' "$PY_SCRIPT")
echo "COMMAND = $COMMAND"

"$PYTHON" "$PY_SCRIPT"
