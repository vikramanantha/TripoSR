#!/usr/bin/env bash
# train_sdf.sh  —  Run train_sdf_head.py
#
#   ./train_sdf.sh                  # precompute then train on all GPUs (default)
#   ./train_sdf.sh --precompute     # precompute only
#   ./train_sdf.sh --train          # train only (dataset must exist)
#
# Python interpreter (first match wins):
#   1) TRAIN_SDF_PYTHON — if set and executable
#   2) $SCRIPT_DIR/.venv/bin/python — if present
#   3) python3 on PATH
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_SCRIPT="$SCRIPT_DIR/train_sdf_head.py"

resolve_python() {
    if [[ -n "${TRAIN_SDF_PYTHON:-}" && -x "${TRAIN_SDF_PYTHON}" ]]; then
        printf '%s\n' "${TRAIN_SDF_PYTHON}"
        return 0
    fi
    local venv_py="$SCRIPT_DIR/.venv/bin/python"
    if [[ -x "$venv_py" ]]; then
        printf '%s\n' "$venv_py"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    echo "train_sdf.sh: no Python found. Options:" >&2
    echo "  - Create $SCRIPT_DIR/.venv (recommended)" >&2
    echo "  - Or: export TRAIN_SDF_PYTHON=/path/to/python" >&2
    echo "  - Or: install python3 on PATH" >&2
    exit 1
}

PYTHON="$(resolve_python)"

# Suppress Python library warnings (FutureWarning from transformers/torch etc.)
# Set PYTHONWARNINGS=default before running to restore them.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# ── Parse flag ───────────────────────────────────────────────────────────────
MODE="both"
for arg in "$@"; do
    case "$arg" in
        --precompute) MODE="precompute" ;;
        --train)      MODE="train" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── GPU detection ─────────────────────────────────────────────────────────────
NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
PORT=$((10000 + RANDOM % 20000))

echo "Mode: $MODE | GPUs: $NGPUS | Port: $PORT"

# ── Launch: torchrun for multi-GPU, plain python for single-GPU ───────────────
# Uses "$PYTHON -m torch.distributed.run" so it always picks up the venv's torch.
run() {
    local cmd="$1"
    if [[ "$NGPUS" -gt 1 ]]; then
        "$PYTHON" -m torch.distributed.run \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="$NGPUS" \
            --master_port="$PORT" \
            "$PY_SCRIPT" --command "$cmd"
    else
        "$PYTHON" "$PY_SCRIPT" --command "$cmd"
    fi
}

cd "$SCRIPT_DIR"

case "$MODE" in
    precompute) "$PYTHON" "$PY_SCRIPT" --command precompute ;;
    train)      run train ;;
    both)       "$PYTHON" "$PY_SCRIPT" --command precompute && run train ;;
esac
