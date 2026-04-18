#!/usr/bin/env bash
# train_sdf.sh  —  Run train_sdf_head.py
#
#   ./train_sdf.sh                  # precompute then train (default)
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

sed_inplace() {
    # GNU sed accepts -i; BSD sed requires -i ''.
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

PYTHON="$(resolve_python)"

# ── Parse optional flag ──────────────────────────────────────────────────────
MODE=""
for arg in "$@"; do
    case "$arg" in
        --precompute) MODE="precompute" ;;
        --train)      MODE="train" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ -n "$MODE" ]]; then
    echo "Overriding COMMAND -> $MODE"
    sed_inplace "s/^COMMAND[[:space:]]*=[[:space:]]*\"[^\"]*\"/COMMAND = \"$MODE\"/" "$PY_SCRIPT"
fi

# Portable read of COMMAND = "..." (no grep -P)
COMMAND="$(awk -F'"' '/^COMMAND[[:space:]]*=/ {print $2; exit}' "$PY_SCRIPT")"
if [[ -z "$COMMAND" ]]; then
    echo "train_sdf.sh: could not parse COMMAND from $PY_SCRIPT" >&2
    exit 1
fi
echo "COMMAND = $COMMAND"

cd "$SCRIPT_DIR"
exec "$PYTHON" "$PY_SCRIPT"
