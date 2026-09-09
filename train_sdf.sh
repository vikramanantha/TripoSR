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
PY_SCRIPT="$SCRIPT_DIR/train_sdf_head_fast.py"

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

# ── Precompute restart prompt ────────────────────────────────────────────────
# run_precompute() only prompts "delete and restart?" when it is a SINGLE
# process — under torchrun it can't (4 ranks would each prompt), so it silently
# resumes instead. Resuming onto a dataset built with different settings is a
# real hazard: samples are only regenerated if FILES ARE MISSING, so a stale
# half of the dataset keeps its old point distribution and (pre-token-caching)
# missing image_tokens.pt, which silently forces the slow training path
# dataset-wide. So ask here, once, before launching any ranks.
prompt_precompute_restart() {
    local ds_dir samples_dir existing
    ds_dir=$("$PYTHON" - "$PY_SCRIPT" <<'PYEOF'
import re, sys, os
src = open(sys.argv[1]).read()
# DATASET_DIR is no longer a string literal - it is computed by
# _resolve_ws_frb_root() so one file works on the host and in the container.
# Exec just that config slice (only `os` is needed) and read the result; fall
# back to the old literal regex for scripts that still hardcode a path.
# Without this the regex silently returns "" and the existing-samples guard
# below is skipped entirely.
val = ""
try:
    i = src.index("_WS_FRB_ROOTS")
    j = src.index("OUTPUT_DIR", i)
    ns = {"os": os}
    exec(src[i:j], ns)
    val = ns.get("DATASET_DIR", "") or ""
except Exception:
    pass
if not val:
    m = re.search(r'^DATASET_DIR\s*=\s*"([^"]+)"', src, re.M)
    val = m.group(1) if m else ""
print(val)
PYEOF
)
    [[ -z "$ds_dir" ]] && return 0
    samples_dir="$ds_dir/samples"
    [[ -d "$samples_dir" ]] || return 0
    # readdir ONLY - do not use find here. find must know each entry's type, and
    # NFSv3 readdir returns DT_UNKNOWN, so it issues one stat() per entry: measured
    # 8+ minutes on 331k samples (and this dataset is headed for 1M) versus 0.58 s
    # for ls -U. The count is the only thing needed, so never stat.
    existing=$(ls -U "$samples_dir" 2>/dev/null | grep -v '^_tmp' | wc -l)
    [[ "$existing" -eq 0 ]] && return 0
    echo "Found $existing existing samples in $samples_dir."
    read -r -p "Delete all and restart? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        rm -rf "$samples_dir"
        echo "Deleted existing samples."
    else
        echo "Keeping existing samples (will skip already-done views)."
        echo "NOTE: mixing samples from different precompute settings is unsupported."
    fi
}

case "$MODE" in
    precompute) prompt_precompute_restart; run precompute ;;
    train)      run train ;;
    both)       prompt_precompute_restart; run precompute && run train ;;
esac
