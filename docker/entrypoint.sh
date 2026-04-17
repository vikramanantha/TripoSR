#!/usr/bin/env bash

set -euo pipefail

sudo ldconfig

PROJECT_DIR="${TRIPOSR_DIR:-$HOME/TripoSR}"
VENV_DIR="$PROJECT_DIR/.venv"
BOOTSTRAP_STAMP="$VENV_DIR/.tripo_bootstrap_v2"

bootstrap_project() {
  if [ ! -d "$PROJECT_DIR" ]; then
    return
  fi

  mkdir -p "$PROJECT_DIR"

  if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" -V >/dev/null 2>&1; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  export VIRTUAL_ENV="$VENV_DIR"
  export PATH="$VENV_DIR/bin:$PATH"
  cd "$PROJECT_DIR"

  if [ ! -f "$BOOTSTRAP_STAMP" ] || ! python -c "import onnxruntime, objaverse, pybullet, pyrender, rembg, torch" >/dev/null 2>&1; then
    python -m pip install --upgrade pip setuptools
    python -m pip install torch torchvision torchaudio
    python -m pip install -r requirements.txt
    python -m pip install onnxruntime objaverse pyrender pybullet PyOpenGL PyOpenGL-accelerate
    touch "$BOOTSTRAP_STAMP"
  fi
}

bootstrap_project

if [ -d "$PROJECT_DIR" ]; then
  export VIRTUAL_ENV="$VENV_DIR"
  export PATH="$VENV_DIR/bin:$PATH"
  cd "$PROJECT_DIR"
fi

# Always source the venv in new interactive shells
if [ ! -f "$HOME/.bashrc.d/venv.sh" ]; then
  mkdir -p "$HOME/.bashrc.d"
  cat > "$HOME/.bashrc.d/venv.sh" <<EOF
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:\$PATH"
EOF
  echo '[ -d "$HOME/.bashrc.d" ] && for f in "$HOME/.bashrc.d"/*.sh; do source "$f"; done' >> "$HOME/.bashrc"
fi

exec /bin/bash