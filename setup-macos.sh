#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3.11}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.11 tidak ditemukan. Install via: conda create -p $ROOT/.venv python=3.11 -y"
  exit 1
fi

if [ ! -x "$ROOT/.venv/bin/python3.11" ]; then
  echo "Membuat environment conda di .venv ..."
  conda create -p "$ROOT/.venv" python=3.11 ffmpeg libsndfile -y -c conda-forge
fi

PY="$ROOT/.venv/bin/python3.11"
PIP="$PY -m pip"

echo "Menginstall TensorFlow + Spleeter ..."
$PIP install -r "$ROOT/requirements-macos.txt" --no-deps
$PIP install -r "$ROOT/requirements.txt"

echo "Selesai. Aktivasi:"
echo "  conda activate $ROOT/.venv"
echo "  python run.py inputs/lagu.mp3"
