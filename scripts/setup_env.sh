#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-reward_as_agent}"

if command -v conda >/dev/null 2>&1; then
  conda env create -n "${ENV_NAME}" -f environment.yml || conda env update -n "${ENV_NAME}" -f environment.yml
  echo "Activate with: conda activate ${ENV_NAME}"
else
  python -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e .
  echo "Activated local virtualenv at .venv"
fi
