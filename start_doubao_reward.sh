#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
exec "${REWARD_PYTHON:-python3}" scripts/serve_doubao.py
