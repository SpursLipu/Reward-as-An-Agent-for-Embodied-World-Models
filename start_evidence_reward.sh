#!/usr/bin/env bash
set -euo pipefail
# Development-only service entrypoint.  It does not inject the required
# WMReward hook used by the frozen v4.1 evaluation protocol; use
# scripts/frozen_v41/evaluate_frozen_v41.py for protocol-faithful offline evaluation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REWARD_AS_AGENT_PORT="${REWARD_AS_AGENT_PORT:-17027}"
export REWARD_AS_AGENT_MAX_TOKENS="${REWARD_AS_AGENT_MAX_TOKENS:-8192}"
export REWARD_AS_AGENT_LLM_TIMEOUT="${REWARD_AS_AGENT_LLM_TIMEOUT:-240}"
export REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP="${REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP:-4}"
exec bash "${SCRIPT_DIR}/start_doubao_reward.sh"
