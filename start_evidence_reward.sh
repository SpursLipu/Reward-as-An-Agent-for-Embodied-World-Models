#!/usr/bin/env bash
set -euo pipefail
# Full service entrypoint: WMReward and tool-grounded Reflection are required.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REWARD_AS_AGENT_PORT="${REWARD_AS_AGENT_PORT:-17027}"
export REWARD_AS_AGENT_MAX_TOKENS="${REWARD_AS_AGENT_MAX_TOKENS:-8192}"
export REWARD_AS_AGENT_LLM_TIMEOUT="${REWARD_AS_AGENT_LLM_TIMEOUT:-240}"
export REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP="${REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP:-4}"
exec bash "${SCRIPT_DIR}/start_doubao_reward.sh"
