#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DOUBAO_PROXY_SCRIPT:-/KAIROS_vepfs-2/KAIROS_vepfs/wuqiang/proxy.sh}" >/dev/null
if [[ -n "${DOUBAO_PROXY_URL:-}" ]]; then
  export http_proxy="${DOUBAO_PROXY_URL}" https_proxy="${DOUBAO_PROXY_URL}"
  export HTTP_PROXY="${DOUBAO_PROXY_URL}" HTTPS_PROXY="${DOUBAO_PROXY_URL}"
fi
export REWARD_AS_AGENT_PROVIDER=doubao
export REWARD_AS_AGENT_API_BASE="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export REWARD_AS_AGENT_MODEL="${ARK_VIDEO_MODEL:-doubao-seed-2-1-pro-260628}"
export REWARD_AS_AGENT_HOST="${REWARD_AS_AGENT_HOST:-0.0.0.0}"
export REWARD_AS_AGENT_PORT="${REWARD_AS_AGENT_PORT:-7024}"
export REWARD_AS_AGENT_DP_SIZE=1
export REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP="${REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP:-4}"
export REWARD_AS_AGENT_MAX_RETRIES="${REWARD_AS_AGENT_MAX_RETRIES:-2}"
export REWARD_AS_AGENT_MAX_TOKENS="${REWARD_AS_AGENT_MAX_TOKENS:-4096}"
export REWARD_AS_AGENT_LLM_TIMEOUT="${REWARD_AS_AGENT_LLM_TIMEOUT:-180}"
export REWARD_AS_AGENT_LOG_ROOT="${REWARD_AS_AGENT_LOG_ROOT:-/increase_kairos_vepfs/increase/lipu/grpo/reward_as_agent_runs/doubao}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
PYTHON="${REWARD_PYTHON:-/increase_kairos_vepfs/increase/lipu/grpo/envs/reward_as_agent/bin/python}"
cd "${SCRIPT_DIR}"
"${PYTHON}" -c 'from reward_as_agent.doubao import load_ark_key; load_ark_key(); print("Ark credential available")'
exec "${PYTHON}" -m reward_as_agent.cli serve
