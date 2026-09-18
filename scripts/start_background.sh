#!/bin/bash
set -euo pipefail

ulimit -n 65535

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

mkdir -p "${REWARD_AS_AGENT_LOG_ROOT:-./runs}"
nohup python -m reward_as_agent.cli serve > "${REWARD_AS_AGENT_LOG_ROOT:-./runs}/reward_as_agent.log" 2>&1 &
echo "Reward as An Agent started. Log: ${REWARD_AS_AGENT_LOG_ROOT:-./runs}/reward_as_agent.log"
