#!/bin/bash
set -euo pipefail

ulimit -n 65535

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

mkdir -p "${REWARD_AS_AGENT_LOG_ROOT:-./runs}"
export REWARD_PHYSICS_COMPLETION=required
export REWARD_WMREWARD_PYTHON="${REWARD_WMREWARD_PYTHON:?Set WMReward CUDA interpreter path}"
export REWARD_WMREWARD_REPO="${REWARD_WMREWARD_REPO:?Set WMReward checkout path}"
export REWARD_WMREWARD_CHECKPOINT="${REWARD_WMREWARD_CHECKPOINT:?Set WMReward checkpoint path}"
export REWARD_WMREWARD_SHA256="${REWARD_WMREWARD_SHA256:?Set WMReward checkpoint SHA256}"
export REWARD_WMREWARD_DEVICE="${REWARD_WMREWARD_DEVICE:-cuda:0}"
nohup python -m reward_as_agent.cli serve > "${REWARD_AS_AGENT_LOG_ROOT:-./runs}/reward_as_agent.log" 2>&1 &
echo "Reward as An Agent started. Log: ${REWARD_AS_AGENT_LOG_ROOT:-./runs}/reward_as_agent.log"
