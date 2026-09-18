#!/usr/bin/env bash
set -euo pipefail

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

export REWARD_PHYSICS_COMPLETION=required
export REWARD_WMREWARD_PYTHON="${REWARD_WMREWARD_PYTHON:?Set WMReward CUDA interpreter path}"
export REWARD_WMREWARD_REPO="${REWARD_WMREWARD_REPO:?Set WMReward checkout path}"
export REWARD_WMREWARD_CHECKPOINT="${REWARD_WMREWARD_CHECKPOINT:?Set WMReward checkpoint path}"
export REWARD_WMREWARD_SHA256="${REWARD_WMREWARD_SHA256:?Set WMReward checkpoint SHA256}"
export REWARD_WMREWARD_DEVICE="${REWARD_WMREWARD_DEVICE:-cuda:0}"

exec python -m reward_as_agent.cli serve
