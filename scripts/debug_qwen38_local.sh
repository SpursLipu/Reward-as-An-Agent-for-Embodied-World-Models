#!/usr/bin/env bash
# Local experiment launcher; paths may be overridden without changing the Agent.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT=/increase_kairos_vepfs/increase/lipu/grpo
if [[ "${1:-}" == vllm ]]; then
  # Use a short writable path for IPC and JIT caches, not the full repository path.
  export TMPDIR="${QWEN_RUNTIME_DIR:-$(mktemp -d /dev/shm/qwen38.XXXXXX)}"
  mkdir -p "$TMPDIR"
  export VLLM_RPC_BASE_PATH="$TMPDIR"
  export TRITON_CACHE_DIR="$TMPDIR/triton" VLLM_CACHE_ROOT="$TMPDIR/vllm"
  export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
  export CUDA_VISIBLE_DEVICES="${QWEN_GPUS:-0,1}"
  launch=(
    "${VLLM_PYTHON:-$TASK_ROOT/paibench/environments/paibench_vllm_0_19_0/bin/python}" -m vllm.entrypoints.openai.api_server \
    --model "${QWEN_MODEL_PATH:-$TASK_ROOT/paibench/models/qwen/Qwen3.8-27B}" \
    --served-model-name qwen38-reward --host 127.0.0.1 --port "${QWEN_PORT:-8038}" \
    --tensor-parallel-size 2 --max-model-len "${QWEN_MAX_MODEL_LEN:-98304}" --max-num-seqs "${QWEN_MAX_NUM_SEQS:-2}" \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --limit-mm-per-prompt '{"image":128,"video":0}' --reasoning-parser qwen3
  )
  exec "${launch[@]}"
elif [[ "${1:-}" == agent ]]; then
  export REWARD_AS_AGENT_PROVIDER=openai REWARD_AS_AGENT_API_BASE=http://127.0.0.1:8038/v1
  export REWARD_AS_AGENT_API_KEY=EMPTY REWARD_AS_AGENT_MODEL=qwen38-reward
  export REWARD_AS_AGENT_HOST=127.0.0.1 REWARD_AS_AGENT_PORT=7038
  export REWARD_AS_AGENT_TEMPERATURE=0 REWARD_AS_AGENT_MAX_TOKENS=8192
  export REWARD_AS_AGENT_SAVE_INPUTS=false
  export REWARD_WMREWARD_WORKERS="${REWARD_WMREWARD_WORKERS:-1}"
  export REWARD_AS_AGENT_LLM_TIMEOUT=600 REWARD_AS_AGENT_CACHE_BYPASS=1
  export REWARD_TASK_CONTRACT_REGISTRY="" REWARD_GATE_POLICY=off REWARD_REQUIREMENT_AUDIT_MODE=joint
  export REWARD_EVIDENCE_FRAMES=32 REWARD_PHYSICS_COMPLETION=required
  export REWARD_EVIDENCE_TRACE_DIR=runs/qwen38_20260918/traces
  export REWARD_WMREWARD_PYTHON="${REWARD_WMREWARD_PYTHON:-$TASK_ROOT/envs/qworld_grpo_kairos31/bin/python}"
  export REWARD_WMREWARD_REPO="${REWARD_WMREWARD_REPO:-$TASK_ROOT/reward_agent_experiments_202609/physics_and_rollout_runs/upstream/WMReward}"
  export REWARD_WMREWARD_CHECKPOINT="${REWARD_WMREWARD_CHECKPOINT:-$TASK_ROOT/reward_agent_experiments_202609/physics_and_rollout_runs/models/wmreward/vitg.pt}"
  export REWARD_WMREWARD_SHA256=67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e
  export CUDA_VISIBLE_DEVICES="${REWARD_WMREWARD_CUDA_VISIBLE_DEVICES:-2}"
  export REWARD_WMREWARD_DEVICE="${REWARD_WMREWARD_DEVICE:-cuda:0}"
  exec "${AGENT_PYTHON:-$TASK_ROOT/envs/reward_as_agent/bin/python}" -m reward_as_agent.cli serve
else
  echo 'Usage: bash scripts/debug_qwen38_local.sh vllm|agent' >&2
  exit 2
fi
