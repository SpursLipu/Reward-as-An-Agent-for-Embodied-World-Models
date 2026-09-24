#!/usr/bin/env bash
# Local experiment launcher; paths may be overridden without changing the Agent.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
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
    --tensor-parallel-size 2 --max-model-len "${QWEN_MAX_MODEL_LEN:-98304}" --max-num-seqs "${QWEN_MAX_NUM_SEQS:-16}" \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --limit-mm-per-prompt '{"image":128,"video":0}' --reasoning-parser qwen3
  )
  exec "${launch[@]}"
elif [[ "${1:-}" == agent ]]; then
  exec bash ./start_qwen_reward.sh
else
  echo 'Usage: bash scripts/debug_qwen38_local.sh vllm|agent' >&2
  exit 2
fi
