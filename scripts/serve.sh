#!/usr/bin/env bash
# Start one vLLM server (V0 engine) for manual testing. Extra args pass through.
#   bash scripts/serve.sh
#   PP=2 PARTITION=24,32 bash scripts/serve.sh --enforce-eager
#   PROFILE_DIR=results/manual/profile bash scripts/serve.sh --enforce-eager
set -euo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
PP=${PP:-2}
TP=${TP:-1}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-32}
GPU_UTIL=${GPU_UTIL:-0.9}

export VLLM_USE_V1=0
[ -n "${PARTITION:-}" ] && export VLLM_PP_LAYER_PARTITION=$PARTITION
[ -n "${PROFILE_DIR:-}" ] && export HYBRID_PP_PROFILE_DIR=$PROFILE_DIR

mkdir -p logs
LOG=logs/serve_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"

vllm serve "$MODEL" \
  --port "$PORT" \
  --pipeline-parallel-size "$PP" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --mamba-ssm-cache-dtype float32 \
  --trust-remote-code \
  "$@" 2>&1 | tee "$LOG"
