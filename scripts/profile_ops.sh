#!/usr/bin/env bash
# How long does every layer / lm_head / sampler take on each stage?
# Short-context workloads at several concurrencies, in two modes:
#   graph : default CUDA graphs -> real stage step times, lm_head, sampler,
#           per-layer times for prefill only
#   eager : --enforce-eager     -> per-layer times for decode too (includes
#           kernel launch overhead; compare stage totals with graph mode)
#   PARTITION=30,26 bash scripts/profile_ops.sh
set -euo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
PARTITION=${PARTITION:-default}
WORKLOADS=${WORKLOADS:-"512:256 2048:256"}
CONCURRENCY=${CONCURRENCY:-"1 8 32"}
OUT=${OUT:-results/ops}

for mode in graph eager; do
  extra=()
  [ "$mode" = eager ] && extra=(--eager)
  for c in $CONCURRENCY; do
    python -m hybrid_pp.runner --model "$MODEL" --pp 2 --profile "${extra[@]}" \
      --partitions "$PARTITION" --workloads $WORKLOADS \
      --num-prompts $((c * 8)) --concurrency "$c" --max-model-len 8192 \
      --out "$OUT/$mode/c$c"
  done
done

for mode in graph eager; do
  for c in $CONCURRENCY; do
    echo "################ $mode, concurrency $c"
    python -m hybrid_pp.analyze_profile "$OUT"/$mode/c$c/*/*/profile \
      --out "$OUT/$mode/c$c/summary"
  done
done
