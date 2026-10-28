#!/usr/bin/env bash
# Motivation data that fits on 2x24GB, after day1.sh passed:
#   bash scripts/day2.sh 2>&1 | tee logs/day2.log
# Step 1 is the figure-4 grid: does the best static partition change with
# the workload? Step 2 fills the profile point day1 lost to an OOM.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/day2}
RUN_SEQS=${RUN_SEQS:-16}
mkdir -p "$OUT" logs

step() { echo; echo "################ $*"; }

step "1. partition x workload grid (figure 4)"
# Attention layers sit at 14/21/30/39, so these splits put 1:3 ... 3:1 of
# them on the two stages. Workloads: short chat, long context,
# prefill-heavy, decode-heavy.
python -m hybrid_pp.runner --model "$MODEL" --pp 2 \
  --partitions 16,40 20,36 24,32 default 32,24 36,20 40,16 \
  --workloads 1024:256 16384:128 8192:16 512:1024 \
  --max-num-seqs "$RUN_SEQS" --num-prompts 64 --concurrency 16 \
  --out "$OUT/grid"
python -m hybrid_pp.analyze_memory "$OUT"/grid/*/server.log

step "2. profile: default split at 16K"
# day1 ran out of memory here by ~0.2 GiB on stage 1 (mamba prefill scratch).
python -m hybrid_pp.runner --model "$MODEL" --pp 2 --profile \
  --workloads 16384:128 --gpu-memory-utilization 0.85 \
  --max-num-seqs "$RUN_SEQS" --num-prompts 32 --concurrency 8 \
  --out "$OUT/profile"
python -m hybrid_pp.analyze_profile "$OUT"/profile/default/*/profile \
  --out "$OUT/profile/summary_default"

echo; echo "done: $OUT"
