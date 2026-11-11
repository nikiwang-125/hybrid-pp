#!/usr/bin/env bash
# PP=4 across slave21 + slave20 (Ray cluster from ray_up.sh), after the PP=4
# correctness check passed. Run inside the container on the Ray head:
#   bash scripts/pp4.sh 2>&1 | tee logs/pp4.log
# SKIP_SWEEP=1 runs the profile step only.
# Ranks on the other machine write their profile traces there (same path):
# rsync them back before analyze_profile.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/pp4}
RUN_SEQS=${RUN_SEQS:-16}
mkdir -p "$OUT" logs

step() { echo; echo "################ $*"; }

if [ -z "${SKIP_SWEEP:-}" ]; then
step "1. sweep: throughput + per-stage memory"
# Attention layers sit at 14/21/30/39. default 14x4 puts 0/2/2/0 on the
# stages, 18,10,10,18 puts 1/1/1/1, 16,12,12,16 puts 1/1/2/0.
python3 -m hybrid_pp.runner --model "$MODEL" --pp 4 \
  --partitions default 18,10,10,18 16,12,12,16 \
  --workloads 1024:256 16384:128 \
  --max-num-seqs "$RUN_SEQS" --num-prompts 64 --concurrency 16 \
  --out "$OUT/sweep"
python3 -m hybrid_pp.analyze_memory "$OUT"/sweep/*/server.log
fi

step "2. profile: per-stage step time and busy ratio"
python3 -m hybrid_pp.runner --model "$MODEL" --pp 4 --profile \
  --partitions default 18,10,10,18 \
  --workloads 1024:128 8192:128 16384:128 \
  --max-num-seqs "$RUN_SEQS" --num-prompts 32 --concurrency 8 \
  --out "$OUT/profile"
for p in default 18-10-10-18; do
  python3 -m hybrid_pp.analyze_profile "$OUT"/profile/$p/*/profile \
    --out "$OUT/profile/summary_$p"
done

echo; echo "done: $OUT"
