#!/usr/bin/env bash
# First session on a rented 2-GPU box: verify that everything works end to end.
# Run after scripts/setup_env.sh, from the repo root:
#   bash scripts/day1.sh 2>&1 | tee logs/day1.log
# Each step can be re-run alone; read the comment above it.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/day1}
mkdir -p "$OUT" logs

step() { echo; echo "################ $*"; }

step "1. environment"
bash scripts/check_env.sh

step "2. static stage analysis (no GPU)"
python -m hybrid_pp.model_info "$MODEL" --pp 2
python -m hybrid_pp.model_info "$MODEL" --pp 4

step "3. upstream behaviour: PP=2 WITHOUT our forward fix (expected to fail)"
# Confirms docs/findings.md bug 1/2 on real hardware. Failure here is fine.
python -m hybrid_pp.correctness --model "$MODEL" --pp 2 \
  --out "$OUT/correctness_nofix" --no-plugin-fix \
  || echo ">> as expected: upstream PP=2 fails or differs (see $OUT/correctness_nofix)"

step "4. PP=2 with fix vs PP=1: outputs must be identical"
python -m hybrid_pp.correctness --model "$MODEL" --pp 2 \
  --out "$OUT/correctness" \
  && echo ">> PASS" || echo ">> FAIL: look at $OUT/correctness/*.server.log"

step "5. uneven partition still correct"
python -m hybrid_pp.correctness --model "$MODEL" --pp 2 --partition 20,36 \
  --out "$OUT/correctness" --skip-reference \
  && echo ">> PASS" || echo ">> FAIL"

step "6. small sweep: 3 partitions x 2 workloads (throughput, per-stage memory)"
python -m hybrid_pp.runner --model "$MODEL" --pp 2 \
  --partitions default 20,36 36,20 \
  --workloads 1024:256 16384:128 \
  --num-prompts 64 --concurrency 16 --out "$OUT/sweep"
python -m hybrid_pp.analyze_memory "$OUT"/sweep/*/server.log

step "7. profiled run: per-layer cost + stage balance"
python -m hybrid_pp.runner --model "$MODEL" --pp 2 --profile \
  --workloads 1024:128 8192:128 \
  --num-prompts 32 --concurrency 8 --out "$OUT/profile"
python -m hybrid_pp.analyze_profile "$OUT"/profile/default/*/profile \
  --out "$OUT/profile/summary"

echo; echo "done. Bring back: logs/day1.log and $OUT/ (tar czf day1.tgz $OUT logs)"
