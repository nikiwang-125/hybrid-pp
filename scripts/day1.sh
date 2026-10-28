#!/usr/bin/env bash
# First session on a rented 2-GPU box: verify that everything works end to end.
# Run after scripts/setup_env.sh, from the repo root:
#   bash scripts/day1.sh 2>&1 | tee logs/day1.log
# Each step can be re-run alone; read the comment above it.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/day1}
# 24GB cards (3090/4090): PP=1 holds all 9B weights (~17GB), keep the
# reference run small. Raise on 48GB+ cards.
CORR_SEQS=${CORR_SEQS:-4}
NGPU=$(nvidia-smi -L | wc -l)
CORR="python -m hybrid_pp.correctness --model $MODEL --max-num-seqs $CORR_SEQS --out $OUT/correctness"
mkdir -p "$OUT" logs

step() { echo; echo "################ $*"; }

step "1. environment"
bash scripts/check_env.sh

step "2. static stage analysis (no GPU)"
python -m hybrid_pp.model_info "$MODEL" --pp 2
python -m hybrid_pp.model_info "$MODEL" --pp 4

# Steps 3a-3c confirm docs/findings.md bugs 3, 1, 2 (bug 4 is fixed with 1) on real hardware. Each
# bug masks the next one, so each run fixes one more. Failure is expected.
step "3a. upstream PP=2 (expect KeyError while loading weights: bug 3)"
$CORR --pp 2 --fix none --tag pp2_upstream \
  || echo ">> as expected: upstream fails (bug 3)"

step "3b. only bug 3 fixed (expect 'too many values to unpack': bug 1)"
$CORR --pp 2 --fix no_forward --tag pp2_no_forward --skip-reference \
  || echo ">> as expected: fails (bug 1)"

step "3c. bugs 3+1+4 fixed, bug 2 kept (expect start OK, outputs differ early)"
$CORR --pp 2 --fix layers_only --tag pp2_layers_only --skip-reference \
  || echo ">> as expected: outputs differ (bug 2)"

step "4. PP=2 with all fixes vs PP=1: outputs must be identical"
$CORR --pp 2 --skip-reference \
  && echo ">> PASS" || echo ">> FAIL: look at $OUT/correctness/pp2.server.log"

step "5. uneven partition still correct"
$CORR --pp 2 --partition 20,36 --skip-reference \
  && echo ">> PASS" || echo ">> FAIL"

step "6. small sweep: 3 partitions x 2 workloads (throughput, per-stage memory)"
# Attention layers sit at 14/21/30/39: default 28,28 puts 2+2 on the stages
# (control), 20,36 puts 1+3 and 36,20 puts 3+1 (expect KV waste to swap).
python -m hybrid_pp.runner --model "$MODEL" --pp 2 \
  --partitions default 20,36 36,20 \
  --workloads 1024:256 16384:128 \
  --num-prompts 64 --concurrency 16 --out "$OUT/sweep"
python -m hybrid_pp.analyze_memory "$OUT"/sweep/*/server.log

step "7. profiled run: per-layer cost + stage balance"
# 36,20 puts 3 of the 4 attention layers on stage 0; its step time should
# pull away from stage 1 as the context grows.
python -m hybrid_pp.runner --model "$MODEL" --pp 2 --profile \
  --partitions default 36,20 \
  --workloads 1024:128 8192:128 16384:128 30720:64 \
  --num-prompts 32 --concurrency 8 --out "$OUT/profile"
for p in default 36-20; do
  python -m hybrid_pp.analyze_profile "$OUT"/profile/$p/*/profile \
    --out "$OUT/profile/summary_$p"
done

if [ "$NGPU" -ge 4 ]; then
  step "8. PP=4 even split: stages 0 and 3 have no attention layer"
  # Upstream should fail at startup (findings.md #3); with our fix it runs.
  HYBRID_PP_FIX_NOATTN=0 $CORR --pp 4 --tag pp4_noattn_unfixed \
    --skip-reference \
    || echo ">> as expected without the NOATTN fix: PP=4 fails"
  $CORR --pp 4 --skip-reference && echo ">> PASS" || echo ">> FAIL"
fi

echo; echo "done. Bring back: logs/day1.log and $OUT/ (tar czf day1.tgz $OUT logs)"
