#!/usr/bin/env bash
# PP=4 per-layer cost vs batch size and context length, plus memory per
# config. Eager mode so decode steps get per-layer timings too.
# Run inside the container on the Ray head, after scripts/pp4.sh:
#   bash scripts/pp4_layers.sh 2>&1 | tee logs/pp4_layers.log
# max-num-seqs follows the concurrency, so each server's startup stage_mem
# line shows how weights / activations / KV / mamba state shift with batch.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/pp4_layers}
PARTITION=${PARTITION:-default}
CONCURRENCY=${CONCURRENCY:-"1 4 16 32"}
WORKLOADS=${WORKLOADS:-"1024:128 4096:128 16384:128 32768:64"}
mkdir -p "$OUT" logs

for c in $CONCURRENCY; do
  n=$((c * 4)); [ "$n" -lt 8 ] && n=8
  echo; echo "################ concurrency $c (max-num-seqs $c, $n prompts)"
  python3 -m hybrid_pp.runner --model "$MODEL" --pp 4 --profile --eager \
    --partitions "$PARTITION" --workloads $WORKLOADS \
    --num-prompts "$n" --concurrency "$c" --max-num-seqs "$c" \
    --max-model-len 40960 \
    --out "$OUT/c$c"
done

for c in $CONCURRENCY; do
  echo; echo "################ summary, concurrency $c"
  python3 -m hybrid_pp.analyze_profile "$OUT"/c$c/*/*/profile \
    --out "$OUT/c$c/summary"
done
python3 -m hybrid_pp.analyze_memory "$OUT"/c*/*/*/server.log
echo; echo "done: $OUT"
