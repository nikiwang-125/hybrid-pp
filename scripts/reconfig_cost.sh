#!/usr/bin/env bash
# Baseline cost of one layout switch (free KV -> load layers -> rebuild KV
# -> recapture graphs -> move in-flight state). Run inside the container:
#   bash scripts/reconfig_cost.sh 2>&1 | tee logs/reconfig.log
# Uses GPU 0 (and GPU 1 for the GPU-to-GPU state copy). ~10 minutes.
set -uo pipefail

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}
OUT=${OUT:-results/reconfig}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
mkdir -p "$OUT" logs

busy=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader)
if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
  echo "GPUs are in use by someone else; ask before running:" >&2
  echo "$busy" >&2
  exit 1
fi

echo "################ 1. micro: free KV / load layers / rebuild KV / move state"
python -m hybrid_pp.reconfig_cost micro --model "$MODEL" --out "$OUT/micro.json"

echo
echo "################ 2. vLLM sleep mode + engine start-up phases (PP=1, GPU 0)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} \
  python -m hybrid_pp.reconfig_cost sleep --model "$MODEL" --out "$OUT/sleep.json" \
  > "$OUT/sleep_engine.log" 2>&1
status=$?
grep -E "round [0-9]|wrote|Error|error" "$OUT/sleep_engine.log" | tail -n 20
echo "-- start-up phases (a full restart = upper bound of one switch)"
grep -E "Loading weights took|Model loading took|Memory profiling takes|# (CUDA|cuda) blocks|Graph capturing finished|init engine|KV Cache is|Sleep mode freed|It took" \
  "$OUT/sleep_engine.log" | sed 's/^.*\] //' | tail -n 20
[ $status -ne 0 ] && echo "sleep run failed (exit $status); see $OUT/sleep_engine.log"
echo
echo "done: $OUT/micro.json $OUT/sleep.json $OUT/sleep_engine.log"
