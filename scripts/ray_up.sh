#!/usr/bin/env bash
# Join slave20 + slave21 into one Ray cluster for PP=4. Run inside the
# container from docker_run.sh on each machine, head first:
#   slave21$ bash scripts/ray_up.sh head
#   slave20$ bash scripts/ray_up.sh worker
#   slave21$ ray status    # expect 4.0 GPU
# Then on the head:
#   PP=4 bash scripts/serve.sh --distributed-executor-backend ray
# Both machines need the same repo commit and the same models in hf_cache.
set -euo pipefail

HEAD_IP=${HEAD_IP:-192.168.0.21}
RAY_PORT=${RAY_PORT:-6379}

case "${1:-}" in
  head)
    ray start --head --port "$RAY_PORT" --node-ip-address "$VLLM_HOST_IP" \
      --num-gpus "$(nvidia-smi -L | wc -l)" ;;
  worker)
    ray start --address "$HEAD_IP:$RAY_PORT" --node-ip-address "$VLLM_HOST_IP" \
      --num-gpus "$(nvidia-smi -L | wc -l)" ;;
  down)
    ray stop ;;
  *)
    echo "usage: $0 head|worker|down" >&2; exit 1 ;;
esac
