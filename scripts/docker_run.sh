#!/usr/bin/env bash
# Dev container on the shared lab boxes (slave20/slave21).
#   bash scripts/docker_run.sh build   # build the image once per machine
#   bash scripts/docker_run.sh         # start the container if needed, open a shell
#   bash scripts/docker_run.sh stop    # remove the container (code and models stay on the host)
# Other people use these GPUs: we refuse to start if any GPU process is
# running; FORCE=1 overrides once the owner has agreed.
set -euo pipefail

# DOCKER="sudo docker" when this account is not in the docker group; the
# container still runs as you, since id -u is evaluated outside sudo.
DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-niki}
NAME=${NAME:-nikihpp}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK_ROOT=$(dirname "$REPO_DIR")
HF_CACHE=${HF_CACHE:-$WORK_ROOT/hf_cache}
CHOME=${CHOME:-$WORK_ROOT/container_home}

case "${1:-shell}" in
  build)
    $DOCKER build -t "$IMAGE" "$REPO_DIR/docker"
    exit ;;
  stop)
    $DOCKER rm -f "$NAME"
    exit ;;
esac

busy=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader)
if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
  echo "GPUs are in use by someone else; ask before running:" >&2
  echo "$busy" >&2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2
  exit 1
fi

if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$NAME"; then
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  mkdir -p "$HF_CACHE" "$CHOME"
  # NCCL, Gloo and Ray must use the 192.168.0.x LAN, not docker0 (172.17.x).
  LAN_IF=$(ip -o -4 addr show | awk '$4 ~ /^192\.168\.0\./ {print $2; exit}')
  LAN_IP=$(ip -o -4 addr show | awk '$4 ~ /^192\.168\.0\./ {split($4, a, "/"); print a[1]; exit}')
  # Run as the host user so files written to the mounted dirs stay ours.
  # --runtime=nvidia rather than --gpus: a `systemctl reload docker` picks up
  # the runtime, while --gpus needs a docker restart that would bounce other
  # users' containers.
  $DOCKER run -d --name "$NAME" \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    --ipc=host --network host --shm-size 16g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/niki -e USER=niki -e LOGNAME=niki \
    -e NCCL_SOCKET_IFNAME="$LAN_IF" -e GLOO_SOCKET_IFNAME="$LAN_IF" \
    -e VLLM_HOST_IP="$LAN_IP" \
    -v "$REPO_DIR":/workspace/hybrid-pp \
    -v "$HF_CACHE":/hf_cache \
    -v "$CHOME":/home/niki \
    "$IMAGE" sleep infinity >/dev/null
  echo "started $NAME (LAN $LAN_IF $LAN_IP)"
fi
exec $DOCKER exec -it "$NAME" bash
