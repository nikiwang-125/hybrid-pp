#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU box.
#   bash scripts/setup_env.sh            # env + vllm + this repo + models
#   SKIP_MODELS=1 bash scripts/setup_env.sh
# In mainland China, export HF_ENDPOINT=https://hf-mirror.com first.
set -euo pipefail

ENV_NAME=${ENV_NAME:-vllm010}
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
MODELS=${MODELS:-"nvidia/NVIDIA-Nemotron-Nano-9B-v2 nvidia/Nemotron-H-8B-Base-8K"}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

if ! command -v conda >/dev/null; then
  echo "conda not found; install Miniconda first" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

# vllm 0.10.1.1 has no upper bound on transformers; 5.x breaks it.
pip install "vllm==${VLLM_VERSION}" "transformers>=4.55,<4.56"
pip install -e "${REPO_DIR}[analysis]"

# vLLM source at the same tag, for reading (not installed).
if [ ! -d "${REPO_DIR}/third_party/vllm" ]; then
  git clone --depth 1 --branch "v${VLLM_VERSION}" \
    https://github.com/vllm-project/vllm.git "${REPO_DIR}/third_party/vllm"
fi

if [ -z "${SKIP_MODELS:-}" ]; then
  # huggingface_hub >= 1.0 renamed huggingface-cli to hf.
  HF_CLI=$(command -v hf || command -v huggingface-cli)
  for m in $MODELS; do
    "$HF_CLI" download "$m"
  done
fi

python -c "import vllm, hybrid_pp; print('vllm', vllm.__version__, '| hybrid_pp ok')"
