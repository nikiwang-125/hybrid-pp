#!/usr/bin/env bash
# Print what matters about the box before running experiments.
set -uo pipefail

echo "== GPUs";        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "== topology";    nvidia-smi topo -m
echo "== CUDA (torch)"
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  {i}: {p.name} sm{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB")
EOF
echo "== packages"
python -c "import vllm; print('vllm', vllm.__version__)"
python - <<'EOF'
from importlib.metadata import entry_points
names = [e.name for e in entry_points(group="vllm.general_plugins")]
print("vllm.general_plugins:", names, "<- must include hybrid_pp")
EOF
echo "== memory / disk"; free -g | head -2; df -h . | tail -1
