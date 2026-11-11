#!/usr/bin/env python3
"""How CUDA graph capture time scales (2026-11-11).

Starts a PP server on the local GPUs with HYBRID_PP_CAPTURE_TIMING=1, reads
the per-stage capture time (ms), stops it. Varies max_num_seqs (V0 derives
the captured batch sizes from it: 1 -> [1], 4 -> [1,2,4], 16 -> [1,2,4,8,16],
32 -> [1,2,4,8,16,24,32]) on an uneven split, so one run shows capture time
against layers per stage and the sweep shows it against batch sizes.

    python3 scripts/graph_capture_cost.py --out results/graph_capture_20261111
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_pp.server import Server, serve_command  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
ap.add_argument("--pp", type=int, default=2)
ap.add_argument("--partition", default="14,42")
ap.add_argument("--max-num-seqs", type=int, nargs="+", default=[1, 4, 16, 32])
ap.add_argument("--max-model-len", type=int, default=16384)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = []
for n in a.max_num_seqs:
    log = os.path.join(a.out, f"seqs{n}.server.log")
    env = dict(os.environ, VLLM_USE_V1="0", HYBRID_PP_CAPTURE_TIMING="1",
               VLLM_PP_LAYER_PARTITION=a.partition)
    cmd = serve_command(a.model, a.port, a.pp, 1, a.max_model_len, n, 0.9, False,
                        ["--distributed-executor-backend", "mp",
                         f"--max-seq-len-to-capture={a.max_model_len}"])
    with Server(cmd, env, log, a.port, timeout_s=900):
        pass
    recs = [json.loads(m) for m in re.findall(r"capture_timing (\{.*\})", open(log).read())]
    for r in recs:
        r["max_num_seqs"] = n
        rows.append(r)
        print(json.dumps(r), flush=True)
json.dump(rows, open(os.path.join(a.out, "capture.json"), "w"), indent=1)
print("wrote", os.path.join(a.out, "capture.json"))
