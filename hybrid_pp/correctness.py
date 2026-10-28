"""Check that pipeline parallelism does not change model outputs.

Starts a PP=1 server and a PP=N server one after the other, sends the same
prompts with greedy decoding one at a time, and compares the completions.

    python -m hybrid_pp.correctness --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        --pp 2 --out results/correctness

Use --skip-reference to reuse an existing reference.json (e.g. when the
reference model needs more GPUs than one). --fix selects which of our
NemotronH PP fixes are active on the PP side (docs/findings.md):
  full         all fixes (default)
  layers_only  bug 3 and 1 fixed, bug 2 kept: starts, but outputs differ
  no_forward   only bug 3 fixed: fails in the startup profile run (bug 1)
  none         upstream: fails while loading weights (bug 3)
"""

import argparse
import json
import os
import re
import sys
import urllib.request

from hybrid_pp.server import Server, serve_command

PROMPTS = [
    "The capital of France is",
    "Explain the difference between a process and a thread.",
    "Write a Python function that checks whether a number is prime.",
    "Translate into Chinese: Pipeline parallelism splits a model by layers.",
    "List the planets of the solar system in order from the sun.",
    "What is 37 * 43? Show your steps.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "def fibonacci(n):",
    "A state space model differs from attention because",
    # Long prompt: exercises multi-chunk mamba prefill and many KV blocks.
    "Repeat after me. " + "The quick brown fox jumps over the lazy dog. " * 300
    + "\nNow say what animal jumped:",
]


FIX_ENV = {
    "full": {},
    "layers_only": {"HYBRID_PP_FIX": "layers_only"},
    "no_forward": {"HYBRID_PP_FIX": "0"},
    "none": {"HYBRID_PP_FIX": "0", "HYBRID_PP_FIX_LOAD": "0"},
}


def last_error(log_path: str) -> str:
    """Last exception line in a server log, e.g. 'KeyError: ...'."""
    err = ""
    with open(log_path, errors="replace") as f:
        for line in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = re.search(r"\b(\w+(Error|Exception): .*)", line)
            if m:
                err = m.group(1).strip()
    return err


def complete(port: int, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


def run(args, pp: int, out_path: str, env_extra: dict) -> list[str]:
    env = dict(os.environ, VLLM_USE_V1="0", **env_extra)
    cmd = serve_command(args.model, args.port, pp, 1, args.max_model_len,
                        args.max_num_seqs, args.gpu_memory_utilization,
                        args.eager, [])
    log = out_path.replace(".json", ".server.log")
    with Server(cmd, env, log, args.port, timeout_s=args.server_timeout):
        outputs = [complete(args.port, args.model, p, args.max_tokens)
                   for p in PROMPTS]
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    return outputs


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--partition", help="VLLM_PP_LAYER_PARTITION for the PP run")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="results/correctness")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--fix", choices=list(FIX_ENV), default="full",
                    help="which hybrid_pp fixes the PP run uses")
    ap.add_argument("--tag", help="output name of the PP run "
                    "(default: pp<N>[_<partition>])")
    ap.add_argument("--server-timeout", type=int, default=600,
                    help="seconds to wait for a server to become healthy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref_path = os.path.join(args.out, "reference.json")
    if args.skip_reference:
        with open(ref_path) as f:
            ref = json.load(f)
    else:
        print("== PP=1 reference run", flush=True)
        ref = run(args, 1, ref_path, {})

    env_extra = dict(FIX_ENV[args.fix])
    if args.partition:
        env_extra["VLLM_PP_LAYER_PARTITION"] = args.partition
    tag = args.tag or (f"pp{args.pp}" + (
        f"_{args.partition.replace(',', '-')}" if args.partition else ""))
    out_path = os.path.join(args.out, f"{tag}.json")
    print(f"== PP={args.pp} run (fix={args.fix})", flush=True)
    try:
        got = run(args, args.pp, out_path, env_extra)
    except (RuntimeError, TimeoutError) as e:
        print(f"FAIL: PP={args.pp} server did not start: {e}")
        print("  last error: "
              + last_error(out_path.replace(".json", ".server.log")))
        sys.exit(1)

    mismatches = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        if a == b:
            continue
        mismatches += 1
        n = common_prefix(a, b)
        print(f"[{i}] differs after {n} chars\n  PP=1: {a[n:n + 80]!r}\n"
              f"  PP={args.pp}: {b[n:n + 80]!r}")
    print(f"{len(ref) - mismatches}/{len(ref)} prompts identical")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
