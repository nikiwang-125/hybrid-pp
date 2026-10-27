"""Run a grid of (pipeline partition x workload) experiments.

For every partition a fresh ``vllm serve`` is started (V0 engine, our plugin
active), each workload is driven with ``vllm bench serve`` on the random
dataset, and the server is stopped. With --profile the server is restarted
per workload so every profile directory holds exactly one workload.

    python -m hybrid_pp.runner --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        --pp 2 --partitions 28,28 24,32 32,24 \
        --workloads 1024:256 8192:256 32768:128 \
        --num-prompts 200 --concurrency 32 --out results/sweep_pp2

Outputs, per run tag ``<partition>/<workload>``:
  server.log          full server log (input to parse_stage_memory.py)
  bench.json          vllm bench serve result (throughput, TTFT, TPOT, ...)
  profile/*.jsonl     with --profile, per-worker step/layer traces
"""

import argparse
import json
import os
import subprocess
import sys

from hybrid_pp.server import Server, serve_command


def bench_command(args, port: int, in_len: int, out_len: int, result_dir: str):
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", args.model,
        "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", str(args.concurrency),
        "--ignore-eos",
        "--trust-remote-code",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", "bench.json",
    ]


def parse_workload(text: str) -> tuple[int, int]:
    in_len, out_len = text.split(":")
    return int(in_len), int(out_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--partitions", nargs="*", default=["default"],
                    help="comma lists for VLLM_PP_LAYER_PARTITION, or "
                    "'default' for vLLM's own split")
    ap.add_argument("--workloads", nargs="+", required=True,
                    help="INPUT_LEN:OUTPUT_LEN, e.g. 8192:256")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--profile", action="store_true",
                    help="enable the hybrid_pp profiler (implies "
                    "--enforce-eager and one server per workload)")
    ap.add_argument("--eager", action="store_true",
                    help="--enforce-eager without profiling")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--serve-arg", action="append", default=[],
                    help="extra argument for vllm serve (repeatable), e.g. "
                    "--serve-arg=--no-enable-chunked-prefill")
    args = ap.parse_args()

    workloads = [parse_workload(w) for w in args.workloads]
    eager = args.eager or args.profile

    for partition in args.partitions:
        p_tag = "default" if partition == "default" else partition.replace(
            ",", "-")
        env = dict(os.environ, VLLM_USE_V1="0")
        if partition != "default":
            env["VLLM_PP_LAYER_PARTITION"] = partition
        # One server for all workloads, unless profiles must stay separate.
        groups = [[w] for w in workloads] if args.profile else [workloads]

        for group in groups:
            run_dir = os.path.join(args.out, p_tag)
            if args.profile:
                run_dir = os.path.join(run_dir, "%d-%d" % group[0])
                env["HYBRID_PP_PROFILE_DIR"] = os.path.join(run_dir, "profile")
            cmd = serve_command(args.model, args.port, args.pp, args.tp,
                                args.max_model_len, args.max_num_seqs,
                                args.gpu_memory_utilization, eager,
                                args.serve_arg)
            print(f"== partition {partition}: starting server", flush=True)
            try:
                with Server(cmd, env, os.path.join(run_dir, "server.log"),
                            args.port):
                    for in_len, out_len in group:
                        result_dir = os.path.join(args.out, p_tag,
                                                  f"{in_len}-{out_len}")
                        os.makedirs(result_dir, exist_ok=True)
                        print(f"   workload {in_len}:{out_len}", flush=True)
                        subprocess.run(bench_command(args, args.port, in_len,
                                                     out_len, result_dir),
                                       check=True)
            except (RuntimeError, TimeoutError,
                    subprocess.CalledProcessError) as e:
                # Keep sweeping; a failed partition is a data point too.
                print(f"!! partition {partition} failed: {e}", file=sys.stderr)
                with open(os.path.join(run_dir, "FAILED"), "w") as f:
                    f.write(str(e) + "\n")

    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
