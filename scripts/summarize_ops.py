"""Compact text summary of scripts/profile_ops.sh results, small enough to
copy by hand from a box without file transfer.

    python scripts/summarize_ops.py                 # default, 30-26, 32-24
    python scripts/summarize_ops.py default 31-25
    python scripts/summarize_ops.py --prefix ops2 28-28 29-27 30-26
"""

import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("partitions", nargs="*", default=["default", "30-26", "32-24"])
ap.add_argument("--prefix", default="ops",
                help="results/<prefix>_<partition>/ (OUT of profile_ops.sh)")
ARGS = ap.parse_args()
PARTITIONS = ARGS.partitions
PREFIX = ARGS.prefix
WORKLOADS = ["512-256", "2048-256"]
CONCURRENCY = [1, 8, 32]
HEAD_OPS = ["embed", "lm_head", "sampler"]


def bench(p, mode, c, w):
    files = glob.glob(f"results/{PREFIX}_{p}/{mode}/c{c}/*/{w}/bench.json")
    return json.load(open(files[0])) if files else None


def summary_csv(p, mode, c, name):
    path = f"results/{PREFIX}_{p}/{mode}/c{c}/summary/{name}"
    return pd.read_csv(path) if os.path.exists(path) else None


print("== 1. graph mode: output tok/s | mean TPOT ms")
for w in WORKLOADS:
    for c in CONCURRENCY:
        cells = []
        for p in PARTITIONS:
            b = bench(p, "graph", c, w)
            cells.append(f"{p}: " + (f"{b['output_throughput']:.0f} | "
                                     f"{b['mean_tpot_ms']:.1f}" if b else "-"))
        print(f"  {w} c{c:<2}  " + "   ".join(cells))

print("\n== 2. graph mode, decode: mean ms per step on each stage")
print("     (total = stage forward + embed/lm_head/sampler)")
for c in CONCURRENCY:
    for p in PARTITIONS:
        d = summary_csv(p, "graph", c, "breakdown.csv")
        if d is None:
            continue
        d = d[d.phase == "decode"]
        if d.empty:
            continue
        d = d.loc[d.groupby("pp_rank").steps.idxmax()]
        parts = []
        for r in d.itertuples():
            ops = " ".join(f"{k}={getattr(r, k):.2f}" for k in HEAD_OPS
                           if k in d.columns and getattr(r, k) > 0)
            parts.append(f"s{int(r.pp_rank)} total={r.total_ms:.2f} {ops}")
        print(f"  c{c:<2} {p:<8} batch~{int(d.batch_bucket.max()):<3} "
              + " | ".join(parts))

print("\n== 3. eager mode, decode: mean ms per op (one layer / one call)")
for c in CONCURRENCY:
    for p in PARTITIONS:
        d = summary_csv(p, "eager", c, "op_by_batch.csv")
        if d is None:
            continue
        d = d[d.phase == "decode"]
        if d.empty:
            continue
        d = d.loc[d.groupby("kind").samples.idxmax()]
        order = {k: i for i, k in enumerate(
            ["embed", "mamba", "attention", "mlp", "lm_head", "sampler"])}
        d = d.sort_values("kind", key=lambda s: s.map(order))
        print(f"  c{c:<2} {p:<8} " + "  ".join(
            f"{r.kind}={r.mean_ms:.3f}" for r in d.itertuples()))

print("\n== 4. eager vs graph: decode step total on the last stage (ms)")
for c in CONCURRENCY:
    cells = []
    for p in PARTITIONS:
        vals = []
        for mode in ("eager", "graph"):
            d = summary_csv(p, mode, c, "breakdown.csv")
            d = None if d is None else d[d.phase == "decode"]
            if d is None or d.empty:
                vals.append("-")
                continue
            last = d[d.pp_rank == d.pp_rank.max()]
            vals.append(f"{last.loc[last.steps.idxmax(), 'total_ms']:.2f}")
        cells.append(f"{p}: {vals[0]} vs {vals[1]}")
    print(f"  c{c:<2} " + "   ".join(cells))

print("\n== 5. graph mode, decode: CPU wall ms per step on each stage")
print("     GPU = stage GPU time; exec = model runner on CPU (last stage waits")
print("     for sampling); recv = waiting for previous stage; gap = waiting")
print("     for the engine between steps")
for c in CONCURRENCY:
    for p in PARTITIONS:
        d = summary_csv(p, "graph", c, "breakdown.csv")
        if d is None or "exec_ms" not in d.columns:
            continue
        d = d[d.phase == "decode"]
        if d.empty:
            continue
        d = d.loc[d.groupby("pp_rank").steps.idxmax()]
        print(f"  c{c:<2} {p:<8} " + " | ".join(
            f"s{int(r.pp_rank)} GPU={r.total_ms:.2f} exec={r.exec_ms:.2f} "
            f"recv={r.recv_ms:.2f} send={r.send_ms:.2f} gap={r.gap_ms:.2f}"
            for r in d.itertuples()))
