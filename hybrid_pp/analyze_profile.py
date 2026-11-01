"""Summarize hybrid_pp profiler traces: per-op cost and per-stage breakdown.

    python -m hybrid_pp.analyze_profile results/prof/default/*/profile \
        --out results/prof/summary

Writes
  ops.csv          one row per (step, op): phase, kind, layer, ms, batch info
  layer_cost.csv   mean ms per op, by phase / kind / context-length bucket
  op_by_batch.csv  mean ms per op, by phase / kind / batch-size bucket
                   (lm_head and sampler scale with batch, not context)
  breakdown.csv    per stage, phase, batch bucket: mean ms per step spent in
                   each kind (sum over that stage's layers) and the stage total
  stages.csv       per stage: step time stats and GPU busy ratio
  layer_cost.png, breakdown.png, stage_time.png   (if matplotlib is installed)

Phases: "prefill" (no decode tokens), "decode" (no prefills), "mixed".
Decode size = mean context length per sequence; prefill size = prefill tokens.
Per-layer rows exist only for eager steps; CUDA-graph decode steps still
contribute stage totals and the embed / lm_head / sampler ops.
"""

import argparse
import glob
import json
import os

import pandas as pd

KIND_ORDER = ["embed", "mamba", "attention", "mlp", "lm_head", "sampler"]


def load_traces(dirs: list[str]):
    metas, steps = [], []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            run = os.path.dirname(os.path.abspath(path))
            pp_rank = None
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    rec["run"] = run
                    if rec["kind"] == "meta":
                        metas.append(rec)
                        pp_rank = rec["pp_rank"]
                    elif rec["kind"] == "step":
                        rec["pp_rank"] = pp_rank
                        steps.append(rec)
    return metas, steps


def phase(step: dict) -> str:
    if not step.get("num_decode_tokens"):
        return "prefill"
    if not step.get("num_prefills"):
        return "decode"
    return "mixed"


def size_of(step: dict, ph: str) -> float:
    if ph == "decode":
        return step["ctx_sum"] / max(step["batch"], 1)
    return step.get("num_prefill_tokens") or 0


def pow2_bucket(series: pd.Series) -> pd.Series:
    return series.clip(lower=1).apply(lambda x: 1 << (int(x).bit_length() - 1))


CPU_COLS = ["call_ms", "recv_ms", "exec_ms", "send_ms", "gap_ms"]


def step_frame(steps: list[dict], skip: int) -> pd.DataFrame:
    rows = []
    for s in steps:
        if s["step"] < skip:
            continue
        ph = phase(s)
        cpu = s.get("cpu") or {}
        rows.append({
            "run": s["run"], "pp_rank": s["pp_rank"], "step": s["step"],
            "graph": s.get("graph", False), "phase": ph,
            "batch": s.get("batch") or 0, "size": size_of(s, ph),
            "t_start_ms": s["t_start_ms"], "dur_ms": s["dur_ms"],
            "total_ms": s.get("total_ms", s["dur_ms"]),
            "call_ms": cpu.get("call_ms"),
            "recv_ms": cpu.get("recv_ms", 0.0) if cpu else None,
            "exec_ms": cpu.get("exec_ms"),
            "send_ms": cpu.get("send_ms", 0.0) if cpu else None,
            "call_start": cpu.get("call_start"),
            "call_end": cpu.get("call_end"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # gap_ms: worker idle between two execute_model calls, i.e. waiting for
    # the engine to hand it the next step
    df = df.sort_values(["run", "pp_rank", "step"])
    prev_end = df.groupby(["run", "pp_rank"]).call_end.shift()
    df["gap_ms"] = (df.call_start - prev_end) * 1e3
    return df


def op_frame(steps: list[dict], skip: int) -> pd.DataFrame:
    rows = []
    for s in steps:
        if s["step"] < skip:
            continue
        ph = phase(s)
        for idx, kind, ms in s.get("layers", []):
            rows.append({
                "run": s["run"], "pp_rank": s["pp_rank"], "step": s["step"],
                "graph": s.get("graph", False), "phase": ph, "kind": kind,
                "layer": idx, "ms": ms, "batch": s.get("batch") or 0,
                "size": size_of(s, ph),
            })
    return pd.DataFrame(rows)


def mean_by(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    df = df.assign(bucket=pow2_bucket(df[bucket_col]))
    return (df.groupby(["phase", "kind", "bucket"])
            .agg(mean_ms=("ms", "mean"), std_ms=("ms", "std"),
                 samples=("ms", "size"))
            .reset_index())


def breakdown(ops: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Mean per-step time of each op kind on each stage."""
    per_step = (ops.groupby(["run", "pp_rank", "step", "kind"]).ms.sum()
                .unstack("kind", fill_value=0.0).reset_index())
    merged = steps.merge(per_step, on=["run", "pp_rank", "step"], how="left")
    kinds = [k for k in KIND_ORDER if k in merged.columns]
    merged[kinds] = merged[kinds].fillna(0.0)
    merged["batch_bucket"] = pow2_bucket(merged["batch"])
    agg = {k: (k, "mean") for k in kinds}
    agg.update(total_ms=("total_ms", "mean"), forward_ms=("dur_ms", "mean"),
               steps=("step", "size"), graph=("graph", "mean"))
    agg.update({c: (c, "mean") for c in CPU_COLS if c in merged.columns})
    out = (merged.groupby(["pp_rank", "phase", "batch_bucket"]).agg(**agg)
           .reset_index())
    # whatever the per-layer events did not cover (graph decode, norms,
    # inter-op gaps) shows up here
    out["other_ms"] = out.total_ms - out[kinds].sum(axis=1)
    return out


def breakdown_steps_only(steps: pd.DataFrame) -> pd.DataFrame:
    """Breakdown without per-op rows (all steps were CUDA-graph replays)."""
    steps = steps.assign(batch_bucket=pow2_bucket(steps["batch"]))
    agg = dict(total_ms=("total_ms", "mean"), forward_ms=("dur_ms", "mean"),
               steps=("step", "size"), graph=("graph", "mean"))
    agg.update({c: (c, "mean") for c in CPU_COLS})
    return (steps.groupby(["pp_rank", "phase", "batch_bucket"]).agg(**agg)
            .reset_index())


def stage_stats(steps: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (run, rank), g in steps.groupby(["run", "pp_rank"]):
        span = (g.t_start_ms + g.total_ms).max() - g.t_start_ms.min()
        out.append({
            "run": run, "pp_rank": rank, "steps": len(g),
            "mean_step_ms": g.total_ms.mean(),
            "p50_step_ms": g.total_ms.median(),
            "p99_step_ms": g.total_ms.quantile(0.99),
            "busy_ratio": g.total_ms.sum() / span if span > 0 else float("nan"),
        })
    return pd.DataFrame(out)


def plot(cost: pd.DataFrame, bd: pd.DataFrame, steps: pd.DataFrame, out: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    layer_kinds = {"mamba", "attention", "mlp"}
    phases = [p for p in ("prefill", "decode")
              if p in set(cost.phase) and
              set(cost[cost.phase == p].kind) & layer_kinds]
    if phases:
        fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 3.6),
                                 squeeze=False)
        for ax, ph in zip(axes[0], phases):
            sub = cost[(cost.phase == ph) & cost.kind.isin(layer_kinds)]
            for kind, g in sub.groupby("kind"):
                ax.plot(g.bucket, g.mean_ms, marker="o", label=kind)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("prefill tokens" if ph == "prefill"
                          else "mean context length per sequence")
            ax.set_ylabel("ms per layer")
            ax.set_title(ph)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "layer_cost.png"), dpi=150)

    if not bd.empty:
        # one bar per stage, for the most common (phase, batch bucket)
        key = bd.sort_values("steps").iloc[-1][["phase", "batch_bucket"]]
        sub = bd[(bd.phase == key.phase) &
                 (bd.batch_bucket == key.batch_bucket)].sort_values("pp_rank")
        kinds = [k for k in KIND_ORDER + ["other_ms"] if k in sub.columns]
        if not kinds:
            kinds = ["total_ms"]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bottom = [0.0] * len(sub)
        for k in kinds:
            vals = sub[k].clip(lower=0).tolist()
            ax.bar([f"stage {r}" for r in sub.pp_rank], vals, bottom=bottom,
                   label=k.replace("_ms", ""))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("ms per step")
        ax.set_title(f"{key.phase}, batch ~{key.batch_bucket}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "breakdown.png"), dpi=150)

    if not steps.empty:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ranks = sorted(steps.pp_rank.unique())
        ax.boxplot([steps[steps.pp_rank == r].total_ms for r in ranks],
                   showfliers=False)
        ax.set_xticks(range(1, len(ranks) + 1), [f"stage {r}" for r in ranks])
        ax.set_ylabel("step time on stage (ms)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "stage_time.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("profile_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-steps", type=int, default=0,
                    help="drop the first N recorded steps per worker "
                    "(memory profiling and graph capture are never recorded)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)

    metas, raw = load_traces(args.profile_dirs)
    if not raw:
        raise SystemExit("no step records found")
    for m in metas:
        print(f"pp_rank {m['pp_rank']}: layers [{m['start_layer']},"
              f"{m['end_layer']}) {m['stage_pattern']}  params "
              f"{m['stage_param_bytes'] / (1 << 30):.2f} GiB")

    steps = step_frame(raw, args.skip_steps)
    ops = op_frame(raw, args.skip_steps)
    stages = stage_stats(steps)
    stages.to_csv(os.path.join(args.out, "stages.csv"), index=False)
    print("\n== stage step time\n" + stages.to_string(index=False))

    cost, bd = pd.DataFrame(), pd.DataFrame()
    if not ops.empty:
        ops.to_csv(os.path.join(args.out, "ops.csv"), index=False)
        cost = mean_by(ops, "size")
        cost.to_csv(os.path.join(args.out, "layer_cost.csv"), index=False)
        by_batch = mean_by(ops, "batch")
        by_batch.to_csv(os.path.join(args.out, "op_by_batch.csv"), index=False)
        bd = breakdown(ops, steps)
        bd.to_csv(os.path.join(args.out, "breakdown.csv"), index=False)
        print("\n== per-stage breakdown (mean ms per step)\n" +
              bd.round(3).to_string(index=False))
        print("\n== op cost by batch size (mean ms per op)\n" +
              by_batch.round(3).to_string(index=False))
    else:
        bd = breakdown_steps_only(steps)
        bd.to_csv(os.path.join(args.out, "breakdown.csv"), index=False)
        print("\n== per-stage breakdown (mean ms per step)\n" +
              bd.round(3).to_string(index=False))
    plot(cost, bd, steps, args.out)


if __name__ == "__main__":
    main()
