"""Summarize hybrid_pp profiler traces: per-layer-type cost and stage balance.

    python -m hybrid_pp.analyze_profile results/layers/default/*/profile \
        --out results/layers/summary

Writes
  layers.csv       one row per (step, layer): phase, kind, ms, batch features
  layer_cost.csv   mean ms per layer, by phase / kind / size bucket
  stages.csv       per pp_rank: step time stats and GPU busy ratio
  layer_cost.png, stage_time.png   (if matplotlib is installed)

Phases: "prefill" (no decode tokens), "decode" (no prefills), "mixed".
Decode size = mean context length per sequence; prefill size = prefill tokens.
"""

import argparse
import glob
import json
import os

import pandas as pd


def load_traces(dirs: list[str]):
    metas, steps = [], []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            run = os.path.dirname(os.path.abspath(path))
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


def layer_rows(steps: list[dict], skip: int) -> pd.DataFrame:
    rows = []
    for s in steps:
        if s["step"] < skip:
            continue
        ph = phase(s)
        for idx, kind, ms in s.get("layers", []):
            rows.append({
                "run": s["run"], "pp_rank": s["pp_rank"], "step": s["step"],
                "phase": ph, "kind": kind, "layer": idx, "ms": ms,
                "batch": s.get("batch"), "size": size_of(s, ph),
                "num_tokens": (s.get("num_prefill_tokens") or 0) +
                              (s.get("num_decode_tokens") or 0),
            })
    return pd.DataFrame(rows)


def stage_rows(steps: list[dict], skip: int) -> pd.DataFrame:
    df = pd.DataFrame([{
        "run": s["run"], "pp_rank": s["pp_rank"], "step": s["step"],
        "t_start_ms": s["t_start_ms"], "dur_ms": s["dur_ms"],
        "phase": phase(s),
    } for s in steps if s["step"] >= skip])
    out = []
    for (run, rank), g in df.groupby(["run", "pp_rank"]):
        span = (g.t_start_ms + g.dur_ms).max() - g.t_start_ms.min()
        out.append({
            "run": run, "pp_rank": rank, "steps": len(g),
            "mean_step_ms": g.dur_ms.mean(), "p50_step_ms": g.dur_ms.median(),
            "p99_step_ms": g.dur_ms.quantile(0.99),
            "busy_ratio": g.dur_ms.sum() / span if span > 0 else float("nan"),
        })
    return df, pd.DataFrame(out)


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    # power-of-two buckets keep prefill and decode sizes on one scale
    df = df.copy()
    df["bucket"] = (2 ** (df["size"].clip(lower=1).apply(
        lambda x: int(x).bit_length() - 1))).astype(int)
    return (df.groupby(["phase", "kind", "bucket"])
            .agg(mean_ms=("ms", "mean"), std_ms=("ms", "std"),
                 samples=("ms", "size"), mean_batch=("batch", "mean"))
            .reset_index())


def plot(cost: pd.DataFrame, steps_df: pd.DataFrame, out: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    phases = [p for p in ("prefill", "decode") if p in set(cost.phase)]
    if phases:
        fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 3.6),
                                 squeeze=False)
        for ax, ph in zip(axes[0], phases):
            sub = cost[cost.phase == ph]
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

    if not steps_df.empty:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ranks = sorted(steps_df.pp_rank.unique())
        ax.boxplot([steps_df[steps_df.pp_rank == r].dur_ms for r in ranks],
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
    ap.add_argument("--skip-steps", type=int, default=2,
                    help="drop the first N steps per worker (memory "
                    "profiling / warm-up)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metas, steps = load_traces(args.profile_dirs)
    if not steps:
        raise SystemExit("no step records found")
    for m in metas:
        print(f"pp_rank {m['pp_rank']}: layers [{m['start_layer']},"
              f"{m['end_layer']}) {m['stage_pattern']}  params "
              f"{m['stage_param_bytes'] / (1 << 30):.2f} GiB")

    layers = layer_rows(steps, args.skip_steps)
    steps_df, stages = stage_rows(steps, args.skip_steps)
    stages.to_csv(os.path.join(args.out, "stages.csv"), index=False)
    print(stages.to_string(index=False))

    cost = pd.DataFrame()
    if not layers.empty:
        layers.to_csv(os.path.join(args.out, "layers.csv"), index=False)
        cost = bucketize(layers)
        cost.to_csv(os.path.join(args.out, "layer_cost.csv"), index=False)
        print(cost.to_string(index=False))
    plot(cost, steps_df, args.out)


if __name__ == "__main__":
    main()
