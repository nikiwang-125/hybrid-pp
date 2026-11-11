"""Line charts for the PP=4 batch x context grid from scripts/pp4_layers.sh.

    python3 scripts/plot_pp4_layers.py results/pp4_layers

Expects <root>/c<C>/<partition>/<in>-<out>/{profile/*.jsonl,server.log,bench.json}
(profile traces of every rank, including the ones written on the other
machine) and optional <root>/gpu_<host>.csv from nvidia-smi. Writes PNGs and
CSVs to <root>/plots.
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_pp.analyze_profile import load_traces, op_frame, step_frame  # noqa: E402

GIB = 1 << 30
# categorical slots 1-4 (stages, layer kinds); ordinal blue ramp for levels
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ORD = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 6,
    "legend.frameon": False, "font.size": 9, "axes.titlesize": 10,
})
_STAGE_RE = re.compile(r"\[hybrid_pp\] stage_mem (\{.*\})")
_KV_RE = re.compile(r"GPU KV cache usage: ([\d.]+)%")
_BLOCKS_RE = re.compile(r"# (\w+) blocks: (\d+)")


def runs(root):
    for d in sorted(glob.glob(os.path.join(root, "c*", "*", "*-*"))):
        c = int(os.path.basename(os.path.dirname(os.path.dirname(d)))[1:])
        inp, out = map(int, os.path.basename(d).split("-"))
        yield d, c, inp, out


def plain_ticks(ax, values):
    from matplotlib.ticker import FixedLocator, NullLocator
    vals = [v for v in values if v > 0]
    if len(vals) > 8:  # continuous x: powers of two
        lo, hi = min(vals), max(vals)
        vals, p = [], 1
        while p <= hi * 2:
            if p >= lo / 2:
                vals.append(p)
            p *= 2
    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{v / 1024:g}K" if v >= 1024 else f"{v:g}" for v in vals])


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def lines(ax, df, x, y, series, palette, label_fmt, logx=False):
    for i, (key, g) in enumerate(sorted(df.groupby(series), key=lambda t: t[0])):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", color=palette[i % len(palette)],
                label=label_fmt(key))
    if logx:
        ax.set_xscale("log", base=2)
        plain_ticks(ax, sorted(df[x].unique()))


def main(root):
    out = os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)
    meta_rows, op_rows, step_rows, bench_rows, mem_rows, kv_rows, pre_rows = [], [], [], [], [], [], []
    for d, c, inp, outlen in runs(root):
        tag = dict(c=c, inp=inp, out=outlen)
        metas, raw = load_traces([os.path.join(d, "profile")])
        ranks = sorted({m["pp_rank"] for m in metas})
        if len(ranks) < 4:
            print(f"WARNING {d}: traces for ranks {ranks} only")
        for m in metas:
            meta_rows.append({**tag, "pp_rank": m["pp_rank"],
                              "pattern": m["stage_pattern"]})
        if raw:
            o = op_frame(raw, 0)
            s = step_frame(raw, 0)
            # V0 with PP=N splits the running requests over N virtual
            # engines, so a full decode step holds ~concurrency/N sequences
            sb = int(s[s.phase == "decode"].batch.max()) if (s.phase == "decode").any() else 0
            op_rows.append(o.assign(sb=sb, **tag))
            for st in raw:
                if st.get("num_decode_tokens") or not st.get("num_prefill_tokens"):
                    continue
                for idx, kind, ms in st.get("layers", []):
                    pre_rows.append({**tag, "pp_rank": st["pp_rank"], "kind": kind,
                                     "ms": ms, "tokens": st["num_prefill_tokens"],
                                     "ctx_end": st.get("ctx_max", 0)})
            step_rows.append(s.assign(sb=sb, **tag))
        # mamba state cache size per rank
        mamba = {}
        for path in glob.glob(os.path.join(d, "profile", "*.jsonl")):
            rank = None
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["kind"] == "meta":
                        rank = rec["pp_rank"]
                    elif rec["kind"] == "mamba_cache":
                        mamba[rank] = rec["bytes"] / GIB
                        break
        bj = os.path.join(d, "bench.json")
        if os.path.exists(bj):
            b = json.load(open(bj))
            bench_rows.append({**tag, **{k: b.get(k) for k in (
                "output_throughput", "total_token_throughput", "mean_ttft_ms",
                "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "duration")}})
        stages, blocks, kv = {}, None, []
        with open(os.path.join(d, "server.log"), errors="replace") as f:
            for line in f:
                m = _STAGE_RE.search(line)
                if m:
                    info = json.loads(m.group(1))
                    stages[info["pp_rank"]] = info
                m = _BLOCKS_RE.search(line)
                if m and m.group(1) != "CPU":
                    blocks = int(m.group(2))
                m = _KV_RE.search(line)
                if m:
                    kv.append(float(m.group(1)))
        for r, info in stages.items():
            mem_rows.append({**tag, "pp_rank": r,
                             "weights_gib": info["weights_gib"],
                             "activation_gib": info["activation_gib"],
                             "kv_budget_gib": info["kv_gib"],
                             "mamba_cache_gib": mamba.get(r),
                             "kv_alloc_gib": (blocks or 0) * info["cache_block_bytes"] / GIB,
                             "attn_layers": info["stage_pattern"].count("*")})
        if kv and stages:
            per_block = max(i["cache_block_bytes"] for i in stages.values())
            kv_rows.append({**tag, "peak_kv_pct": max(kv),
                            "peak_kv_used_gib": max(kv) / 100 * (blocks or 0) * per_block / GIB})

    ops = pd.concat(op_rows) if op_rows else pd.DataFrame()
    steps = pd.concat(step_rows) if step_rows else pd.DataFrame()
    bench, mem, kvdf = map(pd.DataFrame, (bench_rows, mem_rows, kv_rows))
    for name, df in [("bench", bench), ("memory", mem), ("kv_usage", kvdf)]:
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    ctx_lab = lambda v: f"ctx {v // 1024}K"
    bat_lab = lambda v: f"concurrency {v}"
    sbmap = {}
    if not steps.empty:
        sbmap = steps.groupby("c").sb.max().to_dict()
    conc_lab = lambda v: f"conc {v} (step batch {sbmap.get(v, '?')})"
    stage_lab = lambda v: f"stage {v}"

    if not ops.empty:
        # decode steps at full batch (batch == concurrency)
        dec = ops[(ops.phase == "decode") & (ops.batch == ops.sb)]
        g = (dec.groupby(["c", "inp", "kind"])
             .agg(ms=("ms", "mean"), ctx=("size", "mean"), sb=("sb", "first"),
                  n=("ms", "size"))
             .reset_index())
        g.to_csv(os.path.join(out, "decode_layer_cost.csv"), index=False)
        kinds = [k for k in ["mamba", "attention", "mlp"] if k in set(g.kind)]
        for x, series, lab, name, xl in [
                ("inp", "c", conc_lab, "decode_layer_vs_ctx.png", "input length (tokens; context grows by the output length)"),
                ("c", "inp", ctx_lab, "decode_layer_vs_batch.png", "concurrency (decode step batch = concurrency / PP)")]:
            fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.4))
            for ax, k in zip(axes, kinds):
                lines(ax, g[g.kind == k], x, "ms", series, ORD, lab, logx=True)
                ax.set_title(f"decode, one {k} layer")
                ax.set_xlabel(xl)
                ax.set_ylabel("ms per layer per step")
                ax.set_ylim(bottom=0)
            axes[0].legend()
            save(fig, out, name)

        # prefill: one full 2K chunk (chunked prefill), by where the chunk ends
        pre = pd.DataFrame(pre_rows)
        if not pre.empty:
            full = pre[pre.tokens >= pre.tokens.max() * 0.9].copy()
            full["bucket"] = full.ctx_end.clip(lower=1).apply(
                lambda v: 1 << (int(v - 1).bit_length()))
            gp = full.groupby(["kind", "bucket"]).ms.median().reset_index()
            gp.to_csv(os.path.join(out, "prefill_layer_cost.csv"), index=False)
            chunk = int(pre.tokens.max())
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            for i, k in enumerate(["mamba", "attention", "mlp"]):
                s = gp[gp.kind == k].sort_values("bucket")
                if not s.empty:
                    ax.plot(s.bucket, s.ms, marker="o", color=CAT[i], label=k)
            ax.set_xscale("log", base=2)
            plain_ticks(ax, sorted(gp.bucket.unique()))
            ax.set_ylim(bottom=0)
            ax.set_xlabel("context position at the end of the chunk (tokens, rounded up)")
            ax.set_ylabel("ms per layer")
            ax.set_title(f"prefill, one layer on a {chunk}-token chunk")
            ax.legend()
            save(fig, out, "prefill_layer_vs_ctx.png")

        # lm_head / sampler / embed vs batch, decode
        hd = ops[(ops.phase == "decode") & ops.kind.isin(["embed", "lm_head", "sampler"])]
        if not hd.empty:
            gh = hd.groupby(["kind", "batch"]).ms.mean().reset_index()
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            for i, k in enumerate(["embed", "lm_head", "sampler"]):
                s = gh[gh.kind == k].sort_values("batch")
                if not s.empty:
                    ax.plot(s.batch, s.ms, marker="o", color=CAT[i], label=k)
            ax.set_xscale("log", base=2)
            plain_ticks(ax, sorted(gh.batch.unique()))
            ax.set_ylim(bottom=0)
            ax.set_xlabel("decode step batch (sequences)")
            ax.set_ylabel("ms per step")
            ax.set_title("decode, embed (stage 0) / lm_head + sampler (stage 3)")
            ax.legend()
            save(fig, out, "decode_head_vs_batch.png")

    if not steps.empty:
        ds = steps[(steps.phase == "decode") & (steps.batch == steps.sb)]
        gs = (ds.groupby(["c", "inp", "pp_rank"])
              .agg(step_ms=("total_ms", "mean"), ctx=("size", "mean"))
              .reset_index())
        gs.to_csv(os.path.join(out, "decode_stage_time.csv"), index=False)
        cs = sorted(gs.c.unique())
        fig, axes = plt.subplots(1, len(cs), figsize=(3.6 * len(cs), 3.4), sharey=False)
        axes = [axes] if len(cs) == 1 else axes
        for ax, c in zip(axes, cs):
            lines(ax, gs[gs.c == c], "inp", "step_ms", "pp_rank", CAT, stage_lab, logx=True)
            ax.set_title(f"decode step, conc {c} (batch {sbmap.get(c)})")
            ax.set_xlabel("input length (tokens)")
            ax.set_ylabel("ms per step")
            ax.set_ylim(bottom=0)
        axes[0].legend()
        save(fig, out, "decode_stage_time.png")

        imb = (gs.groupby(["c", "inp"])
               .agg(mx=("step_ms", "max"), mean=("step_ms", "mean"), ctx=("ctx", "mean"))
               .reset_index())
        imb["imbalance"] = imb.mx / imb["mean"]
        imb.to_csv(os.path.join(out, "decode_imbalance.csv"), index=False)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        lines(ax, imb, "inp", "imbalance", "c", ORD, conc_lab, logx=True)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("input length (tokens)")
        ax.set_ylabel("slowest stage / mean stage")
        ax.set_title("decode stage imbalance (1.0 = balanced)")
        ax.legend()
        save(fig, out, "decode_imbalance.png")

        bz = []
        for (c, inp, r), g in steps.groupby(["c", "inp", "pp_rank"]):
            span = (g.t_start_ms + g.total_ms).max() - g.t_start_ms.min()
            bz.append(dict(c=c, inp=inp, pp_rank=r, busy=g.total_ms.sum() / span))
        bz = pd.DataFrame(bz)
        bz.to_csv(os.path.join(out, "busy_ratio.csv"), index=False)
        ins = sorted(bz.inp.unique())
        fig, axes = plt.subplots(1, len(ins), figsize=(3.6 * len(ins), 3.4))
        axes = [axes] if len(ins) == 1 else axes
        for ax, inp in zip(axes, ins):
            lines(ax, bz[bz.inp == inp], "c", "busy", "pp_rank", CAT, stage_lab, logx=True)
            ax.set_title(f"GPU busy ratio, ctx {inp // 1024}K")
            ax.set_xlabel("concurrency")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("busy time / wall time")
        axes[0].legend()
        save(fig, out, "busy_ratio.png")

    if not bench.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, (col, title) in zip(axes, [
                ("output_throughput", "output throughput (tok/s)"),
                ("mean_tpot_ms", "mean TPOT (ms)"),
                ("mean_ttft_ms", "mean TTFT (ms)")]):
            lines(ax, bench, "c", col, "inp", ORD, ctx_lab, logx=True)
            ax.set_title(title)
            ax.set_xlabel("concurrency")
            ax.set_ylim(bottom=0)
        axes[0].legend()
        save(fig, out, "bench.png")

    if not mem.empty:
        m1 = mem.sort_values("inp").groupby(["c", "pp_rank"]).first().reset_index()
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
        comps = [("activation_gib", "activation peak"), ("kv_budget_gib", "KV budget"),
                 ("kv_alloc_gib", "KV allocated"), ("mamba_cache_gib", "mamba state")]
        for ax, r in zip(axes, range(4)):
            s = m1[m1.pp_rank == r].sort_values("c")
            for i, (col, lab) in enumerate(comps):
                if s[col].notna().any():
                    ax.plot(s.c, s[col], marker="o", color=CAT[i], label=lab)
            w = s.weights_gib.iloc[0] if len(s) else float("nan")
            na = s.attn_layers.iloc[0] if len(s) else 0
            ax.set_title(f"stage {r} ({na} attn, weights {w:.2f} GiB)" if len(s) else f"stage {r}")
            ax.set_xscale("log", base=2)
            plain_ticks(ax, sorted(m1.c.unique()))
            ax.set_xlabel("max-num-seqs")
        axes[0].set_ylabel("GiB")
        axes[0].legend()
        save(fig, out, "memory_vs_batch.png")

    if not kvdf.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        lines(ax, kvdf, "inp", "peak_kv_used_gib", "c", ORD, bat_lab, logx=True)
        ax.set_xlabel("input length (tokens)")
        ax.set_ylabel("GiB per attention stage")
        ax.set_title("peak KV cache actually used")
        ax.set_ylim(bottom=0)
        ax.legend()
        save(fig, out, "kv_used.png")

    gpu = []
    for path in glob.glob(os.path.join(root, "gpu_*.csv")):
        host = os.path.basename(path)[4:-4]
        df = pd.read_csv(path, header=None, names=["ts", "gpu", "mem", "util"],
                         skipinitialspace=True)
        df["ts"] = pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        df["mem"] = df["mem"].str.replace(" MiB", "").astype(float) / 1024
        df["util"] = df["util"].str.replace(" %", "").astype(float)
        df["dev"] = host + ":" + df.gpu.astype(str)
        gpu.append(df.dropna())
    if gpu:
        gpu = pd.concat(gpu)
        t0 = gpu.ts.min()
        gpu["min"] = (gpu.ts - t0).dt.total_seconds() / 60
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        # rank 0 runs in the engine process on the Ray head (slave21);
        # Ray places ranks in node order. Override with DEV_STAGE if needed.
        dev_stage = json.loads(os.environ.get("DEV_STAGE", json.dumps(
            {"slave21:0": 0, "slave21:1": 1, "slave20:0": 2, "slave20:1": 3})))
        for dev, g in sorted(gpu.groupby("dev"), key=lambda t: dev_stage.get(t[0], 9)):
            st = dev_stage.get(dev)
            col = CAT[st % 4] if st is not None else INK2
            ls = "-" if dev.startswith("slave21") else "--"
            lab = f"stage {st} ({dev})" if st is not None else dev
            axes[0].plot(g["min"], g.mem, color=col, lw=1.2, ls=ls, label=lab)
            axes[1].plot(g["min"], g.util.rolling(15, min_periods=1).mean(),
                         color=col, lw=1.0, ls=ls, label=lab)
        axes[0].set_ylabel("memory used (GiB)")
        axes[0].set_title("nvidia-smi over the whole grid (each plateau = one server; stages 0/3 hold no KV cache)")
        axes[1].set_ylabel("GPU util % (30 s mean)")
        axes[1].set_xlabel("minutes")
        axes[0].legend(ncol=4)
        save(fig, out, "gpu_timeline.png")


if __name__ == "__main__":
    main(sys.argv[1])
