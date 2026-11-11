"""First-cut PP execution model for vLLM V0 (plan steps 5-6).

Three parts:

1. Stage time model, fitted on step traces of *training* partitions only.
   Per phase, one linear model shared by all stages and partitions:
     decode : t = c0 + sum_T n_T * (a_T + b_T * batch) + n_attn * g * ctx_sum
              + embed * (e0 + e1*batch) + head * (h0 + h1*batch)
     prefill: t = c0 + sum_T n_T * (a_T + b_T * tokens) + n_attn * g * sum(len^2)
              + embed * e1*tokens + head * (h0 + h1*seqs)
   n_T = number of mamba / mlp / attention layers in the stage. Because the
   features are layer counts, the model can price partitions never run.

2. Communication: t = a + bytes / BW per boundary (results/comm/comm.json);
   boundary 1->2 is cross-node under the current mapping.

3. Event simulation of V0 semantics (docs/vllm_callchain.md):
   - PP virtual engines (VE), each its own scheduler; a new request goes to
     the VE with fewest unfinished requests.
   - one micro-batch per VE in flight (next step needs sampled tokens).
   - each stage is a FIFO resource; a micro-batch walks stage 0..PP-1.
   - no chunked prefill (max_model_len <= 32K): a step is all-prefill
     (waiting prompts up to the token budget / max_num_seqs) or all-decode.
   - closed-loop client with fixed concurrency (vllm bench serve).
   A per-stage non-GPU overhead (recv/send, metadata, scheduling, sampling) is
   the one calibrated constant, fitted on training partitions' bench.json.

    python3 -m hybrid_pp.pipeline_sim fit  --runs results/part_sweep_20261111 --holdout 15,13,15,13
    python3 -m hybrid_pp.pipeline_sim search --model-file results/part_sweep_20261111/sim_model.json
"""

import argparse
import glob
import heapq
import json
import os
import statistics

import numpy as np

PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
HIDDEN = 4480
KIND = {"M": "mamba", "-": "mlp", "*": "attention"}
MAX_SEQ_CAPTURE = 8192  # vLLM V0 default max_seq_len_to_capture


def stages_of(partition, pattern=PATTERN):
    """partition: list of layer counts -> per-stage composition dicts."""
    out, s = [], 0
    for i, n in enumerate(partition):
        seg = pattern[s:s + n]
        out.append({"mamba": seg.count("M"), "mlp": seg.count("-"),
                    "attention": seg.count("*"), "embed": int(i == 0),
                    "head": int(i == len(partition) - 1)})
        s += n
    return out


def parse_partition(name, nlayers=len(PATTERN), pp=4):
    if name == "default":
        return [nlayers // pp] * pp
    return [int(x) for x in name.replace("-", ",").split(",")]


# ------------------------------------------------------------- features

def feat_decode(st, batch, ctx_sum):
    return [1.0,
            st["mamba"], st["mamba"] * batch,
            st["mlp"], st["mlp"] * batch,
            st["attention"], st["attention"] * batch,
            st["attention"] * ctx_sum,
            st["embed"], st["embed"] * batch,
            st["head"], st["head"] * batch]


def feat_prefill(st, tokens, seqs, sq_sum):
    return [1.0,
            st["mamba"], st["mamba"] * tokens,
            st["mlp"], st["mlp"] * tokens,
            st["attention"], st["attention"] * tokens,
            st["attention"] * sq_sum,
            st["embed"] * tokens,
            st["head"], st["head"] * seqs]


def load_steps(run_dir, skip=2):
    """Yield (stage composition, step record) from one run's profile dir."""
    for f in sorted(glob.glob(os.path.join(run_dir, "profile", "*.jsonl"))):
        meta, n = None, 0
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "meta":
                meta = r
            elif r["kind"] == "step" and meta:
                n += 1
                if n <= skip:
                    continue
                pat = meta["stage_pattern"]
                st = {"mamba": pat.count("M"), "mlp": pat.count("-"),
                      "attention": pat.count("*"),
                      "embed": int(meta["pp_rank"] == 0),
                      "head": int(meta["pp_rank"] == meta["pp_size"] - 1)}
                yield meta["pp_rank"], st, r


def step_phase(r):
    if r.get("num_prefills", 0) and r.get("num_decode_tokens", 0):
        return "mixed"
    return "prefill" if r.get("num_prefills", 0) else "decode"


def step_time(r):
    """GPU span of the stage's step. CPU exec_ms is not usable: with CUDA
    graphs non-last stages return after launch (~1 ms) before the GPU is done.
    Non-GPU per-stage cost (recv/send/metadata) is the calibrated overhead."""
    return r["total_ms"]


def seq_lens(r):
    """Per-sequence scheduled query tokens (prefill) from profile_features."""
    rows = r.get("scheduled_sequences")
    if not rows:
        return None
    lens = [x.get("scheduled_query_tokens") for x in rows if x.get("is_prompt")]
    return lens if lens and all(lens) else None


def rows_for(run_dirs):
    X = {"decode_graph": [], "decode_eager": [], "prefill": []}
    y = {"decode_graph": [], "decode_eager": [], "prefill": []}
    for d in run_dirs:
        for rank, st, r in load_steps(d):
            ph = step_phase(r)
            if ph == "decode":
                ph = "decode_graph" if r.get("graph") else "decode_eager"
                X[ph].append(feat_decode(st, r["batch"], r["ctx_sum"]))
            elif ph == "prefill":
                lens = seq_lens(r)
                tok = r["num_prefill_tokens"]
                sq = sum(l * l for l in lens) if lens else tok * tok / max(r["batch"], 1)
                X[ph].append(feat_prefill(st, tok, r["batch"], sq))
            else:
                continue
            y[ph].append(step_time(r))
    return {k: (np.array(X[k]), np.array(y[k])) for k in X}


def fit_stage_model(run_dirs):
    model = {}
    for ph, (X, y) in rows_for(run_dirs).items():
        if len(y) == 0:
            continue
        # non-negative least squares by clipping would bias; plain lstsq,
        # weighted so each run counts by steps (no reweighting) for now
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        rel = np.abs(pred - y) / y
        model[ph] = {"coef": coef.tolist(), "n": int(len(y)),
                     "train_median_rel_err": float(np.median(rel)),
                     "train_p90_rel_err": float(np.percentile(rel, 90))}
    return model


def predict_stage(model, st, phase, **kw):
    c = np.array(model[phase]["coef"])
    if phase.startswith("decode"):
        f = feat_decode(st, kw["batch"], kw["ctx_sum"])
    else:
        f = feat_prefill(st, kw["tokens"], kw["seqs"], kw["sq_sum"])
    return max(float(np.dot(c, f)), 0.05)


# ------------------------------------------------------------- comm

def load_comm(path):
    rep = json.load(open(path))
    return {k: v["fit_ge_1MiB"] for k, v in rep["pairs"].items()}


def comm_ms(comm, kind, tokens):
    f = comm[kind]
    return f["latency_ms"] + 2 * tokens * HIDDEN * 2 / (f["bandwidth_GBps"] * 1e6)


# ------------------------------------------------------------- simulation

class Req:
    __slots__ = ("id", "inp", "out", "arr", "done_tok", "first", "last",
                 "end", "ve", "prefilled", "itl")

    def __init__(self, i, inp, out, arr):
        self.id, self.inp, self.out, self.arr = i, inp, out, arr
        self.done_tok, self.first, self.last, self.end = 0, None, None, None
        self.ve, self.prefilled, self.itl = None, False, []


def simulate(stage_model, comm, partition, inp, out, num_prompts, concurrency,
             max_num_seqs=16, overhead_ms=0.0, cross=(1,), pp=4,
             max_batched_tokens=None):
    """Return dict with throughput, TTFT, TPOT and per-stage busy time."""
    sts = stages_of(partition)
    pp = len(sts)
    budget = max_batched_tokens or max(16384, 2048)
    reqs = [Req(i, inp, out, 0.0) for i in range(num_prompts)]
    pending = list(reqs)            # not yet sent by the client
    waiting = [[] for _ in range(pp)]
    running = [[] for _ in range(pp)]
    ve_busy = [False] * pp
    stage_free = [0.0] * pp
    busy = [0.0] * pp
    in_flight = 0
    ev = []                         # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, payload)); seq += 1

    def admit(t):
        nonlocal in_flight
        while pending and in_flight < concurrency:
            r = pending.pop(0); r.arr = t; in_flight += 1
            costs = [len(waiting[v]) + len(running[v]) for v in range(pp)]
            v = costs.index(min(costs)); r.ve = v
            waiting[v].append(r)

    def try_start(v, t):
        if ve_busy[v]:
            return
        batch, tok = [], 0
        n_run = len(running[v])
        while waiting[v] and n_run + len(batch) < max_num_seqs and \
                tok + waiting[v][0].inp <= budget:
            r = waiting[v].pop(0); batch.append(r); tok += r.inp
        if batch:
            phase = "prefill"
            kw = dict(tokens=tok, seqs=len(batch),
                      sq_sum=sum(r.inp ** 2 for r in batch))
            ctoks = tok
        elif running[v]:
            batch = list(running[v])
            # V0: CUDA graph only if every seq <= max_seq_len_to_capture (8192)
            # and the batch fits the largest capture size
            graph = (max(r.inp + r.done_tok for r in batch) <= MAX_SEQ_CAPTURE
                     and len(batch) <= max_num_seqs)
            phase = "decode_graph" if graph else "decode_eager"
            if phase not in stage_model:
                phase = "decode_eager" if "decode_eager" in stage_model else "decode_graph"
            kw = dict(batch=len(batch),
                      ctx_sum=sum(r.inp + r.done_tok for r in batch))
            ctoks = len(batch)
        else:
            return
        ve_busy[v] = True
        push(t, "stage", (v, 0, phase, kw, ctoks, batch))

    admit(0.0)
    for v in range(pp):
        try_start(v, 0.0)

    t = 0.0
    while ev:
        t, _, kind, p = heapq.heappop(ev)
        v, s, phase, kw, ctoks, batch = p
        start = max(t, stage_free[s])
        dur = predict_stage(stage_model, sts[s], phase, **kw) + overhead_ms
        end = start + dur
        stage_free[s] = end; busy[s] += dur
        if s + 1 < pp:
            c = comm_ms(comm, "cross" if s in cross else "intra", ctoks)
            push(end + c, "stage", (v, s + 1, phase, kw, ctoks, batch))
            continue
        # last stage done: tokens sampled for the batch
        for r in batch:
            r.done_tok += 1
            if r.first is None:
                r.first = end
            else:
                r.itl.append(end - r.last)
            r.last = end
            if phase == "prefill":
                running[v].append(r)
            if r.done_tok >= r.out:
                running[v].remove(r); r.end = end; in_flight -= 1
        ve_busy[v] = False
        admit(end)
        for u in range(pp):
            try_start(u, end)

    dur_s = max(r.end for r in reqs) / 1000
    ttft = [r.first - r.arr for r in reqs]
    tpot = [(r.end - r.first) / (r.out - 1) for r in reqs if r.out > 1]
    return {"partition": partition, "duration_s": dur_s,
            "output_throughput": num_prompts * out / dur_s,
            "mean_ttft_ms": statistics.mean(ttft),
            "median_ttft_ms": statistics.median(ttft),
            "mean_tpot_ms": statistics.mean(tpot),
            "median_tpot_ms": statistics.median(tpot),
            "stage_busy_frac": [b / (dur_s * 1000) for b in busy]}


# ------------------------------------------------------------- CLI

def bench_of(run_dir):
    f = os.path.join(run_dir, "bench.json")
    return json.load(open(f)) if os.path.exists(f) else None


def run_dirs(root):
    """root/<partition>/<inp>-<out>/ -> list of (partition_name, inp, out, dir)."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*-*"))):
        if not os.path.isdir(os.path.join(d, "profile")):
            continue
        part = os.path.basename(os.path.dirname(d))
        i, o = os.path.basename(d).split("-")
        out.append((part, int(i), int(o), d))
    return out


def cmd_fit(a):
    runs = run_dirs(a.runs)
    hold = a.holdout.replace(",", "-") if a.holdout else None
    train = [r for r in runs if r[0] != hold]
    test = [r for r in runs if r[0] == hold]
    print(f"train runs {len(train)}, holdout runs {len(test)} ({hold})")
    sm = fit_stage_model([r[3] for r in train])
    for ph, m in sm.items():
        print(f"  stage model {ph}: n={m['n']} train median rel err "
              f"{m['train_median_rel_err']:.1%}, p90 {m['train_p90_rel_err']:.1%}")
    # stage-level holdout error
    hold_err = {}
    for ph, (X, y) in rows_for([r[3] for r in test]).items():
        if len(y):
            p = X @ np.array(sm[ph]["coef"])
            hold_err[ph] = float(np.median(np.abs(p - y) / y))
            print(f"  holdout stage {ph}: n={len(y)} median rel err {hold_err[ph]:.1%}")
    comm = load_comm(a.comm)

    def sim_for(r, oh):
        b = bench_of(r[3])
        cfg = json.load(open(os.path.join(a.runs, "args.json"))) \
            if os.path.exists(os.path.join(a.runs, "args.json")) else {}
        return simulate(sm, comm, parse_partition(r[0]), r[1], r[2],
                        cfg.get("num_prompts", a.num_prompts),
                        cfg.get("concurrency", a.concurrency),
                        max_num_seqs=cfg.get("max_num_seqs", a.max_num_seqs),
                        overhead_ms=oh), b

    # calibrate the per-step driver overhead on training runs (grid search on
    # TPOT, which is what the overhead mostly shifts)
    best = None
    for oh in np.arange(0.0, 15.01, 0.25):
        errs = []
        for r in train:
            s, b = sim_for(r, oh)
            if b:
                errs.append(abs(s["median_tpot_ms"] - b["median_tpot_ms"]) / b["median_tpot_ms"])
        e = statistics.median(errs) if errs else float("inf")
        if best is None or e < best[1]:
            best = (float(oh), e)
    oh = best[0]
    print(f"  calibrated non-GPU overhead per stage: {oh:.2f} ms (train median TPOT err {best[1]:.1%})")

    rows = []
    for r in runs:
        s, b = sim_for(r, oh)
        row = {"partition": r[0], "workload": f"{r[1]}:{r[2]}",
               "split": "holdout" if r in test else "train"}
        for k in ("output_throughput", "median_ttft_ms", "median_tpot_ms", "mean_ttft_ms", "mean_tpot_ms"):
            row["pred_" + k] = s[k]
            row["meas_" + k] = b.get(k) if b else None
        row["pred_stage_busy_frac"] = s["stage_busy_frac"]
        rows.append(row)
    print(f"\n{'partition':14s} {'load':10s} {'split':8s} {'thr pred/meas':>18s} "
          f"{'TTFT med pred/meas':>22s} {'TPOT med pred/meas':>22s}")
    for w in rows:
        def pm(k, f="{:.1f}"):
            m = w["meas_" + k]
            return f"{f.format(w['pred_' + k])}/{f.format(m) if m is not None else '-'}"
        print(f"{w['partition']:14s} {w['workload']:10s} {w['split']:8s} "
              f"{pm('output_throughput'):>18s} {pm('median_ttft_ms'):>22s} {pm('median_tpot_ms'):>22s}")
    out = {"stage_model": sm, "holdout_stage_err": hold_err, "overhead_ms": oh,
           "comm": comm, "holdout": hold, "runs": rows}
    path = os.path.join(a.runs, "sim_model.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {path}")


def all_partitions(n=len(PATTERN), pp=4, lo=6):
    def rec(left, k):
        if k == 1:
            if left >= lo:
                yield [left]
            return
        for x in range(lo, left - lo * (k - 1) + 1):
            for rest in rec(left - x, k - 1):
                yield [x] + rest
    return list(rec(n, pp))


def capacity_check(partition, cfg, budget_gib, runtime_gib, slots, need_tokens):
    """Per-stage memory via hybrid_pp.capacity. V0 sizes the KV pool by the
    min over stages of free blocks, so the binding stage is the attention
    stage with the least headroom. Returns (fits, kv_capacity_tokens, rows)."""
    from hybrid_pp.capacity import estimate
    rows = estimate(cfg, partition, [], slots)
    cap, out = None, []
    for r in rows:
        free = budget_gib * 2**30 - r["weights_estimated_bytes"] \
            - r["mamba_allocated_bytes"] - runtime_gib * 2**30
        tok = free / r["kv_bytes_per_token"] if r["kv_bytes_per_token"] else None
        out.append({"pp_rank": r["pp_rank"], "free_gib": free / 2**30,
                    "kv_tokens": tok})
        if free <= 0:
            return False, 0, out
        if tok is not None:
            cap = tok if cap is None else min(cap, tok)
    cap = cap if cap is not None else float("inf")
    return cap >= need_tokens, cap, out


def cmd_search(a):
    m = json.load(open(a.model_file))
    sm, comm, oh = m["stage_model"], m["comm"], m["overhead_ms"]
    loads = [tuple(int(x) for x in w.split(":")) for w in a.workloads]
    parts = all_partitions()
    cfg = None
    if a.config:
        cfg = json.load(open(a.config))
    caps = {}
    if cfg:
        need = a.concurrency * max(i + o for i, o in loads)
        kept = []
        for p in parts:
            ok, cap, rows = capacity_check(p, cfg, a.budget_gib, a.runtime_gib,
                                           a.max_num_seqs, need)
            caps[tuple(p)] = (cap, rows)
            if ok:
                kept.append(p)
        print(f"capacity: {len(kept)}/{len(parts)} partitions fit "
              f"{need} KV tokens (budget {a.budget_gib} GiB, runtime extra {a.runtime_gib} GiB)")
        parts = kept
    # cheap screen: bottleneck stage time for one decode and one prefill step
    res = []
    for p in parts:
        sts = stages_of(p)
        score = 0.0
        for inp, out in loads:
            bd = max(predict_stage(sm, s, "decode_graph" if inp + out <= MAX_SEQ_CAPTURE else "decode_eager", batch=a.max_num_seqs // 4,
                                   ctx_sum=(a.max_num_seqs // 4) * (inp + out // 2)) for s in sts)
            bp = max(predict_stage(sm, s, "prefill", tokens=inp, seqs=1, sq_sum=inp * inp) for s in sts)
            score += bd * out + bp
        res.append((score, p))
    res.sort()
    top = [p for _, p in res[:a.top]]
    for base in ([14] * 4, [18, 10, 10, 18]):
        if base not in top:
            top.append(base)
    print(f"{len(parts)} partitions (min {6} layers/stage); full simulation of {len(top)}")
    table = []
    for p in top:
        row = {"partition": p}
        if tuple(p) in caps:
            cap, crow = caps[tuple(p)]
            row["kv_capacity_tokens"] = cap
            row["stage_free_gib"] = [round(c["free_gib"], 2) for c in crow]
        for inp, out in loads:
            s = simulate(sm, comm, p, inp, out, a.num_prompts, a.concurrency,
                         max_num_seqs=a.max_num_seqs, overhead_ms=oh)
            row[f"{inp}:{out}"] = s
        table.append(row)
    key = f"{loads[0][0]}:{loads[0][1]}"
    table.sort(key=lambda r: -sum(r[f"{i}:{o}"]["output_throughput"] for i, o in loads))
    hdr = "partition          " + f"{'KVcap(tok)':>11s}" + "".join(f"{f'{i}:{o} thr/TTFT/TPOT':>30s}" for i, o in loads)
    print(hdr)
    for r in table:
        kc = r.get("kv_capacity_tokens")
        print(f"{','.join(map(str, r['partition'])):18s} " + (f"{kc:>11.0f}" if kc else f"{'-':>11s}") + "".join(
            f"{r[f'{i}:{o}']['output_throughput']:>12.1f}{r[f'{i}:{o}']['median_ttft_ms']:>9.0f}"
            f"{r[f'{i}:{o}']['median_tpot_ms']:>9.1f}" for i, o in loads))
    if a.out:
        json.dump(table, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--runs", required=True)
    f.add_argument("--holdout", default=None)
    f.add_argument("--comm", default="results/comm/comm.json")
    s = sub.add_parser("search")
    s.add_argument("--model-file", required=True)
    s.add_argument("--workloads", nargs="+", default=["1024:128", "8192:128"])
    s.add_argument("--top", type=int, default=8)
    s.add_argument("--out", default=None)
    s.add_argument("--config", default=None, help="model config.json for the capacity check")
    s.add_argument("--budget-gib", type=float, default=24 * 0.9)
    s.add_argument("--runtime-gib", type=float, default=4.15,
                   help="activation peak + non-torch per stage; 3.7-4.0 + 0.13 GiB measured in stage_mem at max_model_len 16384")
    for p in (f, s):
        p.add_argument("--num-prompts", type=int, default=32)
        p.add_argument("--concurrency", type=int, default=16)
        p.add_argument("--max-num-seqs", type=int, default=16)
    a = ap.parse_args()
    {"fit": cmd_fit, "search": cmd_search}[a.cmd](a)


if __name__ == "__main__":
    main()
