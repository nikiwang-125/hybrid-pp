"""Per-stage memory breakdown from a vLLM server log (needs our plugin).

Reads the ``[hybrid_pp] stage_mem {...}`` lines every worker logs and the
engine's final ``# cuda blocks: N`` line, then reports for each stage how much
of its KV budget is actually used after V0 takes min() over all stages.

    python -m hybrid_pp.analyze_memory results/sweep_pp2/*/server.log
"""

import argparse
import json
import re
import sys

GIB = 1 << 30
_STAGE_RE = re.compile(r"\[hybrid_pp\] stage_mem (\{.*\})")
_BLOCKS_RE = re.compile(r"# (\w+) blocks: (\d+), # CPU blocks: (\d+)")


def parse_log(path: str) -> dict:
    stages, final_blocks = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = _STAGE_RE.search(line)
            if m:
                info = json.loads(m.group(1))
                stages[(info["pp_rank"], info["tp_rank"])] = info
                continue
            m = _BLOCKS_RE.search(line)
            if m:
                final_blocks = int(m.group(2))
    return {"stages": [stages[k] for k in sorted(stages)],
            "num_gpu_blocks": final_blocks}


def summarize(parsed: dict) -> list[dict]:
    blocks = parsed["num_gpu_blocks"]
    rows = []
    for s in parsed["stages"]:
        kv_budget = s.get("kv_gib")
        used = (blocks * s["cache_block_bytes"] / GIB
                if blocks is not None else None)
        rows.append({
            "pp_rank": s["pp_rank"],
            "tp_rank": s["tp_rank"],
            "pattern": s.get("stage_pattern"),
            "weights_gib": s.get("weights_gib"),
            "activation_gib": s.get("activation_gib"),
            "non_torch_gib": s.get("non_torch_gib"),
            "kv_budget_gib": kv_budget,
            "kv_used_gib": used,
            "kv_unused_gib": (kv_budget - used
                              if None not in (kv_budget, used) else None),
            "local_blocks": s["local_num_gpu_blocks"],
            "cache_block_bytes": s["cache_block_bytes"],
        })
    return rows


def _fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {}
    for path in args.logs:
        parsed = parse_log(path)
        if not parsed["stages"]:
            print(f"{path}: no [hybrid_pp] stage_mem lines (plugin not "
                  "installed, or server failed early)", file=sys.stderr)
            continue
        rows = summarize(parsed)
        report[path] = {"num_gpu_blocks": parsed["num_gpu_blocks"],
                        "stages": rows}
        if args.json:
            continue
        print(f"== {path}  (engine num_gpu_blocks = "
              f"{parsed['num_gpu_blocks']})")
        print(f"{'pp':>3} {'tp':>3} {'weights':>8} {'act':>6} {'KV budget':>10}"
              f" {'KV used':>8} {'unused':>8} {'blocks':>8}  pattern")
        for r in rows:
            print(f"{r['pp_rank']:>3} {r['tp_rank']:>3} "
                  f"{_fmt(r['weights_gib']):>8} {_fmt(r['activation_gib']):>6} "
                  f"{_fmt(r['kv_budget_gib']):>10} {_fmt(r['kv_used_gib']):>8} "
                  f"{_fmt(r['kv_unused_gib']):>8} {r['local_blocks']:>8}  "
                  f"{r['pattern']}")
    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
