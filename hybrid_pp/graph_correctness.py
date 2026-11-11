"""Does raising --max-seq-len-to-capture change outputs? (2026-11-11)

Same PP=N server twice: vLLM's default max_seq_len_to_capture (8192, so
decode of sequences > 8192 tokens falls back to eager) and a raised value
(those decodes replay CUDA graphs). Greedy, one request at a time, compare
completions. Long prompts are key-value retrieval ("needle") tasks of
~9K-15K tokens, so every decode step of them is past 8192 and the answer
can be checked for content, not only for equality.

    python3 -m hybrid_pp.graph_correctness --pp 4 --capture 16384 \
        --out results/graph_correctness_20261111
"""

import argparse
import json
import os
import random

from hybrid_pp.correctness import common_prefix, complete
from hybrid_pp.server import Server, serve_command

SHORT = [
    "The capital of France is",
    "Explain the difference between a process and a thread.",
    "def fibonacci(n):",
]


def needle_prompt(n_items, asks, seed):
    rng = random.Random(seed)
    codes = {i: "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
             for i in range(1, n_items + 1)}
    lines = [f"Record {i}: the access code for locker {i} is {codes[i]}." for i in range(1, n_items + 1)]
    q = ", ".join(str(a) for a in asks)
    prompt = ("Below is a list of locker records.\n" + "\n".join(lines) +
              f"\n\nQuestion: what are the access codes for lockers {q}? "
              "Answer with one line per locker in the form 'locker N: CODE'.\nAnswer:\n")
    return prompt, {a: codes[a] for a in asks}


def build_prompts(tok, targets):
    out = []
    for k, target in enumerate(targets):
        n = 200
        while True:  # grow until the prompt reaches the target length
            asks = [7, n // 2, n - 3]
            p, ans = needle_prompt(n, asks, seed=k)
            if len(tok(p).input_ids) >= target:
                break
            n += 50
        out.append({"prompt": p, "answers": ans, "tokens": len(tok(p).input_ids)})
    return out


def run(args, capture, prompts, out_path):
    extra = [f"--max-seq-len-to-capture={capture}"] if capture else []
    cmd = serve_command(args.model, args.port, args.pp, 1, args.max_model_len,
                        args.max_num_seqs, 0.9, False, extra)
    env = dict(os.environ, VLLM_USE_V1="0")
    if args.partition:
        env["VLLM_PP_LAYER_PARTITION"] = args.partition
    log = out_path.replace(".json", ".server.log")
    with Server(cmd, env, log, args.port, timeout_s=900):
        outs = [complete(args.port, args.model, p, args.max_tokens) for p in prompts]
    json.dump(outs, open(out_path, "w"), indent=1, ensure_ascii=False)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("--pp", type=int, default=4)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--capture", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--targets", type=int, nargs="+", default=[9000, 11000, 13000, 15000])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    longs = build_prompts(tok, a.targets)
    prompts = SHORT + [x["prompt"] for x in longs]
    json.dump({"long": [{"tokens": x["tokens"], "answers": x["answers"]} for x in longs]},
              open(os.path.join(a.out, "prompts_meta.json"), "w"), indent=1)
    print("long prompt tokens:", [x["tokens"] for x in longs], flush=True)

    base = run(a, None, prompts, os.path.join(a.out, "capture_default.json"))
    print("default (8192) done", flush=True)
    raised = run(a, a.capture, prompts, os.path.join(a.out, f"capture_{a.capture}.json"))
    print(f"raised ({a.capture}) done", flush=True)

    rows, same = [], 0
    for i, (x, y) in enumerate(zip(base, raised)):
        kind = "short" if i < len(SHORT) else "long"
        r = {"i": i, "kind": kind, "identical": x == y,
             "common_prefix_chars": common_prefix(x, y), "len_default": len(x), "len_raised": len(y)}
        if kind == "long":
            lp = longs[i - len(SHORT)]
            r["tokens"] = lp["tokens"]
            r["answers_found_default"] = sum(c in x for c in lp["answers"].values())
            r["answers_found_raised"] = sum(c in y for c in lp["answers"].values())
            r["answers_total"] = len(lp["answers"])
        same += r["identical"]
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    summary = {"identical": same, "total": len(rows), "rows": rows,
               "config": vars(a)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"identical {same}/{len(rows)}; wrote {a.out}/summary.json")


if __name__ == "__main__":
    main()
