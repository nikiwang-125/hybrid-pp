"""Static per-stage analysis of a NemotronH config (no GPU, no vLLM needed).

Shows, for a given pipeline partition, how many mamba / attention / MLP
layers each stage gets and an analytical estimate of its memory:
weights, KV-cache bytes per token, and mamba state bytes per sequence.

    python -m hybrid_pp.model_info nvidia/NVIDIA-Nemotron-Nano-9B-v2 --pp 2
    python -m hybrid_pp.model_info <model> --pp 4 --partition 12,16,16,12
    python -m hybrid_pp.model_info <model> --pp 2 --candidates 4

<model> is a local directory, a config.json path or a Hugging Face repo id.
"""

import argparse
import itertools
import json
import os

LAYER_KIND = {"M": "mamba", "*": "attention", "-": "mlp"}
GIB = 1 << 30


def load_config(model: str) -> dict:
    if os.path.isfile(model):
        path = model
    elif os.path.isdir(model):
        path = os.path.join(model, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json")
    with open(path) as f:
        return json.load(f)


def default_partition(num_layers: int, pp: int) -> list[int]:
    """Same rule as vllm.distributed.utils.get_pp_indices (v0.10.1.1)."""
    partitions = [num_layers // pp] * pp
    for i in range(2, num_layers % pp + 2):
        partitions[-i] += 1
    return partitions


def parse_partition(text: str, num_layers: int, pp: int) -> list[int]:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != pp or sum(parts) != num_layers:
        raise SystemExit(f"partition {parts} must have {pp} entries "
                         f"summing to {num_layers}")
    return parts


def candidate_partitions(num_layers: int, pp: int, radius: int,
                         step: int = 1) -> list[list[int]]:
    """Contiguous partitions within +-radius layers of the even split."""
    base = default_partition(num_layers, pp)
    offsets = range(-radius, radius + 1, step)
    out = []
    for delta in itertools.product(offsets, repeat=pp - 1):
        parts = [b + d for b, d in zip(base, delta)]
        parts.append(num_layers - sum(parts))
        if min(parts) > 0 and parts not in out:
            out.append(parts)
    return out


def _intermediate_size(cfg: dict, mlp_index: int) -> int:
    size = cfg["intermediate_size"]
    if isinstance(size, list):
        return size[0] if len(size) == 1 else size[mlp_index]
    return size


def layer_param_counts(cfg: dict) -> list[int]:
    """Approximate parameter count of every decoder layer."""
    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or h // n_heads
    kv_heads = cfg["num_key_value_heads"]
    m_inner = cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
    m_conv_dim = m_inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"]

    counts, mlp_index = [], 0
    for ch in cfg["hybrid_override_pattern"]:
        if ch == "M":
            in_proj = h * (m_inner + m_conv_dim + cfg["mamba_num_heads"])
            counts.append(in_proj + m_conv_dim * cfg["conv_kernel"] +
                          m_inner * h + 3 * cfg["mamba_num_heads"] + m_inner + h)
        elif ch == "*":
            counts.append(h * (n_heads + 2 * kv_heads) * head_dim +
                          n_heads * head_dim * h + h)
        else:
            counts.append(2 * h * _intermediate_size(cfg, mlp_index) + h)
            mlp_index += 1
    return counts


def stage_table(cfg: dict, partition: list[int], dtype_bytes: int = 2,
                ssm_state_bytes: int = 4, tp: int = 1) -> list[dict]:
    pattern = cfg["hybrid_override_pattern"]
    h, vocab = cfg["hidden_size"], cfg["vocab_size"]
    head_dim = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    kv_heads = max(1, cfg["num_key_value_heads"] // tp)
    m_heads = cfg["mamba_num_heads"] // tp
    m_inner = m_heads * cfg["mamba_head_dim"]
    m_conv_dim = m_inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"] // tp
    conv_state = (cfg["conv_kernel"] - 1) * m_conv_dim * dtype_bytes
    ssm_state = m_heads * cfg["mamba_head_dim"] * cfg["ssm_state_size"] \
        * ssm_state_bytes
    params = layer_param_counts(cfg)

    rows, start = [], 0
    for rank, n in enumerate(partition):
        end = start + n
        seg = pattern[start:end]
        weight = sum(params[start:end])
        # vLLM v0.10.1.1 NemotronH builds embed_tokens and lm_head on every
        # stage, not only on the first / last one (docs/findings.md).
        weight += 2 * vocab * h
        n_attn, n_mamba = seg.count("*"), seg.count("M")
        rows.append({
            "pp_rank": rank,
            "layers": f"[{start},{end})",
            "pattern": seg,
            "mamba": n_mamba,
            "attention": n_attn,
            "mlp": seg.count("-"),
            "weights_gib": weight * dtype_bytes / tp / GIB,
            "kv_bytes_per_token": 2 * kv_heads * head_dim * dtype_bytes * n_attn,
            "mamba_bytes_per_seq": n_mamba * (conv_state + ssm_state),
        })
        start = end
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'rank':>4} {'layers':>9} {'M':>3} {'*':>3} {'-':>3} "
          f"{'weights':>9} {'KV/token':>10} {'mamba/seq':>10}  pattern")
    for r in rows:
        print(f"{r['pp_rank']:>4} {r['layers']:>9} {r['mamba']:>3} "
              f"{r['attention']:>3} {r['mlp']:>3} {r['weights_gib']:>7.2f}Gi "
              f"{r['kv_bytes_per_token'] / 1024:>8.1f}Ki "
              f"{r['mamba_bytes_per_seq'] / (1 << 20):>8.1f}Mi  {r['pattern']}")
    if any(r["attention"] == 0 for r in rows):
        print("NOTE: a stage has no attention layer; vLLM v0.10.1.1 V0 fails "
              "to start for this partition unless the hybrid_pp plugin's "
              "HYBRID_PP_FIX_NOATTN is on (default).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--partition", help="e.g. 28,28 (VLLM_PP_LAYER_PARTITION)")
    ap.add_argument("--candidates", type=int, metavar="RADIUS",
                    help="list partitions within RADIUS layers of even split")
    ap.add_argument("--step", type=int, default=1,
                    help="layer step for --candidates")
    ap.add_argument("--ssm-state-bytes", type=int, default=4,
                    help="4 for --mamba-ssm-cache-dtype float32, 2 for bf16")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.model)
    pattern = cfg["hybrid_override_pattern"]
    n = len(pattern)

    if args.candidates is not None:
        for parts in candidate_partitions(n, args.pp, args.candidates,
                                          args.step):
            print(",".join(map(str, parts)))
        return

    partition = (parse_partition(args.partition, n, args.pp)
                 if args.partition else default_partition(n, args.pp))
    rows = stage_table(cfg, partition, ssm_state_bytes=args.ssm_state_bytes,
                       tp=args.tp)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{n} layers: {pattern.count('M')} mamba, {pattern.count('*')} "
          f"attention, {pattern.count('-')} mlp; partition "
          f"{','.join(map(str, partition))}")
    print_table(rows)


if __name__ == "__main__":
    main()
