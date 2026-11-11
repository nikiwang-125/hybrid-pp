"""Baseline cost of one layout switch on a single stage (plan step 7).

Moving k layers onto a GPU whose free memory is held by the KV pool means:
free the KV pool -> load the k layers' weights -> rebuild a smaller KV pool
(-> re-capture CUDA graphs, since they bake in the KV addresses) and, for
requests in flight, move the layers' KV / Mamba state to the new owner.

  micro : plain PyTorch, mirrors vLLM V0 allocation (CacheEngine zeros one
          tensor per attention layer; MambaCacheManager holds fp32 ssm state
          per max_num_seqs). Times every step for attention vs Mamba layers.
  sleep : real vLLM engine (PP=1) with enable_sleep_mode: how long releasing
          and restoring weights / KV takes, and what memory stays resident.

    python -m hybrid_pp.reconfig_cost micro --out results/reconfig/micro.json
    python -m hybrid_pp.reconfig_cost sleep --out results/reconfig/sleep.json
"""

import argparse
import json
import os
import time

import torch

GiB = 2**30
MiB = 2**20


def sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


class Timer:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __enter__(self):
        sync_all()
        self.t = time.perf_counter()

    def __exit__(self, *a):
        sync_all()
        self.rec.setdefault(self.name, []).append(
            (time.perf_counter() - self.t) * 1000)


def model_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, local_files_only=True)


# ---------------------------------------------------------------- micro

def layer_tensor_names(index, layer):
    prefix = f"backbone.layers.{layer}."
    return sorted(n for n in index if n.startswith(prefix))


def load_from_disk(mdir, index, names, dev):
    from safetensors import safe_open
    by_file = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    out = {}
    for f, ns in by_file.items():
        with safe_open(os.path.join(mdir, f), framework="pt", device="cpu") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n).to(dev, non_blocking=False)
    return out


def alloc_kv(n_layers, gib, dev, kv_heads=8, head_dim=128, block=16):
    per_block_layer = 2 * block * kv_heads * head_dim * 2  # bf16 K+V
    nb = max(int(gib * GiB // (per_block_layer * n_layers)), 1)
    return [torch.zeros((2, nb, block, kv_heads, head_dim),
                        dtype=torch.bfloat16, device=dev)
            for _ in range(n_layers)], nb


def micro(args):
    cfg = json.load(open(os.path.join(model_dir(args.model), "config.json")))
    mdir = model_dir(args.model)
    index = json.load(open(os.path.join(
        mdir, "model.safetensors.index.json")))["weight_map"]
    pat = cfg["hybrid_override_pattern"]
    dev = torch.device("cuda:0")
    peer = torch.device("cuda:1") if torch.cuda.device_count() > 1 else None
    nh, hd, N = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"]
    conv_dim = nh * hd + 2 * cfg["n_groups"] * N
    kv_tok = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2

    report = {"config": vars(args), "gpu": torch.cuda.get_device_name(0),
              "peer": torch.cuda.get_device_name(1) if peer else None,
              "p2p": bool(peer) and torch.cuda.can_device_access_peer(0, 1),
              "cases": {}}

    for kind, sym in (("attention", "*"), ("mamba", "M"), ("mlp", "-")):
        layers = [i for i, c in enumerate(pat) if c == sym][:args.layers]
        names = [n for l in layers for n in layer_tensor_names(index, l)]
        rec = {}
        wbytes = 0
        # host copy of the weights, pinned: the case where the new owner
        # already staged them in RAM
        pinned = {n: t.pin_memory() for n, t in
                  load_from_disk(mdir, index, names, "cpu").items()}
        wbytes = sum(t.numel() * t.element_size() for t in pinned.values())

        for r in range(args.repeat + 1):  # first round is warm-up
            rr = rec if r else {}
            torch.cuda.empty_cache()
            kv, nb = alloc_kv(args.kv_layers, args.kv_gib, dev)
            with Timer(rr, "1_free_kv_pool"):
                del kv
                torch.cuda.empty_cache()
            with Timer(rr, "2a_load_weights_disk"):
                w = load_from_disk(mdir, index, names, dev)
            del w
            torch.cuda.empty_cache()
            with Timer(rr, "2b_load_weights_pinned_ram"):
                w = {n: t.to(dev, non_blocking=True) for n, t in pinned.items()}
            new_gib = args.kv_gib - wbytes / GiB
            with Timer(rr, "3_rebuild_kv_pool"):
                kv, nb2 = alloc_kv(args.kv_layers, new_gib, dev)
            del kv, w
            torch.cuda.empty_cache()

            # state carried by in-flight requests for these layers
            if kind == "mamba":
                st = [torch.empty((args.seqs, nh, hd, N), dtype=torch.float32, device=dev),
                      torch.empty((args.seqs, conv_dim, cfg["conv_kernel"] - 1),
                                  dtype=torch.bfloat16, device=dev)] * len(layers)
            elif kind == "attention":
                st = [torch.empty((2, args.seqs * args.ctx, cfg["num_key_value_heads"],
                                   cfg["head_dim"]), dtype=torch.bfloat16, device=dev)
                      for _ in layers]
            else:
                st = []
            sbytes = sum(t.numel() * t.element_size() for t in st)
            if st:
                if peer:
                    with Timer(rr, "4a_state_to_peer_gpu"):
                        moved = [t.to(peer) for t in st]
                    del moved
                with Timer(rr, "4b_state_gpu_cpu_gpu"):
                    moved = [t.cpu().to(dev) for t in st]
                del moved
            del st
            torch.cuda.empty_cache()

        case = {"layers": layers, "weight_MiB": wbytes / MiB,
                "state_MiB": sbytes / MiB, "kv_blocks_before": nb,
                "kv_blocks_after": nb2,
                "ms": {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)}
                       for k, v in sorted(rec.items())}}
        report["cases"][kind] = case
        print(f"\n== move {len(layers)} {kind} layers {layers}: weights "
              f"{wbytes / MiB:.0f} MiB, in-flight state {sbytes / MiB:.0f} MiB "
              f"({args.seqs} seqs" + (f" x {args.ctx} tok)" if kind == "attention" else ")"))
        print(f"   KV pool {args.kv_gib:.1f} GiB on {args.kv_layers} attn layers: "
              f"{nb} -> {nb2} blocks")
        for k, v in case["ms"].items():
            print(f"   {k:<28} {v['mean']:9.1f} ms  (min {v['min']:.1f}, max {v['max']:.1f})")
    return report


# ---------------------------------------------------------------- sleep

def sleep(args):
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(model=args.model, enable_sleep_mode=True, trust_remote_code=True,
              max_model_len=args.max_model_len, max_num_seqs=args.max_seqs,
              gpu_memory_utilization=args.gpu_util, enforce_eager=args.eager)
    rep = {"config": vars(args), "engine_init_s": time.perf_counter() - t0}
    sp = SamplingParams(temperature=0, max_tokens=32)
    prompt = ["The capital of France is", "Pipeline parallelism splits a model"]

    def used():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / GiB

    def gen():
        return [o.outputs[0].text for o in llm.generate(prompt, sp, use_tqdm=False)]

    ref = gen()
    rep["used_GiB_awake"] = used()
    rounds = []
    for r in range(args.repeat):
        d = {}
        t = time.perf_counter(); llm.sleep(level=1); d["sleep_l1_s"] = time.perf_counter() - t
        d["used_GiB_asleep"] = used()
        t = time.perf_counter(); llm.wake_up(tags=["weights"]); d["wake_weights_s"] = time.perf_counter() - t
        d["used_GiB_weights_only"] = used()
        t = time.perf_counter(); llm.wake_up(tags=["kv_cache"]); d["wake_kv_s"] = time.perf_counter() - t
        d["used_GiB_awake"] = used()
        out = gen()
        d["same_output"] = out == ref
        rounds.append(d)
        print(f"round {r}: " + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
    rep["rounds"] = rounds
    rep["note"] = ("resident while asleep = memory outside the weights/kv_cache "
                   "pools: Mamba state (allocated during profile_run), CUDA "
                   "graph pools, activations workspace, CUDA context")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["micro", "sleep"])
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    # micro
    ap.add_argument("--layers", type=int, default=3, help="layers moved per switch")
    ap.add_argument("--kv-gib", type=float, default=6.0, help="KV pool before the switch")
    ap.add_argument("--kv-layers", type=int, default=2, help="attention layers on the stage")
    ap.add_argument("--seqs", type=int, default=32, help="in-flight requests")
    ap.add_argument("--ctx", type=int, default=4096, help="tokens per request (attention KV)")
    # sleep
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    rep = micro(args) if args.mode == "micro" else sleep(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
