"""Stage-boundary send/recv time, intra-node vs cross-node (plan step 4).

The tensors are what NemotronH passes between PP stages: IntermediateTensors
{hidden_states, residual}, each [tokens, hidden] bf16. Two Ray actors (one GPU
each) form a 2-rank NCCL group and exchange them.

Two timings per size, both measured on rank 0 with host wall clock after
torch.cuda.synchronize():
  pingpong_ms / 2 : rank0 send -> rank1 recv -> rank1 send -> rank0 recv.
                    Both sides are already waiting, so this is transfer
                    time without "downstream not ready" waits.
  oneway_ms       : barrier, then rank0 send, rank1 recv; time on rank1
                    from after the barrier to recv complete. Includes the
                    barrier skew, kept only as a cross-check.
vLLM additionally sends a small metadata dict over the CPU (gloo) group per
step; that cost is not included here (see note in the output JSON).

    python3 -m hybrid_pp.comm_bench --out results/comm/comm.json
"""

import argparse
import json
import os
import socket
import statistics
import time

import ray

DECODE = [1, 2, 4, 8, 16, 32, 64]
PREFILL = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


@ray.remote(num_gpus=1)
class Peer:
    def setup(self, rank, addr, port):
        import torch
        import torch.distributed as dist
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{addr}:{port}",
                                rank=rank, world_size=2)
        self.rank = rank
        return socket.gethostname(), torch.cuda.get_device_name(0), \
            os.environ.get("CUDA_VISIBLE_DEVICES")

    def ip(self):
        return ray.util.get_node_ip_address()

    def free_port(self):
        s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
        return p

    def run(self, tokens, hidden, warmup, iters):
        import torch
        import torch.distributed as dist
        t = [torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
             for _ in range(2)]  # hidden_states, residual
        r = [torch.empty_like(x) for x in t]
        peer = 1 - self.rank

        def pingpong():
            if self.rank == 0:
                for x in t: dist.send(x, peer)
                for x in r: dist.recv(x, peer)
            else:
                for x in r: dist.recv(x, peer)
                for x in t: dist.send(x, peer)
            torch.cuda.synchronize()

        def oneway():
            dist.barrier(); torch.cuda.synchronize()
            s = time.perf_counter()
            if self.rank == 0:
                for x in t: dist.send(x, peer)
            else:
                for x in r: dist.recv(x, peer)
            torch.cuda.synchronize()
            return (time.perf_counter() - s) * 1000

        for _ in range(warmup):
            pingpong()
        pp = []
        for _ in range(iters):
            dist.barrier(); torch.cuda.synchronize()
            s = time.perf_counter(); pingpong()
            pp.append((time.perf_counter() - s) * 1000 / 2)
        ow = [oneway() for _ in range(iters)]
        return {"pingpong_half_ms": pp, "oneway_ms": ow}


def summarize(v):
    v = sorted(v)
    return {"median": statistics.median(v), "p10": v[len(v) // 10],
            "p90": v[(9 * len(v)) // 10], "min": v[0], "n": len(v)}


def fit(points):
    """Least squares t = a + bytes / bw over (bytes, ms)."""
    n = len(points)
    mx = sum(b for b, _ in points) / n
    my = sum(t for _, t in points) / n
    sxx = sum((b - mx) ** 2 for b, _ in points)
    slope = sum((b - mx) * (t - my) for b, t in points) / sxx
    a = my - slope * mx
    return {"latency_ms": a, "bandwidth_GBps": 1 / slope / 1e6}


def pair(nodes, hidden, warmup, iters):
    peers = [Peer.options(resources={f"node:{ip}": 0.001}).remote() for ip in nodes]
    addr = nodes[0]
    port = ray.get(peers[0].free_port.remote())
    info = ray.get([p.setup.remote(i, addr, port) for i, p in enumerate(peers)])
    rows = []
    for phase, sizes in (("decode", DECODE), ("prefill", PREFILL)):
        for tok in sizes:
            res = ray.get([p.run.remote(tok, hidden, warmup, iters) for p in peers])
            nbytes = 2 * tok * hidden * 2
            row = {"phase": phase, "tokens": tok, "bytes": nbytes,
                   "pingpong_half_ms": summarize(res[0]["pingpong_half_ms"]),
                   "oneway_ms": summarize(res[1]["oneway_ms"])}
            rows.append(row)
            print(f"  {phase:7s} {tok:6d} tok {nbytes / 2**20:8.2f} MiB  "
                  f"half-RTT {row['pingpong_half_ms']['median']:8.3f} ms  "
                  f"one-way {row['oneway_ms']['median']:8.3f} ms", flush=True)
    for p in peers:
        ray.kill(p)
    pts = [(r["bytes"], r["pingpong_half_ms"]["median"]) for r in rows]
    return {"nodes": nodes, "peers": info, "rows": rows,
            "fit_all": fit(pts),
            "fit_ge_1MiB": fit([x for x in pts if x[0] >= 2**20])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=4480)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--head", default="192.168.0.21")
    ap.add_argument("--other", default="192.168.0.20")
    ap.add_argument("--only", choices=["intra", "cross"])
    args = ap.parse_args()
    import hybrid_pp
    # ship this package so the actor on the other machine imports the same code
    ray.init(address="auto", logging_level="ERROR",
             runtime_env={"py_modules": [os.path.dirname(hybrid_pp.__file__)]})
    rep = {"config": vars(args),
           "note": "NCCL tensors only; vLLM's per-step metadata over gloo "
                   "is not included. Calibrate against real pipeline traces.",
           "pairs": {}}
    cases = {"intra": [args.head, args.head], "cross": [args.head, args.other]}
    for name, nodes in cases.items():
        if args.only and name != args.only:
            continue
        print(f"== {name}: {nodes}", flush=True)
        rep["pairs"][name] = pair(nodes, args.hidden, args.warmup, args.iters)
        f = rep["pairs"][name]
        print(f"   fit (>=1MiB): latency {f['fit_ge_1MiB']['latency_ms']:.3f} ms, "
              f"bandwidth {f['fit_ge_1MiB']['bandwidth_GBps']:.2f} GB/s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
