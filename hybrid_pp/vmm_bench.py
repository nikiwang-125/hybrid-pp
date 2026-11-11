"""Partial KV shrink with CUDA virtual memory vs. free-and-reallocate (2026-11-11).

Answers: can part of a KV pool be handed to new layer weights without
moving the pool, losing the KV that stays, or invalidating CUDA graphs?

  vmm   : reserve a virtual range for KV, map it page by page (cuMemCreate /
          cuMemMap). Shrink = unmap + release the last pages; the freed
          physical memory backs the new weights (own virtual range); then
          grow back. KV base address never changes.
  torch : today's vLLM V0 path. Free the KV tensor, empty_cache, allocate
          the weights, allocate a smaller KV tensor (new address, contents
          gone).
Both: a CUDA graph that reads the kept part of KV is captured before the
shrink and replayed after it; its result is checked.

    python3 -m hybrid_pp.vmm_bench --out results/vmm_20261111/vmm.json
"""

import argparse
import ctypes
import json
import os
import statistics
import time

import torch

MiB = 2**20
cu = ctypes.CDLL("libcuda.so.1")
CUdeviceptr = ctypes.c_uint64
CUhandle = ctypes.c_uint64


class AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", Location), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", AllocFlags)]


class AccessDesc(ctypes.Structure):
    _fields_ = [("location", Location), ("flags", ctypes.c_int)]


def ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} failed: CUresult {r}")


class VMM:
    def __init__(self, dev):
        self.prop = Prop()
        self.prop.type = 1                 # CU_MEM_ALLOCATION_TYPE_PINNED
        self.prop.location.type = 1        # CU_MEM_LOCATION_TYPE_DEVICE
        self.prop.location.id = dev
        g = ctypes.c_size_t()
        ck(cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self.prop), 0), "granularity")
        self.gran = g.value
        self.acc = AccessDesc()
        self.acc.location.type, self.acc.location.id = 1, dev
        self.acc.flags = 3                 # PROT_READWRITE

    def reserve(self, size):
        p = CUdeviceptr()
        ck(cu.cuMemAddressReserve(ctypes.byref(p), ctypes.c_size_t(size), ctypes.c_size_t(0),
                                  CUdeviceptr(0), ctypes.c_ulonglong(0)), "reserve")
        return p.value

    def map_page(self, va, size):
        h = CUhandle()
        ck(cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(self.prop),
                          ctypes.c_ulonglong(0)), "create")
        ck(cu.cuMemMap(CUdeviceptr(va), ctypes.c_size_t(size), ctypes.c_size_t(0), h,
                       ctypes.c_ulonglong(0)), "map")
        ck(cu.cuMemSetAccess(CUdeviceptr(va), ctypes.c_size_t(size), ctypes.byref(self.acc),
                             ctypes.c_size_t(1)), "access")
        return h.value

    def unmap_page(self, va, size, h):
        ck(cu.cuMemUnmap(CUdeviceptr(va), ctypes.c_size_t(size)), "unmap")
        ck(cu.cuMemRelease(CUhandle(h)), "release")


def as_tensor(va, nbytes):
    import cupy
    mem = cupy.cuda.UnownedMemory(va, nbytes, owner=None)
    arr = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(mem, 0))
    return torch.as_tensor(arr, device="cuda")


def sync():
    torch.cuda.synchronize()


def timed(rec, name, fn):
    sync(); t = time.perf_counter(); v = fn(); sync()
    rec.setdefault(name, []).append((time.perf_counter() - t) * 1000)
    return v


def graph_reading(t):
    """Capture a graph that reduces over tensor t; returns (graph, out)."""
    out = torch.zeros((), dtype=torch.int64, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out.copy_(t.view(torch.int64).sum())  # reinterpret bytes, no cast copy
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(t.view(torch.int64).sum())  # reinterpret bytes, no cast copy
    return g, out


def run_vmm(a, rec, checks):
    vm = VMM(torch.cuda.current_device())
    page = max(a.page_mib * MiB, vm.gran)
    kv_bytes = (a.kv_mib * MiB // page) * page
    w_bytes = -(-a.weights_mib * MiB // page) * page
    n_pages, n_free = kv_bytes // page, w_bytes // page
    kv_va = vm.reserve(kv_bytes)
    handles = [vm.map_page(kv_va + i * page, page) for i in range(n_pages)]
    kv = as_tensor(kv_va, kv_bytes)
    kv.fill_(29)
    keep = (n_pages - n_free) * page
    g, out = graph_reading(kv[:keep])
    g.replay(); sync(); ref = out.item()
    w_va = vm.reserve(w_bytes)
    src = torch.full((w_bytes,), 41, dtype=torch.uint8, pin_memory=True)
    for r in range(a.rounds):
        tail = range(n_pages - n_free, n_pages)
        timed(rec, "1_shrink_kv_unmap_release", lambda: [vm.unmap_page(kv_va + i * page, page, handles[i]) for i in tail])
        wh = timed(rec, "2_map_weight_pages", lambda: [vm.map_page(w_va + i * page, page) for i in range(n_free)])
        w = as_tensor(w_va, w_bytes)
        timed(rec, "3_copy_weights_from_pinned", lambda: w.copy_(src, non_blocking=True))
        timed(rec, "4_replay_graph_on_kept_kv", lambda: g.replay())
        checks.append({"round": r, "kv_base_unchanged": kv.data_ptr() == kv_va,
                       "graph_result_ok": out.item() == ref,
                       "kept_kv_intact": bool(torch.all(kv[:keep] == 29).item()),
                       "weights_ok": bool(torch.all(w == 41).item())})
        del w
        timed(rec, "5_release_weight_pages", lambda: [vm.unmap_page(w_va + i * page, page, h) for i, h in enumerate(wh)])
        def regrow():
            for i in tail:
                handles[i] = vm.map_page(kv_va + i * page, page)
        timed(rec, "6_grow_kv_back", regrow)
    return {"page_bytes": page, "kv_pages": n_pages, "pages_handed_to_weights": n_free,
            "kv_bytes": kv_bytes, "weight_bytes": w_bytes}


def run_torch(a, rec, checks):
    kv_bytes, w_bytes = a.kv_mib * MiB, a.weights_mib * MiB
    keep = kv_bytes - w_bytes
    src = torch.full((w_bytes,), 41, dtype=torch.uint8, pin_memory=True)
    kv = torch.full((kv_bytes,), 29, dtype=torch.uint8, device="cuda")
    for r in range(a.rounds):
        g, out = graph_reading(kv[:keep])
        g.replay(); sync(); ref = out.item(); base = kv.data_ptr()
        def free():
            nonlocal kv
            kv = None
            torch.cuda.empty_cache()
        timed(rec, "1_free_kv_and_empty_cache", free)
        w = timed(rec, "2_alloc_weights", lambda: torch.empty(w_bytes, dtype=torch.uint8, device="cuda"))
        timed(rec, "3_copy_weights_from_pinned", lambda: w.copy_(src, non_blocking=True))
        kv = timed(rec, "4_alloc_smaller_kv_zeroed", lambda: torch.zeros(keep, dtype=torch.uint8, device="cuda"))
        checks.append({"round": r, "kv_base_unchanged": kv.data_ptr() == base,
                       "kept_kv_intact": bool(torch.all(kv == 29).item()),
                       "graph_valid": "no: graph points at freed memory, must be recaptured (not replayed: unsafe)"})
        del w, g
        kv = None; torch.cuda.empty_cache()
        kv = torch.full((kv_bytes,), 29, dtype=torch.uint8, device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-mib", type=int, default=6144)
    ap.add_argument("--weights-mib", type=int, default=844, help="3 Mamba layers of Nemotron-H 9B")
    ap.add_argument("--page-mib", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    torch.cuda.init(); torch.zeros(1, device="cuda")
    rep = {"config": vars(a), "gpu": torch.cuda.get_device_name(0)}
    for name, fn in (("vmm", run_vmm), ("torch", run_torch)):
        rec, checks = {}, []
        info = fn(a, rec, checks) or {}
        rep[name] = {"info": info, "checks": checks,
                     "ms_median_excl_first": {k: statistics.median(v[1:] or v) for k, v in rec.items()},
                     "ms_first": {k: v[0] for k, v in rec.items()}}
        print(f"== {name} {info}")
        for k, v in rep[name]["ms_median_excl_first"].items():
            print(f"   {k:30s} {v:9.2f} ms (first {rep[name]['ms_first'][k]:.2f})")
        print("   checks:", checks[-1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
