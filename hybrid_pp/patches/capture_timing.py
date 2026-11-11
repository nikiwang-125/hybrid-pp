"""Millisecond timing of V0 CUDA graph capture per stage.

vLLM logs "Graph capturing finished in N secs" rounded to whole seconds.
With HYBRID_PP_CAPTURE_TIMING=1 this wraps ModelRunner.capture_model and
logs one line per worker (HYBRID_PP_CAPTURE_NO_GC=1 additionally skips the
per-graph gc.collect() calls during capture):

  [hybrid_pp] capture_timing {"pp_rank": .., "virtual_engines": ..,
      "batch_sizes": [...], "graphs": .., "stage_layers": .., "ms": ..,
      "graph_mem_mib": ..}
"""

import functools
import gc
import json
import logging
import os
import time

import torch

logger = logging.getLogger("vllm.hybrid_pp")


def apply() -> None:
    from vllm.worker.model_runner import GPUModelRunnerBase

    orig = GPUModelRunnerBase.capture_model
    if getattr(orig, "_hybrid_pp_wrapped", False):
        return

    @functools.wraps(orig)
    def capture_model(self, kv_caches):
        from vllm.distributed import get_pp_group
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]
        no_gc = os.environ.get("HYBRID_PP_CAPTURE_NO_GC", "0") not in ("0", "", "false")
        real_collect = gc.collect
        if no_gc:
            # torch.cuda.graph.__enter__ and vLLM's capture loop each call a
            # full gc.collect() per graph; skip them only while capturing.
            gc.collect = lambda *a, **k: 0
        t0 = time.perf_counter()
        try:
            out = orig(self, kv_caches)
        finally:
            gc.collect = real_collect
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        free1 = torch.cuda.mem_get_info()[0]
        sizes = list(self.vllm_config.compilation_config.cudagraph_capture_sizes or [])
        model = getattr(self.model, "model", None)
        start = getattr(model, "start_layer", None)
        end = getattr(model, "end_layer", None)
        rec = {"pp_rank": get_pp_group().rank_in_group,
               "virtual_engines": len(kv_caches),
               "batch_sizes": sizes,
               "graphs": len(kv_caches) * len(sizes),
               "stage_layers": (end - start) if start is not None and end is not None else None,
               "ms": round(ms, 1),
               "no_gc": no_gc,
               "graph_mem_mib": round((free0 - free1) / 2**20, 1)}
        logger.warning("[hybrid_pp] capture_timing %s", json.dumps(rec))
        return out

    capture_model._hybrid_pp_wrapped = True
    GPUModelRunnerBase.capture_model = capture_model
