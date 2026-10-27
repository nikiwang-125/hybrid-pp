"""Per-stage KV-cache accounting on the V0 worker (vLLM v0.10.1.1).

V0 decides the KV-cache size like this (see docs/findings.md):
  * each worker profiles its own memory and returns
    ``available_kv_bytes // cache_block_size`` blocks, where
    ``cache_block_size`` only counts the attention layers on that stage;
  * the executor takes ``min()`` over all workers and every stage allocates
    that many blocks.

This module wraps ``Worker.determine_num_available_blocks`` to

1. log one ``[hybrid_pp] stage_mem {json}`` line per worker, so
   ``scripts/parse_stage_memory.py`` can compute how much memory each stage
   leaves unused after the ``min()``;
2. (fix) let a stage that holds no attention layer opt out of the ``min()``.
   Upstream returns 0 blocks for such a stage, so the whole engine fails with
   "No available memory for the cache blocks". This happens e.g. for
   Nemotron Nano 9B v2 with PP=4 and the default even split.
"""

import json
import logging
import re

logger = logging.getLogger("vllm.hybrid_pp")  # inherits vLLM log handlers

# Large enough never to win the executor's min() against a real stage.
_NO_ATTENTION_BLOCKS = 1 << 40

_PROFILE_RE = re.compile(
    r"= (?P<instance>[\d.]+)GiB\s+model weights take (?P<weights>[\d.]+)GiB;"
    r" non_torch_memory takes (?P<non_torch>[-\d.]+)GiB;"
    r" PyTorch activation peak memory takes (?P<activation>[-\d.]+)GiB;"
    r" the rest of the memory reserved for KV Cache is (?P<kv>[-\d.]+)GiB")


class _CaptureProfileLog(logging.Filter):
    """Grabs the memory-profiling message the worker logs anyway."""

    def __init__(self):
        super().__init__()
        self.fields = {}

    def filter(self, record: logging.LogRecord) -> bool:
        m = _PROFILE_RE.search(record.getMessage())
        if m:
            self.fields = {k + "_gib": float(v) for k, v in m.groupdict().items()}
        return True


def _stage_layers(worker):
    model = getattr(worker.model_runner, "model", None)
    inner = getattr(model, "model", None)
    start = getattr(inner, "start_layer", None)
    end = getattr(inner, "end_layer", None)
    pattern = getattr(getattr(inner, "config", None), "hybrid_override_pattern",
                      None)
    return start, end, pattern


def apply(fix_no_attention: bool, log_stage_memory: bool) -> None:
    from vllm.distributed import (get_pp_group,
                                  get_tensor_model_parallel_rank)
    from vllm.worker import worker as worker_mod

    orig = worker_mod.Worker.determine_num_available_blocks

    def determine_num_available_blocks(self):
        capture = _CaptureProfileLog()
        worker_mod.logger.addFilter(capture)
        try:
            num_gpu_blocks, num_cpu_blocks = orig(self)
        finally:
            worker_mod.logger.removeFilter(capture)

        block_bytes = self.get_cache_block_size_bytes()
        no_attention = (block_bytes == 0
                        and not self.cache_config.is_attention_free)

        if log_stage_memory:
            start, end, pattern = _stage_layers(self)
            info = {
                "pp_rank": get_pp_group().rank_in_group,
                "tp_rank": get_tensor_model_parallel_rank(),
                "start_layer": start,
                "end_layer": end,
                "stage_pattern": (pattern[start:end]
                                  if pattern is not None and start is not None
                                  else None),
                "block_size_tokens": self.cache_config.block_size,
                "cache_block_bytes": block_bytes,
                "local_num_gpu_blocks": num_gpu_blocks,
                **capture.fields,
            }
            logger.info("[hybrid_pp] stage_mem %s", json.dumps(info))

        if no_attention and fix_no_attention:
            logger.info("[hybrid_pp] stage without attention layers: "
                        "excluded from the KV block min()")
            return _NO_ATTENTION_BLOCKS, _NO_ATTENTION_BLOCKS
        return num_gpu_blocks, num_cpu_blocks

    worker_mod.Worker.determine_num_available_blocks = (
        determine_num_available_blocks)
    logger.info("[hybrid_pp] patched Worker.determine_num_available_blocks")
