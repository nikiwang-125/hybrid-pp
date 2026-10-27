"""Per-step and per-layer GPU timing for NemotronH, one JSONL file per worker.

Enabled by HYBRID_PP_PROFILE_DIR. Requires ``--enforce-eager``: with CUDA
graphs the decode steps replay a captured graph and never reach Python, so
they would be missing from the trace (steps during graph capture are skipped).

Timing uses CUDA events only, never a device synchronize, so the pipeline's
overlap between stages is not disturbed. Events are resolved lazily once the
GPU has passed them.

Record format (one JSON object per line):
  {"kind": "meta", pp_rank, tp_rank, start_layer, end_layer, stage_pattern,
   stage_param_bytes, ...}
  {"kind": "mamba_cache", "bytes": ...}            (first step only)
  {"kind": "step", step, t_start_ms, dur_ms, num_prefills, num_prefill_tokens,
   num_decode_tokens, batch, ctx_sum, ctx_max, wall,
   layers: [[layer_idx, "mamba"|"attention"|"mlp", ms], ...]}
``t_start_ms`` is on this GPU's timeline, relative to the first recorded step.
"""

import atexit
import functools
import json
import logging
import os
import time

import torch

logger = logging.getLogger("vllm.hybrid_pp")  # inherits vLLM log handlers

_LAYER_TYPES = {
    "NemotronHMambaDecoderLayer": "mamba",
    "NemotronHMLPDecoderLayer": "mlp",
    "NemotronHAttentionDecoderLayer": "attention",
}

_recorder = None


def _new_event():
    return torch.cuda.Event(enable_timing=True)


class _Recorder:

    def __init__(self, out_dir: str, profile_layers: bool):
        self.out_dir = out_dir
        self.profile_layers = profile_layers
        self.file = None
        self.base_event = None
        self.step = 0
        self.pending = []
        self.current = None
        self.wrote_mamba_cache = False

    # -- output -------------------------------------------------------------

    def _write(self, obj):
        self.file.write(json.dumps(obj) + "\n")

    def _open(self, model):
        from vllm.distributed import (get_pp_group,
                                      get_tensor_model_parallel_rank)
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tensor_model_parallel_rank()
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir,
                            f"pp{pp_rank}_tp{tp_rank}_pid{os.getpid()}.jsonl")
        self.file = open(path, "a", buffering=1)
        pattern = model.config.hybrid_override_pattern
        self._write({
            "kind": "meta",
            "pp_rank": pp_rank,
            "pp_size": get_pp_group().world_size,
            "tp_rank": tp_rank,
            "start_layer": model.start_layer,
            "end_layer": model.end_layer,
            "stage_pattern": pattern[model.start_layer:model.end_layer],
            "stage_param_bytes": sum(p.numel() * p.element_size()
                                     for p in model.parameters()),
            "device": torch.cuda.get_device_name(),
            "profile_layers": self.profile_layers,
            "wall": time.time(),
        })
        atexit.register(self.flush)
        logger.info("[hybrid_pp] profiling to %s", path)

    def _drain(self, block: bool = False):
        while self.pending:
            rec = self.pending[0]
            if block:
                rec["end"].synchronize()
            elif not rec["end"].query():
                return
            self.pending.pop(0)
            out = rec["info"]
            out["t_start_ms"] = self.base_event.elapsed_time(rec["start"])
            out["dur_ms"] = rec["start"].elapsed_time(rec["end"])
            out["layers"] = [[idx, kind, s.elapsed_time(e)]
                             for idx, kind, s, e in rec["layers"]]
            self._write(out)

    def flush(self):
        if self.file is not None and not self.file.closed:
            self._drain(block=True)
            self.file.flush()

    # -- hooks --------------------------------------------------------------

    def begin_step(self, model, mamba_cache_params):
        if self.file is None:
            self._open(model)
            self.base_event = _new_event()
            self.base_event.record()
        if not self.wrote_mamba_cache and mamba_cache_params is not None:
            self.wrote_mamba_cache = True
            nbytes = sum(t.numel() * t.element_size()
                         for t in (mamba_cache_params.conv_state,
                                   mamba_cache_params.ssm_state))
            self._write({"kind": "mamba_cache", "bytes": nbytes})

        info = {"kind": "step", "step": self.step, "wall": time.time()}
        info.update(_batch_features())
        start = _new_event()
        start.record()
        self.current = {"info": info, "start": start, "layers": []}
        self.step += 1

    def end_step(self):
        end = _new_event()
        end.record()
        self.current["end"] = end
        self.pending.append(self.current)
        self.current = None
        self._drain()


def _batch_features():
    """Batch shape from the V0 attention metadata (host-side fields only)."""
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    if md is None:
        return {}
    seq_lens = getattr(md, "seq_lens", None) or []
    return {
        "num_prefills": getattr(md, "num_prefills", None),
        "num_prefill_tokens": getattr(md, "num_prefill_tokens", None),
        "num_decode_tokens": getattr(md, "num_decode_tokens", None),
        "batch": len(seq_lens),
        "ctx_sum": int(sum(seq_lens)),
        "ctx_max": int(max(seq_lens)) if seq_lens else 0,
    }


def _active():
    return (_recorder is not None and _recorder.current is not None
            and not torch.cuda.is_current_stream_capturing())


def _wrap_layer(cls, kind):
    orig_init, orig_forward = cls.__init__, cls.forward

    @functools.wraps(orig_init)
    def __init__(self, config, layer_idx, *args, **kwargs):
        orig_init(self, config, layer_idx, *args, **kwargs)
        self._hybrid_pp_idx = layer_idx

    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        if not (_active() and _recorder.profile_layers):
            return orig_forward(self, *args, **kwargs)
        start, end = _new_event(), _new_event()
        start.record()
        out = orig_forward(self, *args, **kwargs)
        end.record()
        _recorder.current["layers"].append(
            (self._hybrid_pp_idx, kind, start, end))
        return out

    cls.__init__, cls.forward = __init__, forward


def apply(out_dir: str, profile_layers: bool) -> None:
    global _recorder
    from vllm.model_executor.models import nemotron_h

    _recorder = _Recorder(out_dir, profile_layers)

    for name, kind in _LAYER_TYPES.items():
        _wrap_layer(getattr(nemotron_h, name), kind)

    model_cls = nemotron_h.NemotronHModel
    orig_forward = model_cls.forward  # already the PP-fixed forward if enabled

    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            return orig_forward(self, *args, **kwargs)
        mamba_cache_params = (args[2] if len(args) > 2 else
                              kwargs.get("mamba_cache_params"))
        _recorder.begin_step(self, mamba_cache_params)
        out = orig_forward(self, *args, **kwargs)
        _recorder.end_step()
        return out

    model_cls.forward = forward
    logger.info("[hybrid_pp] NemotronH profiler enabled (layers=%s)",
                profile_layers)
