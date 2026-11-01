"""Per-step and per-op GPU timing for NemotronH, one JSONL file per worker.

Enabled by HYBRID_PP_PROFILE_DIR. What gets timed:

  * every decoder layer ("mamba" / "attention" / "mlp") -- only in steps that
    run eagerly, i.e. all steps with ``--enforce-eager``, otherwise prefill
    steps only (CUDA-graph decode steps replay a captured graph and never
    reach the Python layers);
  * "embed" (first stage), "lm_head" (``compute_logits``: lm_head GEMM plus
    logits processing) and "sampler" (last stage) -- in every step, because
    V0 runs them outside the CUDA graph;
  * the whole stage forward of every step, eager or graph (``graph`` flag);
  * CPU wall time of the worker's ``execute_model`` call around each step,
    split into waiting to receive from the previous stage, the model runner
    (input prep, launches and, on the last stage, the sampler's GPU->CPU
    sync and output building) and sending to the next stage (``cpu``).

Memory profiling and CUDA-graph capture are not recorded.

Timing uses CUDA events only, never a device synchronize, so the overlap
between pipeline stages is not disturbed. Events are resolved lazily once the
GPU has passed them. Note that in eager mode an op's time also includes any
GPU idle time waiting for the CPU to launch its kernels; for small decode
batches that launch overhead can dominate, so compare with graph mode.

Record format (one JSON object per line):
  {"kind": "meta", pp_rank, tp_rank, start_layer, end_layer, stage_pattern,
   stage_param_bytes, ...}
  {"kind": "mamba_cache", "bytes": ...}            (first eager step only)
  {"kind": "step", step, graph, t_start_ms, dur_ms, total_ms, num_prefills,
   num_prefill_tokens, num_decode_tokens, batch, ctx_sum, ctx_max, wall,
   layers: [[layer_idx, kind, ms], ...],
   cpu: {call_start, call_end (epoch s), call_ms, recv_ms, exec_ms, send_ms}}
``dur_ms`` covers the stage's model forward; ``total_ms`` runs until the end
of the last op of the step (lm_head / sampler on the last stage). Ops outside
the decoder layers use layer_idx -1. ``t_start_ms`` is on this GPU's
timeline, relative to the first recorded step.
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
        self.current = None  # step whose model forward is running
        self.last = None  # latest finished forward, still collecting ops
        self.paused = 0
        self.call = None  # CPU timings of the running execute_model call
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
            tail = rec["tail"]
            if block:
                tail.synchronize()
            elif not tail.query():
                return
            self.pending.pop(0)
            if rec is self.last:
                self.last = None
            out = rec["info"]
            out["t_start_ms"] = self.base_event.elapsed_time(rec["start"])
            out["dur_ms"] = rec["start"].elapsed_time(rec["end"])
            out["total_ms"] = rec["start"].elapsed_time(tail)
            out["layers"] = [[idx, kind, s.elapsed_time(e)]
                             for idx, kind, s, e in rec["layers"]]
            self._write(out)

    def flush(self):
        if self.file is not None and not self.file.closed:
            self._drain(block=True)
            self.file.flush()

    # -- hooks --------------------------------------------------------------

    def recording(self) -> bool:
        return not self.paused and not torch.cuda.is_current_stream_capturing()

    def begin_step(self, model, mamba_cache_params, graph: bool):
        if self.file is None:
            self._open(model)
            self.base_event = _new_event()
            self.base_event.record()
        # The previous step's lm_head / sampler ops are all issued by now.
        self.last = None
        self._drain()
        if not self.wrote_mamba_cache and mamba_cache_params is not None:
            self.wrote_mamba_cache = True
            nbytes = sum(t.numel() * t.element_size()
                         for t in (mamba_cache_params.conv_state,
                                   mamba_cache_params.ssm_state))
            self._write({"kind": "mamba_cache", "bytes": nbytes})

        info = {"kind": "step", "step": self.step, "graph": graph,
                "wall": time.time()}
        info.update(_batch_features())
        start = _new_event()
        start.record()
        self.current = {"info": info, "start": start, "layers": []}
        self.step += 1

    def end_step(self):
        end = _new_event()
        end.record()
        rec = self.current
        rec["end"] = rec["tail"] = end
        self.pending.append(rec)
        self.current, self.last = None, rec

    def add_op(self, idx: int, kind: str, start, end):
        rec = self.current or self.last
        if rec is None:
            return
        rec["layers"].append((idx, kind, start, end))
        if rec is self.last:
            rec["tail"] = end


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


def _timed(fn, kind: str, idx_of=lambda self: -1, layer: bool = False):
    """Wrap ``fn(self, ...)`` so each call is recorded as one op."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        rec = _recorder
        if (not rec.recording() or (layer and not rec.profile_layers)
                or (rec.current is None and rec.last is None)):
            return fn(self, *args, **kwargs)
        start, end = _new_event(), _new_event()
        start.record()
        out = fn(self, *args, **kwargs)
        end.record()
        rec.add_op(idx_of(self), kind, start, end)
        return out

    return wrapper


def _wrap_layer(cls, kind):
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, config, layer_idx, *args, **kwargs):
        orig_init(self, config, layer_idx, *args, **kwargs)
        self._hybrid_pp_idx = layer_idx

    cls.__init__ = __init__
    cls.forward = _timed(cls.forward, kind,
                         idx_of=lambda self: self._hybrid_pp_idx, layer=True)


def _wrap_step(cls, get_inner, get_mamba, graph: bool):
    orig = cls.forward

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        if not _recorder.recording():
            return orig(self, *args, **kwargs)
        _recorder.begin_step(get_inner(self), get_mamba(args, kwargs), graph)
        out = orig(self, *args, **kwargs)
        _recorder.end_step()
        return out

    cls.forward = forward


def _cpu_timed(fn, key: str):
    """Add the wall time of ``fn`` to the running execute_model call."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        call = _recorder.call
        if call is None:
            return fn(self, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            call[key] = call.get(key, 0.0) + (time.perf_counter() - t0) * 1e3

    return wrapper


def _wrap_worker_call(cls):
    """Time one worker step on the CPU and attach it to the step's record."""
    orig = cls.execute_model

    @functools.wraps(orig)
    def execute_model(self, *args, **kwargs):
        rec = _recorder
        if not rec.recording():
            return orig(self, *args, **kwargs)
        step_before = rec.step
        rec.call = {"call_start": time.time()}
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            call, rec.call = rec.call, None
            call["call_ms"] = (time.perf_counter() - t0) * 1e3
            call["call_end"] = time.time()
            # rec.last stays pending until the next step begins
            if rec.step > step_before and rec.last is not None:
                rec.last["info"]["cpu"] = call

    cls.execute_model = execute_model


def _wrap_paused(cls, name):
    orig = getattr(cls, name)

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        _recorder.paused += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _recorder.paused -= 1

    setattr(cls, name, wrapper)


def apply(out_dir: str, profile_layers: bool) -> None:
    global _recorder
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.sampler import Sampler
    from vllm.model_executor.models import nemotron_h
    from vllm.worker import model_runner, worker_base

    _recorder = _Recorder(out_dir, profile_layers)

    for name, kind in _LAYER_TYPES.items():
        _wrap_layer(getattr(nemotron_h, name), kind)

    model_cls = nemotron_h.NemotronHModel
    model_cls.get_input_embeddings = _timed(model_cls.get_input_embeddings,
                                            "embed")
    lm_cls = nemotron_h.NemotronHForCausalLM
    lm_cls.compute_logits = _timed(lm_cls.compute_logits, "lm_head")
    Sampler.forward = _timed(Sampler.forward, "sampler")

    # Eager steps: NemotronHModel.forward (already the PP-fixed one if
    # enabled). Graph steps: the CUDA graph replay of the whole stage.
    _wrap_step(model_cls, lambda self: self,
               lambda args, kwargs: (args[2] if len(args) > 2 else
                                     kwargs.get("mamba_cache_params")),
               graph=False)
    _wrap_step(model_runner.CUDAGraphRunner, lambda self: self.model.model,
               lambda args, kwargs: None, graph=True)

    for name in ("profile_run", "capture_model"):
        _wrap_paused(model_runner.GPUModelRunnerBase, name)

    group_cls = parallel_state.GroupCoordinator
    group_cls.recv_tensor_dict = _cpu_timed(group_cls.recv_tensor_dict,
                                            "recv_ms")
    group_cls.send_tensor_dict = _cpu_timed(group_cls.send_tensor_dict,
                                            "send_ms")
    model_runner.ModelRunner.execute_model = _cpu_timed(
        model_runner.ModelRunner.execute_model, "exec_ms")
    _wrap_worker_call(worker_base.LocalOrDistributedWorkerBase)

    logger.info("[hybrid_pp] NemotronH profiler enabled (layers=%s)",
                profile_layers)
