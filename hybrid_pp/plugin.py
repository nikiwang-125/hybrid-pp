"""vLLM general plugin: applies our patches in every vLLM process.

Registered through the ``vllm.general_plugins`` entry point (see
pyproject.toml), so ``pip install -e .`` is enough; no vLLM source edits.

Environment switches (all optional):
  HYBRID_PP_FIX_LOAD=0       disable the NemotronH load_weights fix (bug 3)
  HYBRID_PP_FIX=0            disable the NemotronH forward fix (bugs 1, 2, 4)
  HYBRID_PP_FIX=layers_only  fix bugs 1 and 4, keep bug 2 (wrong residual)
  HYBRID_PP_FIX_NOATTN=0     disable the fix for stages without attention layers
  HYBRID_PP_STAGE_MEM_LOG=0  disable the per-stage memory log line
  HYBRID_PP_PROFILE_DIR=dir  enable per-step / per-layer CUDA-event profiling
  HYBRID_PP_PROFILE_LAYERS=0 profile whole steps only (lower overhead)
"""

import logging
import os

logger = logging.getLogger("vllm.hybrid_pp")  # inherits vLLM log handlers

_registered = False


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


def register() -> None:
    # vLLM may call plugins more than once per process.
    global _registered
    if _registered:
        return
    _registered = True

    from hybrid_pp.patches import nemotron_h_pp, stage_memory

    if _flag("HYBRID_PP_FIX_LOAD"):
        nemotron_h_pp.apply_load_fix()
    fix = os.environ.get("HYBRID_PP_FIX", "1")
    if fix == "layers_only":
        nemotron_h_pp.apply_forward_fix(keep_residual=False)
    elif _flag("HYBRID_PP_FIX"):
        nemotron_h_pp.apply_forward_fix()
    if _flag("HYBRID_PP_FIX_NOATTN") or _flag("HYBRID_PP_STAGE_MEM_LOG"):
        stage_memory.apply(fix_no_attention=_flag("HYBRID_PP_FIX_NOATTN"),
                           log_stage_memory=_flag("HYBRID_PP_STAGE_MEM_LOG"))
    if os.environ.get("HYBRID_PP_PROFILE_DIR"):
        from hybrid_pp import profiler
        profiler.apply(out_dir=os.environ["HYBRID_PP_PROFILE_DIR"],
                       profile_layers=_flag("HYBRID_PP_PROFILE_LAYERS"))
