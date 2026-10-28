"""Fix NemotronH pipeline parallelism (vLLM v0.10.1.1).

Upstream bugs (still present in v0.11.0), see docs/findings.md. Each one
masks the next, so they can be switched on/off separately:

3. ``NemotronHModel.load_weights`` never calls ``is_pp_missing_parameter``.
   Every stage receives the full checkpoint, and the first weight of a layer
   owned by another stage raises ``KeyError: 'layers.N...'`` at startup.
1. The forward loop runs over ``self.layers`` -- every layer of the model,
   including the ``PPMissingLayer`` placeholders that belong to other stages.
   A placeholder returns its first kwarg (``positions``, a 1-D tensor), so
   ``hidden_states, residual = layer(...)`` raises
   ``ValueError: too many values to unpack (expected 2)`` during the
   startup profile run.
2. ``residual = None`` is set unconditionally after reading the
   intermediate tensors, so non-first stages drop the residual stream
   received from the previous stage and produce wrong outputs.

``apply_load_fix`` fixes 3. ``apply_forward_fix(keep_residual=True)`` fixes 1
and 2; with ``keep_residual=False`` it fixes only 1 and reproduces 2, which
lets a server start and shows the wrong outputs bug 2 causes.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger("vllm.hybrid_pp")  # inherits vLLM log handlers


def apply_load_fix() -> None:
    from vllm.model_executor.models import nemotron_h
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    orig = nemotron_h.NemotronHModel.load_weights

    def load_weights(self, weights):
        # Names here are relative to NemotronHModel ("layers.N...."), the same
        # form get_pp_missing_layer_names() returns for this module.
        return orig(self, ((name, w) for name, w in weights
                           if not is_pp_missing_parameter(name, self)))

    nemotron_h.NemotronHModel.load_weights = load_weights
    logger.info("[hybrid_pp] patched NemotronHModel.load_weights for PP")


def apply_forward_fix(keep_residual: bool = True) -> None:
    from vllm import envs
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba2_metadata import (
        prepare_mamba2_metadata)
    from vllm.model_executor.models import nemotron_h
    from vllm.sequence import IntermediateTensors

    mamba_layer_cls = nemotron_h.NemotronHMambaDecoderLayer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        attn_metadata = get_forward_context().attn_metadata

        if not envs.VLLM_USE_V1:
            mamba2_metadata = prepare_mamba2_metadata(
                chunk_size=self.config.chunk_size,
                attn_metadata=attn_metadata,
            )
        else:
            # v1 get mamba2_metadata from forward_context
            mamba2_metadata = None

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # keep_residual=False reproduces upstream bug 2.
            residual = intermediate_tensors["residual"] if keep_residual \
                else None

        # The V0 mamba cache only holds this stage's mamba layers, indexed
        # from 0 in layer order.
        mamba_idx = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            layer_mamba_cache_params = None
            if isinstance(layer, mamba_layer_cls):
                if mamba_cache_params:
                    layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                        mamba_idx)
                mamba_idx += 1

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states

    nemotron_h.NemotronHModel.forward = forward
    logger.info("[hybrid_pp] patched NemotronHModel.forward for PP "
                "(keep_residual=%s)", keep_residual)
