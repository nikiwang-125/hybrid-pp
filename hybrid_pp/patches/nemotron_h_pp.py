"""Fix NemotronHModel.forward for pipeline parallelism (vLLM v0.10.1.1).

Upstream bugs (still present in v0.11.0), see docs/findings.md:

1. The layer loop runs over ``self.layers`` -- every layer of the model,
   including the ``PPMissingLayer`` placeholders that belong to other stages.
   A placeholder returns its first kwarg (``positions``), so
   ``hidden_states, residual = layer(...)`` fails as soon as PP > 1.
2. ``residual = None`` is set unconditionally after reading the
   intermediate tensors, so non-first stages drop the residual stream
   received from the previous stage and produce wrong outputs.

The replacement below only iterates over this stage's layers and keeps the
received residual. Everything else matches the upstream forward.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger("vllm.hybrid_pp")  # inherits vLLM log handlers


def apply() -> None:
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
            residual = intermediate_tensors["residual"]

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
    logger.info("[hybrid_pp] patched NemotronHModel.forward for PP")
