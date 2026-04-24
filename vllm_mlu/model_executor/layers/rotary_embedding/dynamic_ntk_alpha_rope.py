# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_alpha_rope import DynamicNTKAlphaRotaryEmbedding

from vllm_mlu.model_executor.layers.rotary_embedding.base import MLURotaryEmbedding


class MLUDynamicNTKAlphaRotaryEmbedding(MLURotaryEmbedding, DynamicNTKAlphaRotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_alpha: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_alpha = scaling_alpha
        MLURotaryEmbedding.__init__(
            self, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )