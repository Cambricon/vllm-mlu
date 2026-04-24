# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Tuple
import torch

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    yarn_get_mscale,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)
from vllm.platforms import current_platform

from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.model_executor.layers.rotary_embedding.base import MLURotaryEmbedding


class MLUDeepseekScalingRotaryEmbedding(MLURotaryEmbedding, DeepseekScalingRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        inverse: bool = False,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        self.inverse = inverse
        MLURotaryEmbedding.__init__(
            self, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def forward_mlu_rot(self, input, position_ids, interleaved, discrete, cu_seq_lens, max_seq_len):
        """only one input rotary implementation"""
        if input is None:
            return None
        if self.rotary_dim < self.head_size:
            input_pass = input[..., self.rotary_dim:]
        input_rot = input[..., :self.rotary_dim]
        input_rot = mlu_ops.rotary_embedding(
            input_rot,
            self.sin_,
            self.cos_,
            position_ids,
            cu_seq_lens,
            interleaved,
            discrete,
            False,
            max_seq_len
        )

        if self.rotary_dim < self.head_size:
            input = torch.cat((input_rot, input_pass), dim=-1)
        else:
            input = input_rot

        return input

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        only_prefill: bool | None = False,
        only_decode: bool | None = False,
        discrete: bool | None = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        position_ids, interleaved, discrete = self.get_param(positions, discrete)
        
        cu_seq_lens = MLURotaryEmbedding.cu_seq_lens
        max_seq_len = MLURotaryEmbedding.max_seq_len

        # for MLA
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            _, attn_metadata = next(iter(attn_metadata.items()))
        if isinstance(attn_metadata, MLACommonMetadata):
            if only_prefill:
                cu_seq_lens = MLURotaryEmbedding.prefill_cu_seq_lens
                max_seq_len = MLURotaryEmbedding.prefill_max_seq_len
            elif only_decode:
                cu_seq_lens = MLURotaryEmbedding.decode_cu_seq_lens
                max_seq_len = MLURotaryEmbedding.decode_max_seq_len

        query = self.forward_mlu_rot(query, position_ids, interleaved, discrete, cu_seq_lens, max_seq_len)
        key = self.forward_mlu_rot(key, position_ids, interleaved, discrete, cu_seq_lens, max_seq_len)

        return query, key

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                dtype=torch.float,
                device=current_platform.device_type,
            )
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        device = current_platform.device_type
        inv_freq_mask = ((
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor).to(device)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=current_platform.device_type,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale * (-1 if self.inverse else 1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    forward = MLURotaryEmbedding.forward
    forward_native = forward_oot