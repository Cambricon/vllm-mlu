# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Union

import torch

from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import LinearScalingRotaryEmbedding

from vllm_mlu.model_executor.layers.rotary_embedding.base import MLURotaryEmbedding

class MLULinearScalingRotaryEmbedding(MLURotaryEmbedding, LinearScalingRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factors: list[float] | float,
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: list[float] = scaling_factors  # noqa
        MLURotaryEmbedding.__init__(
            self, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: dict[float, int]

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        device = current_platform.device_type
        if self.is_neox_style:
            half_dim = self.rotary_dim // 2
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, self.rotary_dim, 1, dtype=torch.float32, device=device)
                    % half_dim * 2 / self.rotary_dim)
            )
        else:
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, self.rotary_dim, 1, dtype=torch.float32, device=device)
                    // 2 * 2 / self.rotary_dim
                )
            )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: list[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: list[int] = []
        device = current_platform.device_type
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float, device=device)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)