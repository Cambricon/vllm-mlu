# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.model_executor.layers.rotary_embedding.base import MLURotaryEmbedding


class MLUMRotaryEmbedding(MLURotaryEmbedding, MRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        # YaRN parameters.
        *,
        scaling_factor: float | None = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        if self.scaling_factor is not None:
            # Get n-d magnitude scaling corrected for interpolation
            self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        else:
            self.mscale = 1.0

        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        MLURotaryEmbedding.__init__(
            self,
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def _apply_mrope(self, positions):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        num_section = len(self.mrope_section)
        mrope_section = self.mrope_section * 2
        def _apply(x):
            x = torch.cat([
                m[i % num_section]
                for i, m in enumerate(x.split(mrope_section, dim=-1))
            ],
                            dim=-1)
            return x
        return _apply(cos), _apply(sin)

    def _apply_interleaved_mrope(self, positions):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        """
        mrope_section = self.mrope_section
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        def _apply(x):
            x_t = x[0].clone()
            x_t[..., 1:mrope_section[1] * 3:3] = x[1, ..., 1:mrope_section[1] * 3:3]
            x_t[..., 2:mrope_section[2] * 3:3] = x[2, ..., 2:mrope_section[2] * 3:3]
            offset = self.rotary_dim // 2
            x_t[..., 1 + offset:mrope_section[1] * 3 + offset:3] = x[1, ..., 1 + offset:mrope_section[1] * 3 + offset:3]
            x_t[..., 2 + offset:mrope_section[2] * 3 + offset:3] = x[2, ..., 2 + offset:mrope_section[2] * 3 + offset:3]
            return x_t
        return _apply(cos), _apply(sin)

    def precompute_sin_cos_cache(
        self,
        positions: torch.Tensor
    ):
        '''
        call this function before forward decoder layers
        precompute sin/cos cache for mrope
        '''
        if positions.ndim == 1:
            return
        assert positions.ndim == 2
        assert self.mrope_section
        if self.mrope_interleaved:
            cos, sin = self._apply_interleaved_mrope(positions)
        else:
            cos, sin = self._apply_mrope(positions)
        self.mrope_cos_cache = cos
        self.mrope_sin_cache = sin
        self.mrope_cu_seq_lens = torch.zeros(2, dtype=torch.int32, device=positions.device)
        num_tokens = positions.shape[-1]
        self.mrope_cu_seq_lens[1] = num_tokens

    def forward_oot(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2
        if positions.ndim == 1:
            return MLURotaryEmbedding.forward_oot(self, positions, x)
        assert self.mrope_cos_cache is not None and self.mrope_sin_cache is not None,\
            "call precompute_sin_cos_cache first!"
        num_tokens = positions.shape[-1]
        x = mlu_ops.rotary_embedding(x,
                                     self.mrope_sin_cache,
                                     self.mrope_cos_cache,
                                     None,
                                     self.mrope_cu_seq_lens,
                                     not self.is_neox_style,
                                     False,
                                     False,
                                     num_tokens)
        return x

    forward = MLURotaryEmbedding.forward
