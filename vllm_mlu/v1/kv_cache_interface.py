# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils import get_dtype_size

from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

logger = init_logger(__name__)


@dataclass
class MLUFullAttentionSpec(FullAttentionSpec):
    # Use to record k_cache info for XS model
    index_head_dim: int = 0
    index_n_heads: int = 0

    @property
    def type_id(self) -> str:
        return f"mlu_full_attention_{self.block_size}_{self.page_size_bytes}"

    @property
    def cache_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    @property
    def scale_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        scale_size_bytes = 0
        if self.dtype in [torch.int8, torch.uint8]:
            scale_size_bytes = coef * self.block_size * self.num_kv_heads \
                * get_dtype_size(torch.float32)
        return scale_size_bytes

    @property
    def index_cache_size_bytes(self) -> int:
        return self.block_size * self.index_n_heads * self.index_head_dim \
            * get_dtype_size(self.dtype)

    @property
    def page_size_bytes(self) -> int:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: caculate kv_cache_scale size when kv_cache_dtype=int8
        '''
        return self.cache_size_bytes + self.scale_size_bytes + self.index_cache_size_bytes
        '''
        ==================
        End of MLU Hijack
        ==================
        '''


@dataclass
class MLUSlidingWindowSpec(SlidingWindowSpec):

    @property
    def type_id(self) -> str:
        return f"mlu_sliding_window_{self.sliding_window}_{self.block_size}_{self.page_size_bytes}"  # noqa

    @property
    def cache_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    @property
    def scale_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        scale_size_bytes = 0
        if self.dtype in [torch.int8, torch.uint8]:
            scale_size_bytes = coef * self.block_size * self.num_kv_heads \
                * get_dtype_size(torch.float32)
        return scale_size_bytes

    @property
    def page_size_bytes(self) -> int:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: caculate kv_cache_scale size when kv_cache_dtype=int8
        '''
        return self.cache_size_bytes + self.scale_size_bytes
        '''
        ==================
        End of MLU Hijack
        ==================
        '''