# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from dataclasses import dataclass
from typing_extensions import Self

import torch

from math import prod
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size
from vllm_mlu.mlu_hijack_utils import MluHijackObject

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    MambaSpec,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class MLUFullAttentionSpec(FullAttentionSpec):

    @property
    def type_id(self) -> str:
        return f"mlu_full_attention_{self.block_size}_{self.page_size_bytes}"

    @property
    def cache_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @property
    def scale_size_bytes(self) -> int:
        scale_size_bytes = 0
        if self.dtype in [torch.int8, torch.uint8]:
            scale_size_bytes = (
                2
                * self.block_size
                * self.num_kv_heads
                * get_dtype_size(torch.float32)
            )
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


@dataclass(frozen=True)
class MLUMLAAttentionSpec(MLAAttentionSpec):
    # Use to record k_cache info for DSA indexer
    index_head_dim: int = 0
    index_n_heads: int = 0

    @property
    def type_id(self) -> str:
        return f"mlu_mla_attention_{self.block_size}_{self.page_size_bytes}"

    @property
    def cache_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @property
    def scale_size_bytes(self) -> int:
        scale_size_bytes = 0
        if self.dtype in [torch.int8, torch.uint8]:
            scale_size_bytes = (
                self.block_size
                * self.num_kv_heads
                * get_dtype_size(torch.float32)
            )
        return scale_size_bytes

    @property
    def index_cache_size_bytes(self) -> int:
        return (
            self.block_size
            * self.index_n_heads
            * self.index_head_dim
            * get_dtype_size(self.dtype)
        )

    @property
    def page_size_bytes(self) -> int:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: caculate kv_cache_scale size when kv_cache_dtype=int8
        @brief: caculate indexer cache size for deepseek v3.2
        '''
        return self.cache_size_bytes + self.scale_size_bytes + self.index_cache_size_bytes
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            index_head_dim=specs[0].index_head_dim,
            index_n_heads=specs[0].index_n_heads,
        )


@dataclass(frozen=True)
class MLUSlidingWindowSpec(SlidingWindowSpec):

    @property
    def type_id(self) -> str:
        return f"mlu_sliding_window_{self.sliding_window}_{self.block_size}_{self.page_size_bytes}"  # noqa

    @property
    def cache_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @property
    def scale_size_bytes(self) -> int:
        scale_size_bytes = 0
        if self.dtype in [torch.int8, torch.uint8]:
            scale_size_bytes = (
                2
                * self.block_size
                * self.num_kv_heads
                * get_dtype_size(torch.float32)
            )
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

@property
def vllm__v1__kv_cache_interface__MambaSpec__page_size_bytes(self) -> int:
    page_size = sum(
        prod(shape) * get_dtype_size(dtype)
        for (shape, dtype) in zip(self.shapes, self.dtypes)
    )
    if self.page_size_padded is not None:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support qwen3-next
        '''
        # assert self.page_size_padded >= page_size
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        return self.page_size_padded
    return page_size

MluHijackObject.apply_hijack(MambaSpec,
                             MambaSpec.page_size_bytes,
                             vllm__v1__kv_cache_interface__MambaSpec__page_size_bytes)
