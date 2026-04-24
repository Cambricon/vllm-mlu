# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.attention import AttentionType
from vllm.attention.backends.abstract import MLAAttentionImpl
from vllm.attention.layer import Attention, MLAAttention, _init_kv_cache_quant
from vllm.attention.selector import get_attn_backend

from vllm.config.cache import CacheConfig
from vllm.config.vllm import QuantizationConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm_mlu.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu.v1.kv_cache_interface import (
    MLUFullAttentionSpec,
    MLUMLAAttentionSpec,
    MLUSlidingWindowSpec,
)

@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kwargs: dict[str, Any] = {},
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add return for self.impl.forward and it's param kwargs
    '''
    output = self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        kwargs=kwargs,
    )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    return output

class Attention_MluHijack(Attention):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        if self.sliding_window is not None:
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: replace SlidingWindowSpec with MLUSlidingWindowSpec.
            '''
            return MLUSlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
        else:
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: replace FullAttentionSpec with MLUFullAttentionSpec.
            '''
            return MLUFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

class MLAAttention_MluHijack(MLAAttention):
    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: insert num_kv_heads for mlu platform
        '''
        self.head_size = qk_nope_head_dim + qk_rope_head_dim
        self.num_kv_heads = extra_impl_args.pop("num_kv_heads", None)
        if self.num_kv_heads is None:
            self.num_kv_heads = num_heads

        self.decoder_attn_dtype = None
        decoder_attn_dtype = get_current_vllm_config().mlu_config.decoder_attn_dtype
        if decoder_attn_dtype in ["int8", "fp8_e4m3", "fp8"]:
            self.decoder_attn_dtype = (
                torch.int8 if decoder_attn_dtype == "int8"
                else torch.float8_e4m3fn
            )
            extra_impl_args['decoder_attn_dtype'] = self.decoder_attn_dtype
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(
            self, quant_config, prefix, kv_cache_dtype, calculate_kv_scales
        )

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla=True,
            use_sparse=use_sparse,
        )
        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
            self.num_heads,
            self.head_size,
            self.scale,
            self.num_kv_heads,
            None, # alibi_slops
            None, # sliding_window
            kv_cache_dtype,
            None, # logits_soft_cap
            AttentionType.DECODER, # attn_dtype
            None, # kv_sharing_target_layer_name
            **extra_impl_args,
        )
        self.dtype = dtype

        self.use_direct_call = not current_platform.opaque_attention_op()

        if current_platform.is_out_of_tree():
            self.use_direct_call = False

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support kv8 and deepseek v3.2
        '''
        self.kv_cache = [
            [torch.tensor([]), torch.tensor([]), torch.tensor([])]
            for _ in range(
                get_current_vllm_config().parallel_config.pipeline_parallel_size
            )
        ]
        self.impl.use_mla = True
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        self.use_sparse = use_sparse

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: replace MLAAttentionSpec with MLUMLAAttentionSpec.
        '''
        index_head_dim, index_n_heads = 0, 0
        if vllm_config.model_config.hf_text_config.model_type == "deepseek_v32":
            index_head_dim = vllm_config.model_config.hf_text_config.index_head_dim
            index_n_heads = 1
            
        if vllm_config.model_config.hf_text_config.model_type == "deepseek_v4":
            index_head_dim = vllm_config.model_config.hf_text_config.index_head_dim
            index_n_heads = 1
            
        return MLUMLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
        kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)

        assert not self.use_direct_call, "MLU-V1 does not support direct call."
        if self.attn_backend.accept_output_buffer:
            output_lse = None
            output_shape = (output_shape if output_shape is not None else query.shape)
            output_shape = [output_shape[0], self.num_heads * self.v_head_dim]

            output = torch.empty(
                output_shape,
                dtype=self.dtype if query.dtype == torch.int8 else query.dtype,
                device=query.device,
            )
            hidden_size = output_shape[-1]
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.v_head_dim)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.v_head_dim)
            if not kwargs:
                torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name
                )
                attn_output_list = output
            else:
                attn_output_list = unified_attention_with_output(
                    query, key, value, output, self.layer_name, kwargs=kwargs)
            if isinstance(attn_output_list, (list, tuple)) and len(attn_output_list) > 1:
                output_lse = attn_output_list[1]
            if output_lse is not None:
                return output.view(-1, hidden_size), output_lse
            else:
                return output.view(-1, hidden_size)
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
        else:
            return torch.ops.vllm.unified_attention(
                query, key, value, self.layer_name
            )

MluHijackObject.apply_hijack(
    Attention,
    Attention.get_kv_cache_spec,
    Attention_MluHijack.get_kv_cache_spec,
)
MluHijackObject.apply_hijack(
    MLAAttention,
    MLAAttention.__init__,
    MLAAttention_MluHijack.__init__,
)
MluHijackObject.apply_hijack(
    MLAAttention,
    MLAAttention.get_kv_cache_spec,
    MLAAttention_MluHijack.get_kv_cache_spec,
)
MluHijackObject.apply_hijack(
    MLAAttention,
    MLAAttention.forward,
    MLAAttention_MluHijack.forward,
)
