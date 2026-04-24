# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

"""Inference-only MOE model."""
from typing import Optional, Any, List, Dict

import torch
from torch import nn

from vllm.distributed import (
    divide,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu._mlu_utils import *
from vllm_mlu.distributed.parallel_state import(
    cnclep_dispatch, cnclep_combine)
from vllm_mlu.model_executor.layers.sparse_moe_mlp import SparseMoeMlp

class LongCatSparseMoeMlp(SparseMoeMlp):
    """
    sparse moe mlp layer specific to longcat model
    """
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        up_proj_name: str,
        is_gated: bool,
        down_proj_name: str,
        has_bias: bool,
        skip_bias_add: bool = False,
        renormalize:bool = False,
        hidden_act: str = "silu",
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        is_use_fused_moe: bool = False,
        expert_group: Optional[int] = 1,
        topk_group: Optional[int] = 1,
        scoring_func: str = "softmax",
        topk_method: str = "",
        routed_scaling_factor: float = 1.0,
        tp_group: Any = None,
        use_all2all: bool = False,
        num_zero_experts: int = 0,
        ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_proj_name=up_proj_name,
            is_gated=is_gated,
            down_proj_name=down_proj_name,
            has_bias=has_bias,
            skip_bias_add=skip_bias_add,
            renormalize=renormalize,
            hidden_act=hidden_act,
            params_dtype=params_dtype,
            quant_config=quant_config,
            is_use_fused_moe=is_use_fused_moe,
            expert_group=expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            topk_method=topk_method,
            routed_scaling_factor=routed_scaling_factor,
            tp_group=tp_group,
            use_all2all=use_all2all,
            init_avg_moe=False,
        )
        self.num_zero_experts = num_zero_experts
        self.total_experts_including_zero = self.num_total_experts + self.num_zero_experts
        self.use_quant_all2all = use_all2all and quant_config is not None
        self.zero_expert_size = divide(self.num_zero_experts, self.moe_ep_size)
        self.start_zero_expert_id = (
            self.num_total_experts + self.moe_ep_rank * ((self.num_zero_experts + self.moe_ep_size - 1) // self.moe_ep_size)
        )

        if VLLM_AVG_MOE_EN and not SparseMoeMlp.is_expert_avg:
            n_tokens = SparseMoeMlp.max_batched_token * self.dp_size
            expert_group = self.moe_ep_size
            val = 1.0 / float(self.total_experts_including_zero)
            SparseMoeMlp.reduce_weight = torch.full((n_tokens, top_k), val, device="mlu", dtype=torch.float32)
            if VLLM_RANDOM_MOE_EN:
                import numpy as np
                # example deepseekv2: experts 160 topk 6
                # avg list: 92,   8,  88,  45,  99,   9,... 118, 142, 116,  57, 104,   6,......
                array = np.stack([np.random.permutation(self.total_experts_including_zero)[:top_k] for _ in range(n_tokens)])
                table = torch.from_numpy(array.flatten()).to(device="mlu", dtype=torch.int32)
            else:
                # example deepseekv2: experts 160
                # avg list: 0,20,40,60,80...120,140,  1,21,...121,141, 2...142,  ......  19,...159,  0,20,......
                import math
                batch_table = math.ceil(n_tokens * top_k / self.total_experts_including_zero) * self.total_experts_including_zero
                hi_val = batch_table // self.total_experts_including_zero
                table = (torch.arange(hi_val * num_experts, device="mlu", dtype=torch.int32) % num_experts).view(
                    hi_val, expert_group, num_experts // expert_group).transpose(1, 2)
                if self.num_zero_experts > 0:
                    # Longcat model, for avg expert, we choose eight non-zero experts and four zero
                    # experts for each token accorrding to the paper.
                    assert num_experts == 512 and num_zero_experts == 256 and top_k == 12
                    assert num_zero_experts % expert_group == 0
                    non_zero_expert_num_per_token = 8
                    zero_expert_num_per_token = 4
                    zero_expert_table = torch.arange(
                        num_experts, num_experts + num_zero_experts, dtype=table.dtype, device=table.device).view(
                        expert_group, num_zero_experts // expert_group).transpose(0, 1).flatten()
                    non_zero_expert_table = table[0].flatten()
                    token_expert_list = []
                    for idx in range(0, num_experts // non_zero_expert_num_per_token):
                        token_expert_list.append(non_zero_expert_table[
                            idx * non_zero_expert_num_per_token:
                            idx * non_zero_expert_num_per_token + non_zero_expert_num_per_token])
                        token_expert_list.append(zero_expert_table[
                            idx * zero_expert_num_per_token:
                            idx * zero_expert_num_per_token + zero_expert_num_per_token])
                    avg_expert_table = torch.cat(token_expert_list)
                    table = avg_expert_table.repeat(hi_val)
            SparseMoeMlp.expert_id = table.flatten()[:n_tokens * top_k].view(n_tokens, top_k)
            SparseMoeMlp.is_expert_avg = True


    def forward_experts_nofused_longcat(
            self, hidden_states, total_num_experts, total_num_experts_per_rank, 
            topk_indices=None, topk_weights=None, residual_=None):
        assert self.moe_ep_size == 1
        assert not self.use_all2all
        expand_gather_idx, scatter_idx, expand_token_count, cusum_token_count = mlu_ops.moe_gen_idx(
            topk_indices.to(torch.int32), total_num_experts)
        # no expert is routed, then expand_gather_idx, expand_scatter_idx has no item,
        # expand_token_count and expand_cusum_token_count has item but the value is all zero
        # so this rank should only return final_hidden_states with zero value
        if cusum_token_count[-1] == 0:
            final_hidden_states = torch.zeros_like(hidden_states,
                                                   dtype=hidden_states.dtype,
                                                   device=hidden_states.device)
            return final_hidden_states

        expand_hidden_states = mlu_ops.moe_expand_input(
            hidden_states, expand_gather_idx, cusum_token_count, 
            start_expert_id=self.start_expert_id,
            expert_size=self.end_expert_id - self.start_expert_id)
        expand_hidden_states_zero = mlu_ops.moe_expand_input(
            hidden_states, expand_gather_idx, cusum_token_count, 
            start_expert_id=self.start_zero_expert_id,
            expert_size=self.zero_expert_size)

        expand_output_list = []
        expand_cusum_token_count = cusum_token_count[self.start_expert_id:self.end_expert_id +
                                                     1] - cusum_token_count[self.start_expert_id]
                                
        for expert_idx, num_tokens_per_expert in enumerate(expand_token_count[:self.num_total_experts]):
            if num_tokens_per_expert > 0:
                expert_hidden_states = expand_hidden_states[
                    expand_cusum_token_count[expert_idx]:expand_cusum_token_count[expert_idx + 1]]
                if expert_idx < self.num_total_experts:
                    expert_output = self.experts[expert_idx](expert_hidden_states)
                else:
                    expert_output = expert_hidden_states
                expert_output = expert_output[0] if isinstance(expert_output, (tuple, list)) else expert_output
                expand_output_list.append(expert_output)
        expand_output = torch.cat(expand_output_list, dim=0)
        num_normal_tokens = cusum_token_count[self.num_total_experts]
        expand_hidden_states[:num_normal_tokens] = expand_output
        # reduce normal experts
        final_hidden_states = mlu_ops.moe_combine_result(
            expand_hidden_states, topk_weights, scatter_idx,
            residual_, cusum_token_count, start_expert_id=self.start_expert_id,
            expert_size=self.end_expert_id - self.start_expert_id, bias=None)
        # reduce zero experts
        if self.moe_ep_size > 1 or self.moe_tp_rank == 0:
            final_hidden_states = mlu_ops.moe_combine_result(
                expand_hidden_states_zero, topk_weights, scatter_idx,
                final_hidden_states, cusum_token_count, start_expert_id=self.start_zero_expert_id,
                expert_size=self.zero_expert_size, bias=None,
                output=final_hidden_states)
        return final_hidden_states

    # no compute-communication parallel, for prototyping only, not in actual use.
    # subject to becoming stale
    def forward_all2all_int8_longcat(
            self, hidden_states, total_num_experts, total_num_experts_per_rank, 
            topk_indices=None, topk_weights=None, residual_=None):
        ori_input_shape = hidden_states.shape
        dtype = hidden_states.dtype
        self.pack_params()
        self.pack_params_after_loading()
        w1=self.w13
        w2=self.w2
        bias2=self.b2
        input_smooth=self.a13_scale_all_experts
        act_smooth=self.a2_scale
        w1_scale=self.w13_scale
        w2_scale=self.w2_scale
        act_mode=self.hidden_act
        quant_input=None

        max_m = hidden_states.shape[0]

        reduce_weight = topk_weights
        expert_id = topk_indices

        expand_idx, combine_idx, token_count, cusum_token_count \
            = mlu_ops.moe_gen_idx(expert_id, total_num_experts)

        num_token_expand = hidden_states.shape[0] * self.top_k
        dispatch_bytes = num_token_expand * self.dispatch_token_size

        dispatch_send_token_tensor = (
            self.dispatch_send_buffer[:dispatch_bytes]
            .view(num_token_expand, self.dispatch_token_size)
        )

        quant_size = self.hidden_size
        quant_input = dispatch_send_token_tensor[:, : quant_size]
        input_scale = dispatch_send_token_tensor[:, quant_size :].view(torch.float32)
        quant_input, input_scale = mlu_ops.moe_quantize(
            hidden_states, input_smooth, None, token_count[:self.num_total_experts], 
            expand_idx, None,
            output=quant_input,
            output_scale=input_scale)
        expand_hidden_states_zero = mlu_ops.moe_expand_input(
            hidden_states, expand_idx, cusum_token_count, 
            start_expert_id=self.num_total_experts,
            expert_size=self.num_zero_experts)

        dispatch_send_layout = mlu_ops.moe_all2all_gen_send_layout(
            token_count[:self.num_total_experts], self.moe_ep_size)

        cnclep_dispatch(self.dispatch_token_size, 
                        num_token_expand, 
                        dispatch_send_layout, 
                        token_count[:self.num_total_experts], 
                        self.dispatch_recv_layout, 
                        self.dispatch_recv_token_num) 

        recv_token_num = self.dispatch_recv_token_num.view(
            self.moe_ep_size, self.num_experts_per_rank)
        pad_num = self.max_num_tokens_per_rank

        (
            gather_by_expert_index,
            gather_by_rank_index,
            tokens_per_local_expert,
            token_sum
        ) = mlu_ops.moe_all2all_gen_gather_index(recv_token_num, pad_num)

        max_tokens_bytes_recv = self.max_num_tokens_recv * self.dispatch_token_size
        dispatch_recv_token_tensor = (
            self.dispatch_recv_buffer[:max_tokens_bytes_recv]
            .view(self.max_num_tokens_recv, self.dispatch_token_size))
        
        mlu_ops.gather_split(dispatch_recv_token_tensor, 
                             gather_by_expert_index,
                             token_sum,
                             self.quant_input_recv,
                             self.input_scale_recv)

        max_m = self.max_num_tokens_per_expert
        gemm_out = mlu_ops.smooth_quant_group_gemm(self.quant_input_recv, w1,
                                                   tokens_per_local_expert,
                                                   None, None, None, None,
                                                   self.input_scale_recv.view(torch.float32).flatten(),
                                                   w1_scale, dtype, max_m)

        # continue reusing self.quant_input_recv and self.input_scale_recv
        quant_input = self.quant_input_recv[:, :gemm_out.shape[-1] // 2]
        input_scale_fp32 = self.input_scale_recv.view(torch.float32).flatten()[:gemm_out.shape[0]]
        quant_input, input_scale = mlu_ops.moe_quantize(gemm_out, act_smooth, None,
                                                        tokens_per_local_expert,
                                                        output=quant_input,
                                                        output_scale=input_scale_fp32,
                                                        act_mode=act_mode,
                                                        is_gated=self.is_gated)

        gemm_out = mlu_ops.smooth_quant_group_gemm(quant_input, w2,
                                                   tokens_per_local_expert,
                                                   None, None, None, None, input_scale, w2_scale, dtype, max_m)

        combine_send_token_tensor = self.combine_send_buffer.view(self.max_num_tokens_recv, -1).view(hidden_states.dtype)
        mlu_ops.gather_split(gemm_out,
                             gather_by_rank_index,
                             token_sum,
                             combine_send_token_tensor,
                             None)

        combine_send_layout = mlu_ops.moe_all2all_gen_send_layout(self.dispatch_recv_token_num, self.moe_ep_size)
        combine_recv_layout = self.dispatch_recv_layout

        # combine
        combine_args = dict(
            token_byte=self.hidden_size * 2,
            token_num=num_token_expand,
            send_src_layout=combine_send_layout,
            send_dst_layout=combine_recv_layout,
            send_token=None,
            recv_token=None)

        cnclep_combine(**combine_args)
       
        numel_recv = num_token_expand * self.hidden_size
        recv_token = (self.combine_recv_buffer.view(hidden_states.dtype)[:numel_recv]
                      .view(num_token_expand, self.hidden_size))

        residual_ = None
        output = mlu_ops.moe_combine_result(recv_token, reduce_weight, combine_idx,
                     residual_, cusum_token_count, start_expert_id=0,
                     expert_size=self.num_total_experts, bias=bias2, output=hidden_states)
        assert self.moe_ep_size > 1
        # zero expert reduce
        output = mlu_ops.moe_combine_result(
            expand_hidden_states_zero, reduce_weight, combine_idx,
            output, cusum_token_count, self.num_total_experts,
            self.num_zero_experts, output=hidden_states)

        return output.view(ori_input_shape)

    # no compute-communication parallel, for prototyping only, not in actual use.
    # subject to becoming stale
    def forward_all2all_bf16_longcat(
            self, hidden_states, total_num_experts, total_num_experts_per_rank, 
            topk_indices=None, topk_weights=None, residual_=None):
        is_fp8_quant = isinstance(self.quant_config, Fp8Config)
        ori_input_shape = hidden_states.shape
        dtype = hidden_states.dtype
        self.pack_params()
        self.pack_params_after_loading()
        w1=self.w13
        w2=self.w2
        bias1=self.b13
        bias2=self.b2
        gated=self.is_gated
        act_mode=self.hidden_act

        max_m = hidden_states.shape[0]
        reduce_weight = topk_weights
        expert_id = topk_indices

        # gen_idx
        expand_idx, combine_idx, token_count, cusum_token_count = \
            mlu_ops.moe_gen_idx(expert_id, total_num_experts)
        num_token_expand = hidden_states.shape[0] * self.top_k
        dispatch_bytes = num_token_expand * self.dispatch_token_size

        dispatch_send_token_tensor = (
            self.dispatch_send_buffer[:dispatch_bytes]
            .view(num_token_expand, self.dispatch_token_size)
            .view(hidden_states.dtype)
        )

        expand_hidden_states = mlu_ops.moe_expand_input(
            hidden_states, expand_idx, cusum_token_count, start_expert_id=0, 
            expert_size=self.num_total_experts)
        expand_hidden_states_zero = mlu_ops.moe_expand_input(
            hidden_states, expand_idx, cusum_token_count, 
            start_expert_id=self.num_total_experts,
            expert_size=self.num_zero_experts)

        dispatch_send_token_tensor.copy_(expand_hidden_states)

        dispatch_send_layout = mlu_ops.moe_all2all_gen_send_layout(
            token_count[:self.num_total_experts], self.moe_ep_size)

        cnclep_dispatch(self.dispatch_token_size, 
                        num_token_expand, 
                        dispatch_send_layout, 
                        token_count[:self.num_total_experts], 
                        self.dispatch_recv_layout, 
                        self.dispatch_recv_token_num,
                        use_quant_dispatch=False,
        )

        recv_token_num = self.dispatch_recv_token_num.view(
            self.moe_ep_size, self.num_experts_per_rank)
        pad_num = self.max_num_tokens_per_rank

        (
            gather_by_expert_index,
            gather_by_rank_index,
            tokens_per_local_expert,
            token_sum
        ) = mlu_ops.moe_all2all_gen_gather_index(recv_token_num, pad_num)

        max_tokens_bytes_recv = self.max_num_tokens_recv * self.dispatch_token_size
        dispatch_recv_token_tensor = (
            self.dispatch_recv_buffer[:max_tokens_bytes_recv]
            .view(self.max_num_tokens_recv, self.dispatch_token_size)
            .view(hidden_states.dtype)
        )

        
        self.quant_input_recv = self.quant_input_recv.view(hidden_states.dtype)
        mlu_ops.gather_split(dispatch_recv_token_tensor, 
                             gather_by_expert_index,
                             token_sum,
                             self.quant_input_recv)

        max_m = self.max_num_tokens_per_expert
        gemm_out = mlu_ops.group_gemm(
            self.quant_input_recv, w1, tokens_per_local_expert,
            None, None, None, None, max_m)
        act_out = mlu_ops.moe_active(
            gemm_out, act_mode, gated)
        gemm_out = mlu_ops.group_gemm(
            act_out, w2, tokens_per_local_expert,
            None, None, None, None, max_m)

        combine_send_token_tensor = self.combine_send_buffer.view(
            self.max_num_tokens_recv, -1).view(hidden_states.dtype)
        mlu_ops.gather_split(gemm_out,
                             gather_by_rank_index,
                             token_sum,
                             combine_send_token_tensor,
                             None)

        combine_send_layout = mlu_ops.moe_all2all_gen_send_layout(
            self.dispatch_recv_token_num, self.moe_ep_size)
        combine_recv_layout = self.dispatch_recv_layout

        combine_args = dict(
            token_byte=self.hidden_size * 2,
            token_num=num_token_expand,
            send_src_layout=combine_send_layout,
            send_dst_layout=combine_recv_layout,
            send_token=None,
            recv_token=None,
            use_quant_dispatch=False,
        )

        cnclep_combine(**combine_args)

        numel_recv = num_token_expand * self.hidden_size
        recv_token = (self.combine_recv_buffer.view(hidden_states.dtype)[:numel_recv]
                      .view(num_token_expand, self.hidden_size))

        residual_ = None
        output = mlu_ops.moe_combine_result(recv_token, reduce_weight, combine_idx,
                     residual_, cusum_token_count, start_expert_id=0,
                     expert_size=self.num_total_experts, bias=bias2, output=hidden_states)
        # zero expert reduce
        output = mlu_ops.moe_combine_result(
            expand_hidden_states_zero, reduce_weight, combine_idx,
            output, cusum_token_count, self.num_total_experts,
            self.num_zero_experts, output=hidden_states)
        return output.view(ori_input_shape)

    def forward_before_dispatch(self, hidden_states: torch.Tensor,
                                topk_indices: torch.Tensor):
        # gate and softmax topk is called in router for longcat
        # other models can do these operations here
        expand_idx, combine_idx, token_count, cusum_token_count = mlu_ops.moe_gen_idx(
            topk_indices, self.total_experts_including_zero)

        num_token_expand = hidden_states.shape[0] * self.top_k
        dispatch_bytes = num_token_expand * self.dispatch_token_size
        dispatch_send_token_tensor = (
            self.dispatch_send_buffer[:dispatch_bytes]
            .view(num_token_expand, self.dispatch_token_size)
        )
        if self.use_quant_all2all:
            hidden_states_stride = self.hidden_size
            quant_input = dispatch_send_token_tensor[:, : hidden_states_stride]
            input_scale = dispatch_send_token_tensor[:, hidden_states_stride :].view(torch.float32)
            # expand input + quantize
            quant_input, input_scale = mlu_ops.moe_quantize(
                hidden_states, self.a13_scale_all_experts, None,
                token_count[:self.num_total_experts], 
                expand_idx, None,
                output=quant_input,
                output_scale=input_scale)
            # expand input of zero-expert
            expand_hidden_states_zero = mlu_ops.moe_expand_input(
                hidden_states, expand_idx, cusum_token_count, 
                start_expert_id=self.num_total_experts,
                expert_size=self.num_zero_experts)
        else:
            expand_hidden_states = mlu_ops.moe_expand_input(
                hidden_states, expand_idx, cusum_token_count, start_expert_id=0,
                expert_size=self.num_total_experts)
            dispatch_send_token_tensor = dispatch_send_token_tensor.view(
                hidden_states.dtype)
            dispatch_send_token_tensor.copy_(expand_hidden_states)
            del expand_hidden_states
            expand_hidden_states_zero = mlu_ops.moe_expand_input(
                hidden_states, expand_idx, cusum_token_count,
                start_expert_id=self.num_total_experts,
                expert_size=self.num_zero_experts)

        dispatch_send_layout = mlu_ops.moe_all2all_gen_send_layout(
            token_count[:self.num_total_experts], self.moe_ep_size)

        return combine_idx, token_count, cusum_token_count, dispatch_send_layout, expand_hidden_states_zero

    def forward_dispatch(self, token_num: int, dispatch_send_layout: torch.Tensor,
                         token_count: torch.Tensor):
        num_token_expand = token_num * self.top_k
        cnclep_dispatch(self.dispatch_token_size,
                        num_token_expand,
                        dispatch_send_layout,
                        token_count[:self.num_total_experts],
                        self.dispatch_recv_layout,
                        self.dispatch_recv_token_num,
                        use_quant_dispatch=self.use_quant_all2all)

    def forward_before_combine(self, hidden_states_dtype: torch.dtype):
        recv_token_num = self.dispatch_recv_token_num.view(
            self.moe_ep_size, self.num_experts_per_rank)

        (
            gather_by_expert_index,
            gather_by_rank_index,
            tokens_per_local_expert,
            token_sum,
            cusum_token_count
        ) = mlu_ops.moe_all2all_gen_gather_index(
            recv_token_num, self.max_num_tokens_per_rank,
            return_cusum_token_count=True)

        max_tokens_bytes_recv = self.max_num_tokens_recv * self.dispatch_token_size
        dispatch_recv_token_tensor = (
            self.dispatch_recv_buffer[:max_tokens_bytes_recv]
            .view(self.max_num_tokens_recv, self.dispatch_token_size))

        max_m = self.max_num_tokens_per_expert
        if self.use_quant_all2all:
            mlu_ops.gather_split(dispatch_recv_token_tensor,
                                 gather_by_expert_index,
                                 token_sum,
                                 self.quant_input_recv,
                                 self.input_scale_recv)
            # OPT: input_scale_recv_flatten can reuse self.input_scale_recv
            input_scale_recv_flatten = self.input_scale_recv.view(torch.float32).flatten()
            gemm_out = mlu_ops.smooth_quant_group_gemm(self.quant_input_recv, self.w13,
                                                       tokens_per_local_expert,
                                                       None, None, None, None,
                                                       input_scale_recv_flatten,
                                                       self.w13_scale, hidden_states_dtype, max_m)

            quant_input = self.quant_input_recv[:, :gemm_out.shape[-1] // 2]
            input_scale_fp32 = input_scale_recv_flatten[:gemm_out.shape[0]]
            quant_input, input_scale = mlu_ops.moe_quantize(gemm_out, self.a2_scale, None,
                                                            tokens_per_local_expert,
                                                            output=quant_input,
                                                            output_scale=input_scale_fp32,
                                                            act_mode=self.hidden_act,
                                                            is_gated=self.is_gated)

            gemm_out = mlu_ops.smooth_quant_group_gemm(quant_input, self.w2, tokens_per_local_expert,
                                                       None, None, None, None, input_scale, self.w2_scale,
                                                       hidden_states_dtype, max_m)
        else:
            dispatch_recv_token_tensor = dispatch_recv_token_tensor.view(hidden_states_dtype)
            self.input_recv = self.input_recv.view(hidden_states_dtype)
            mlu_ops.gather_split(dispatch_recv_token_tensor,
                                 gather_by_expert_index,
                                 token_sum,
                                 self.input_recv)
            gemm_out = mlu_ops.group_gemm(
                self.input_recv, self.w13, tokens_per_local_expert,
                None, None, None, None, max_m)
            act_out = self.input_recv[:, :gemm_out.shape[-1] // 2]
            act_out = mlu_ops.moe_active(
                gemm_out, self.hidden_act, self.is_gated, output=act_out,
                bias=None, cusum_token_count=cusum_token_count,
                start_expert_id=0, expert_size=self.num_experts_per_rank)
            gemm_out = mlu_ops.group_gemm(
                act_out, self.w2, tokens_per_local_expert,
                None, None, None, None, max_m)

        combine_send_token_tensor = self.combine_send_buffer.view(
            self.max_num_tokens_recv, -1).view(hidden_states_dtype)
        mlu_ops.gather_split(gemm_out,
                             gather_by_rank_index,
                             token_sum,
                             combine_send_token_tensor,
                             None)

        combine_send_layout = mlu_ops.moe_all2all_gen_send_layout(
            self.dispatch_recv_token_num, self.moe_ep_size)

        return combine_send_layout

    def forward_combine(self, token_num: int, combine_send_layout: torch.Tensor):
        num_token_expand = token_num * self.top_k
        # combine_recv_layout(self.dispatch_recv_layout) is calculated when cnclep_dispatch
        # because dispatch and combine are inverse operation
        cnclep_combine(token_byte=self.hidden_size * 2,
                       token_num=num_token_expand,
                       send_src_layout=combine_send_layout,
                       send_dst_layout=self.dispatch_recv_layout,
                       send_token=None,
                       recv_token=None,
                       use_quant_dispatch=self.use_quant_all2all)

    def forward_after_combine(self, token_num: int,
                              reduce_weight: torch.Tensor,
                              combine_idx: torch.Tensor,
                              cusum_token_count: torch.Tensor,
                              expand_hidden_states_zero: torch.Tensor,
                              output_tensor_dtype: torch.dtype,
                              output_tensor: Optional[torch.Tensor] = None,
                              residual: Optional[torch.Tensor] = None):
        num_token_expand = token_num * self.top_k
        numel_recv = num_token_expand * self.hidden_size
        recv_token = (self.combine_recv_buffer.view(output_tensor_dtype)[:numel_recv]
                      .view(num_token_expand, self.hidden_size))

        output = mlu_ops.moe_combine_result(recv_token, reduce_weight, combine_idx,
                     residual, cusum_token_count, start_expert_id=0,
                     expert_size=self.num_total_experts, bias=self.b2, output=output_tensor)
        output = mlu_ops.moe_combine_result(
            expand_hidden_states_zero, reduce_weight, combine_idx,
            output, cusum_token_count, self.num_total_experts,
            self.num_zero_experts, output=output_tensor)

        return output

    # no compute-communication parallel, for prototyping only, not in actual use.
    # subject to becoming stale
    def forward_group_experts_longcat(
            self, hidden_states, total_num_experts, total_num_experts_per_rank, 
            topk_indices=None, topk_weights=None, residual_=None,
            expand_idx=None, combine_idx=None, token_count=None, cusum_token_count=None):
        is_fp8_quant = isinstance(self.quant_config, Fp8Config)
        ori_input_shape = hidden_states.shape
        dtype = hidden_states.dtype
        self.pack_params()
        self.pack_params_after_loading()
        w1=self.w13
        w2=self.w2
        bias1=self.b13
        bias2=self.b2
        input_smooth=self.a13_scale
        act_smooth=self.a2_scale
        w1_scale=self.w13_scale
        w2_scale=self.w2_scale
        gated=self.is_gated
        act_mode=self.hidden_act
        quant_input=None

        start_expert_id=self.start_expert_id
        expert_size = w1.size(0)
        max_m = hidden_states.shape[0]
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        residual_ = residual_.view(-1, residual_.size(-1)) if residual_ is not None else None
        # Check smooth quant parameters.
        per_token_sq = False
        if not is_fp8_quant:
            check_list = [input_smooth, act_smooth, w1_scale, w2_scale]
            if all(x is not None for x in check_list):
                per_token_sq = True

            if not (all(x is None for x in check_list) or all(x is not None for x in check_list)):
                raise ValueError("input_smooth, act_smooth, w1_scale and w2_scale must be present "
                                "and absent at the same time.")

        expert_id = topk_indices
        reduce_weight = topk_weights

        # gen_idx
        if expert_id is not None:
            expand_idx, combine_idx, token_count, cusum_token_count = mlu_ops.moe_gen_idx(expert_id, total_num_experts)

        # check quant
        if is_fp8_quant and self.quant_config.activation_quant_method == 'per_token':
            raise NotImplementedError
        elif per_token_sq:
            expand_hidden_states = mlu_ops.moe_expand_input(
                hidden_states, expand_idx, cusum_token_count, 
                start_expert_id=start_expert_id,
                expert_size=expert_size)
            expand_hidden_states_zero = mlu_ops.moe_expand_input(
                hidden_states, expand_idx, cusum_token_count, 
                start_expert_id=self.start_zero_expert_id,
                expert_size=self.zero_expert_size)
            quant_input, input_scale = mlu_ops.moe_quantize(
                expand_hidden_states, input_smooth, None, 
                token_count[start_expert_id:start_expert_id+expert_size])
        else:
            expand_hidden_states = mlu_ops.moe_expand_input(hidden_states, expand_idx,
                    cusum_token_count, start_expert_id, expert_size)
            expand_hidden_states_zero = mlu_ops.moe_expand_input(hidden_states, expand_idx,
                    cusum_token_count, self.start_zero_expert_id, self.zero_expert_size)

        if (is_fp8_quant and self.quant_config.activation_quant_method == 'per_token') or per_token_sq:
            gemm_out = mlu_ops.smooth_quant_group_gemm(
                quant_input, w1, 
                token_count[start_expert_id:start_expert_id+expert_size],
                None, None, None, None, input_scale, w1_scale, dtype, max_m)
        else:
            gemm_out = mlu_ops.group_gemm(expand_hidden_states, w1,
                                          token_count[start_expert_id:start_expert_id+expert_size],
                                          None, None, None, None, max_m)

        # add_bias_active
        if is_fp8_quant and self.quant_config.activation_quant_method == 'per_token':
            raise NotImplementedError
        elif per_token_sq:
            quant_input = quant_input[:, :gemm_out.shape[-1] // 2]
            input_scale = input_scale[:gemm_out.shape[0]]
            quant_input, input_scale = mlu_ops.moe_quantize(gemm_out, act_smooth, None,
                                                            token_count[start_expert_id:start_expert_id+expert_size],
                                                            output=quant_input,
                                                            output_scale=input_scale,
                                                            act_mode=act_mode,
                                                            is_gated=self.is_gated)

        if ((is_fp8_quant and self.quant_config.activation_quant_method == 'per_token') 
            or per_token_sq):
            # Remove the reference to gemm_out tensor.
            # If that was the only reference, the tensor’s memory becomes eligible for deallocation
            # So that we can reuse this memory for the new allocation of next gemm operation
            # del gemm_out
            gemm_out = mlu_ops.smooth_quant_group_gemm(
                quant_input, w2,
                token_count[start_expert_id:start_expert_id+expert_size],
                None, None, None, None, input_scale, w2_scale, dtype, max_m,
                output=expand_hidden_states)
        else:
            act_out = mlu_ops.moe_active(
                gemm_out, act_mode, gated, gemm_out[:,:gemm_out.shape[-1]//2], 
                bias1, cusum_token_count, start_expert_id, expert_size)
            gemm_out = mlu_ops.group_gemm(
                act_out, w2, token_count[start_expert_id:start_expert_id+expert_size],
                None, None, None, None, max_m,
                output=expand_hidden_states)

        output = mlu_ops.moe_combine_result(
            gemm_out, reduce_weight, combine_idx,
            residual_, cusum_token_count, start_expert_id,
            expert_size, bias2)
        if self.moe_ep_size > 1 or self.moe_tp_rank == 0:
            output = mlu_ops.moe_combine_result(
                expand_hidden_states_zero, reduce_weight, combine_idx,
                output, cusum_token_count, self.start_zero_expert_id,
                self.zero_expert_size, bias2,
                output=output)
        return output.view(ori_input_shape)