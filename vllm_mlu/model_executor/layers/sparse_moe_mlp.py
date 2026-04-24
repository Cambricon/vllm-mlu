# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

"""Inference-only MOE model."""
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_tensor_parallel_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_expert_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_dp_group,
    divide,
)
from vllm.distributed.utils import divide
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe import fused_grouped_topk
from vllm.utils.torch_utils import get_dtype_size
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu._mlu_utils import *
from vllm_mlu.model_executor.layers.feed_forward import FeedForward
from vllm_mlu.model_executor.layers.quantization.smoothquant import SmoothQuantConfig
from vllm_mlu.model_executor.layers.quantization.weightonly import WeightOnlyConfig
from vllm_mlu.distributed.parallel_state import(
    CnclEP, cnclep_dispatch, cnclep_combine)

from vllm_mlu.distributed.parallel_state import(
    CnclEP, cnclep_dispatch, cnclep_combine)

@dataclass
class MoeGroupInfo:
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    moe_tp_size: int
    moe_tp_rank: int
    moe_ep_size: int
    moe_ep_rank: int
    moe_group: Any
    moe_kwargs: dict

    def __init__(self):
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_rank = get_dp_group().rank_in_group
        self.dp_size = get_dp_group().world_size

        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        self.moe_tp_rank = get_moe_tensor_parallel_rank()
        self.moe_tp_group = get_moe_tensor_parallel_group()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.moe_ep_group = get_moe_expert_parallel_group()
        self.moe_group = self.moe_ep_group if self.moe_ep_size > 1 else self.moe_tp_group
        self.moe_kwargs = {"tp_group": self.moe_tp_group}


class SqrtSoftPlusTopK(torch.nn.Module):

    def __init__(self,
                 score_func: str,
                 use_hash: bool,
                 n_routed_experts: int,
                 n_activated_experts: int,
                 route_scale: float,
                 vocab_size: int,
                 prefix: str = ""):
        super().__init__()
        self.topk = n_activated_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.use_hash = use_hash
        self.n_routed_experts = n_routed_experts
        self.vocab_size = vocab_size
        if self.use_hash:
            self.tid2eid = nn.Parameter(
                torch.randint(0,
                              self.n_activated_experts,
                              (self.vocab_size, self.n_activated_experts),
                              dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.tid2eid = None
            self.bias = nn.Parameter(torch.empty(self.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, scores: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.score_func == "sqrtsoftplus"
        return mlu_ops.moe_softplus_topk(
            scores,
            self.topk,
            input_ids,
            self.tid2eid,
            self.bias,
            self.route_scale,
        )


# This is used by the Deepseek-V2 and Deepseek-V3 model
'''
=============================
Modify by vllm_mlu
=============================
@brief: comment out decorator torch.compiler to avoid triton bug for torch_mlu 2.9.1
'''
# @torch.compile(
#     dynamic=True,
#     backend=current_platform.simple_compile_backend,
#     options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
# )
'''
==================
End of MLU Hijack
==================
'''
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and current_platform.is_cuda()
        and num_expert_group <= 32
        and topk <= 32
        and e_score_correction_bias is not None
    ):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class SparseMoeMlp(nn.Module):
    """
    Tensor Parallel evenly splits each expert's weight and distributes them to different ranks,
    which means each rank holds partial weight of all experts.
    While Expert Parallel evenly distributes some of the experts' full weight to different ranks,
    which means each rank holds part of the experts' full weight.

    As a result, each rank in the Tensor Parallel group receives all tokens' hidden states for all experts,
    then computes using the partial weights, while for Expert Parallel, each rank only receives
    part of tokens' hidden states for experts on this rank, then computes using the full weights.

    When both Tensor Parallel and Expert Parallel are enabled, each rank handles
    a portion of the expert weights matrices (as in EP mode) and these weights are further sliced
    across ranks (as in TP mode). This hybrid approach aims to balance the workload more evenly across ranks,
    enhancing efficiency and reducing the likelihood of bottlenecks associated with EP mode alone.
    """
    reduce_weight : torch.Tensor = None
    expert_id     : torch.Tensor = None
    is_expert_avg : bool = False
    max_batched_token : int = 2048
    random_idx    : int = 0

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
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        is_use_fused_moe: bool = False,
        expert_group: int | None = 1,
        topk_group: int | None = 1,
        scoring_func: str = "softmax",
        topk_method: str = "",
        routed_scaling_factor: float = 1.0,
        tp_group: Any = None,
        use_all2all: bool = False,
        use_hash: bool = False,
        vocab_size: int = 0,
        prefix: str = "",
        init_avg_moe: bool = True,
    ):
        super().__init__()
        if tp_group is None:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        else:
            self.tp_rank = tp_group.rank_in_group
            self.tp_size = tp_group.world_size
            self.tp_group = tp_group
        self.use_hash = use_hash
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj_name = up_proj_name
        self.is_gated = is_gated
        self.down_proj_name = down_proj_name
        self.has_bias = has_bias
        self.renormalize = renormalize
        self.hidden_act = hidden_act
        self.quant_config = quant_config
        self.is_use_fused_moe = is_use_fused_moe
        self.expert_group = expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.use_all2all = use_all2all
        self.vocab_size = vocab_size
        # fused_moe doesn't support weightonly quantization
        if isinstance(quant_config, WeightOnlyConfig):
            self.is_use_fused_moe = False

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # [num_bytes_hidden_states, num_bytes_reduce_weights, num_bytes_expert_id]
        self.precompute_dim_bytes_list: List[int] | None = None
        # sum(self.precompute_dim_bytes_list)
        self.precompute_dim_bytes = -1

        moe_group_info = MoeGroupInfo()
        self.moe_tp_size = moe_group_info.moe_tp_size
        self.moe_tp_rank = moe_group_info.moe_tp_rank
        self.moe_ep_size = moe_group_info.moe_ep_size
        self.moe_ep_rank = moe_group_info.moe_ep_rank
        self.dp_size = moe_group_info.dp_size
        self.dp_rank = moe_group_info.dp_rank
        self.moe_group = moe_group_info.moe_group
        self.moe_kwargs = moe_group_info.moe_kwargs

        vllm_config = get_current_vllm_config()
        model_config = getattr(vllm_config, "model_config", None)
        hf_text_config = getattr(model_config, "hf_text_config", None)
        self.model_type = getattr(hf_text_config, "model_type", "")

        if (init_avg_moe and
            VLLM_AVG_MOE_EN and not SparseMoeMlp.is_expert_avg):
            n_tokens = SparseMoeMlp.max_batched_token * self.dp_size
            expert_group = self.moe_ep_size
            val = 1.0 / float(num_experts)
            SparseMoeMlp.reduce_weight = torch.full((n_tokens, top_k), val, device="mlu", dtype=torch.float32)
            import math
            if VLLM_RANDOM_MOE_EN:
                import numpy as np
                # example deepseekv2: experts 160 topk 6
                # avg list: 92,   8,  88,  45,  99,   9,... 118, 142, 116,  57, 104,   6,......
                array = np.stack([np.random.permutation(num_experts)[:top_k] for _ in range(n_tokens)])
                table = torch.from_numpy(array.flatten()).to(device="mlu", dtype=torch.int32)
            else:
                # example deepseekv2: experts 160
                # avg list: 0,20,40,60,80...120,140,  1,21,...121,141, 2...142,  ......  19,...159,  0,20,......
                batch_table = math.ceil(n_tokens * top_k / num_experts) * num_experts
                hi_val = batch_table // num_experts
                table = (torch.arange(hi_val * num_experts, device="mlu", dtype=torch.int32) % num_experts).view(
                    hi_val, expert_group, num_experts // expert_group).transpose(1, 2)
            SparseMoeMlp.expert_id = table.flatten()[:n_tokens * top_k].view(n_tokens, top_k)
            SparseMoeMlp.is_expert_avg = True
        # NOTE: The bias for fc2 is only applied on tp_rank 0. If we added it on all nodes the allreduce() would
        # contain multiple copies of the bias. The bias on other node will be ignored, and may be set to nullptr
        self.skip_bias_add = True if self.moe_tp_rank > 0 else False

        assert self.num_total_experts >= self.moe_ep_size, (
            f"need num_total_experts:{self.num_total_experts} >= moe_ep_size:{self.moe_ep_size}")

        assert self.intermediate_size % self.moe_tp_size == 0, (
            f"need intermediate_size:{self.intermediate_size} % moe_tp_size:{self.moe_tp_size} == 0")

        self.num_experts_per_rank = (self.num_total_experts + self.moe_ep_size - 1) // self.moe_ep_size
        if self.moe_ep_rank + 1 == self.moe_ep_size and self.num_total_experts % self.moe_ep_size:
            self.num_experts_per_rank = self.num_total_experts % self.moe_ep_size

        self.start_expert_id = self.moe_ep_rank * ((self.num_total_experts + self.moe_ep_size - 1) // self.moe_ep_size)
        self.end_expert_id = self.start_expert_id + self.num_experts_per_rank

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=self.params_dtype,
            quant_config=None,
        )
        if self.is_deepseek_v4:
            self.deepseekv4_topk = SqrtSoftPlusTopK(
                score_func=self.scoring_func,
                use_hash=self.use_hash,
                n_routed_experts=self.num_total_experts,
                n_activated_experts=self.top_k,
                route_scale=self.routed_scaling_factor,
                vocab_size=self.vocab_size,
                prefix=f"{prefix}.topk",
            )
        if topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.num_total_experts, device="mlu"))
        else:
            self.gate.e_score_correction_bias = None
        self.is_fp8_block_wise = (isinstance(self.quant_config, Fp8Config)
                                  and (self.quant_config.weight_block_size is not None))
        if self.is_fp8_block_wise:
            self.experts = FusedMoE(
                num_experts=self.num_experts_per_rank,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                reduce_results=False,
                renormalize=self.renormalize,
                quant_config=self.quant_config,
                use_grouped_topk=True,
                num_expert_group=self.expert_group,
                topk_group=self.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias)
        else:
            self.experts = nn.ModuleList([
                FeedForward(hidden_size=self.hidden_size,
                            intermediate_size=self.intermediate_size,
                            hidden_act=self.hidden_act,
                            up_proj_name=self.up_proj_name,
                            is_gated=self.is_gated,
                            down_proj_name=self.down_proj_name,
                            bias=self.has_bias,
                            quant_config=self.quant_config,
                            skip_bias_add=self.skip_bias_add,
                            reduce_results=False,
                            prefix=f"experts.{idx}",
                            **self.moe_kwargs) for idx in range(self.num_experts_per_rank)
            ])

        self.init_pack_param()

    @property
    def is_deepseek_v4(self):
        return self.scoring_func == 'sqrtsoftplus'

    @property
    def is_kimi_k2(self):
        kimi_k2_scoring_func = "sigmoid"
        kimi_k2_expert_group_num = 1
        kimi_k2_experts_num = 384
        return (self.scoring_func == kimi_k2_scoring_func
                and self.expert_group == kimi_k2_expert_group_num
                and self.num_total_experts == kimi_k2_experts_num)
    
    @property
    def is_glm4_moe(self):
        return self.model_type == "glm4_moe"

    def init_pack_param(self):
        self.w13 = None
        self.w2 = None
        self.b13 = None
        self.b2 = None
        self.w13_scale = None
        self.w2_scale = None
        self.a13_scale = None
        self.a13_scale_all_experts = None
        self.a2_scale = None
        self.pack_params_done = False
        self.pack_params_after_loading_done = False


    def map_param_data(self, param_list, is_use_first_data=False):
        if len(param_list) == 0:
            return None

        if is_use_first_data or len(param_list) == 1:
            first_data = param_list[0].data
            for param in param_list[1: -1]:
                param.data = first_data
            if is_use_first_data:
                out_param = first_data.view_as(param_list[0])
            else:
                out_param = first_data.view(len(param_list), *first_data.shape)
        else:
            packed_param = torch._utils._flatten_dense_tensors(param_list)
            data_list = torch._utils._unflatten_dense_tensors(packed_param, param_list)
            for data, param in zip(data_list, param_list):
                param.data = data
            out_param = packed_param.view(len(param_list), *data_list[0].shape)

        torch.mlu.empty_cache()

        return out_param


    def pack_unquantized_params(self, w13, w2, b13, b2):
        for expert in self.experts:
            up_proj = getattr(expert, self.up_proj_name)
            down_proj = getattr(expert, self.down_proj_name)
            w13.append(up_proj.weight)
            w2.append(down_proj.weight)
            if self.has_bias:
                b13.append(up_proj.bias)
                b2.append(down_proj.bias)


    def pack_smoothquant_params(self, w13, w2, b13, b2, w13_scale, w2_scale, a13_scale, a2_scale):
        for expert in self.experts:    
            up_proj = getattr(expert, self.up_proj_name)
            down_proj = getattr(expert, self.down_proj_name)
            w13.append(up_proj.qweight)
            w2.append(down_proj.qweight)
            if self.has_bias:
                b13.append(up_proj.bias)
                b2.append(down_proj.bias)
            w13_scale.append(up_proj.per_channel_scale)
            w2_scale.append(down_proj.per_channel_scale)
            if self.quant_config.input_quant_method == "per_token":
                a13_scale.append(up_proj.smooth)
                a2_scale.append(down_proj.smooth)
            else:
                a13_scale.append(up_proj.scale_to_int)
                a2_scale.append(down_proj.scale_to_int)


    def pack_weightonly_params(self, w13, w2, b13, b2, w13_scale, w2_scale):
        for expert in self.experts:
            up_proj = getattr(expert, self.up_proj_name)
            down_proj = getattr(expert, self.down_proj_name)
            w13.append(up_proj.qweight)
            w2.append(down_proj.qweight)
            if self.has_bias:
                b13.append(up_proj.bias)
                b2.append(down_proj.bias)
            w13_scale.append(up_proj.scales)
            w2_scale.append(down_proj.scales)

    def pack_fp8_params_without_activation_scheme(self, w13, w2, b13, b2, w13_scale, w2_scale):
        for expert in self.experts:
            up_proj = getattr(expert, self.up_proj_name)
            down_proj = getattr(expert, self.down_proj_name)
            w13.append(up_proj.weight)
            w2.append(down_proj.weight)
            if self.has_bias:
                b13.append(up_proj.bias)
                b2.append(down_proj.bias)
            w13_scale.append(up_proj.weight_scale)
            w2_scale.append(down_proj.weight_scale)


    def pack_params(self):
        if self.pack_params_done or self.is_fp8_block_wise:
            return

        w13 = []
        w2 = []
        b13 = []
        b2 = []
        w13_scale = []
        w2_scale = []
        a13_scale = []
        a2_scale = []

        if self.quant_config is None:
            self.pack_unquantized_params(w13, w2, b13, b2)
        elif isinstance(self.quant_config, SmoothQuantConfig):
            self.pack_smoothquant_params(w13, w2, b13, b2, w13_scale, w2_scale, a13_scale, a2_scale)
        elif isinstance(self.quant_config, WeightOnlyConfig):
            self.pack_weightonly_params(w13, w2, b13, b2, w13_scale, w2_scale)
        elif isinstance(self.quant_config, Fp8Config) and self.quant_config.activation_scheme == 'dynamic':
            self.pack_fp8_params_without_activation_scheme(w13, w2, b13, b2, w13_scale, w2_scale)
        else:
            raise ValueError(f'Unsupported quantization:{self.quant_config}')

        # pack weight
        self.w13 = self.map_param_data(w13)
        self.w2 = self.map_param_data(w2)

        # pack bias
        if self.has_bias:
            self.b13 = self.map_param_data(b13)
            # NOTE: The bias for fc2 is only applied on tp_rank 0. If we added it on all nodes the allreduce() would
            # contain multiple copies of the bias. The bias on other node will be ignored, and may be set to nullptr
            if self.skip_bias_add is False:
                self.b2 = self.map_param_data(b2)


        # pack weight scale
        if len(w13_scale) > 0:
            self.w13_scale = self.map_param_data(w13_scale)
        if len(w2_scale) > 0:
            self.w2_scale = self.map_param_data(w2_scale)

        # pack activate scale
        if len(a13_scale) > 0:
            self.a13_scale = self.map_param_data(a13_scale)
        if len(a2_scale) > 0:
            self.a2_scale = self.map_param_data(a2_scale)

        self.pack_params_done = True

    def pack_params_after_loading(self):
        if self.pack_params_after_loading_done or self.is_fp8_block_wise:
            return

        if isinstance(self.quant_config, SmoothQuantConfig) and self.quant_config.group_size > 1 and self.is_use_fused_moe:
            assert self.w13_scale is not None and self.w2_scale is not None, "w13_scale and w2_scale must be not None"
            self.w13_scale = self.w13_scale.permute(2, 0, 1).contiguous()
            self.w2_scale = self.w2_scale.permute(2, 0, 1).contiguous()

        # pack smooth variables for moe_quantize if fp8
        # FIXME: replace smooth to None after tmo supports.
        if isinstance(self.quant_config, Fp8Config):
            expert_size = self.w13.shape[0]
            fp8_smooth_2_hidden_size = self.w13.shape[1] // 2 if self.is_gated else self.w13.shape[1]
            self.fp8_smooth_1 = torch.ones([expert_size, self.hidden_size], device=self.w13.device, dtype=torch.float32)
            self.fp8_smooth_2 = torch.ones([expert_size, fp8_smooth_2_hidden_size], device=self.w13.device, dtype=torch.float32)

        self.pack_params_done = True
        self.pack_params_after_loading_done = True

    def get_precompute_dim_bytes_list(self, hidden_states_dtype: torch.dtype) -> List[int]:
        '''
        get the number of bytes of the hidden dimension corresponding to
        hidden_states, reduce_weight, and expert_id, respectively.
        '''
        if not self.precompute_dim_bytes_list:
            hidden_states_size = self.hidden_size * get_dtype_size(hidden_states_dtype)
            reduce_weights_size = self.top_k * get_dtype_size(torch.float)
            expert_id_size = self.top_k * get_dtype_size(torch.int32)
            self.precompute_dim_bytes_list = [
                hidden_states_size, reduce_weights_size, expert_id_size
            ]
        return self.precompute_dim_bytes_list

    def get_precompute_dim_bytes(self, hidden_states_dtype: torch.dtype) -> int:
        '''
        get the hidden dimension in bytes for a packed hidden states that 
        include 
        [hidden_states | reduce_weights | expert_id]
        '''
        if self.precompute_dim_bytes < 0:
            self.precompute_dim_bytes = sum(self.get_precompute_dim_bytes_list(hidden_states_dtype))
        return self.precompute_dim_bytes

    def reduce_results(self, final_hidden_states: torch.Tensor, reduce_results: bool = True):
        if reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states, self.moe_group)
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        orig_hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # expert_logits: [num_tokens, self.num_experts_per_rank]
        expert_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.forward_experts(hidden_states, expert_logits, residual)
        final_hidden_states = self.reduce_results(final_hidden_states)
        output = final_hidden_states.view(orig_hidden_states_shape)
        return output

    def precompute_weight_expert_id(
        self,
        packed: torch.Tensor,
    ) -> torch.Tensor:
        '''
        pre compute gate and softmax_topk/sigmoid_topk, and fill the weight and 
        expert_id part as below
        in  = [ hidden_states | ------ | --------- ] 
              [     bf16      |  fp32  |   int32   ]

        out = [ hidden_states | weight | expert_id ]
              [     bf16      |  fp32  |   int32   ]
        '''
        hidden_states_size, weight_size, expert_id_size = self.get_precompute_dim_bytes_list(packed.dtype)
        packed_int8 = packed.view(torch.int8)
        hidden_states = packed_int8[:, : hidden_states_size].view(packed.dtype)
        router_logits, _ = self.gate(hidden_states)
        topk=self.top_k
        renormalized=self.renormalize
        reduce_weight = packed_int8[:, hidden_states_size : hidden_states_size + weight_size].view(torch.float)
        expert_id = packed_int8[:, hidden_states_size + weight_size :].view(torch.int32)
        if self.scoring_func == "softmax":
            reduce_weight, expert_id = mlu_ops.moe_softmax_topk(router_logits, topk, renormalized, self.expert_group,
                                                                self.topk_group, route_scale=self.routed_scaling_factor,
                                                                reduce_weight=reduce_weight, 
                                                                expert_id=expert_id)
        elif self.scoring_func == "sigmoid":
            reduce_weight, expert_id = mlu_ops.moe_sigmoid_topk(router_logits, topk, renormalized,
                                                                self.expert_group, self.topk_group,
                                                                self.routed_scaling_factor,
                                                                self.gate.e_score_correction_bias,
                                                                reduce_weight=reduce_weight, 
                                                                expert_id=expert_id)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        return packed

    def forward_experts(self, hidden_states, expert_logits, residual: torch.Tensor | None = None,
                        shared_output: torch.Tensor | None = None,
                        input_ids: torch.Tensor | None = None):
        assert not (residual is not None and shared_output is not None)
        residual_ = None if self.tp_rank > 0 else residual

        # change only for deepseek_model without residual_
        if shared_output is not None:
            residual_ = shared_output

        if self.is_fp8_block_wise:
            output = self.experts(hidden_states=hidden_states,
                                router_logits=expert_logits) * self.routed_scaling_factor
            if residual_ is not None:
                output = output + residual_
            return output

        use_forward_group_experts = (self.is_use_fused_moe
                                     and (
                                         self.is_kimi_k2
                                         or self.is_glm4_moe
                                         or self.is_deepseek_v4
                                         or self.expert_group != 1)
                                    )
        if use_forward_group_experts:
            final_hidden_states = self.forward_group_experts(
                hidden_states,
                expert_logits,
                residual_,
                input_ids=input_ids,
            )
        elif self.is_use_fused_moe:
            self.pack_params()
            self.pack_params_after_loading()
            final_hidden_states = mlu_ops.fused_moe(hidden_states=hidden_states,
                                                    gating_output=expert_logits,
                                                    w1=self.w13,
                                                    w2=self.w2,
                                                    bias1=self.b13,
                                                    bias2=self.b2,
                                                    residual=residual_,
                                                    input_smooth=self.a13_scale,
                                                    act_smooth=self.a2_scale,
                                                    w1_scale=self.w13_scale,
                                                    w2_scale=self.w2_scale,
                                                    topk=self.top_k,
                                                    renormalize=self.renormalize,
                                                    gated=self.is_gated,
                                                    act_mode=self.hidden_act,
                                                    start_expert_id=self.start_expert_id,
                                                    avg_moe=VLLM_AVG_MOE_EN,
                                                    class_reduce_weight=SparseMoeMlp.reduce_weight,
                                                    class_expert_id=SparseMoeMlp.expert_id,
                                                    )
        else:
            final_hidden_states = self.forward_experts_nofused(hidden_states, expert_logits)
            if residual_ is not None:
                final_hidden_states = final_hidden_states + residual_
        return final_hidden_states


    def forward_experts_nofused(self, hidden_states, expert_logits):
        hidden_states_shape = hidden_states.shape
        if self.scoring_func == "softmax":
            topk_values, topk_indices = self.topk_softmax(expert_logits)
        elif self.scoring_func == "sigmoid":
            gating_output = expert_logits.to(torch.float32)
            gating_output = gating_output.view(-1, gating_output.size(-1))
            topk_values, topk_indices = grouped_topk(hidden_states, gating_output, self.top_k, self.renormalize,
                                                    self.expert_group, self.topk_group, self.scoring_func,
                                                    self.routed_scaling_factor, self.gate.e_score_correction_bias)
            topk_values = topk_values.to(hidden_states.dtype)
            topk_indices = topk_indices.to(torch.int64)
        expand_gather_idx, scatter_idx, expand_token_count, cusum_token_count = self.generate_gather_idx(
            topk_indices)
        # no expert is routed, then expand_gather_idx, expand_scatter_idx has no item,
        # expand_token_count and expand_cusum_token_count has item but the value is all zero
        # so this rank should only return final_hidden_states with zero value
        if expand_gather_idx.numel() == 0:
            final_hidden_states = torch.zeros_like(hidden_states,
                                                   dtype=hidden_states.dtype,
                                                   device=hidden_states.device)
            return final_hidden_states

        expand_hidden_states = self.expand_input(hidden_states, expand_gather_idx)

        expand_output_list = []
        expand_cusum_token_count = cusum_token_count[self.start_expert_id:self.end_expert_id +
                                                     1] - cusum_token_count[self.start_expert_id]
        for expert_idx, num_tokens_per_expert in enumerate(expand_token_count):
            if num_tokens_per_expert > 0:
                expert_hidden_states = expand_hidden_states[
                    expand_cusum_token_count[expert_idx]:expand_cusum_token_count[expert_idx + 1]]
                expert_output = self.experts[expert_idx](expert_hidden_states)
                expert_output = expert_output[0] if isinstance(expert_output, (tuple, list)) else expert_output
                expand_output_list.append(expert_output)
        expand_output = torch.cat(expand_output_list, dim=0)
        final_hidden_states = self.combine_moe(expand_output, scatter_idx, cusum_token_count, hidden_states_shape,
                                               topk_values)

        return final_hidden_states

    def forward_group_experts(self, hidden_states, gating_output, residual_, input_ids: torch.Tensor | None = None):
        # determine if hidden_states packs reduce_weight and expert_id in it, 
        # and if so, extract them.
        orig_dtype = hidden_states.dtype
        device = hidden_states.device
        hidden_states_int8 = hidden_states.view(torch.int8)
        hidden_states_size, weight_size, _ = self.get_precompute_dim_bytes_list(orig_dtype)
        packed_dim = self.get_precompute_dim_bytes(orig_dtype)
        is_precompute_weight_expert_id: bool = (hidden_states_int8.shape[1] == packed_dim)
        if is_precompute_weight_expert_id:
            assert gating_output is None
            hidden_states = hidden_states_int8[:, : hidden_states_size].view(orig_dtype)
            reduce_weight = hidden_states_int8[:, hidden_states_size : hidden_states_size + weight_size].view(torch.float)
            expert_id = hidden_states_int8[:, hidden_states_size + weight_size :].view(torch.int32)

        is_fp8_quant = isinstance(self.quant_config, Fp8Config)
        ori_input_shape = hidden_states.shape
        dtype = hidden_states.dtype
        self.pack_params()
        self.pack_params_after_loading()
        w1=self.w13.to(device) if self.w13 is not None else None
        w2=self.w2.to(device) if self.w2 is not None else None
        bias1=self.b13.to(device) if self.b13 is not None else None
        bias2=self.b2.to(device) if self.b2 is not None else None
        input_smooth=self.a13_scale.to(device) if self.a13_scale is not None else None
        act_smooth=self.a2_scale.to(device) if self.a2_scale is not None else None
        w1_scale=self.w13_scale.to(device) if self.w13_scale is not None else None
        w2_scale=self.w2_scale.to(device) if self.w2_scale is not None else None
        topk=self.top_k
        renormalized=self.renormalize
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

        # softmax_topk
        if not is_precompute_weight_expert_id:
            gating_output = gating_output.view(-1, gating_output.size(-1))
            if self.scoring_func == "softmax":
                reduce_weight, expert_id = mlu_ops.moe_softmax_topk(gating_output, topk, renormalized, self.expert_group,
                                                                    self.topk_group, route_scale=self.routed_scaling_factor)
            elif self.scoring_func == "sigmoid":
                reduce_weight, expert_id = mlu_ops.moe_sigmoid_topk(gating_output, topk, renormalized,
                                                                    self.expert_group, self.topk_group,
                                                                    self.routed_scaling_factor,
                                                                    self.gate.e_score_correction_bias)
            elif self.scoring_func == "sqrtsoftplus":
                assert hasattr(self,"deepseekv4_topk")
                reduce_weight, expert_id = self.deepseekv4_topk(
                    gating_output,
                    input_ids,
                )
            else:
                raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        if VLLM_RANDOM_MOE_EN: 
            n_tokens = hidden_states.shape[0]
            token_len = SparseMoeMlp.expert_id.size(0)
            SparseMoeMlp.random_idx = 0 if token_len == n_tokens else (SparseMoeMlp.random_idx+1) % (token_len-n_tokens)
            n_tokens = hidden_states.shape[0]
            reduce_weight = SparseMoeMlp.reduce_weight[:n_tokens]
            expert_id     = SparseMoeMlp.expert_id[SparseMoeMlp.random_idx: SparseMoeMlp.random_idx + n_tokens]
        elif VLLM_AVG_MOE_EN:
            n_tokens = hidden_states.shape[0]
            reduce_weight = SparseMoeMlp.reduce_weight[:n_tokens]
            expert_id     = SparseMoeMlp.expert_id[:n_tokens]
        # gen_idx
        expand_idx, combine_idx, token_count, cusum_token_count = mlu_ops.moe_gen_idx(expert_id, self.num_total_experts)
        # check quant
        if is_fp8_quant and self.quant_config.activation_quant_method == 'per_token':
            quant_input, input_scale = mlu_ops.moe_quantize(
                hidden_states,
                self.fp8_smooth_1,
                zero=None,
                token_count=token_count[start_expert_id:start_expert_id+expert_size],
                gather_index=expand_idx,
                gather_index_start_position=cusum_token_count[start_expert_id].unsqueeze(0),
                output=None,
                output_scale=None,
                dynamic_quant=True,
                quant_type=torch.float8_e4m3fn
            )
        elif per_token_sq:
            quant_input, input_scale = mlu_ops.moe_quantize(hidden_states,
                input_smooth, None, token_count[start_expert_id:start_expert_id+expert_size], expand_idx,
                cusum_token_count[start_expert_id].unsqueeze(0))
        else:
            expand_hidden_states = mlu_ops.moe_expand_input(
                hidden_states,
                expand_idx,
                cusum_token_count,
                start_expert_id,
                expert_size,
            )

        if (is_fp8_quant and self.quant_config.activation_quant_method == 'per_token') or per_token_sq:
            gemm_out = mlu_ops.smooth_quant_group_gemm(quant_input, w1,
                                                        token_count[start_expert_id:start_expert_id+expert_size],
                                                        None, None, None, None,
                                                        input_scale, w1_scale, dtype, max_m)
        else:
            gemm_out = mlu_ops.group_gemm(expand_hidden_states, w1,
                                       token_count[start_expert_id:start_expert_id+expert_size],
                                       None, None, None, None, max_m)
        # add_bias_active
        if is_fp8_quant and self.quant_config.activation_quant_method == 'per_token':
            act_out = mlu_ops.moe_active(gemm_out, act_mode, gated, gemm_out[:,:gemm_out.shape[-1]//2], bias=bias1, cusum_token_count=cusum_token_count, start_expert_id=start_expert_id, expert_size=expert_size)
            quant_input, input_scale = mlu_ops.moe_quantize(
                act_out,
                self.fp8_smooth_2,
                zero=None,
                token_count=token_count[start_expert_id:start_expert_id+expert_size],
                gather_index=None,
                gather_index_start_position=None,
                output=quant_input[:,:act_out.shape[-1]],
                output_scale=None,
                dynamic_quant=True,
                quant_type=torch.float8_e4m3fn
            )
        elif per_token_sq:
            quant_input = quant_input[:, :gemm_out.shape[-1] // 2]
            input_scale = input_scale[:gemm_out.shape[0]]
            quant_input, input_scale = mlu_ops.moe_quantize(gemm_out, act_smooth, None,
                                                            token_count[start_expert_id:start_expert_id+expert_size],
                                                            output=quant_input,
                                                            output_scale=input_scale,
                                                            act_mode=act_mode,
                                                            is_gated=self.is_gated)

        if (is_fp8_quant and self.quant_config.activation_quant_method == 'per_token') or per_token_sq:
            # Remove the reference to gemm_out tensor.
            # If that was the only reference, the tensor’s memory becomes eligible for deallocation
            # So that we can reuse this memory for the new allocation of next gemm operation
            del gemm_out
            gemm_out = mlu_ops.smooth_quant_group_gemm(quant_input, w2,
                                                    token_count[start_expert_id:start_expert_id+expert_size],
                                                    None, None, None, None, input_scale, w2_scale, dtype, max_m)
        else:
            act_out = mlu_ops.moe_active(gemm_out, act_mode, gated, gemm_out[:,:gemm_out.shape[-1]//2], bias1, cusum_token_count, start_expert_id, expert_size)
            gemm_out = mlu_ops.group_gemm(act_out, w2,
                                       token_count[start_expert_id:start_expert_id+expert_size],
                                       None, None, None, None, max_m)

        # we reuse the memory of hidden_states to store the output
        output = mlu_ops.moe_combine_result(
            gemm_out, reduce_weight, combine_idx,
            residual_, cusum_token_count, start_expert_id,
            expert_size, bias2,
            output=hidden_states if not is_precompute_weight_expert_id else None)
        return output.view(ori_input_shape)


    def topk_softmax(self, expert_logits):
        # expert_logits: [num_tokens, self.num_experts_per_rank]
        # topk_values: [num_tokens, self.top_k]
        # topk_indices: [num_tokens, self.top_k]
        if self.renormalize:
            topk_values, topk_indices = torch.topk(expert_logits, self.top_k, dim=-1)
            topk_values = torch.softmax(topk_values, -1)
        else:
            router_probs = torch.softmax(expert_logits, -1)
            topk_values, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        return topk_values, topk_indices


    def generate_gather_idx(self, topk_indices):
        device = topk_indices.device
        # gather_expand_idx: [num_tokens * self.top_k]
        sorted_expert_id, indices = topk_indices.flatten().sort()
        gather_idx = indices // self.top_k

        seqs = torch.arange(indices.numel(), dtype=indices.dtype, device=indices.device)
        scatter_idx=torch.zeros((indices.numel(),), dtype=seqs.dtype, device=seqs.device).scatter(0, indices, seqs)

        # token_count: [self.num_experts_per_rank]
        partial_token_index, partial_token_count = sorted_expert_id.unique(sorted=True, return_counts=True)
        zero_token_count = torch.zeros(self.num_total_experts, dtype=partial_token_count.dtype, device=device)
        token_count = zero_token_count.scatter(dim=0, index=partial_token_index, src=partial_token_count)
        # cusum_token_count: [self.num_experts_per_rank + 1]
        cusum_token_count = torch.cat(
            [torch.tensor([0], dtype=token_count.dtype, device=device),
             token_count.cumsum(dim=0)])

        num_tokens_before_expert = cusum_token_count[self.start_expert_id]
        num_tokens_including_expert = cusum_token_count[self.end_expert_id]

        expand_gather_idx = gather_idx[num_tokens_before_expert:num_tokens_including_expert]
        expand_token_count = token_count[self.start_expert_id:self.end_expert_id]

        return expand_gather_idx, scatter_idx, expand_token_count, cusum_token_count


    def expand_input(self, hidden_states, expand_gather_idx):
        expand_hidden_states = hidden_states[expand_gather_idx]
        return expand_hidden_states


    def combine_moe(self, expand_output, scatter_idx, cusum_token_count, hidden_states_shape, topk_values):
        num_tokens, hidden_size = hidden_states_shape
        num_tokens_before_expert = cusum_token_count[self.start_expert_id]
        num_tokens_after_expert = cusum_token_count[-1] - cusum_token_count[self.end_expert_id]

        expand_output_before_expert = torch.zeros((num_tokens_before_expert, hidden_size),
                                                   dtype=expand_output.dtype,
                                                   device=expand_output.device)
        expand_output_after_expert = torch.zeros((num_tokens_after_expert, hidden_size),
                                                   dtype=expand_output.dtype,
                                                   device=expand_output.device)
        unscatted_output = torch.cat([expand_output_before_expert, expand_output, expand_output_after_expert], dim=0)
        scatter_output = unscatted_output[scatter_idx]
        hidden_states_weight = topk_values.flatten().unsqueeze(-1)
        weighted_hidden_states = scatter_output * hidden_states_weight
        unreduced_hidden_states = weighted_hidden_states.view(num_tokens, self.top_k, hidden_size)
        final_hidden_states = unreduced_hidden_states.sum(dim=1)

        return final_hidden_states

    def prepare_for_cnclep(self, cnclep: CnclEP) -> None:
        if cnclep.use_quant_dispatch:
            self.prepare_for_cnclep_quant_dispatch(cnclep)
        else:
            self.prepare_for_cnclep_bf16(cnclep)

    def prepare_for_cnclep_bf16(self, cnclep: CnclEP) -> None:
        # prepare buffers for the forward process
        buffer = cnclep.buffer
        self.dispatch_send_buffer = buffer.dispatch_send_token_tensor
        self.dispatch_recv_buffer = buffer.dispatch_recv_token_tensor
        self.combine_send_buffer = buffer.combine_send_token_tensor
        self.combine_recv_buffer = buffer.combine_recv_token_tensor
        self.max_num_tokens_per_rank = cnclep.max_num_tokens_per_rank

        # get sizes in bytes
        self.dispatch_token_size = self.config.hidden_size * 2
        # [nranks, 2]
        self.dispatch_recv_layout = torch.empty((self.moe_ep_size, 2), dtype=torch.int32, device="mlu")
        # [num_total_experts]
        self.dispatch_recv_token_num = torch.empty((self.num_total_experts), dtype=torch.int32, device="mlu")

        self.max_num_tokens_recv = self.max_num_tokens_per_rank * self.moe_ep_size
        self.max_num_tokens_per_expert = divide(self.max_num_tokens_recv, self.top_k)

        # input to the first groupgemm, in which tokens are ordered by experts.
        input_recv_size = self.max_num_tokens_recv * self.dispatch_token_size
        self.input_recv = (
            self.combine_send_buffer[:input_recv_size]
            .view(self.max_num_tokens_recv, self.dispatch_token_size)
        )
        # kept for code without compute-communication parallel, which may have
        # become stale.
        self.quant_input_recv = self.input_recv

    def prepare_for_cnclep_quant_dispatch(self, cnclep: CnclEP) -> None:
        # prepare smooth parameter for _all_ experts globally, which would be needed during
        # input quantization before dispatch.
        assert self.a13_scale is not None, "a13_scale has not been loaded"
        self.a13_scale_all_experts = torch.zeros((self.num_total_experts, self.hidden_size),
                                                  dtype=self.a13_scale.dtype,
                                                  device=self.a13_scale.device)
        torch.distributed.all_gather_into_tensor(self.a13_scale_all_experts,
                                                 self.a13_scale,
                                                 group=self.moe_group.device_group,
                                                 async_op=False)

        # prepare buffers for the forward process
        buffer = cnclep.buffer
        self.dispatch_send_buffer = buffer.dispatch_send_token_tensor
        self.dispatch_recv_buffer = buffer.dispatch_recv_token_tensor
        self.combine_send_buffer = buffer.combine_send_token_tensor
        self.combine_recv_buffer = buffer.combine_recv_token_tensor
        self.max_num_tokens_per_rank = cnclep.max_num_tokens_per_rank

        # get sizes in bytes
        self.quant_size = self.hidden_size
        self.scale_size = get_dtype_size(torch.float32)
        self.dispatch_token_size = self.quant_size + self.scale_size
        # [nranks, 2]
        self.dispatch_recv_layout = torch.empty((self.moe_ep_size, 2), dtype=torch.int32, device="mlu")
        # [num_total_experts]
        self.dispatch_recv_token_num = torch.empty((self.num_total_experts), dtype=torch.int32, device="mlu")

        self.max_num_tokens_recv = self.max_num_tokens_per_rank * self.moe_ep_size
        self.max_num_tokens_per_expert = divide(self.max_num_tokens_recv, self.top_k)

        quant_input_recv_size = self.max_num_tokens_recv * self.quant_size
        input_scale_recv_size = self.max_num_tokens_recv * self.scale_size
        self.quant_input_recv = (
            self.combine_send_buffer[:quant_input_recv_size]
            .view(self.max_num_tokens_recv, self.quant_size))
        self.input_scale_recv = (
            self.combine_send_buffer[quant_input_recv_size : quant_input_recv_size + input_scale_recv_size]
            .view(self.max_num_tokens_recv, self.scale_size))

    def forward_all2all(
        self,
        hidden_states: torch.Tensor,
        gate: ReplicatedLinear,
        streams: Optional[Dict[str, torch.mlu.Stream]] = None,
        shared_experts: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """forward with all2all."""
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
        topk=self.top_k
        renormalized=self.renormalize
        act_mode=self.hidden_act
        quant_input=None

        start_expert_id=self.start_expert_id
        expert_size = w1.size(0)
        max_m = hidden_states.shape[0]
        gating_output, _ = gate(hidden_states)
        gating_output = gating_output.view(-1, gating_output.size(-1))
        if self.scoring_func == "softmax":
            reduce_weight, expert_id = mlu_ops.moe_softmax_topk(gating_output, topk, renormalized, self.expert_group,
                                                                self.topk_group, route_scale=self.routed_scaling_factor)
        elif self.scoring_func == "sigmoid":
            reduce_weight, expert_id = mlu_ops.moe_sigmoid_topk(gating_output, topk, renormalized,
                                                                self.expert_group, self.topk_group,
                                                                self.routed_scaling_factor,
                                                                self.gate.e_score_correction_bias)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        if VLLM_AVG_MOE_EN:
            # get dp rank
            dp_rank = get_dp_group().rank_in_group
            tp_rank = get_tp_group().rank_in_group
            global_rank = dp_rank * get_tp_group().world_size + tp_rank
            n_tokens = hidden_states.shape[0]
            reduce_weight = SparseMoeMlp.reduce_weight[:n_tokens]
            if self.use_all2all and VLLM_RANDOM_MOE_EN:
                expert_id = SparseMoeMlp.expert_id[global_rank * n_tokens : (global_rank+1) * n_tokens]
            elif self.use_all2all:
                expert_id = SparseMoeMlp.expert_id[dp_rank * n_tokens: dp_rank * n_tokens + n_tokens]
            else:
                expert_id = SparseMoeMlp.expert_id[:n_tokens]
        
        expand_idx, combine_idx, token_count, cusum_token_count \
            = mlu_ops.moe_gen_idx(expert_id, self.num_total_experts)

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
            hidden_states, input_smooth, None, token_count, expand_idx, None,
            output=quant_input,
            output_scale=input_scale)

        dispatch_send_layout = mlu_ops.moe_all2all_gen_send_layout(token_count, self.moe_ep_size)

        cnclep_dispatch(self.dispatch_token_size, 
                        num_token_expand, 
                        dispatch_send_layout, 
                        token_count, 
                        self.dispatch_recv_layout, 
                        self.dispatch_recv_token_num) 

        recv_token_num = self.dispatch_recv_token_num.view(self.moe_ep_size, self.num_experts_per_rank)
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

        shared_output = None
        if shared_experts is not None:
            parallelize_shared_expert = streams is not None
            if parallelize_shared_expert:
                compute_stream = streams['shared']
                comm_stream = streams['routed']
                curr_stream = torch.mlu.current_stream()
                compute_stream.wait_stream(curr_stream)
                comm_stream.wait_stream(curr_stream)

                with torch.mlu.stream(compute_stream):
                    shared_output = shared_experts(hidden_states, use_tp_weight=False)

                with torch.mlu.stream(comm_stream):
                    cnclep_combine(**combine_args)

                curr_stream.wait_stream(compute_stream)
                curr_stream.wait_stream(comm_stream)
            else:
                shared_output = shared_experts(hidden_states, use_tp_weight=False)
                cnclep_combine(**combine_args)
        else:
            cnclep_combine(**combine_args)

        numel_recv = num_token_expand * self.hidden_size
        recv_token = (self.combine_recv_buffer.view(hidden_states.dtype)[:numel_recv]
                      .view(num_token_expand, self.hidden_size))

        residual_ = shared_output
        output = mlu_ops.moe_combine_result(recv_token, reduce_weight, combine_idx,
                     residual_, None, start_expert_id,
                     expert_size, bias2, output=hidden_states)

        return output.view(ori_input_shape)
