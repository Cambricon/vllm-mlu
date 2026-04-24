# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import (
    Any, List, Tuple, Optional, Dict, Union, ClassVar, Literal,
    Protocol, overload, runtime_checkable)
from typing_extensions import TypeIs

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_gather_into_list,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed import (
    get_tp_group,
    get_pp_group,
    get_dp_group,
    get_data_parallel_group_rank,
    get_data_parallel_group_world_size,
    get_dense_mlp_tp_world_size,
    get_tp_world_world_size,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_logits_tp_world_size,
    get_parallel_rank_with_group,
    get_tp_world_group,
    get_tp_world_rank,
    GroupCoordinator,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_mlu.mlu_forward_context import MLUDPMetadata
from vllm_mlu.model_executor.layers.sparse_moe_mlp import SparseMoeMlp
from vllm_mlu.v1.attention.backends.utils import get_common_metadata

logger = init_logger(__name__)

# alias after refactor
DataParallelRuntimeParams = MLUDPMetadata


def enable_data_parallel():
    return get_dp_group().world_size > 1


def enable_emb_logits_custom_parallel():
    return get_logits_tp_world_size() != get_tensor_model_parallel_world_size()


def enable_dense_mlp_custom_parallel():
    return get_dense_mlp_tp_world_size() != get_tp_world_world_size()


def get_runtime_infos_per_dp_group(
        num_tokens: int, num_requests: int, all_prefill: bool, seq_lens: List[int], 
        device: torch.device, vllm_config: VllmConfig) -> Tuple[List[int], List[bool]]:
    dp_tensor = torch.tensor([num_tokens, num_requests, int(all_prefill)]).to(device, non_blocking=True)
    outputs = tensor_model_parallel_all_gather_into_list(dp_tensor, get_dp_group())
    outputs = torch.cat(outputs).tolist() # d2h
    dp_world_size = get_data_parallel_group_world_size()
    dp_is_prefill, dp_query_lens, dp_group_bs, seq_len_per_batch = [], [], [], []
    for i in range(0, 3 * dp_world_size, 3):
        dp_query_lens.append(outputs[i])
        dp_group_bs.append(outputs[i + 1])
        dp_is_prefill.append(bool(outputs[i + 2]))

    # Only run communication if mcc is enabled and is prefill.
    if vllm_config.mlu_config.is_dpsk_mcc_enabled and all(dp_is_prefill):
        assert len(seq_lens) == num_requests
        seq_len_per_batch = [torch.empty([bs], dtype=dp_tensor.dtype, device=device) for bs in dp_group_bs]
        seq_lens_tensor = torch.tensor(seq_lens, dtype=dp_tensor.dtype, device=device)
        torch.distributed.all_gather(seq_len_per_batch, seq_lens_tensor, group=get_dp_group().device_group)
        seq_len_per_batch=torch.cat(seq_len_per_batch).tolist()
    else:
        seq_len_per_batch = [0] * sum(dp_group_bs)

    return dp_query_lens, dp_group_bs, dp_is_prefill, seq_len_per_batch


def get_deepseek_layer_split_list(
    dp_query_lens: List[int], dp_group_bs: List[int]
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    if len(dp_query_lens) != len(dp_group_bs) or len(dp_query_lens) != get_data_parallel_group_world_size():
        logger.warning(f"dp_query_lens length: {len(dp_query_lens)} != dp_group_bs length: {len(dp_group_bs)}, "
                       f"disable deepseek layer split")
        return None, None, None
    emb_query_lens, logits_batch_sizes, dense_attn_token_split_list = None, None, None
    all_dp_query_lens, all_dp_group_bs = [], []
    for i in range(len(dp_query_lens)):
        all_dp_query_lens.extend([dp_query_lens[i]] * get_tensor_model_parallel_world_size())
        all_dp_group_bs.extend([dp_group_bs[i]] * get_tensor_model_parallel_world_size())
    if get_logits_tp_world_size() != get_tensor_model_parallel_world_size():
        slice_start = get_tp_world_rank() // get_logits_tp_world_size() * get_logits_tp_world_size()
        slice_end = slice_start + get_logits_tp_world_size()
        emb_query_lens = all_dp_query_lens[slice_start:slice_end]
        logits_batch_sizes = all_dp_group_bs[slice_start:slice_end]
    if get_dense_mlp_tp_world_size() != get_tp_world_world_size():
        slice_start = get_tp_world_rank() // get_dense_mlp_tp_world_size() * get_dense_mlp_tp_world_size()
        slice_end = slice_start + get_dense_mlp_tp_world_size()
        dense_attn_token_split_list = all_dp_query_lens[slice_start:slice_end]
    return emb_query_lens, logits_batch_sizes, dense_attn_token_split_list


def get_dp_metadata(
    num_tokens: int,
    data_parallel_size: int,
    data_parallel_rank: int,
    tensor_parallel_size: int,
    prefill_dispatch_use_RS_AG: bool,
) -> DataParallelRuntimeParams:
    """
    Get dp params when dummy run or capture model graph. These two cases do not have
    dp_params when forward call, because we do not want to hijack to much.
    """
    dp_query_lens = [num_tokens] * data_parallel_size
    in_prefill = get_forward_context().attn_metadata is None  # dummy run
    dp_is_prefill = [in_prefill] * data_parallel_size
    emb_query_lens, logits_batch_sizes, dense_attn_token_split_list = None, None, None
    if get_logits_tp_world_size() != get_tensor_model_parallel_world_size():
        emb_query_lens = [num_tokens] * get_logits_tp_world_size()
        logits_batch_sizes = None  # dummy run and capture model does not contain logits
    if get_dense_mlp_tp_world_size() != get_tp_world_world_size():
        dense_attn_token_split_list = [num_tokens] * get_dense_mlp_tp_world_size()

    return MLUDPMetadata.make_oot(data_parallel_rank,
                                  data_parallel_size,
                                  tensor_parallel_size,
                                  dp_query_lens,
                                  dp_is_prefill,
                                  prefill_dispatch_use_RS_AG,
                                  emb_query_lens=emb_query_lens,
                                  logits_batch_sizes=logits_batch_sizes,
                                  dense_attn_token_split_list=dense_attn_token_split_list)


def remove_paddings_after_all_gather(
    hidden_states: torch.Tensor,
    padding_to_token_num: int,
    token_num_list: List[int],
) -> torch.Tensor:
    dp_group_tensors = []
    offset = 0
    for token_num in token_num_list:
        if token_num != 0:
            dp_group_tensors.append(hidden_states[offset:offset+token_num])
        offset += padding_to_token_num
    if len(dp_group_tensors) == 1:
        hidden_states = dp_group_tensors[0]
    else:
        hidden_states = torch.cat(dp_group_tensors)
    return hidden_states


def tensor_model_parallel_all_gather_dp(
    group_num_tokens: List[int],
    rank: int,
    hidden_states: Optional[torch.Tensor],
    group: GroupCoordinator,
    hidden_size: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None) -> torch.Tensor:
    """
    All gather in the group.
    Input is a 2-D tensor, and can have different shape in the first dim,
    for example, [4, 7, 5, 8], [2, 5, 4, 0].
    """
    num_tokens_equal = all(x == group_num_tokens[0] for x in group_num_tokens)
    if num_tokens_equal:
        hidden_states = tensor_model_parallel_all_gather(
            input_=hidden_states, dim=0, tp_group=group)
    else:
        max_num_tokens = max(group_num_tokens)
        num_padding = max_num_tokens - group_num_tokens[rank]
        if num_padding > 0:
            if hidden_states is None:
                hidden_states = torch.empty((max_num_tokens, hidden_size),
                                            dtype=dtype, device=device)
            else:
                hidden_states = F.pad(hidden_states, (0, 0, 0, num_padding))
        hidden_states = tensor_model_parallel_all_gather(
            input_=hidden_states, dim=0, tp_group=group)
        hidden_states = remove_paddings_after_all_gather(
            hidden_states, max_num_tokens, group_num_tokens)
    return hidden_states

def tensor_model_parallel_all_gather_op_v2(
    input_: torch.Tensor,
    dim_size_list: List[int],
    group_coordinator: GroupCoordinator,
    non_leading_dim_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    All gather the input tensor across model parallel group with only communication ops.

    Note: compared to `tensor_model_parallel_all_gather_dp`, this method supports different
    sizes in the first dim, and does not involve padding operation.
    """
    all_size_equal = all([dim_size == dim_size_list[0] for dim_size in dim_size_list])

    output_shape = (sum(dim_size_list), non_leading_dim_size)
    output = torch.empty(output_shape, device=device, dtype=dtype)

    if input_ is None:
        input_ = torch.empty((0, non_leading_dim_size), device=device, dtype=dtype)

    if all_size_equal:
        torch.distributed.all_gather_into_tensor(
            output, input_, group=group_coordinator.device_group)
    else:
        # Note: torch.split splits the tensor into chunks. And each chunk
        # is a view of the original tensor.
        tensor_list = torch.split(output, dim_size_list, dim=0)
        torch.distributed.all_gather(
            list(tensor_list), input_, group=group_coordinator.device_group)
    return output

def process_post_attention_communication(
    hidden_states: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    tp_group: Any = None,
):
    """
    Processes distributed communication operations after attention computation.

    This function performs necessary communication operations after attention computation
    to ensure data synchronization across different parallel groups.
    Supports two modes:
    1. Tensor parallel mode: Uses tp_group for all-reduce and all-gather operations
    2. Data parallel mode: Uses reduce-scatter and all-gather for global synchronization

    Args:
        hidden_states: Hidden states tensor after attention computation, can be None
        dp_params: Data parallel runtime parameters containing token distribution and padding info
        hidden_size: Dimension size of hidden states
        dtype: Data type of the tensor
        device: Device where the tensor is located
        tp_group: Tensor parallel group, if None uses data parallel mode

    Returns:
        Hidden states tensor after communication synchronization processing

    Note:
        - When prefill_pad_to_token_num != -1, padding and unpadding operations will be performed
        - Function selects optimal communication path based on token count and parallel strategy
    """
    if tp_group is not None:
        if dp_params.token_num != 0:
            hidden_states = tensor_model_parallel_all_reduce(
                hidden_states)
        hidden_states = tensor_model_parallel_all_gather_dp(
            group_num_tokens=dp_params.dense_attn_token_split_list,
            rank=get_parallel_rank_with_group(tp_group),
            hidden_states=hidden_states,
            group=tp_group,
        )
    else:
        if dp_params.prefill_pad_to_token_num != -1:
            # pad hidden_states to use reduce_scatter and global all gather
            pad_num = dp_params.prefill_pad_to_token_num - dp_params.token_num
            if pad_num != 0:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_num))

            hidden_states = tensor_model_parallel_reduce_scatter(
                hidden_states, dim=0)

            hidden_states = tensor_model_parallel_all_gather_dp(
                group_num_tokens=dp_params.attn_token_split_list_reduce_scatter,
                rank=get_tp_world_rank(),
                hidden_states=hidden_states,
                group=get_tp_world_group(),
            )

            # get origin hidden_states for moe compute
            hidden_states = remove_paddings_after_all_gather(
                hidden_states, dp_params.prefill_pad_to_token_num,
                dp_params.token_split_list)
        else:
            hidden_states = tensor_model_parallel_all_reduce(
                hidden_states)

            all_gather_group = get_dp_group()
            all_gather_rank = get_data_parallel_group_rank()
            hidden_states = tensor_model_parallel_all_gather_dp(
                dp_params.token_split_list, all_gather_rank, hidden_states,
                all_gather_group, hidden_size, dtype, device)

    return hidden_states

def dp_model_forward(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    embedding_layer: nn.Module,
    model_norm_layer: nn.Module,
    start_layer: int,
    end_layer: int,
    layers: List[nn.Module],
    layer_input_norm_name: str,
    prefill_dispatch_use_RS_AG: bool,
    streams: Optional[Dict[str, torch.mlu.Stream]] = None,
) -> Union[torch.Tensor, IntermediateTensors]:
    """run model with dp."""
    if dp_params is None:
        dp_params = get_dp_metadata(positions.numel(),
                                    get_data_parallel_group_world_size(),
                                    get_data_parallel_group_rank(),
                                    get_tensor_model_parallel_world_size(),
                                    prefill_dispatch_use_RS_AG)

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if embedding_layer.__class__.__name__ == "DPVocabParallelEmbedding":
                hidden_states = embedding_layer(input_ids, dp_params=dp_params)
            else:
                hidden_states = embedding_layer(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for i in range(start_layer, end_layer):
        is_first_layer = (i == start_layer)
        is_last_layer = (i == end_layer - 1)
        next_input_layernorm = None
        if not is_last_layer:
            next_input_layernorm = getattr(layers[i+1], layer_input_norm_name)
        hidden_states, residual = layers[i](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            dp_params=dp_params,
            is_first_layer=is_first_layer,
            is_last_layer=is_last_layer,
            streams=streams,
            next_input_layernorm=next_input_layernorm,
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    hidden_states = model_norm_layer(hidden_states)
    return hidden_states

def dp_layer_forward(
    input_norm: nn.Module,
    self_attn: nn.Module,
    post_norm: nn.Module,
    mlp: nn.Module,
    mlp_kwargs: List[Dict[str, Any]],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    hidden_size: int,
    hidden_states_dtype: torch.dtype,
    is_first_layer: bool = False,
    is_last_layer: bool = False,
    next_input_layernorm: Optional[nn.Module] = None,
    enable_all2all: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    run layer with dp. dispatch all2all or rs+ag or common.
    
    For mlp_kwargs, because all2all forward args is often different with common mlp args.
    So here we decide that the mlp_kwargs[-1] is always all2all kwargs. For example:

    Deepseek enable all2all, mlp_kwargs will be: [{mlp common forward kwargs}, {mlp all2all kwargs}].
    Deepseek does not enable all2all, mlp_kwargs will be: [{mlp common forward kwargs}].
    """
    if dp_params.layer_use_reduce_scatter:
        common_metadata = get_common_metadata()
        is_decode_only = common_metadata is not None and common_metadata.is_decode_only
        use_all2all = enable_all2all and is_decode_only and isinstance(mlp, SparseMoeMlp)
        forward_func = _dp_forward_layer_all2all if use_all2all else _dp_forward_layer_rs_ag
        hidden_states, residual = forward_func(input_norm,
                                               self_attn,
                                               post_norm,
                                               mlp,
                                               mlp_kwargs,
                                               positions,
                                               hidden_states,
                                               residual,
                                               dp_params,
                                               is_first_layer,
                                               is_last_layer,
                                               next_input_layernorm)
    else:
        hidden_states, residual = _dp_forward_layer_common(input_norm,
                                                           self_attn,
                                                           post_norm,
                                                           mlp,
                                                           mlp_kwargs,
                                                           positions,
                                                           hidden_states,
                                                           residual,
                                                           dp_params,
                                                           hidden_size,
                                                           hidden_states_dtype)
    return hidden_states, residual

def _dp_forward_layer_rs_ag(
    input_norm: nn.Module,
    self_attn: nn.Module,
    post_norm: nn.Module,
    mlp: nn.Module,
    mlp_kwargs: List[Dict[str, Any]],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    is_first_layer: bool,
    is_last_layer: bool,
    next_input_layernorm: List[Optional[nn.Module]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """run layer with rs+ag."""
    if residual is None:
        residual = hidden_states

    # We move the input_layernorm of i+1 layer to the end of i layer.
    # But for the first layer, we need to do input_layernorm first.
    if is_first_layer:
       hidden_states = input_norm(hidden_states)

    # Self Attention
    hidden_states = self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # add residual here for the first layer
    if is_first_layer and get_tensor_model_parallel_rank() == 0:
        hidden_states = hidden_states + residual

    hidden_states = tensor_model_parallel_reduce_scatter(
        hidden_states, dim=0)

    # move norm between rs and ag
    if is_first_layer:
        residual = hidden_states
        hidden_states = post_norm(hidden_states)
    else:
        hidden_states, residual = post_norm(hidden_states, residual)

    hidden_states = tensor_model_parallel_all_gather_dp(
        group_num_tokens=dp_params.attn_token_split_list_reduce_scatter,
        rank=get_tp_world_rank(),
        hidden_states=hidden_states,
        group=get_tp_world_group(),
    )

    # mlp, use all cards
    hidden_states = mlp(hidden_states, **mlp_kwargs[0])

    hidden_states = tensor_model_parallel_reduce_scatter(
        hidden_states, dim=0, tp_group=get_tp_world_group())

    if is_last_layer:
        hidden_states = hidden_states + residual
        residual = None
    else:
        # To reduce layernorm computation, we move the layernorm of i+1 layer to
        # the end of i layer. Besides, we fuse residual addition into layernorm.
        assert next_input_layernorm is not None
        hidden_states, residual = next_input_layernorm(hidden_states, residual)

    hidden_states = tensor_model_parallel_all_gather_dp(
        group_num_tokens=dp_params.moe_token_split_list_reduce_scatter,
        rank=get_tensor_model_parallel_rank(),
        hidden_states=hidden_states,
        group=get_tp_group(),
    )

    return hidden_states, residual

def _dp_forward_layer_all2all(
    input_norm: nn.Module,
    self_attn: nn.Module,
    post_norm: nn.Module,
    mlp: nn.Module,
    mlp_kwargs: List[Dict[str, Any]],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    is_first_layer: bool,
    is_last_layer: bool,
    next_input_layernorm: List[Optional[nn.Module]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """run layer with all2all."""
    if residual is None:
        residual = hidden_states

    # We move the input_layernorm of i+1 layer to the end of i layer.
    # But for the first layer, we need to do input_layernorm first.
    if is_first_layer:
        hidden_states = input_norm(hidden_states)

    # Self Attention
    hidden_states = self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # add residual here for the first layer
    if is_first_layer and get_tensor_model_parallel_rank() == 0:
        hidden_states = hidden_states + residual

    hidden_states = tensor_model_parallel_reduce_scatter(
        hidden_states, dim=0)

    # move norm between rs and ag
    if is_first_layer:
        residual = hidden_states
        hidden_states = post_norm(hidden_states)
    else:
        # add residual in norm for other layers
        hidden_states, residual = post_norm(hidden_states, residual)

    hidden_states = mlp.forward_all2all(hidden_states, **mlp_kwargs[-1])

    if is_last_layer:
        hidden_states = hidden_states + residual
        residual = None
    else:
        # To reduce layernorm computation, we move the layernorm of i+1 layer to
        # the end of i layer. Besides, we fuse residual addition into layernorm.
        assert next_input_layernorm is not None
        hidden_states, residual = next_input_layernorm(hidden_states, residual)

    hidden_states = tensor_model_parallel_all_gather_dp(
        group_num_tokens=dp_params.moe_token_split_list_reduce_scatter,
        rank=get_tensor_model_parallel_rank(),
        hidden_states=hidden_states,
        group=get_tp_group(),
    )

    return hidden_states, residual

def _dp_forward_layer_common(
    input_norm: nn.Module,
    self_attn: nn.Module,
    post_norm: nn.Module,
    mlp: nn.Module,
    mlp_kwargs: List[Dict[str, Any]],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    dp_params: DataParallelRuntimeParams,
    hidden_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """run layer with common."""
    if residual is None:
        residual = hidden_states

    hidden_states = input_norm(hidden_states)
    hidden_states = self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # add residual here
    if get_tensor_model_parallel_rank() == 0:
        hidden_states = hidden_states + residual

    hidden_states = process_post_attention_communication(
        hidden_states, dp_params, hidden_size, dtype, positions.device, None
    )

    residual = hidden_states[dp_params.token_num_offset:
                             dp_params.token_num_offset + dp_params.token_num]

    hidden_states = post_norm(hidden_states)

    hidden_states = mlp(hidden_states, **mlp_kwargs[0])

    hidden_states = tensor_model_parallel_all_reduce(
        hidden_states, tp_group=get_tp_world_group())

    # add residual here
    hidden_states = hidden_states[dp_params.token_num_offset:
                                  dp_params.token_num_offset+dp_params.token_num]
    hidden_states = hidden_states + residual
    residual = hidden_states

    return hidden_states, residual
