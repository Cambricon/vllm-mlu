# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from collections import OrderedDict, deque

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.attention.backends.gdn_attn import (GDNAttentionMetadataBuilder,
                                                 GDNAttentionMetadata,
                                                 )
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              compute_causal_conv1d_metadata,
                                              split_decodes_and_prefills,)


class DeviceAwareLocalIdMapper:
    def __init__(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.global_to_local: OrderedDict[int, int] = OrderedDict()
        self.local_to_global = {}
        self.available_local_ids = deque(range(batch_size))

    def batch_get_local_ids(self, global_id_tensor: torch.Tensor) -> torch.Tensor:
        original_device = global_id_tensor.device
        original_shape = global_id_tensor.shape

        flat_global_cpu = global_id_tensor.cpu().numpy().ravel()
        num_elements = flat_global_cpu.size
        local_ids_cpu = torch.empty(num_elements, dtype=global_id_tensor.dtype)

        g2l = self.global_to_local
        unique_miss_set = set()

        # Pass 1: handle hits and collect unique misses
        for i, gid in enumerate(flat_global_cpu):
            if gid in g2l:
                local_id = g2l[gid]
                local_ids_cpu[i] = local_id
                g2l.move_to_end(gid)
            else:
                local_ids_cpu[i] = -1
                unique_miss_set.add(gid)

        # Pass 2: assign local IDs to unique new global IDs
        new_mappings = {}
        available = self.available_local_ids
        local_to_global = self.local_to_global

        for gid in unique_miss_set:
            if len(g2l) >= self.batch_size:
                old_gid, old_local = g2l.popitem(last=False)
                available.append(old_local)
                local_to_global.pop(old_local, None)
            new_local = available.popleft()
            g2l[gid] = new_local
            local_to_global[new_local] = gid
            new_mappings[gid] = new_local

        # Pass 3: fill in all miss positions
        for i, gid in enumerate(flat_global_cpu):
            if local_ids_cpu[i].item() == -1:
                local_ids_cpu[i] = new_mappings[gid]

        return local_ids_cpu.to(original_device).view(original_shape)

    def reset(self):
        self.global_to_local.clear()
        self.local_to_global.clear()
        self.available_local_ids = deque(range(self.batch_size))

def vllm__v1__attention__bachends__GDNAttentionMetadataBuilder____init__(
    self,
    kv_cache_spec: AttentionSpec,
    layer_names: list[str],
    vllm_config: VllmConfig,
    device: torch.device,
):
    assert isinstance(kv_cache_spec, MambaSpec)
    self.vllm_config = vllm_config
    self.compilation_config = vllm_config.compilation_config
    self.speculative_config = vllm_config.speculative_config
    self.kv_cache_spec = kv_cache_spec
    if self.speculative_config:
        self.num_spec = self.speculative_config.num_speculative_tokens
    else:
        self.num_spec = 0
    self.use_spec_decode = self.num_spec > 0
    self._init_reorder_batch_threshold(1, self.use_spec_decode)

    self.use_full_cuda_graph = (
        self.compilation_config.cudagraph_mode.has_full_cudagraphs()
    )
    self.decode_cudagraph_max_bs = min(
        self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
        self.compilation_config.max_cudagraph_capture_size,
    )

    self.spec_state_indices_tensor = torch.empty(
        (self.decode_cudagraph_max_bs, self.num_spec + 1),
        dtype=torch.int32,
        device=device,
    )
    self.non_spec_state_indices_tensor = torch.empty(
        (self.decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    self.spec_sequence_masks = torch.empty(
        (self.decode_cudagraph_max_bs,),
        dtype=torch.bool,
        device=device,
    )
    self.spec_token_indx = torch.empty(
        (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
        dtype=torch.int32,
        device=device,
    )
    self.non_spec_token_indx = torch.empty(
        (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
        dtype=torch.int32,
        device=device,
    )
    self.spec_query_start_loc = torch.empty(
        (self.decode_cudagraph_max_bs + 1,),
        dtype=torch.int32,
        device=device,
    )
    self.non_spec_query_start_loc = torch.empty(
        (self.decode_cudagraph_max_bs + 1,),
        dtype=torch.int32,
        device=device,
    )
    self.num_accepted_tokens = torch.empty(
        (self.decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: support qwen3-next
    '''
    self.mapper = DeviceAwareLocalIdMapper(self.vllm_config.mlu_config.mamba_support_max_batch_size)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__v1__attention__bachends__GDNAttentionMetadataBuilder__build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
) -> GDNAttentionMetadata:
    m = common_attn_metadata

    query_start_loc = m.query_start_loc
    context_lens = m.num_computed_tokens_cpu
    context_lens_tensor = context_lens.to(query_start_loc.device)
    nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

    if (
        not self.use_spec_decode
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
        .sum()
        .item()
        == 0
    ):
        spec_sequence_masks = None
        num_spec_decodes = 0
    else:
        spec_sequence_masks = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = spec_sequence_masks.sum().item()
        if num_spec_decodes == 0:
            spec_sequence_masks = None
        else:
            spec_sequence_masks = spec_sequence_masks.to(
                query_start_loc.device, non_blocking=True
            )

    if spec_sequence_masks is None:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
        )
        num_spec_decode_tokens = 0
        spec_token_indx = None
        non_spec_token_indx = None
        spec_state_indices_tensor = None
        non_spec_state_indices_tensor = m.block_table_tensor[:, 0]
        spec_query_start_loc = None
        non_spec_query_start_loc = query_start_loc
        num_accepted_tokens = None
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        non_spec_query_lens = query_lens[~spec_sequence_masks]
        num_decodes = (non_spec_query_lens == 1).sum().item()
        num_prefills = non_spec_query_lens.size(0) - num_decodes
        num_decode_tokens = num_decodes
        num_prefill_tokens = non_spec_query_lens.sum().item() - num_decode_tokens
        num_spec_decode_tokens = (
            query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
        )

        if num_prefills == 0 and num_decodes == 0:
            spec_token_size = min(
                num_spec_decodes * (self.num_spec + 1),
                query_start_loc[-1].item(),
            )
            spec_token_indx = torch.arange(
                spec_token_size,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_state_indices_tensor = m.block_table_tensor[:, : self.num_spec + 1]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc
            non_spec_query_start_loc = None
        else:
            spec_token_masks = torch.repeat_interleave(
                spec_sequence_masks, query_lens
            )
            index = torch.argsort(spec_token_masks)
            num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
            non_spec_token_indx = index[:num_non_spec_tokens]
            spec_token_indx = index[num_non_spec_tokens:]

            spec_state_indices_tensor = m.block_table_tensor[
                spec_sequence_masks, : self.num_spec + 1
            ]
            non_spec_state_indices_tensor = m.block_table_tensor[
                ~spec_sequence_masks, 0
            ]

            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
            )
            non_spec_query_start_loc = torch.zeros(
                query_lens.size(0) - num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[~spec_sequence_masks],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )

        assert num_accepted_tokens is not None
        num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

    if num_prefills > 0:
        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks is not None:
            has_initial_state = has_initial_state[~spec_sequence_masks]
        nums_dict, batch_ptr, token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(non_spec_query_start_loc)
        )
    else:
        has_initial_state = None
    num_actual_tokens = (
        num_prefill_tokens + num_decode_tokens + num_spec_decode_tokens
    )

    # prepare tensors for cudagraph
    #
    # With speculative decoding, the xgrammar backend may rollback tokens
    # and causing some sequences has less draft tokens than self.num_spec.
    #
    # In above cases, the max possible batch size for n tokens, can be
    # min(n, cudagraph_max_bs).
    if (
        self.use_full_cuda_graph
        and num_prefills == 0
        and num_decodes == 0
        and num_spec_decodes <= self.decode_cudagraph_max_bs
        and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
    ):
        num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
        batch_size = min(self.decode_cudagraph_max_bs, num_actual_tokens)

        self.spec_state_indices_tensor[:num_spec_decodes].copy_(
            spec_state_indices_tensor, non_blocking=True
        )
        spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

        self.spec_sequence_masks[:num_spec_decodes].copy_(
            spec_sequence_masks, non_blocking=True
        )
        spec_sequence_masks = self.spec_sequence_masks[:batch_size]
        spec_sequence_masks[num_spec_decodes:].fill_(False)

        assert non_spec_token_indx is not None and spec_token_indx is not None
        self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
            non_spec_token_indx, non_blocking=True
        )
        non_spec_token_indx = self.non_spec_token_indx[
            : non_spec_token_indx.size(0)
        ]

        self.spec_token_indx[: spec_token_indx.size(0)].copy_(
            spec_token_indx, non_blocking=True
        )
        spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

        self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
            spec_query_start_loc, non_blocking=True
        )
        spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
        spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
        spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

        self.num_accepted_tokens[:num_spec_decodes].copy_(
            num_accepted_tokens, non_blocking=True
        )
        num_accepted_tokens = self.num_accepted_tokens[:batch_size]
        num_accepted_tokens[num_spec_decodes:].fill_(1)

    if (
        self.use_full_cuda_graph
        and num_prefills == 0
        and num_spec_decodes == 0
        and num_decodes <= self.decode_cudagraph_max_bs
    ):
        num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
        batch_size = num_actual_tokens

        self.non_spec_state_indices_tensor[:num_decodes].copy_(
            non_spec_state_indices_tensor, non_blocking=True
        )
        non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
            :batch_size
        ]
        non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

        self.non_spec_query_start_loc[: num_decodes + 1].copy_(
            non_spec_query_start_loc, non_blocking=True
        )
        non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
        non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
        non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: support qwen3-next
    '''
    non_spec_state_indices_tensor = self.mapper.batch_get_local_ids(non_spec_state_indices_tensor)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    attn_metadata = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=has_initial_state,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=non_spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    return attn_metadata

MluHijackObject.apply_hijack(GDNAttentionMetadataBuilder,
                             GDNAttentionMetadataBuilder.__init__,
                             vllm__v1__attention__bachends__GDNAttentionMetadataBuilder____init__)

MluHijackObject.apply_hijack(GDNAttentionMetadataBuilder,
                             GDNAttentionMetadataBuilder.build,
                             vllm__v1__attention__bachends__GDNAttentionMetadataBuilder__build)
