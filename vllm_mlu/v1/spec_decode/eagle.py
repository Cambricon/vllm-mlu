# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
import ast
from dataclasses import replace
from importlib.util import find_spec
import numpy as np
import torch
import torch.nn as nn
from typing import Any, List, Optional
from vllm.config.vllm import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

from vllm.v1.spec_decode.eagle import EagleProposer, PADDING_SLOT_ID, logger
from vllm.v1.utils import CpuGpuBuffer

from vllm_mlu.compilation.mlu_graph import MLUGraphWrapper
from vllm_mlu.v1.attention.backends.mla.flashmla import FlashMLAMetadataBuilder
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata, get_common_metadata_from_attn_metadata,
    get_common_metadata, COMMON_METADATA_STR)
from vllm_mlu.model_executor.models.sp_utils import set_sp_forward_context
from vllm_mlu._mlu_utils import *
from vllm_mlu.v1.attention.backends.utils import MLUInferMode


class MluEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.draft_indexer_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []
        self.indexer_layer_names: list[str] = []

        self.use_cuda_graph = True

        compilation_config = self.vllm_config.compilation_config
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            cudagraph_mode = compilation_config.cudagraph_mode
            if cudagraph_mode != CUDAGraphMode.NONE and not cudagraph_mode.has_mode(
                CUDAGraphMode.PIECEWISE
            ):
                logger.warning(
                    "Currently the eagle proposer only supports cudagraph_mode "
                    "PIECEWISE, if you want the drafter to use cuda graphs, "
                    "please set compilation_config.cudagraph_mode to PIECEWISE "
                    "or FULL_AND_PIECEWISE"
                )
            self.use_cuda_graph = (
                not self.speculative_config.enforce_eager
            )

        self.cudagraph_batch_sizes = (
            (sorted(self.vllm_config.compilation_config.cudagraph_capture_sizes))
            if self.use_cuda_graph
            else []
        )

        self.use_cuda_graph = self.use_cuda_graph and bool(self.cudagraph_batch_sizes)
        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.uses_mrope = self.vllm_config.model_config.uses_mrope
        if self.uses_mrope:
            # M-RoPE need (3, max_num_tokens)
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens), dtype=torch.int64, device=device
            )
        else:
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: tmo positions need to be int32
            '''
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device=device)
            '''
            =============================
            End of MLU Hijack
            =============================
            '''
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # Determine allowed attention backends once during initialization.
        from vllm.attention.backends.registry import AttentionBackendEnum

        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            rocm_types = [TritonAttentionMetadata, FlashAttentionMetadata]
            # ROCM_AITER_FA is an optional backend
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)
            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1,
            len(self.tree_choices) + 1,
            device=device,
            dtype=torch.int32,
        ).repeat(max_batch_size, 1)
        self.arange = torch.arange(max_num_slots_for_arange,
                                    device=device,
                                    dtype=torch.int32)
        '''
        =============================
        Modify by vllm_mlu
        @brief: Now kv_cache is stored in groups, need to get the corresponding group_id
        FIXME: need to be removed after update https://github.com/vllm-project/vllm/pull/20022
        =============================
        '''
        self.kv_cache_group_id = None
        '''
        =============================
        End of MLU Hijack
        =============================
        '''
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # Determine allowed attention backends once during initialization.
        from vllm.attention.backends.registry import AttentionBackendEnum

        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            rocm_types = [TritonAttentionMetadata, FlashAttentionMetadata]
            # ROCM_AITER_FA is an optional backend
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)
            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1,
            len(self.tree_choices) + 1,
            device=device,
            dtype=torch.int32,
        ).repeat(max_batch_size, 1)


    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: MLUCommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
        # [num_tokens]
        token_indices: torch.Tensor,
        time_markers: List = [],
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids
        hidden_states_indices = last_token_indices

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        # FIXME: need to consider multiple kv_cache_groups
        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0,
        )

        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        per_layer_attn_metadata[COMMON_METADATA_STR] = common_attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        cudagraph_runtime_mode = CUDAGraphMode.NONE
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            num_input_tokens = num_tokens

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            start = torch.mlu.Event(enable_timing=True)
            start.record()

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Use full graph with draft model
        @brief: Add set_sp_forward_context for sequence parallel.
        '''
        use_full_graph = False
        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            uniform_decode=True,
        )

        if batch_descriptor in self.model.concrete_cudagraph_entries:
            cudagraph_runtime_mode = CUDAGraphMode.FULL
            use_full_graph = True

        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        with set_forward_context(per_layer_attn_metadata, self.vllm_config,
                num_tokens=num_tokens, cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor if use_full_graph else None), \
            set_sp_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens,
            ):
            ret_hidden_states = self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                is_running_drafter=use_full_graph
            )
            if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
                end = torch.mlu.Event(enable_timing=True)
                end.record()
                time_markers.append([start, end])
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        '''
        =============================
        End of MLU Hijack
        =============================
        '''
        sample_hidden_states = last_hidden_states[hidden_states_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]

        '''
        =============================
        Modify by vllm_mlu
        =============================
        '''
        hidden_states = sample_hidden_states
        '''
        =============================
        End of MLU Hijack
        =============================
        '''

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        input_batch_size = batch_size

        if common_attn_metadata.infer_mode != MLUInferMode.DECODE_ONLY:
            seq_lens_cpu = torch.ones(input_batch_size, dtype=torch.int32,)
            cu_num_tokens = torch.cumsum(seq_lens_cpu, dim=0)
            query_start_loc_cpu = torch.empty(input_batch_size + 1, dtype=torch.int32)
            query_start_loc_cpu[0] = 0
            query_start_loc_cpu[1:] = cu_num_tokens
            seq_start_loc_cpu = self.arange[:input_batch_size + 1]
            common_attn_metadata_k = MLUCommonAttentionMetadata.build(
                query_start_loc=query_start_loc_cpu.to(self.device, non_blocking=True),
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu.to(self.device, non_blocking=True),
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
                num_reqs=common_attn_metadata.num_reqs,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                slot_mapping=common_attn_metadata.slot_mapping,
                seq_start_loc=seq_start_loc_cpu.to(self.device, non_blocking=True),
                is_start_loc_match=False, # not prefill
                max_query_len=1,
                num_actual_tokens=input_batch_size,
                num_input_tokens=input_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                has_prefill_reqs=common_attn_metadata.infer_mode == MLUInferMode.CHUNKED,
            )
        else:
            common_attn_metadata_k = common_attn_metadata
            common_attn_metadata_k.num_actual_tokens = batch_size
            common_attn_metadata_k.num_input_tokens = batch_size
            common_attn_metadata_k.max_query_len = 1
            common_attn_metadata_k.query_start_loc = self.arange[: batch_size + 1]
            common_attn_metadata_k.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata_k.seq_lens += 1
            # This is an out-of-place operation to avoid modifying the original tensor.
            common_attn_metadata_k.seq_lens_cpu = common_attn_metadata_k.seq_lens_cpu + 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.

            common_attn_metadata_k.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            common_attn_metadata_k.num_computed_tokens_cpu = (
                common_attn_metadata_k.seq_lens_cpu - 1
            )

            # Compute the slot mapping.
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // self.block_size
            else:
                block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata_k.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata_k.slot_mapping = (
                    block_ids * self.block_size + clamped_positions[0] % self.block_size
                )
            else:
                common_attn_metadata_k.slot_mapping = (
                    block_ids * self.block_size + clamped_positions % self.block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata_k.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )

            # Rebuild attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=common_attn_metadata_k, draft_index=token_index + 1
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
            per_layer_attn_metadata[COMMON_METADATA_STR] = common_attn_metadata_k

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: record latency
            @brief: add set_sp_forward_context for sequence parallel.
            '''
            if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
                start = torch.mlu.Event(enable_timing=True)
                start.record()
            '''
            =============================
            End of MLU Hijack
            =============================
            '''
            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                        self.vllm_config,
                                        num_tokens=input_batch_size
            ), set_sp_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                input_batch_size,
            ):
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                )
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: adapt to different methods
                '''
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
                '''
                =============================
                End of MLU Hijack
                =============================
                '''
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: record latency
            '''
            if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
                end = torch.mlu.Event(enable_timing=True)
                end.record()
                time_markers.append([start, end])
            '''
            =============================
            End of MLU Hijack
            =============================
            '''
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: MLUCommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor
    ) -> tuple[MLUCommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #         [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #         [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #         [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                  q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                  q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu \
            - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = (query_start_loc_cpu[1:] -
                                 query_start_loc_cpu[:-1])
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                                  new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = self.token_arange_np[:total_num_tokens] \
            - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                   // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,         // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2]  // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True)

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add seq_start_loc compute, use MLUCommonAttentionMetadata
        '''
        new_seq_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_seq_start_loc_np = new_seq_start_loc_cpu.numpy()
        np.cumsum(new_seq_lens_cpu.numpy(), out=new_seq_start_loc_np[1:])

        spec_common_attn_metadata = MLUCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device,
                                                       non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            seq_start_loc=new_seq_start_loc_cpu.to(device, non_blocking=True),
            num_input_tokens=total_num_tokens,
            num_prefill_query_tokens=total_num_tokens,
            num_prefill_kv_tokens=total_num_tokens,
            infer_mode=common_attn_metadata.infer_mode,
        )
        '''
        =============================
        End of MLU Hijack
        =============================
        '''

        return spec_common_attn_metadata, token_indices

    def load_model(
        self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys())

        from vllm.compilation.backends import set_model_tag
        with set_model_tag("eagle_head"):
            self.model = get_model(vllm_config=self.vllm_config,
                                    model_config=draft_model_config)
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: use graph wrapper for draft model
            '''
            self.model = MLUGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
            '''
            =============================
            End of MLU Hijack
            =============================
            '''

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

        if supports_multimodal(target_model):
            # handle multimodality
            self.model.config.image_token_index = (
                target_model.config.image_token_index)
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: only eagle and eagle3 need to share embed_tokens with the target model
        '''
        if self.method in ["eagle", "eagle3"] or self.vllm_config.model_config.hf_config.model_type == "glm4_moe":
            # share embed_tokens with the target model if needed
            if get_pp_group().world_size == 1 \
                and self.model.model.embed_tokens.weight.shape \
                    == target_language_model.model.embed_tokens.weight.shape:
                logger.info(
                    "Assuming the EAGLE head shares the same vocab embedding" \
                    " with the target model."
                )
                del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_language_model.model.embed_tokens
            else:
                logger.info(
                    "The EAGLE head's vocab embedding will be loaded separately" \
                    " from the target model."
                )
        '''
        =============================
        End of MLU Hijack
        =============================
        '''
        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method not in ["eagle3", "longcat_flash_mtp"] and \
                hasattr(target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head

            target_lm_head = target_model.lm_head
            if target_lm_head is None:
                logger.warning("Target model lm_head is None")
                return
            if self.vllm_config.model_config.hf_config.model_type == "glm4_moe":
                self._process_moe_mtp_layers(target_lm_head)

    def _process_moe_mtp_layers(self, target_lm_head):
        # For GLM4 MoE MTP models, share weights with all MTP layer shared_head.head
        # instead of replacing the module (to preserve DPParallelLMHead functionality)
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return
        for layer_name, layer in self.model.model.layers.items():
            if not (hasattr(layer, "shared_head") and hasattr(layer.shared_head, "head")):
                continue
            if not (hasattr(target_lm_head, "weight") and hasattr(layer.shared_head.head, "weight")):
                continue
            if layer.shared_head.head.weight.shape != target_lm_head.weight.shape:
                logger.debug(
                    f"Skipping weight sharing for layer {layer_name}: "
                    f"shape mismatch (mtp: {layer.shared_head.head.weight.shape}, "
                    f"target: {target_lm_head.weight.shape})"
                )
                continue
            # Safe replacement
            del layer.shared_head.head
            layer.shared_head.head = target_lm_head
            logger.info(f"Replaced MTP layer {layer_name} shared_head.head with target lm_head")

    @torch.inference_mode()
    def dummy_run(
        self,
        attn_metadata: Any,
        num_tokens: int,
        use_cudagraphs=True,
    ) -> None:
        # Determine if CUDA graphs should be used for this run.
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph
        if cudagraphs_enabled and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @breif: add set_sp_forward_context for sequence parallel.
        @brief: capture drafter model
        '''
        cudagraph_runtime_mode = (CUDAGraphMode.FULL if cudagraphs_enabled
            else CUDAGraphMode.NONE)

        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            uniform_decode=True,
        )

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            batch_descriptor=batch_descriptor,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        ), set_sp_forward_context(None, self.vllm_config, num_tokens):
            if self.supports_mm_inputs:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self._get_positions(num_tokens),
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
                is_running_drafter=True
            )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def validate_same_kv_cache_group(
        self,
        kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: get kv_cache_group_id and filter kv_cache_groups
        '''
        eagle_cache_groups = set(kv_cache_groups[layer_name]
                            for layer_name in self.attn_layer_names
                            if layer_name in kv_cache_groups)
        assert len(eagle_cache_groups) == 1, (
            "All eagle layers should belong to the same kv cache group")
        self.kv_cache_group_id = next(iter(eagle_cache_groups))
        '''
        =============================
        End of MLU Hijack
        =============================
        '''

    def prepare_inputs_padded(
        self,
        common_attn_metadata: MLUCommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[MLUCommonAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_draft_tokens_gpu = torch.cat(
            [
                spec_decode_metadata.cu_num_draft_tokens[0:1],
                spec_decode_metadata.cu_num_draft_tokens[1:]
                - spec_decode_metadata.cu_num_draft_tokens[:-1],
            ]
        )

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add seq_start_loc compute, use MLUCommonAttentionMetadata
        '''
        new_seq_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_seq_start_loc_np = new_seq_start_loc_cpu.numpy()
        np.cumsum(common_attn_metadata.seq_lens.cpu().numpy(), out=new_seq_start_loc_np[1:])

        spec_common_attn_metadata = MLUCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            seq_start_loc=new_seq_start_loc_cpu.to(self.device, non_blocking=True),
            num_input_tokens=total_num_tokens,
            num_prefill_query_tokens=total_num_tokens,
            num_prefill_kv_tokens=total_num_tokens,
            infer_mode=common_attn_metadata.infer_mode,
        )
        '''
        =============================
        End of MLU Hijack
        =============================
        '''

        token_indices_to_sample = (
            common_attn_metadata.query_start_loc[1:] - 1 - num_rejected_tokens_gpu
        )
        return spec_common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu


    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for EAGLE layers.

        Returns:
            The metadata builders for EAGLE layers.

        Raises:
            AssertionError: If no metadata builders are found for EAGLE layers.
        """
        builder = None
        chosen_layer = self.attn_layer_names[0]

        """
        =============================
        Modify by vllm_mlu
        =============================
        @brief: replace attn metadata name to prefill_attn name
        """
        if self.draft_model_config.is_deepseek_mla and chosen_layer.endswith("self_attn.attn"):
            chosen_layer = chosen_layer.replace(
                "self_attn.attn", "self_attn.mla_attn")
        """
        =================
        End of MLU Hijack
        =================
        """
        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            "Failed to find attention metadata builder for EAGLE layers."
        )
        return builder