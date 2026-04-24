# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from typing import List, Optional, Any
import copy

import torch
import torch.nn.functional as F
from vllm.config.vllm import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.flash_attn import (CommonAttentionMetadata,
                                                   FlashAttentionMetadata)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, logger

from vllm.distributed.communication_op import tensor_model_parallel_all_gather_into_list
from vllm.distributed import (
    get_logits_tp_world_size,
    get_logits_tp_group,
    get_tensor_model_parallel_world_size,
)
from vllm_mlu.v1.attention.backends.flash_attn import pad_attn_metadata
from vllm_mlu.v1.attention.backends.mla.flashmla import FlashMLAMetadataBuilder
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata, COMMON_METADATA_STR)
from vllm_mlu._mlu_utils import *
from vllm_mlu.v1.attention.backends.utils import MLUInferMode

from vllm_mlu.mlu_forward_context import MLUDPMetadata
from vllm_mlu.v1.spec_decode.eagle import MluEagleProposer
from vllm_mlu.model_executor.models.dp_utils import (
    enable_data_parallel,
    DataParallelRuntimeParams
)

class DPMluEagleProposer(MluEagleProposer):

    def get_logits_batch_sizes(self, batch_size: int) -> Optional[List[int]]:
        tp_world_size, logits_batch_sizes = get_logits_tp_world_size(), None
        if tp_world_size != get_tensor_model_parallel_world_size():
            tp_tensor = torch.tensor([batch_size]).to(self.runner.device)
            outputs = tensor_model_parallel_all_gather_into_list(tp_tensor, get_logits_tp_group())
            # Convert device tensor to host list
            outputs = torch.cat(outputs).tolist()
            logits_batch_sizes = [outputs[i] for i in range(tp_world_size)]
        return logits_batch_sizes

    def propose_ds_execute_dummy_batch(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        dp_params: DataParallelRuntimeParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # num_scheduled_tokens
        num_tokens = target_token_ids.shape[0]
        input_ids = self.input_ids[:num_tokens]
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]

        # always skip attn compute
        attn_metadata: Optional[dict[str, Any]] = None

        # Get graph capture related infomation for deepseek model.

        with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=target_positions,
                hidden_states=target_hidden_states,
                intermediate_tensors=None,
                inputs_embeds=None,
                dp_params=dp_params,
            )
        if dp_params is not None:
            dp_params.logits_batch_split_list = self.get_logits_batch_sizes(num_tokens)
        _ = self.model.compute_logits(hidden_states, dp_params=dp_params)

        if self.num_speculative_tokens == 1:
            return
        '''
        =============================
        Modify by vllm_mlu
        @brief: support k > 1, need run draft model k-1 times
        =============================
        '''
        # support k > 1
        for _ in range(self.num_speculative_tokens - 1):
            new_dp_params = self.runner._get_data_parallel_metadata(
                num_tokens, num_tokens, True, [1] * num_tokens)
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=target_positions,
                    hidden_states=target_hidden_states,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                    dp_params=new_dp_params,
                )
                _ = self.model.compute_logits(hidden_states, dp_params=new_dp_params)
        '''
        =============================
        End of MLU Hijack
        =============================
        '''

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
        whole_block_table: torch.Tensor,
        main_model_dp_params: Optional[DataParallelRuntimeParams] = None,
        time_markers: List =[],
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

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Use full graph with draft model and pad batch_size for dp
        '''
        dp_group_max_token_num = max(main_model_dp_params.token_split_list)
        if dp_group_max_token_num <= self.vllm_config.compilation_config.max_cudagraph_capture_size:
            batch_descriptor_num_tokens = self.vllm_config.pad_for_cudagraph(dp_group_max_token_num)
            captured_already = True
        else:
            batch_descriptor_num_tokens = num_tokens
            captured_already = False

        # Determine if we can use full graph
        decode_only = all(not prefill for prefill in main_model_dp_params.dp_is_prefill)
        # FIXME(wangchao2): disable mtp graph for ds3.2 with dp fow now(core dump)
        is_dsv32 = self.vllm_config.model_config.hf_config.model_type == "deepseek_v32"
        use_full_graph = (self.use_cuda_graph
                          and decode_only and captured_already and not is_dsv32)
        if (self.use_cuda_graph and decode_only and not use_full_graph and not is_dsv32):
            logger.warning_once(
                f"Select MLU-V1 Full-MLUGraph mode with drafter, however running in " +
                f"eager mode: decode_only={decode_only}, captured_already={captured_already}, " +
                f"num_tokens={num_tokens}."
            )

        cudagraph_runtime_mode = CUDAGraphMode.FULL if use_full_graph else CUDAGraphMode.NONE
        batch_descriptor = BatchDescriptor(
            num_tokens=batch_descriptor_num_tokens,
            uniform_decode=True,
        )

        # dp pad batch_size
        if use_full_graph:
            K = self.num_speculative_tokens
            num_input_tokens = batch_descriptor_num_tokens
            padded_batch_size = num_input_tokens // (K + 1)
        else:
            padded_batch_size = batch_size
            num_input_tokens = num_tokens

        # change attn metadata num_actual_tokens
        attn_metadata.num_actual_tokens = num_input_tokens

        common_attn_metadata_copy = None
        # copy common_attn_metadata when k>1 for draft model,
        # because dp pad batch_size will change common_attn_metadata
        if self.num_speculative_tokens > 1:
            common_attn_metadata_copy = copy.deepcopy(common_attn_metadata)
        # pad attn metadata
        if use_full_graph and enable_data_parallel() and num_input_tokens != num_tokens:
            assert self.runner is not None
            # Update attention metadata.
            pad_attn_metadata(
                attn_metadata,
                common_attn_metadata,
                whole_block_table,
                self.runner,
                num_tokens,
                num_input_tokens,
                batch_size,
                padded_batch_size,
            )

            # Update input ids, pad with 0 if necessary.
            token_pad_size = num_input_tokens - num_tokens
            assert token_pad_size >= 0
            # Update target hidden states, pad with zeros if necessary.
            if token_pad_size > 0:
                target_hidden_states = F.pad(
                    target_hidden_states,
                    (0, 0, 0, token_pad_size),
                    value=0.0
                )

            # Update positions, pad with zeros if necessary.
            if token_pad_size > 0:
                target_positions = F.pad(
                    target_positions,
                    (0, token_pad_size),
                    value=0
                )

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        per_layer_attn_metadata[COMMON_METADATA_STR] = common_attn_metadata

        # copy inputs to buffer for cudagraph
        self.positions[:num_input_tokens] = target_positions
        self.hidden_states[:num_input_tokens] = target_hidden_states

        kwargs = {} if main_model_dp_params is None else {"dp_params": main_model_dp_params}
        if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            start = torch.mlu.Event(enable_timing=True)
            start.record()
        with set_forward_context(per_layer_attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens,
                                 batch_descriptor=batch_descriptor if use_full_graph else None,
                                 cudagraph_runtime_mode=cudagraph_runtime_mode):
            if use_full_graph:
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    intermediate_tensors=None,
                    inputs_embeds=None,
                    is_running_drafter=True,
                    **kwargs,
                )
            else:
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    **kwargs,
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
        if main_model_dp_params is not None:
            # Ensure main_model_dp_params has required attribute before assignment
            if hasattr(main_model_dp_params, 'logits_batch_split_list'):
                main_model_dp_params.logits_batch_split_list = self.get_logits_batch_sizes(batch_size)
            else:
                raise AttributeError("dp_params must have 'logits_batch_split_list' attribute")

        sample_hidden_states = last_hidden_states[hidden_states_indices]
        logits = self.model.compute_logits(sample_hidden_states, dp_params=main_model_dp_params)

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
        hidden_states = last_hidden_states[hidden_states_indices]
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
            common_attn_metadata_k = common_attn_metadata_copy
            common_attn_metadata_k.num_actual_tokens = batch_size
            common_attn_metadata_k.num_input_tokens = batch_size
            common_attn_metadata_k.max_query_len = 1
            common_attn_metadata_k.query_start_loc = self.arange[: batch_size + 1]
            common_attn_metadata_k.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()
        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        for token_index in range(self.num_speculative_tokens - 1):
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: get dp_params for draft model
            '''
            # dp_params for draft model
            if main_model_dp_params is not None:
                dp_params = self.runner._get_data_parallel_metadata(
                    input_batch_size,
                    input_batch_size,
                    common_attn_metadata.is_decode_only,
                    [1] * input_batch_size
                )
            kwargs = {} if main_model_dp_params is None else {"dp_params": dp_params}
            '''
            =============================
            End of MLU Hijack
            =============================
            '''
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
                                     num_tokens=input_batch_size):
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                    **kwargs,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            if VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
                end = torch.mlu.Event(enable_timing=True)
                end.record()
                time_markers.append([start, end])
            '''
            =============================
            End of MLU Hijack
            =============================
            '''
            hidden_states = last_hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                                dp_params=dp_params)

            # TODO(wenlong): get more than one token for tree attention
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
