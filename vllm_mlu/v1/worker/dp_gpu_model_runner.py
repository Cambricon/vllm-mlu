# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from copy import copy
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple, Any

import torch
import numpy as np
import cnpx

from vllm.distributed.parallel_state import (
    get_tp_group, get_pp_group)
from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group
from vllm.distributed import (
    divide, get_moe_expert_parallel_world_size
)
from vllm.config import VllmConfig, CUDAGraphMode
from vllm.forward_context import set_forward_context, BatchDescriptor
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.gpu_model_runner import ExecuteModelState
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput, GrammarOutput

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu.v1.attention.backends.flash_attn import pad_attn_metadata
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata, unpad_common_attn_metadata,
    get_common_metadata_from_attn_metadata, MLUInferMode)
from vllm_mlu.v1.worker.gpu_model_runner import (
    MLUModelRunner, AsyncMLUModelRunnerOutput, apply_grammar_bitmask)
from vllm_mlu.mlu_forward_context import MLUDPMetadata
from vllm_mlu.model_executor.models.dp_utils import (
    enable_emb_logits_custom_parallel,
    get_runtime_infos_per_dp_group,
    get_deepseek_layer_split_list,
)

from vllm_mlu.model_executor.models.dp_utils import (
    DataParallelRuntimeParams
)
from vllm_mlu.model_executor.layers.sparse_moe_mlp import SparseMoeMlp
from vllm_mlu.distributed.parallel_state import (
    init_cnclep, get_cnclep
)

from vllm_mlu._mlu_utils import *
import vllm_mlu._mlu_utils as mlu_envs

logger = init_logger(__name__)


class DPMLUModelRunner(MLUModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        vllm_config.mlu_config.enable_custom_data_parallel_opt = True
        super().__init__(vllm_config, device)
        self.use_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and not self.model_config.enforce_eager)
        if not self.use_cuda_graph and not self.model_config.enforce_eager:
            logger.warning("Can not use cudagraph for dp mlu model runner. Dp mlu model runner can "
                           "only support cudagraph_mode with CUDAGraphMode.FULL_DECODE_ONLY.")
        self.use_all2all = self.mlu_config.decode_dispatch_combine_use_all2all
        if self.use_all2all:
            assert get_moe_expert_parallel_world_size() > 1, (
                "all2all requires that expert parallel is enabled")
            kwargs = self.make_cnclep_kwargs()
            init_cnclep(**kwargs)
            if self.model_config.is_longcat_flash:
                kwargs_bf16 = self.make_cnclep_kwargs(use_quant_dispatch=False)
                init_cnclep(**kwargs_bf16)
        self.dp_metadata = None

    def _get_data_parallel_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        is_decode_only: bool,
        query_len_per_batch: Optional[List[int]],
    ) -> "MLUDPMetadata":
        (dp_query_lens, dp_group_bs, dp_is_prefill,
         seq_len_per_batch) = get_runtime_infos_per_dp_group(
            num_tokens,
            num_reqs,
            not is_decode_only,
            query_len_per_batch,
            self.device,
            self.vllm_config,
        )
        (emb_query_lens, logits_batch_sizes,
         dense_attn_token_split_list) = get_deepseek_layer_split_list(
            dp_query_lens,
            dp_group_bs,
        )
        return MLUDPMetadata.make_oot(
            self.parallel_config.data_parallel_rank,
            self.parallel_config.data_parallel_size,
            self.parallel_config.tensor_parallel_size,
            dp_query_lens,
            dp_is_prefill,
            self.vllm_config.mlu_config.prefill_dispatch_use_RS_AG,
            seq_lens=(seq_len_per_batch if all(dp_is_prefill) else None),
            batch_sizes=dp_group_bs,
            emb_query_lens=emb_query_lens,
            logits_batch_sizes=logits_batch_sizes,
            dense_attn_token_split_list=dense_attn_token_split_list,
        )

    def _get_dp_graph_info(self,
                           K: int,
                           num_scheduled_tokens: int,
                           dp_metadata: "MLUDPMetadata"):
        """
        Check if the DeepSeek model can enter graph mode and retrieve input
        tokens and batch.

        This function also applies to other eligible MoE models with DP enabled,
        reusing the same graph mode compatibility logic.

        Returns:
            tuple: Contains three elements:
                num_input_tokens: Retrieved input token
                num_input_batchs: Retrieved input batch
                use_graph: Whether the model can use graph mode
        """
        if (self.use_cuda_graph
            and all(not prefill for prefill in dp_metadata.dp_is_prefill)
            and all(token_num <= self.cudagraph_batch_sizes[-1]
                    for token_num in dp_metadata.token_split_list)):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    max(dp_metadata.token_split_list))
            assert num_input_tokens % (K + 1) == 0, \
                f"num_input_tokens ({num_input_tokens}) must be divisible by (K + 1) = {K + 1}"
            num_input_batchs = num_input_tokens // (1 + K)
            use_graph = True
        else:
            num_input_batchs = self.input_batch.num_reqs
            num_input_tokens = num_scheduled_tokens
            use_graph = False
        return num_input_tokens, num_input_batchs, use_graph

    @torch.inference_mode()
    def moe_dp_execute_dummy_batch(
        self, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        # MUST do comm across dp group first when enable data parallel.
        # Here we set dummy run state as prefill only to prevent other dp
        # group use graph.
        dp_metadata = self._get_data_parallel_metadata(
            num_tokens, num_reqs, False, [num_tokens // num_reqs] * num_reqs
        )

        # always skip attn compute
        attn_metadata: Optional[Dict[str, Any]] = None

        input_ids = self.input_ids.gpu[:num_tokens]
        positions = self.positions.gpu[:num_tokens]
        with self.maybe_randomize_inputs(input_ids), set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=None):
            hidden_states = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                dp_params=dp_metadata,
            )

        kwargs = ({"dp_params": dp_metadata}
                  if enable_emb_logits_custom_parallel() else {})
        self.model.compute_logits(
            hidden_states[:num_tokens], **kwargs)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            target_token_ids = input_ids
            target_positions = positions
            # hidden_states no need to be sliced
            target_hidden_states = hidden_states
            self.drafter.propose_ds_execute_dummy_batch(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                dp_params=dp_metadata)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("dp_gpu_model_runner: preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if not num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: add mlu_infer_mode.
                @brief: prepare mlu dp metadata in _prepare_inputs instead of ubatch_slices
                  and num_tokens_across_dp.
                '''
                max_computed_tokens = np.max(self.input_batch.num_computed_tokens_cpu[:num_reqs])
                self.mlu_infer_mode = MLUInferMode.build(
                    max_query_len=max_num_scheduled_tokens,
                    max_computed_tokens=max_computed_tokens,
                    uniform_decode_query_len=self.uniform_decode_query_len,
                )

                num_tokens_across_dp = None
                (
                    logits_indices,
                    spec_decode_metadata,
                    ubatch_slices,
                    dp_metadata,
                ) = self._prepare_inputs(
                    scheduler_output, num_scheduled_tokens_np, max_num_scheduled_tokens
                )
                self.dp_metadata = dp_metadata
                '''
                ==================
                End of MLU Hijack
                ==================
                '''

                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and ubatch_slices is None:
                    # Pre-compute cascade attention prefix lengths
                    # NOTE: Must be AFTER _prepare_inputs uses self.input_batch state
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        scheduler_output.num_common_prefix_blocks,
                    )

                # TODO(lucas): move cudagraph dispatching here:
                #   https://github.com/vllm-project/vllm/issues/23789

                total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                attn_metadata, spec_decode_common_attn_metadata = (
                    self._build_attention_metadata(
                        total_num_scheduled_tokens=total_num_scheduled_tokens,
                        max_num_scheduled_tokens=max_num_scheduled_tokens,
                        num_reqs=num_reqs,
                        ubatch_slices=ubatch_slices,
                        logits_indices=logits_indices,
                        use_spec_decode=use_spec_decode,
                        scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
                        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                        mlu_infer_mode=self.mlu_infer_mode,
                    )
                )

                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: pad attn metadata for mlu grpah.
                @brief: pad num_input_tokens based on all dp groups and spec decode.
                @brief: add dp_params to model_kwargs.
                '''
                dp_can_use_graph = False
                if self.use_cuda_graph:
                    num_input_tokens_dp, num_padded_reqs, dp_can_use_graph = self._get_dp_graph_info(
                        self.num_spec_tokens, num_scheduled_tokens, dp_metadata)
                    if dp_can_use_graph:
                        # all layers share the same attn_metadata
                        assert len(self.kv_cache_config.kv_cache_groups) == 1
                        attn_metadata_val = next(iter(attn_metadata.values()))
                        common_metadata = get_common_metadata_from_attn_metadata(attn_metadata)
                        block_table = self.input_batch.block_table[0]
                        pad_attn_metadata(
                            attn_metadata_val, common_metadata, block_table, self,
                            num_scheduled_tokens, num_input_tokens_dp, num_reqs, num_padded_reqs)

                dp_rank = self.parallel_config.data_parallel_rank
                if ubatch_slices:
                    assert num_tokens_across_dp is not None
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                    self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
                elif num_tokens_across_dp is not None:
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                else:
                    num_input_tokens = (
                        num_input_tokens_dp if dp_can_use_graph else num_scheduled_tokens)

                (
                    input_ids,
                    inputs_embeds,
                    positions,
                    intermediate_tensors,
                    model_kwargs,
                    ec_connector_output,
                ) = self._preprocess(
                    scheduler_output, num_input_tokens, intermediate_tensors
                )

                model_kwargs["dp_params"] = dp_metadata
                '''
                ==================
                End of MLU Hijack
                ==================
                '''

            uniform_decode = (
                max_num_scheduled_tokens == self.uniform_decode_query_len
            ) and (num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=len(self.input_batch.lora_id_to_lora_request) > 0,
            )
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    batch_descriptor,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                )
            )

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: check if we can use cudagraph using dp_can_use_graph.
        '''
        if not dp_can_use_graph:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False
            
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: debug disagg cnpx.
        '''
        if mlu_envs.VLLM_DISAGG_CNPX_EXECUTE:
            self.execute_cnpx_mark = cnpx.rangeStart("DP_" + str(self.parallel_config.data_parallel_rank) + "_TP_" \
                                     + str(get_tensor_model_parallel_rank()) + "_execute_model" + \
                                     ("_no_graph" if cudagraph_runtime_mode == CUDAGraphMode.NONE else ""))
        if mlu_envs.VLLM_DISAGG_CNPX_REQUEST:
            self.request_cnpx_mark.clear()
            for req in scheduler_output.scheduled_new_reqs:
                self.request_cnpx_mark[req.req_id] = cnpx.rangeStart(req.req_id)
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self.request_cnpx_mark[req_id] = cnpx.rangeStart(req_id)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        if mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            start = torch.mlu.Event(enable_timing=True)
            start.record()

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("dp_gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("dp_gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: support embed logits custom parallel.
                '''
                sample_hidden_states = hidden_states[logits_indices]
                logits_kwargs = ({"dp_params": dp_metadata}
                                 if enable_emb_logits_custom_parallel() else {})
                logits = self.model.compute_logits(sample_hidden_states, **logits_kwargs)
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

        self.time_markers = []
        if mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            end = torch.mlu.Event(enable_timing=True)
            end.record()
            self.time_markers.append([start, end])

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            kv_connector_output,
        )
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncMLUModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(
            sampled_token_ids: torch.Tensor | list[np.ndarray],
        ) -> None:
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    whole_block_table=self.input_batch.block_table[0],
                    main_model_dp_params=self.dp_metadata,
                )
        use_padded_batch_for_eagle = (
            self.speculative_config
            and self.speculative_config.use_eagle()
            and not self.speculative_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            self.speculative_config
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len
            )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Force `input_fits_in_drafter` to be True to ensure that `self.uniform_decode_query_len` tokens are scheduled per batch during model execution.
                This is required for graph validation and to keep the batch token count consistent with `self.uniform_decode_query_len` immediately after the prefill stage.
        '''
        # input_fits_in_drafter = spec_decode_common_attn_metadata and (
        #     spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
        #     <= effective_drafter_max_model_len
        # )
        input_fits_in_drafter = True
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if use_padded_batch_for_eagle:
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        spec_decode_common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if (
            self.speculative_config
            and not use_padded_batch_for_eagle
            and input_fits_in_drafter
        ):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
            )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: supoort disagg for mlu.
        '''
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()
            get_kv_transfer_group().clear_connector_metadata()
            
        if mlu_envs.VLLM_DISAGG_CNPX_EXECUTE:
            current_stream = torch.mlu.current_stream()
            current_stream.synchronize()
            cnpx.rangeEnd(self.execute_cnpx_mark)
        if mlu_envs.VLLM_DISAGG_CNPX_REQUEST:
            current_stream = torch.mlu.current_stream()
            current_stream.synchronize()
            for req in scheduler_output.scheduled_new_reqs:
                cnpx.rangeEnd(self.request_cnpx_mark[req.req_id])
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                cnpx.rangeEnd(self.request_cnpx_mark[req_id])
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncMLUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        common_attn_metadata: MLUCommonAttentionMetadata,
        whole_block_table: torch.Tensor,
        main_model_dp_params: Optional[DataParallelRuntimeParams] = None,
    ) -> list[list[int]]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: draft model will build new FlashMLAMetadata,
                so just unpad common_attn_metadata here.
        '''
        unpad_common_attn_metadata(
            common_metadata=common_attn_metadata,
            num_reqs=self.input_batch.num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self.propose_ngram_draft_token_ids(
                sampled_token_ids)
        elif self.speculative_config.method == "medusa":
            assert isinstance(self.drafter, MedusaProposer)
            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            spec_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # TODO(woosuk): Refactor the loop.
            if self.speculative_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                # TODO(woosuk): Support M-RoPE.
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
                num_rejected_tokens_gpu = None
                token_indices = None
            else:
                if self.speculative_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                else:
                    common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count,
                        )
                    )
                target_token_ids = self.input_ids.gpu[token_indices]
                target_positions = self._get_positions(token_indices)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[token_indices]
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: add debug info for draft accepted rate
                '''
                if mlu_envs.VLLM_MTP_DEBUG:
                    batch_total_draft = sum(spec_decode_metadata.num_draft_tokens)
                    batch_total_rejected = sum(num_rejected_tokens_gpu)
                    self.total_draft_tokens += batch_total_draft
                    self.total_accepted_tokens += (
                        batch_total_draft - batch_total_rejected)
                    if batch_total_draft > 0:
                        batch_accept_rate = (
                            batch_total_draft - batch_total_rejected
                            ) / batch_total_draft
                        print(f"Batch Accept Rate: {batch_accept_rate:.4f}, "
                              f"Total Accept Rate: {self.get_accept_rate():.4f}")
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
            if self.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: keep full scheduled tokens for draft model compute
            '''
            target_token_ids = target_token_ids[:num_scheduled_tokens]
            target_positions = target_positions[:num_scheduled_tokens]
            target_hidden_states = target_hidden_states[:num_scheduled_tokens]
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

            spec_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                num_rejected_tokens=num_rejected_tokens_gpu,
                token_indices=token_indices,
                whole_block_table=whole_block_table,
                main_model_dp_params=main_model_dp_params,
                time_markers=self.time_markers,
            )
        return spec_token_ids

    def make_cnclep_kwargs(self, use_quant_dispatch: bool = True) -> dict[Any, Any]:

        K = (self.drafter.num_speculative_tokens
            if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer)
            else 0)
        seq_len = K + 1
        config = self.model_config.hf_config
        num_experts = (config.n_routed_experts if hasattr(config, "n_routed_experts")
                       else config.num_experts)
        topk = getattr(config, "num_experts_per_tok", None) or getattr(config, "moe_topk", None) 
        assert topk is not None, "failed to get topk from config"
        hidden_size = config.hidden_size
        dispatch_token_size = hidden_size * get_dtype_size(self.dtype)
        if use_quant_dispatch:
            dispatch_token_size = hidden_size * get_dtype_size(torch.int8) + get_dtype_size(torch.float32)
        combine_token_size = hidden_size * get_dtype_size(self.dtype)

        max_num_seqs_per_dp = self.scheduler_config.max_num_seqs
        # max number of tokens that an ep rank could send
        max_num_tokens_per_rank = divide(max_num_seqs_per_dp * seq_len * topk,
                                         self.parallel_config.tensor_parallel_size)

        return dict(dispatch_token_size=dispatch_token_size,
                    combine_token_size=combine_token_size,
                    max_num_tokens_per_rank=max_num_tokens_per_rank,
                    num_global_experts=num_experts,
                    use_quant_dispatch=use_quant_dispatch)

    def prepare_all2all_buffer_for_model(
        self, model: torch.nn.Module) -> None:
        """
        Prepare all2all buffer for the model.
        """
        if not self.use_all2all:
            return

        moe_modules = [
            module for module in self.model.modules()
            if isinstance(module, SparseMoeMlp)
        ]
        if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
            draft_moes = [
                module for module in self.drafter.model.modules()
                if isinstance(module, SparseMoeMlp) and not mlu_envs.VLLM_MTP_NO_QUANT
            ]
            moe_modules.extend(draft_moes)
        for module in moe_modules:
            if self.load_config.load_format == "dummy":
                module.pack_params()
                module.pack_params_after_loading()
            use_quant_dispatch = module.quant_config is not None
            module.prepare_for_cnclep(get_cnclep(use_quant_dispatch=use_quant_dispatch))

    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model()
        if self.use_all2all:
            self.prepare_all2all_buffer_for_model(self.model)
