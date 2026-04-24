# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0

from copy import copy
import gc
import time
from contextlib import contextmanager
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Tuple, cast
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cnpx

import vllm.envs as envs
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
    get_tensor_model_parallel_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
    supports_eagle3,
    supports_multimodal_pruning,
)
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    supports_dynamo,
    weak_ref_tensor,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    split_attn_metadata,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    FullAttentionSpec,
    MambaSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
    LogprobsLists,
    SamplerOutput,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import (
    AsyncGPUModelRunnerOutput,
    ExecuteModelState,
    GPUModelRunner,
    PerLayerAttnMetadata,
)
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    MultiModalBudget,
    bind_kv_cache,
    is_residual_scattered_for_sp,
    sanity_check_mm_encoder_outputs,
    scatter_mm_placeholders,
)
if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile", globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile")

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu.compilation.mlu_graph import MLUGraphWrapper
from vllm_mlu.distributed.parallel_state import mlu_graph_capture
from vllm_mlu.model_executor.layers.feed_forward import FeedForward
from vllm_mlu.model_executor.layers.rotary_embedding.base import MLURotaryEmbedding
from vllm_mlu.model_executor.layers.sparse_moe_mlp import SparseMoeMlp
from vllm_mlu.v1.kv_cache_interface import MLUMLAAttentionSpec
from vllm_mlu.v1.attention.backends.flash_attn import FlashAttentionMetadata, pad_attn_metadata
from vllm_mlu.v1.attention.backends.utils import (
    COMMON_METADATA_STR,
    MLUCommonAttentionMetadata,
    MLUInferMode,
    get_common_metadata,
    unpad_common_attn_metadata)
from vllm_mlu.model_executor.models.sp_utils import set_sp_forward_context
from vllm_mlu.v1.sample.sampler import MluSampler
from vllm_mlu.v1.spec_decode.dp_eagle import DPMluEagleProposer
from vllm_mlu.v1.spec_decode.eagle import MluEagleProposer
import vllm_mlu._mlu_utils as mlu_envs

logger = init_logger(__name__)


_NUM_WARMUP_ITERS = 2


def _model_forward_pre_hook(self, args, kwargs):
    '''
    This hook function will be called before model.forward
    '''
    assert len(args) == 0 and len(kwargs) > 0, \
        f"The pre-forward's expected inputs are not passed by kwargs. " + \
        f"Expected len(args)=0, len(kwargs)>0, " + \
        f"now, len(args)={len(args)}, len(kwargs)={len(kwargs)}."

    common_metadata: MLUCommonAttentionMetadata = get_common_metadata()

    if common_metadata:
        # Prepare attributes for all rope in model
        MLURotaryEmbedding.set_mlu_var_v1(common_metadata=common_metadata)

    if self.config.model_type == "deepseek_v4":
        args, kwargs = self.update_forward_args(args, kwargs)

    return (args, kwargs)


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncMLUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: torch.Tensor | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.mlu.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.mlu.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.mlu.current_stream()
        with torch.mlu.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    seq = sorted(input_batch.req_id_to_index.items(), key=lambda x: x[1])
    for req_id, batch_index in seq:
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(
            scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
        )
        if req_id in grammar_output.structured_output_request_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(
            scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
        )
        if req_id in struct_out_req_batch_indices:
            logit_index = struct_out_req_batch_indices[req_id]
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device as tensor.
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    index_tensor = None
    if not skip_out_indices:
        # xgrammar expects a python list of indices but it will actually work with
        # a tensor. If we copy the tensor ourselves here we can do it in a non_blocking
        # manner and there should be no cpu sync within xgrammar.
        index_tensor = torch.tensor(
            out_indices, dtype=torch.int32, device="cpu", pin_memory=True
        )
        index_tensor = index_tensor.to(logits.device, non_blocking=True)

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: remove index_put_ from inductor lowering denylist to
    avoid torch.compile error when using xgrammar
    '''
    from torch_mlu._inductor import remove_from_lowering_denylist
    remove_from_lowering_denylist([torch.ops.aten.index_put_])
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)

class MLUModelRunner(GPUModelRunner):

    def _init_kv_state(
        self,
    ):
        hf_config = self.model_config.hf_config
        if hf_config.model_type != "deepseek_v4":
            return

        CACHED_STATE_NUM = self.scheduler_config.max_num_seqs
        hf_config.cached_state_num = CACHED_STATE_NUM
        self.kv_state_free_slots = set(range(CACHED_STATE_NUM))
        self.req_id_to_kv_state = dict()

    def _insert_req_id(
        self,
        scheduler_output: "SchedulerOutput",
    ):
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            assert req_id not in self.req_id_to_kv_state, \
                f"try to insert req_id: {req_id}, which has been stored int kv_state."
            assert self.kv_state_free_slots, "fail to allocate kv states"
            slot = self.kv_state_free_slots.pop()
            self.req_id_to_kv_state[req_id] = slot

    def _remove_req_id(
        self,
        scheduler_output: "SchedulerOutput",
    ):
        for req_id in scheduler_output.finished_req_ids:
            slot = self.req_id_to_kv_state.pop(req_id)
            self.kv_state_free_slots.add(slot)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.mlu_config = vllm_config.mlu_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 0
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Sampler
        """
        =============================
        Modify by vllm_mlu
        =============================
        @brief: use tmo topk_topp_sampler to sample.
        """
        sampler_cls = (MluSampler
                       if self.model_config.is_deepseek_mla or self.model_config.is_longcat_flash
                       else Sampler)
        self.sampler = sampler_cls(logprobs_mode=self.model_config.logprobs_mode)
        """
        =================
        End of MLU Hijack
        =================
        """

        """
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Add an extra field to indicate infer mode.
        """
        self.mlu_infer_mode = MLUInferMode.PREFILL_ONLY
        """
        =================
        End of MLU Hijack
        =================
        """

        self.eplb_state: EplbState | None = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer | SuffixDecodingProposer | EagleProposer | MedusaProposer
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: Use MluEagleProposer instead of EagleProposer
                '''
                if vllm_config.mlu_config.enable_custom_data_parallel_opt:
                    proposer_cls = DPMluEagleProposer
                else:
                    proposer_cls = MluEagleProposer
                self.drafter = proposer_cls(self.vllm_config, self.device, self)
                self.previous_hidden_states = torch.zeros(
                    (self.max_num_tokens, self.hidden_size),
                    dtype=self.dtype,
                    device=self.device)
                '''
                =============================
                End of MLU Hijack
                =============================
                '''
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        self.comm_stream = torch.mlu.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        custom_logitsprocs = model_config.logits_processors
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Adjust `max_model_len` to expand input_batch.token_ids_cpu_tensor, prevent
                overflow when the total length (including speculative tokens) exceeds max_model_len.
        '''
        max_model_len_revise=max(self.max_model_len, self.max_encoder_len)
        if self.num_spec_tokens > 1:
            max_model_len_revise = max_model_len_revise + self.num_spec_tokens - 1
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max_model_len_revise,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            dcp_kv_cache_interleave_size=self.parallel_config.dcp_kv_cache_interleave_size,
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        self.use_async_scheduling = self.scheduler_config.async_scheduling
        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.mlu.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.mlu.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.mlu.Stream()
            self.prepare_inputs_event = torch.mlu.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: change postions dtype from int64 to int32
        @brief: add seq_start_loc buffer for chunk fa
        '''
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.seq_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )

        self.prefill_enable_mlugraph = self.mlu_config.prefill_enable_mlugraph
        self.prefill_mlugraph_batch_size = self.mlu_config.prefill_mlugraph_batch_size
        self.prefill_mlugraph_seq_len = self.mlu_config.prefill_mlugraph_seq_len
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Add kv_state buffer for deepseekv4
        '''
        self._init_kv_state()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_indices = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )
        self.num_discarded_requests = 0

        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            self.is_mm_embed = self._make_buffer(self.max_num_tokens, dtype=torch.bool)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: change postions dtype from int64 to int32
            '''
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int32
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            )
            if self.supports_mm_inputs
            else None
        )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: enable reorder batch to ensure correct splitting
            between prefill chunks and decode tokens in chunked prefill mode.
        '''
        self.reorder_batch_threshold: int | None = self.uniform_decode_query_len
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self.transfer_event = torch.mlu.Event()
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: change sampled_token_ids dtype from int64 to int32
        @brief: add draft accepted counter
        '''
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.mlu.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.mlu.Stream | None = None
        if self.use_async_scheduling and self.num_spec_tokens:
            self.valid_sampled_token_count_event = torch.mlu.Event()
            self.valid_sampled_token_count_copy_stream = torch.mlu.Stream()
        self.valid_sampled_token_count_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        
        self.execute_cnpx_mark = None
        self.request_cnpx_mark = {}
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support qwen3-next
        '''
        self.mamba_block_num = 1
        self.mamba_tensor_size = 0
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""
        self.device_properties = torch.mlu.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.mlu.synchronize()

    
    def get_accept_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: only support pad tokens in decode mode.
        '''
        if (
            self.mlu_infer_mode == MLUInferMode.DECODE_ONLY
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and hasattr(self, "cudagraph_batch_sizes")
            and self.cudagraph_batch_sizes
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            # Use CUDA graphs.
            # Add padding to the batch size.
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (
            self.compilation_config.pass_config.enable_sequence_parallelism
            and tp_size > 1
        ):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        max_num_scheduled_tokens: int,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
        UBatchSlices | None,
        torch.Tensor | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
            ubatch_slices, num_tokens_across_dp,
        ]
        """
        self._insert_req_id(scheduler_output)

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # # Get the number of scheduled tokens for each request.
        # req_ids = self.input_batch.req_ids
        # tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        # num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        # max_num_scheduled_tokens = max(tokens)


        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = self._get_num_input_tokens(num_tokens_unpadded)
        uniform_decode = (
            max_num_scheduled_tokens == self.uniform_decode_query_len
        ) and (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)

        # Disable DP padding when running eager to avoid excessive padding when
        # running prefills. This lets us set enforce_eager on the prefiller in
        # a P/D setup and still use CUDA graphs (enabled by this padding) on the
        # decoder.
        allow_dp_padding = self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: build mlu dp metadata for dp opt.
        '''
        ubatch_slices, num_tokens_across_dp = None, None
        if self.vllm_config.mlu_config.enable_custom_data_parallel_opt:
            cur_num_reqs = (num_reqs if self.mlu_infer_mode.is_prefill_only
                            else num_reqs * (1 + self.num_spec_tokens))
            query_len_per_batch = (self.query_start_loc.np[1:] -
                                   self.query_start_loc.np[:-1]).tolist()
            dp_metadata = self._get_data_parallel_metadata(
                total_num_scheduled_tokens, cur_num_reqs,
                self.mlu_infer_mode.is_decode_only, query_len_per_batch[:cur_num_reqs]
            )
            # replace num_tokens_across_dp with dp_metadata for return
            num_tokens_across_dp = dp_metadata
        else:
            ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens_unpadded,
                parallel_config=self.parallel_config,
                allow_microbatching=True,
                allow_dp_padding=allow_dp_padding,
                num_tokens_padded=num_tokens_padded,
                uniform_decode=uniform_decode,
                num_scheduled_tokens_per_request=num_scheduled_tokens,
            )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add seq_start_loc for chunk fa.
        '''
        self.seq_start_loc.np[0] = 0
        self.seq_start_loc.np[1:num_reqs + 1] = np.cumsum(self.seq_lens.np[:num_reqs])
        self.seq_start_loc.np[num_reqs + 1 :].fill(self.seq_start_loc.np[num_reqs])
        self.seq_start_loc.copy_to_gpu()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = (
            discard_request_indices
        )

        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (
                    len(draft_token_ids)
                    if (
                        self.input_batch.num_computed_tokens_cpu[req_idx]
                        >= self.input_batch.num_prompt_tokens[req_idx]
                    )
                    else -1
                )
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
            ubatch_slices,
            num_tokens_across_dp,
        )

    def get_model(self) -> nn.Module:
        # get raw model out of the cudagraph wrapper.
        if isinstance(self.model, (MLUGraphWrapper, UBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        # Batch the multi-modal inputs using the helper method.
        mm_kwargs, mm_hashes_pos = self._batch_mm_kwargs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: v1 offline benchmark
        '''
        self.mm_time_markers = []

        if mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            mm_start = torch.mlu.Event(enable_timing=True)
            mm_start.record()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)
        encoder_outputs = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
            multimodal_cpu_fields=model.multimodal_cpu_fields,
        ):
            curr_group_outputs = []

            # EVS-related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                for video_mm_kwargs_item in filter(
                    lambda item: item.modality == "video", mm_kwargs
                ):
                    _, _, micro_batch_mm_inputs = next(
                        group_mm_kwargs_by_modality(
                            [video_mm_kwargs_item],
                            device=self.device,
                            pin_memory=self.pin_memory,
                            merge_by_field_config=model.merge_by_field_config,
                            multimodal_cpu_fields=model.multimodal_cpu_fields,
                        )
                    )

                    micro_batch_outputs = model.embed_multimodal(
                        **micro_batch_mm_inputs
                    )

                    curr_group_outputs.extend(micro_batch_outputs)
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.
                curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: v1 offline benchmark
        '''
        if encoder_outputs and mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
            mm_end = torch.mlu.Event(enable_timing=True)
            mm_end.record()
            self.mm_time_markers.append([mm_start, mm_end])
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Cache the encoder outputs by mm_hash
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        self._remove_req_id(scheduler_output)

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: supoort disagg for mlu.
            '''
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=getattr(new_req_data, 'new_token_ids', None) or [],
                lora_request=new_req_data.lora_request,
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            self.requests[req_id] = req_state

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # prev_num_draft_len is used in async scheduling mode with
            # spec decode. it indicates if need to update num_computed_tokens
            # of the request. for example:
            # fist step: num_computed_tokens = 0, spec_tokens = [],
            # prev_num_draft_len = 0.
            # second step: num_computed_tokens = 100(prompt lenth),
            # spec_tokens = [a,b], prev_num_draft_len = 0.
            # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
            # prev_num_draft_len = 2.
            # num_computed_tokens in first step and second step does't contain
            # the spec tokens length, but in third step it contains the
            # spec tokens length. we only need to update num_computed_tokens
            # when prev_num_draft_len > 0.
            if req_state.prev_num_draft_len:
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens[req_index] = end_idx
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, []
            )
            num_spec_tokens = len(spec_token_ids)
            # For async scheduling, token_ids_cpu assigned from
            # spec_token_ids are placeholders and will be overwritten in
            # _prepare_input_ids.
            if num_spec_tokens:
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index
                ] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

            # When speculative decoding is used with structured output,
            # the scheduler can drop draft tokens that do not
            # conform to the schema. This can result in
            # scheduler_output.scheduled_spec_decode_tokens being empty,
            # even when speculative decoding is enabled.
            self.input_batch.spec_token_ids[req_index] = spec_token_ids

            # there are no draft tokens with async scheduling,
            # we clear the spec_decoding info in scheduler_output and
            # use normal sampling but rejection_sampling.
            if self.use_async_scheduling:
                req_state.prev_num_draft_len = num_spec_tokens
                if num_spec_tokens and self._draft_token_ids is None:
                    scheduler_output.total_num_scheduled_tokens -= num_spec_tokens
                    scheduler_output.num_scheduled_tokens[req_id] -= num_spec_tokens
                    scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: cache the unverified spec_decode token
            '''
            req_id = request.req_id
            req_index = self.input_batch.req_id_to_index.get(req_id)
            assert req_index is not None
            num_tokens = self.input_batch.num_tokens[req_index]
            end_token_index = num_tokens
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, []
            )
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: clear time markers before execute model.
        '''
        self.time_markers = []
        self.mm_time_markers = []
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )   
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("gpu_model_runner: preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        return make_empty_encoder_model_runner_output(scheduler_output)

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
                '''
                max_computed_tokens = np.max(self.input_batch.num_computed_tokens_cpu[:num_reqs])
                self.mlu_infer_mode = MLUInferMode.build(
                    max_query_len=max_num_scheduled_tokens,
                    max_computed_tokens=max_computed_tokens,
                    uniform_decode_query_len=self.uniform_decode_query_len,
                )
                '''
                ==================
                End of MLU Hijack
                ==================
                '''

                (
                    logits_indices,
                    spec_decode_metadata,
                    ubatch_slices,
                    num_tokens_across_dp,
                ) = self._prepare_inputs(
                    scheduler_output, num_scheduled_tokens_np, max_num_scheduled_tokens
                )

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

                dp_rank = self.parallel_config.data_parallel_rank
                if ubatch_slices:
                    assert num_tokens_across_dp is not None
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                    self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
                elif num_tokens_across_dp is not None:
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                else:
                    '''
                    =============================
                    Modify by vllm_mlu
                    =============================
                    pad num_input_tokens after supporting pad decode graph.
                    '''
                    max_num_tokens = (
                        self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
                    )
                    capture_already = False
                    K = 0
                    if hasattr(self.speculative_config, "num_speculative_tokens"):
                        K = self.speculative_config.num_speculative_tokens
                    if (hasattr(self, 'cudagraph_batch_sizes') and
                        self.cudagraph_batch_sizes is not None):
                        decode_cudagraph_batch_sizes = [
                            x
                            for x in self.cudagraph_batch_sizes
                            if max_num_tokens >= x >= self.uniform_decode_query_len
                        ]
                        capture_already = len(decode_cudagraph_batch_sizes) > 0 and \
                            num_reqs*(1+K) <= max(decode_cudagraph_batch_sizes)
                    if self.mlu_infer_mode == MLUInferMode.DECODE_ONLY and not \
                        all(x == K + 1 for x in scheduler_output.num_scheduled_tokens.values()):
                        capture_already = False
                    if capture_already:
                        num_input_tokens = self._get_num_input_tokens(
                            scheduler_output.total_num_scheduled_tokens
                        )
                    else:
                        num_input_tokens = scheduler_output.total_num_scheduled_tokens
                    '''
                    ==================
                    End of MLU Hijack
                    ==================
                    '''

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

                '''
                =============================
                Modify by vllm_mlu
                =============================
                @breif: add padding for attention metadata in decode graph.
                '''
                # padding
                self._padding_attn_metadata(attn_metadata,
                    input_ids, inputs_embeds, capture_already,
                    num_input_tokens, num_scheduled_tokens)
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
            @breif: add prefill graph & capture already check
            '''
            if (self.prefill_enable_mlugraph and
                attn_metadata.get(COMMON_METADATA_STR) is not None and
                attn_metadata[COMMON_METADATA_STR].infer_mode == MLUInferMode.PREFILL_ONLY):
                cudagraph_runtime_mode = CUDAGraphMode.FULL
            if not capture_already:
                cudagraph_runtime_mode = CUDAGraphMode.NONE
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

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @breif: add set_sp_forward_context for sequence parallel.
        '''
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
            set_sp_forward_context(
                attn_metadata,
                self.vllm_config,
                num_input_tokens,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            if self.model_config.hf_config.model_type == "deepseek_v4":
                model_kwargs["batch_to_kv_state"] = torch.tensor([
                    self.req_id_to_kv_state[req_id] for req_id in self.input_batch._req_ids
                ], dtype=torch.int32, device=input_ids.device)
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                prefill_enable_mlugraph=self.prefill_enable_mlugraph,
                **model_kwargs,
            )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
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
                    '''
                    =============================
                    Modify by vllm_mlu
                    =============================
                    @breif: add time markers for pooling model
                    '''
                    if mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN:
                        end = torch.mlu.Event(enable_timing=True)
                        end.record()
                        self.time_markers.append([start, end])
                    '''
                    ==================
                    End of MLU Hijack
                    ==================
                    '''
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
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
    
    def response_remote_alloc_once(self) -> None:
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            kv_connector.response_remote_alloc_once()

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
        sampled_token_ids: torch.Tensor | list[np.ndarray],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: MLUCommonAttentionMetadata,
    ) -> torch.Tensor | list[list[int]]:
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
            num_scheduled_tokens=num_scheduled_tokens
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if self.speculative_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.req_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                self.input_batch.spec_decode_unsupported_reqs,
            )
        elif self.speculative_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(self.input_batch, sampled_token_ids)
        elif self.speculative_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + tokens.shape[0] - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

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
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
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
            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                num_rejected_tokens=num_rejected_tokens_gpu,
                token_indices=token_indices,
                time_markers=self.time_markers,
            )

        return draft_token_ids

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief:
        1. Set max_batched_token for SparseMoeMlp when enable avg moe.
        2. modify rope's max_position_embeddings to max_model_len.
        Those MUST be set before init model.
        '''
        if mlu_envs.VLLM_AVG_MOE_EN:
            logger.warning("Inference with Moe avg dispatch, "
                "it's only for deepseek-v3/r1 model's performance test,"
                " and will result in precision anomalies. Be careful!")
            SparseMoeMlp.max_batched_token = max(self.model_config.max_model_len,
                                                 self.scheduler_config.max_num_batched_tokens)
        MLURotaryEmbedding.max_seq_len = self.model_config.max_model_len
        MLURotaryEmbedding.max_model_len = self.model_config.max_model_len
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        global_expert_loads, old_global_expert_indices_per_model, rank_mapping = (
            EplbState.get_eep_state(self.parallel_config)
            if eep_scale_up
            else (None, None, None)
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0
        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: register model pre forward for rope optimization
            '''
            self.model.register_forward_pre_hook(_model_forward_pre_hook, with_kwargs=True)
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )
            if hasattr(self, "drafter"):
                logger.info_once("Loading drafter model...")
                self.drafter.load_model(self.model)
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: Apply forward prehook to draft model.
                '''
                self.drafter.model.register_forward_pre_hook(_model_forward_pre_hook, with_kwargs=True)
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
                if (
                    hasattr(self.drafter, "model")
                    and is_mixture_of_experts(self.drafter.model)
                    and self.parallel_config.enable_eplb
                ):
                    logger.info_once(
                        "EPLB is enabled for drafter model %s.",
                        self.vllm_config.speculative_config.draft_model_config.model,
                    )

                    global_expert_load = (
                        global_expert_loads[eplb_models]
                        if global_expert_loads
                        else None
                    )
                    old_global_expert_indices = (
                        old_global_expert_indices_per_model[eplb_models]
                        if old_global_expert_indices_per_model
                        else None
                    )
                    if self.eplb_state is None:
                        self.eplb_state = EplbState(self.parallel_config, self.device)
                    self.eplb_state.add_model(
                        self.drafter.model,
                        self.vllm_config.speculative_config.draft_model_config,
                        global_expert_load,
                        old_global_expert_indices,
                        rank_mapping,
                    )
                    eplb_models += 1

            if self.use_aux_hidden_state_outputs:
                if not supports_eagle3(self.get_model()):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )

                # Try to get auxiliary layers from speculative config,
                # otherwise use model's default layers
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if aux_layers:
                    logger.info(
                        "Using auxiliary layers from speculative config: %s",
                        aux_layers,
                    )
                else:
                    aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                self.model.set_aux_hidden_state_layers(aux_layers)
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info_once(
            "Model loading took %.4f GiB memory and %.6f seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
            scope="local",
        )
        prepare_communication_buffer_for_model(self.model)
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and self.model_config.multimodal_config.is_multimodal_pruning_enabled()
        )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            global_expert_load = (
                global_expert_loads[eplb_models] if global_expert_loads else None
            )
            old_global_expert_indices = (
                old_global_expert_indices_per_model[eplb_models]
                if old_global_expert_indices_per_model
                else None
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
            and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        if (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.enable_dbo
        ):
            self.model = MLUGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.enable_dbo:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: remove the prompt_logprobs for final chunk request
                '''
                del prompt_logprobs_dict[req_id]
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _build_attention_metadata(
        self,
        total_num_scheduled_tokens: int,
        max_num_scheduled_tokens: int,
        num_reqs: int,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        scheduled_encoder_inputs: dict[str, list[int]] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        mlu_infer_mode: MLUInferMode | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        logits_indices_padded = None
        num_logits_indices = 0
        if logits_indices is not None:
            num_logits_indices = logits_indices.size(0)
            if self.cache_config.kv_sharing_fast_prefill:
                logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                    logits_indices
                )

        # update seq_lens of decode reqs under DCP.
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.dcp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs)

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # Used in the below loop
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs + 1]
        seq_lens = self.seq_lens.gpu[:num_reqs]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs
        ]
        dcp_local_seq_lens = (
            self.dcp_local_seq_lens.gpu[:num_reqs] if self.dcp_world_size > 1 else None
        )
        spec_decode_common_attn_metadata = None

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_gid, kv_cache_group in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduled_encoder_inputs or {},
                kv_cache_group.kv_cache_spec,
                num_reqs,
            )

            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode.
                blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(-1)
            """
            =============================
            Modify by vllm_mlu
            =============================
            @brief: replace CommonAttentionMetadata with MLUCommonAttentionMetadata
            """
            common_attn_metadata = MLUCommonAttentionMetadata(
                    query_start_loc=query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    max_seq_len=max_seq_len,
                    block_table_tensor=blk_table_tensor,
                    slot_mapping=slot_mapping,
                    causal=True,
                    dcp_local_seq_lens=dcp_local_seq_lens,
                    seq_start_loc=self.seq_start_loc.gpu[: num_reqs + 1],
                    seq_start_loc_cpu=self.seq_start_loc.cpu[: num_reqs + 1],
                    infer_mode=mlu_infer_mode,
                    num_prefill_query_tokens=total_num_scheduled_tokens,
                    num_prefill_kv_tokens=total_num_scheduled_tokens,
                )
            """
            =================
            End of MLU Hijack
            =================
            """
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    """
                    =============================
                    Modify by vllm_mlu
                    =============================
                    @brief: replace attn metadata name to prefill_attn name
                    """
                    attn_layer_name = self.drafter.attn_layer_names[0]
                    if self.model_config.is_deepseek_mla and attn_layer_name.endswith("self_attn.attn"):
                        attn_layer_name = attn_layer_name.replace(
                            "self_attn.attn", "self_attn.mla_attn")
                    if attn_layer_name in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = common_attn_metadata
                    """
                    =================
                    End of MLU Hijack
                    =================
                    """
                else:
                    spec_decode_common_attn_metadata = common_attn_metadata

            for attn_gid, attn_group in enumerate(self.attn_groups[kv_cache_gid]):
                cascade_attn_prefix_len = (
                    cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                    if cascade_attn_prefix_lens
                    else 0
                )
                builder = attn_group.get_metadata_builder()

                extra_attn_metadata_args = {}
                if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs],
                        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                            :num_reqs
                        ],
                    )

                if ubatch_slices is not None:
                    common_attn_metadata_list = split_attn_metadata(
                        ubatch_slices, common_attn_metadata
                    )
                    for ubid, common_attn_metadata in enumerate(
                        common_attn_metadata_list
                    ):
                        builder = attn_group.get_metadata_builder(ubatch_id=ubid)
                        if for_cudagraph_capture:
                            attn_metadata_i = builder.build_for_cudagraph_capture(
                                common_attn_metadata
                            )
                        else:
                            attn_metadata_i = builder.build(
                                common_prefix_len=cascade_attn_prefix_len,
                                common_attn_metadata=common_attn_metadata,
                            )
                        for layer_name in kv_cache_group.layer_names:
                            assert type(attn_metadata) is list
                            attn_metadata[ubid][layer_name] = attn_metadata_i
                else:
                    assert isinstance(attn_metadata, dict)
                    if for_cudagraph_capture:
                        attn_metadata_i = builder.build_for_cudagraph_capture(
                            common_attn_metadata
                        )
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=cascade_attn_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            **extra_attn_metadata_args,
                        )
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i
                """
                =============================
                Modify by vllm_mlu
                =============================
                @brief: bind decode_attn metadata to prefill_attn
                """
                for layer_name in attn_group.layer_names:
                    if (
                        self.model_config.is_deepseek_mla
                        and layer_name.endswith("self_attn.mla_attn")
                    ):
                        prefill_attn_name = layer_name.replace(
                            "self_attn.mla_attn", "self_attn.attn"
                        )
                        attn_metadata[prefill_attn_name] = attn_metadata[layer_name]

                    # matches self_attn.0.attn or self_attn.1.attn for longcat-flash
                    if (
                        self.model_config.is_longcat_flash
                        and (match := re.match(r".*self_attn\.(0|1)\.mla_attn$", layer_name))
                    ):
                        # Extract the captured digit (0 or 1)
                        digit = match.group(1)
                        prefill_attn_name = layer_name.replace(
                            f"self_attn.{digit}.mla_attn",
                            f"self_attn.{digit}.attn"
                        )
                        attn_metadata[prefill_attn_name] = attn_metadata_i
                """
                =================
                End of MLU Hijack
                =================
                """
            """
            =============================
            Modify by vllm_mlu
            =============================
            @brief: Add common_attn_metadata to attn_metadata
            """
            attn_metadata[COMMON_METADATA_STR] = common_attn_metadata
            """
            =================
            End of MLU Hijack
            =================
            """

        return attn_metadata, spec_decode_common_attn_metadata

    def _padding_attn_metadata(
        self,
        attn_metadata: MLACommonMetadata | FlashAttentionMetadata,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        captured_already: bool,
        num_input_tokens: int,
        num_scheduled_tokens: int
    ) -> None:
        common_metadata = attn_metadata[COMMON_METADATA_STR]
        decode_only = common_metadata.is_decode_only
        if decode_only and captured_already:
            # If the model is decode only, we can use full graph.
            # use_full_graph = use_full_graph # and captured_already
            # Update attn_metadata for full graph.
            K = 0
            if (self.speculative_config is not None
                and self.speculative_config.num_speculative_tokens > 0
            ):
                K = self.speculative_config.num_speculative_tokens

            if num_input_tokens != num_scheduled_tokens:
                for kv_cache_group_id, kv_cache_group_spec in enumerate(
                        self.kv_cache_config.kv_cache_groups):
                    block_table = self.input_batch.block_table[kv_cache_group_id]
                    first_layer_name = kv_cache_group_spec.layer_names[0]
                    attn_metadata_i = attn_metadata[first_layer_name]
                    num_reqs = self.input_batch.num_reqs
                    num_padded_reqs = self.vllm_config.pad_for_cudagraph(num_reqs * (1 + K)) // (1 + K)
                    pad_attn_metadata(
                        attn_metadata_i, common_metadata, block_table,
                        self, num_scheduled_tokens, num_input_tokens,
                        num_reqs, num_padded_reqs,
                    )


    """
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add prefill input parameters
        @parameters: is_capturing_prefill, prefill_batch_size, prefill_seq_len
    """
    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        is_capturing_prefill: bool = False,
        prefill_batch_size: int = None,
        prefill_seq_len: int = None,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        # Disable DP padding when running eager
        allow_dp_padding = self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
            num_tokens_unpadded=total_num_scheduled_tokens,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=allow_microbatching,
            allow_dp_padding=allow_dp_padding,
            num_tokens_padded=total_num_scheduled_tokens,
            uniform_decode=uniform_decode,
            num_scheduled_tokens_per_request=num_scheduled_tokens,
        )
        num_tokens_after_padding = num_tokens
        if num_tokens_across_dp is not None:
            dp_rank = self.parallel_config.data_parallel_rank
            num_tokens_after_padding = int(num_tokens_across_dp[dp_rank])

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            """
            =============================
            Modify by vllm_mlu
            =============================
            @brief: use prefill_seq_len to build seq_lens
                when prefill capture
            """
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            elif is_capturing_prefill:
                seq_lens = prefill_seq_len
            else:
                seq_lens = max_query_len
            """
            =================
            End of MLU Hijack
            =================
            """
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()
            """
            =============================
            Modify by vllm_mlu
            =============================
            @brief: compute seq_start_loc and mlu_infer_mode.
            @brief: use prefill_batch_size to build seq_start_loc
            """
            cu_seqlens_k = np.cumsum(self.seq_lens.np[:num_reqs])
            self.seq_start_loc.np[0] = 0
            self.seq_start_loc.np[1 : num_reqs + 1] = cu_seqlens_k
            self.seq_start_loc.copy_to_gpu()

            max_computed_tokens = np.max(self.input_batch.num_computed_tokens_cpu[:num_reqs])
            mlu_infer_mode = MLUInferMode.build(
                max_query_len=max_query_len,
                max_computed_tokens=max_computed_tokens,
                uniform_decode_query_len=self.uniform_decode_query_len)
            if is_capturing_prefill:
                attn_metadata, _ = self._build_attention_metadata(
                    total_num_scheduled_tokens=num_tokens,
                    max_num_scheduled_tokens=max_query_len,
                    num_reqs=prefill_batch_size,
                    ubatch_slices=ubatch_slices,
                    for_cudagraph_capture=True,
                    mlu_infer_mode=MLUInferMode.PREFILL_ONLY,
                )
            else:
                attn_metadata, _ = self._build_attention_metadata(
                    total_num_scheduled_tokens=num_tokens,
                    max_num_scheduled_tokens=max_query_len,
                    num_reqs=num_reqs,
                    ubatch_slices=ubatch_slices,
                    for_cudagraph_capture=True,
                    mlu_infer_mode=mlu_infer_mode,
                )
            """
            =================
            End of MLU Hijack
            =================
            """

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_after_padding <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            else:
                input_ids = self.input_ids.gpu[:num_tokens_after_padding]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_after_padding]
            else:
                positions = self.positions.gpu[:num_tokens_after_padding]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_after_padding, None, False
                )

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                        has_lora=activate_lora and self.lora_config is not None,
                    )
                )
                if not is_profile
                else (CUDAGraphMode.NONE, None)
            )
            """
            =============================
            Modify by vllm_mlu
            =============================
            @brief: adjust cudagraph mode for
                prefill graph capture
            """
            if is_capturing_prefill:
                _cg_mode = cudagraph_runtime_mode
            """
            =================
            End of MLU Hijack
            =================
            """
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert (
                    cudagraph_runtime_mode == CUDAGraphMode.NONE
                    or cudagraph_runtime_mode == _cg_mode
                ), (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
                )
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            '''
            =============================
            Modify by vllm_mlu
            =============================
            @breif: add set_sp_forward_context for sequence parallel.
            '''
            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices,
                ),
                set_sp_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens_after_padding,
                ),
            ):
                if self.model_config.hf_config.model_type == "deepseek_v4":
                    assert self.kv_state_free_slots, \
                        "At least one slot is needed to run dummy model"
                    model_kwargs["batch_to_kv_state"] = torch.tensor([
                            list(self.kv_state_free_slots)[0]
                        ] * num_reqs,
                        dtype=torch.int32,
                        device=input_ids.device,
                    )
                outputs = self.model(
                    is_capturing_prefill=is_capturing_prefill,
                    prefill_batch_size=prefill_batch_size,
                    prefill_seq_len=prefill_seq_len,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                use_cudagraphs = (
                    cudagraph_runtime_mode == CUDAGraphMode.FULL
                    and not self.speculative_config.enforce_eager
                )

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    attn_metadata,
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                )

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    def _capture_cudagraphs(
        self,
        is_capturing_prefill: bool = False,
        prefill_batch_size: int = 0,
        prefill_seq_len: int = 0,
        compilation_cases: list[tuple[int, bool]] = []  ,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        uniform_decode: bool = False,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "prefill" if is_capturing_prefill else "decode",
                    cudagraph_runtime_mode.name,
                ),
            )
            if (self.speculative_config is not None
                and self.speculative_config.num_speculative_tokens > 0
            ):
                compilation_cases = tqdm(
                    compilation_cases,
                    disable=not self.load_config.use_tqdm_on_load,
                    desc="Capturing CUDA draft graphs ({}, {})".format(
                        "decode",
                        cudagraph_runtime_mode.name,
                    ),
                )

        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens, activate_lora in compilation_cases:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.enable_dbo
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
                self._dummy_run(
                    num_tokens=num_tokens,
                    is_capturing_prefill=is_capturing_prefill,
                    prefill_batch_size=prefill_batch_size,
                    prefill_seq_len=prefill_seq_len,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=uniform_decode,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    activate_lora=activate_lora,
                )
            self._dummy_run(
                num_tokens=num_tokens,
                is_capturing_prefill=is_capturing_prefill,
                prefill_batch_size=prefill_batch_size,
                prefill_seq_len=prefill_seq_len,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=uniform_decode,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                activate_lora=activate_lora,
            )
        self.maybe_remove_all_loras(self.lora_config)

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), mlu_graph_capture(device=self.device):
            start_free_gpu_memory = torch.mlu.mem_get_info()[0]
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None

            if self.lora_config:
                if self.compilation_config.cudagraph_specialize_lora:
                    lora_cases = [True, False]
                else:
                    lora_cases = [True]
            else:
                lora_cases = [False]

            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: prefill graph capture
            '''
            if self.prefill_enable_mlugraph:
                # capture prefill mlugraph
                batch_size = self.prefill_mlugraph_batch_size
                seq_len = self.prefill_mlugraph_seq_len
                num_tokens = batch_size * seq_len
                assert num_tokens <= self.scheduler_config.max_num_batched_tokens
                assert batch_size <= self.scheduler_config.max_num_seqs
                logger.info("Capture prefill mlugraph for batch size "
                                f"{batch_size} and seq len {seq_len}")
                prefill_compilation_cases = list(
                    product([num_tokens], lora_cases)
                )
                self._capture_cudagraphs(
                    is_capturing_prefill=True,
                    prefill_batch_size=batch_size,
                    prefill_seq_len=seq_len,
                    compilation_cases=prefill_compilation_cases,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=False,
                )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()
                # make sure we capture the largest batch size first
                compilation_cases = list(
                    product(reversed(self.cudagraph_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False,
                )

            # Capture full cudagraph for uniform decode batches if we
            # don't already have full mixed prefill-decode cudagraphs.
            if (
                cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and hasattr(self, 'cudagraph_batch_sizes')
                and cudagraph_mode.separate_routine()
            ):
                max_num_tokens = (
                    self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
                )
                decode_cudagraph_batch_sizes = [
                    x
                    for x in self.cudagraph_batch_sizes
                    if max_num_tokens >= x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    product(reversed(decode_cudagraph_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                )

            torch.mlu.synchronize()
            end_free_gpu_memory = torch.mlu.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _allocate_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support qwen3-next, deepseek v3.2 indexer cache and mlu kv8
        '''

        kv_cache_group = kv_cache_config.kv_cache_groups[0]

        if self.mlu_config.enable_mamba_split_page_size:
            # hybrid attention, try to find full attention
            for group in kv_cache_config.kv_cache_groups:
                if isinstance(group.kv_cache_spec, FullAttentionSpec):
                    kv_cache_group = group
                    break
            self.mamba_block_num = self.mlu_config.mamba_support_max_batch_size
            self.mamba_tensor_size = (kv_cache_group.kv_cache_spec.page_size_bytes \
                * self.mlu_config.mamba_to_attn_block_ratio * self.mamba_block_num)
            logger.info(f"one linear attn layer cache tensor size {self.mamba_tensor_size}")
        
        kv_cache_raw_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
            num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
            if kv_cache_spec.dtype in [torch.int8, torch.uint8]:
                # mlu kv8
                assert isinstance(kv_cache_spec, AttentionSpec)
                cache_ = torch.zeros(
                    num_blocks * kv_cache_spec.cache_size_bytes,
                    dtype=torch.int8, device=self.device,
                )
                scale_ = torch.zeros(
                    num_blocks * kv_cache_spec.scale_size_bytes,
                    dtype=torch.int8, device=self.device,
                )
            else:
                # not mlu kv8
                cache_ = torch.zeros(
                    num_blocks * kv_cache_spec.cache_size_bytes,
                    dtype=torch.int8,
                    device=self.device
                )
                scale_ = torch.tensor([], dtype=torch.int8, device=self.device)

            if (isinstance(kv_cache_spec, MLUMLAAttentionSpec)
                and kv_cache_spec.index_n_heads > 0):
                index_cache_ = torch.zeros((num_blocks *
                                            kv_cache_spec.index_cache_size_bytes),
                                            dtype=torch.int8,
                                            device=self.device)
            else:
                index_cache_ = torch.tensor([], dtype=torch.int8, device=self.device)

            for layer_name in kv_cache_tensor.shared_by:
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: support qwen3-next
                '''
                if self.mlu_config.enable_mamba_split_page_size:
                    if 'linear_attn' in layer_name:
                        mamba_tensor = torch.zeros(
                            self.mamba_tensor_size, dtype=torch.int8, device=self.device
                        )
                        kv_cache_raw_tensors[layer_name] = [mamba_tensor, scale_, index_cache_]
                    else:
                        kv_cache_raw_tensors[layer_name] = [cache_, scale_, index_cache_]
                else:
                    kv_cache_raw_tensors[layer_name] = [cache_, scale_, index_cache_]
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support mlu kv8 and deepseek v3.2 indexer
        '''
        kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                cache_, scale_, index_cache_ = raw_tensor
                total_numel = cache_.numel() + scale_.numel() + index_cache_.numel()
                assert total_numel % kv_cache_spec.page_size_bytes == 0
                num_blocks = total_numel // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    cache_ = (
                        cache_
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                    # Reshape kv cache scale tensor
                    if dtype in [torch.int8, torch.uint8]:
                        kv_cache_scale_shape = attn_backend.get_kv_cache_scale_shape(
                            kernel_num_blocks,
                            kernel_block_size,
                            kv_cache_spec.num_kv_heads,
                        )
                        scale_ = (
                            scale_
                            .view(torch.float32)
                            .view(kv_cache_scale_shape)
                        )
                    # Reshape index_cache
                    if (isinstance(kv_cache_spec, MLUMLAAttentionSpec)
                        and kv_cache_spec.index_n_heads > 0):
                        index_cache_shape = (
                            kernel_num_blocks,
                            kv_cache_spec.index_n_heads,
                            kernel_block_size,
                            kv_cache_spec.index_head_dim,
                        )
                        index_cache_ = index_cache_.view(dtype).view(index_cache_shape)
                    kv_caches[layer_name] = [cache_, scale_, index_cache_]
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    cache_ = kv_cache_raw_tensors[layer_name]
                    raw_tensor, scale_, index_cache_ = cache_
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        '''
                        =============================
                        Modify by vllm_mlu
                        =============================
                        @brief: support qwen3-next
                        '''
                        if self.mlu_config.enable_mamba_split_page_size:
                            num_element_per_page *= self.mlu_config.mamba_to_attn_block_ratio
                            num_blocks = self.mamba_block_num
                        '''
                        ==================
                        End of MLU Hijack
                        ==================
                        '''
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support qwen3-next
        '''
        if has_attn and has_mamba and not self.mlu_config.enable_mamba_split_page_size:
            self._update_hybrid_attention_mamba_layout(kv_caches)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        return kv_caches

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(
            kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
        )

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: bind kv cache to deepseek prefill attn
        '''
        if self.model_config.is_deepseek_mla:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            for layer_name, kv_cache in kv_caches.items():
                if layer_name.endswith("self_attn.mla_attn"):
                    layer_name = layer_name.replace(
                        "self_attn.mla_attn", "self_attn.attn")
                forward_context[layer_name].kv_cache = [kv_cache]

        # matches self_attn.0.attn or self_attn.1.attn
        if self.model_config.is_longcat_flash:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            for layer_name, kv_cache in kv_caches.items():
                if (match := re.match(r".*self_attn\.(0|1)\.mla_attn$", layer_name)):
                    digit = match.group(1)  # Extract the captured digit (0 or 1)
                    layer_name = layer_name.replace(
                        f"self_attn.{digit}.mla_attn",
                        f"self_attn.{digit}.attn"
                    )
                forward_context[layer_name].kv_cache = [kv_cache]
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        return kv_caches

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        # block_size = self.vllm_config.cache_config.block_size
        # use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: skip deepseek prefill attn init kv_cache
            '''
            if (
                self.model_config.is_deepseek_mla
                and layer_name.endswith("self_attn.attn")
            ):
                continue
            # matches self_attn.0.attn or self_attn.1.attn
            if (
                self.model_config.is_longcat_flash
                and re.match(r".*self_attn\.(0|1)\.attn$", layer_name)
            ):
                continue
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def reset_capture_context(self,
                              prefill_enable_mlugraph: bool,
                              batch_size: int,
                              input_len: int):
        self.graph_runners = {}
        self.context_graph_runner = None
        self.graph_memory_pool = None

        # reset prefill mlugraph infos
        self.prefill_enable_mlugraph = prefill_enable_mlugraph
        self.prefill_mlugraph_batch_size = batch_size
        self.prefill_mlugraph_seq_len = input_len

        gc.collect()
        torch.mlu.empty_cache()

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        '''
        =============================
        Modify by vllm_mlu
        @brief:  replace current stream for MLU device.
        =======
        '''
        default_stream = torch.mlu.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.mlu.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)


    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[np.ndarray],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        discard_sampled_tokens_req_indices = self.discard_request_indices.np[
            : self.num_discarded_requests
        ]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)
        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        valid_sampled_token_ids: list[np.ndarray]
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)] = np.array([])
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        cu_num_accepted_tokens = (
            [0] if spec_decode_metadata and logprobs_tensors else None
        )
        for req_idx in range(num_sampled_tokens):
            sampled_ids: np.ndarray | None
            if self.use_async_scheduling:
                sampled_ids = (
                    np.array([-1]) if req_idx not in invalid_req_indices_set else None
                )
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = (
                sampled_ids.shape[0] if sampled_ids is not None else 0
            )

            if cu_num_accepted_tokens is not None:
                cu_num_accepted_tokens.append(
                    cu_num_accepted_tokens[-1] + num_sampled_ids
                )

            if sampled_ids is None or num_sampled_ids == 0:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            '''
            =============================
            Modify by vllm_mlu
            @brief:  end_idx may exceed max_model_len for sepculative tokens in MTP mode.
            =======
            '''
            num_async_sched_tokens = 1 if self.use_async_scheduling else 0
            max_model_len = self.num_spec_tokens + self.max_model_len + num_async_sched_tokens
            assert end_idx <= max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{max_model_len}"
            )
            if end_idx > self.max_model_len:
                end_idx = self.max_model_len
                sampled_ids = sampled_ids[:end_idx - start_idx]
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (
            logprobs_tensors.tolists(cu_num_accepted_tokens)
            if not self.use_async_scheduling and logprobs_tensors is not None
            else None
        )

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

