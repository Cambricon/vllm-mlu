# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.attention.backends.utils import MLADims
from vllm.config import ModelConfig
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend, MLACommonPrefillMetadata,
    MLACommonDecodeMetadata, MLACommonMetadata,
    MLACommonMetadataBuilder, M, QueryLenSupport,
    use_cudnn_prefill, use_flashinfer_prefill,
    use_trtllm_ragged_deepseek_prefill,
    FlashInferPrefillMetadata,
    CudnnPrefillMetadata,
    MLACommonImpl,
    CUDNN_WORKSPACE_SIZE
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, split_decodes_and_prefills,
    infer_global_hyperparameters, get_per_layer_parameters,
    )
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.v1.attention.backends.flash_attn import MLUFlashAttentionImpl
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata, get_common_metadata,
    MLUInferMode)
from vllm.distributed.parallel_state import get_dcp_group, is_global_first_rank
from vllm.platforms import current_platform
from vllm import envs
try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache  # noqa: F401

    flashinfer_available = True
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object

    flashinfer_available = False

logger = init_logger(__name__)



from vllm_mlu.mlu_hijack_utils import MluHijackObject


class MLACommonBackend_MluHijack(MLACommonBackend):
    
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576, 512]
    

def get_mla_dims(model_config: ModelConfig) -> MLADims:
    hf_text_config = model_config.hf_text_config
    
    if model_config.hf_text_config.model_type == "deepseek_v4":
        return MLADims(
            q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
            kv_lora_rank=hf_text_config.head_dim,
            qk_nope_head_dim=hf_text_config.head_dim - hf_text_config.rope_head_dim,
            qk_rope_head_dim=hf_text_config.rope_head_dim,
            v_head_dim=hf_text_config.head_dim,
        )

    return MLADims(
        q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
        kv_lora_rank=hf_text_config.kv_lora_rank,
        qk_nope_head_dim=hf_text_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_text_config.qk_rope_head_dim,
        v_head_dim=hf_text_config.v_head_dim,
    )

    
class MLACommonMetadataBuilder_MluHijack(MLACommonMetadataBuilder):

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[M] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (
            metadata_cls if metadata_cls is not None else MLACommonMetadata
        )
        self.kv_cache_spec = kv_cache_spec
        scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.device = device

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.aot_schedule = current_platform.is_cuda()
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.dcp_local_block_size = parallel_config.dcp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size

        # Don't try to access the runner on AMD
        if self.aot_schedule:
            self.page_size = self.kv_cache_spec.block_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )

        if self.dcp_world_size > 1:
            # Note(hc): The local kvcache is incomplete when DCP is triggered,
            # an additional kvcache allgather across the DCP group is therefore
            # required, so the workspace has to be enlarged by 1/DCP relative
            # to the original TP allocation.
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size
                    + self.chunked_prefill_workspace_size // self.dcp_world_size,
                    self.model_config.get_head_size(),
                ),
                dtype=self.model_config.dtype,
                device=device,
            )
        else:
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.model_config.get_head_size(),
                ),
                dtype=self.model_config.dtype,
                device=device,
            )

        self._use_cudnn_prefill = use_cudnn_prefill()
        self._use_fi_prefill = use_flashinfer_prefill()
        self._use_trtllm_ragged_prefill = use_trtllm_ragged_deepseek_prefill()
        self.prefill_metadata_cls = (
            FlashInferPrefillMetadata
            if self._use_fi_prefill
            else CudnnPrefillMetadata
            if self._use_cudnn_prefill
            else MLACommonPrefillMetadata
        )

        if self._use_fi_prefill:
            self._workspace_buffer = torch.empty(
                envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=device,
            )

            self._fi_prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
            self._fi_prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = []

            self._global_hyperparameters = infer_global_hyperparameters(
                get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl)
            )

        if self._use_trtllm_ragged_prefill:
            self._workspace_buffer = torch.empty(
                envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=device,
            )

        if self._use_cudnn_prefill:
            self.cudnn_workspace = torch.empty(
                CUDNN_WORKSPACE_SIZE * scheduler_config.max_num_seqs,
                dtype=torch.int8,
                device=device,
            )

        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode, supports_dcp_with_varlen
        )

        # Validate consistency between query_len_support and reorder_batch_threshold
        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )
    
    
MluHijackObject.apply_hijack(MLACommonBackend,
                             MLACommonBackend.get_supported_head_sizes,
                             MLACommonBackend_MluHijack.get_supported_head_sizes)
MluHijackObject.apply_hijack(MLACommonMetadataBuilder,
                             MLACommonMetadataBuilder.__init__,
                             MLACommonMetadataBuilder_MluHijack.__init__)


class FlashMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (1, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_scale_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
    ) -> tuple[int, ...]:
        return (1, num_blocks, num_kv_heads, block_size)
    
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576, 512]


@dataclass
class FlashMLAPrefillMetadata(MLACommonPrefillMetadata):
    num_prefills: int = -1 # for gather_cache
    max_seq_len: int = -1 # for attn forward

    @property
    def block_tables(self):
        return self.block_table

    @property
    def context_chunk_cu_seq_lens(self):
        if self.chunked_context is None:
            return None
        return self.chunked_context.cu_seq_lens

    @property
    def context_chunk_starts(self):
        if self.chunked_context is None:
            return None
        return self.chunked_context.starts

    @property
    def context_chunk_seq_tot(self):
        if self.chunked_context is None:
            return None
        return self.chunked_context.seq_tot

    @property
    def context_chunk_max_seq_lens(self):
        if self.chunked_context is None:
            return None
        return self.chunked_context.max_seq_lens

    @property
    def context_chunk_workspace(self):
        if self.chunked_context is None:
            return None
        return self.chunked_context.workspace


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor

    # add for mlu rope and attn forward
    query_start_loc: torch.Tensor # for rope
    max_query_len: int # for rope
    max_seq_len:int = -1 # for attn forward


@dataclass
class FlashMLAMetadata(MLACommonMetadata):
    num_prefill_tokens: Optional[int] = None


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # ^ TODO(matt): tune this

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None
        self.is_fp8_kvcache = vllm_config.cache_config.cache_dtype.startswith("fp8")

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: 1. set decoder_query_len for mtp
        @brief: 2. init chunk workspace for prefix_caching only
        @brief: 3. set prefill_metadata_cls
        @brief: 4. add deepseek v3.2 infos
        '''
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = (speculative_config.num_speculative_tokens
                                       if speculative_config is not None else 0)
        self.decoder_query_len = 1 + self.num_speculative_tokens

        self.max_model_len = self.model_config.max_model_len
        self.is_deepseek_v32 = self.model_config.hf_text_config.model_type == "deepseek_v32"

        self.enable_caching = cache_config.enable_prefix_caching
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        if (not self.is_deepseek_v32 and not self.chunked_prefill_enabled and
                (mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED and self.enable_caching)):
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(
                    8 * self.model_config.max_model_len, 4 *
                    scheduler_config.max_num_seqs * cache_config.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                scheduler_config.max_num_seqs * cache_config.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 self.model_config.get_head_size()),
                dtype=self.model_config.dtype,
                device=device,
            )

        self.prefill_metadata_cls = FlashMLAPrefillMetadata
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0
        # mlu v1 mtp forces decoder_query_len = 1 for k > 1, so we should set again
        self.decoder_query_len = 1 + self.num_speculative_tokens

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: record prefill and decode requests and token nums to call
            chunked fa and single-query attn respectively in forward.
            @Notes: decodes need all prompt tokens are computed.
            '''
            req_index = input_batch.req_id_to_index.get(req_id)
            all_prompt_tokens_has_computed = (
                input_batch.num_computed_tokens_cpu[req_index] >=
                input_batch.num_prompt_tokens[req_index])
            if num_tokens <= self.decoder_query_len and all_prompt_tokens_has_computed:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        return modified_batch

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
    ) -> FlashMLADecodeMetadata:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: set tile_scheduler_metadata and num_splits to None.
        @brief: set dcp_tot_seq_lens_device.
        '''
        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            tile_scheduler_metadata=None,
            num_splits=None,
            dcp_tot_seq_lens=None,
            # for mlu
            max_seq_len=max_seq_len,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def build_for_cudagraph_capture(
            self, common_attn_metadata: MLUCommonAttentionMetadata) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MLA.
        """
        m = common_attn_metadata
        if m.infer_mode == MLUInferMode.DECODE_ONLY:
            assert m.num_reqs * m.max_query_len == m.num_actual_tokens, \
                "MLA only supports decode-only full CUDAGraph capture. " \
                "Make sure all cudagraph capture sizes <= max_num_seq."

        return self.build(0, m)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: MLUCommonAttentionMetadata,
              fast_build: bool = False,
              input_batch: "InputBatch" = None) -> M:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        num_computed_tokens_cpu = (common_attn_metadata.seq_lens_cpu -
                                   query_seq_lens_cpu)
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support normal and mtp input split
        '''
        if input_batch is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
                split_decodes_and_prefills(common_attn_metadata,
                                           self.decoder_query_len)
        else:
            num_decodes, num_prefills = input_batch.split_decodes_and_prefills()
            num_decode_tokens = common_attn_metadata.query_start_loc_cpu[num_decodes].item()
            num_prefill_tokens = num_tokens - num_decode_tokens
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: avoid buffer missing when prefill_only + mlugraph
            '''
            if num_decodes > 0:
                prefill_query_start_loc = query_start_loc[
                    reqs_start:] - query_start_loc[reqs_start]
            else:
                prefill_query_start_loc= query_start_loc
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

            chunked_context_metadata = None
            if ((self.chunked_prefill_enabled or
                        (mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED and
                         self.enable_caching and
                         common_attn_metadata.is_chunked)
                    ) and num_prefills > 0 and max_context_len_cpu > 0):
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                if self.is_deepseek_v32:
                    max_context_chunk = self.max_model_len
                else:
                    max_context_chunk = (self.chunked_prefill_workspace_size //
                                         num_prefills_with_context_cpu)

                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_cache` kernel cannot handle
                    # `context_chunk_starts` that are not aligned to page_size
                    max_context_chunk = round_down(max_context_chunk,
                                                   self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like
                #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = \
                    torch.arange(num_chunks, dtype=torch.int32) \
                    .unsqueeze(1).expand(-1, num_prefills) \
                    * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)

                chunked_context_metadata_cls = \
                        FlashMLAPrefillMetadata.ChunkedContextMetadata

                chunked_context_metadata = \
                    chunked_context_metadata_cls(
                    cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                    starts=chunk_starts.to(device, non_blocking=True),
                    seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                    max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                    seq_lens=chunk_seq_lens,
                    workspace=getattr(self, "chunked_prefill_workspace", None),
                )

                if not self.is_deepseek_v32:
                    assert max(chunked_context_metadata.max_seq_lens) <= \
                        self.chunked_prefill_workspace_size

            prefill_metadata = self.prefill_metadata_cls(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
                # for mlu
                num_prefills=num_prefills,
                max_seq_len=common_attn_metadata.seq_lens_cpu[reqs_start:].max().item(),
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens=seq_lens[:num_decodes],
                query_start_loc=query_start_loc[:num_decodes + 1],
                max_query_len=query_seq_lens_cpu[:num_decodes].max().item(),
                max_seq_len=common_attn_metadata.seq_lens_cpu[:num_decodes].max().item(),
            )

        attn_metadata = self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata

    def can_run_in_cudagraph(
            self, common_attn_metadata: MLUCommonAttentionMetadata) -> bool:
        return common_attn_metadata.max_query_len == self.decoder_query_len

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class FlashMLAImpl(MLUFlashAttentionImpl):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        kwargs: Optional[dict[str, Any]] = {},
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        out_lse = None

        # use default common metadata if kwargs does not have common_metadata
        common_metadata: MLUCommonAttentionMetadata = kwargs.get("common_metadata", None)
        if common_metadata is None:
            common_metadata = get_common_metadata()

        only_prefill = kwargs.get("only_prefill", False)
        only_decode = kwargs.get("only_decode", False)
        attn_bias = kwargs.get("attn_bias", None)

        assert only_prefill != only_decode, "only_prefill and only_decode cannot be True and False at the same time."

        if only_prefill:
            cu_seqlens_q = attn_metadata.prefill.query_start_loc
            cu_seqlens_kv = common_metadata.query_start_loc
            seqused_k = common_metadata.seq_lens[attn_metadata.num_decodes:]
            max_seqlen_q = attn_metadata.prefill.max_query_len
            max_seqlen_k = attn_metadata.prefill.max_seq_len
            block_table = attn_metadata.prefill.block_table
            num_actual_tokens = attn_metadata.num_prefill_tokens
        else:
            cu_seqlens_q = None # nouse
            cu_seqlens_kv = None # nouse
            seqused_k = common_metadata.seq_lens[:attn_metadata.num_decodes]
            max_seqlen_q = None # nouse
            max_seqlen_k = common_metadata.max_seq_len
            block_table = attn_metadata.decode.block_table
            num_actual_tokens = attn_metadata.num_decode_tokens

        skip_process_cache = ((self.use_mla
                              and (common_metadata.is_prefill_only
                                   or self.use_fused_mla_qkv
                                   or only_prefill))
                              or self.kv_sharing_target_layer_name is not None)

        kv_cache_, kv_cache_scale_, kv_cache_index_ = kv_cache
        key_cache = kv_cache_[0]
        value_cache = None if self.use_mla else kv_cache_[1]
        key_cache_scale, value_cache_scale = None, None
        if kv_cache_scale_.numel() > 0:
            key_cache_scale = kv_cache_scale_[0]
            value_cache_scale = None if self.use_mla else kv_cache_scale_[1]
        if not skip_process_cache:
            if is_quantized_kv_cache(self.kv_cache_dtype):
                mlu_ops.quant_to_paged_cache(
                    k=key[:num_actual_tokens],
                    v=(None if self.use_mla else value[:num_actual_tokens]),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    k_cache_quant_scale=key_cache_scale,
                    v_cache_quant_scale=value_cache_scale,
                    slot_mapping=attn_metadata.slot_mapping.flatten(),
                )
            else:
                mlu_ops.reshape_paged_cache(
                    k=key[:num_actual_tokens],
                    v=(None if self.use_mla else value[:num_actual_tokens]),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping.flatten()
                )

        alibi_slopes = None if self.alibi_slopes is None else \
                        self.alibi_slopes.repeat(seqused_k.shape[0], 1)

        if kwargs.get("model_type", "") == "deepseek_v32":
            from vllm_mlu.model_executor.models.sp_utils import get_sp_forward_context
            sp_context = get_sp_forward_context()
            if sp_context is not None and sp_context.is_v32:
                num_actual_tokens = sp_context.sp_attn_metadata.num_prefill_tokens
            decode_query = query[:num_actual_tokens].view(-1, self.num_heads, self.head_size)
            head_size_v = value.shape[-1] if self.use_mla else self.head_size
            decode_output = output[:num_actual_tokens].view(-1, self.num_heads, head_size_v)
            decode_query = query.unsqueeze(1)  # see tokens as batch dim
            decode_output = decode_output.unsqueeze(1)
            q_quant_scale = kwargs.get("q_quant_scale", None)
            if q_quant_scale is not None:
                q_quant_scale = q_quant_scale[:num_actual_tokens].view(-1, self.num_heads)
                q_quant_scale = q_quant_scale.unsqueeze(1)
            mlu_ops.single_query_cached_kv_attn(
                q=decode_query,
                k_cache=key_cache,
                v_cache=value_cache,
                out=decode_output,
                block_tables=kwargs.get("new_block_tables", None),
                context_lens=kwargs.get("new_context_lens", None),
                k_cache_quant_scale=key_cache_scale,
                v_cache_quant_scale=value_cache_scale,
                alibi_slopes=alibi_slopes,
                max_contxt_len=kwargs.get("index_topk", None),
                windows_size_left=(-1 if self.sliding_window is None else self.sliding_window[0]),
                windows_size_right=(-1 if self.sliding_window is None else self.sliding_window[0]),
                softmax_scale=self.scale,
                head_size_v=(-1 if not self.use_mla else head_size_v),
                compute_dtype=compute_dtype,
                q_quant_scale=q_quant_scale,
                decoder_attn_dtype=self.decoder_attn_dtype,
                mask=attn_bias,
            )
            return output

        if common_metadata.is_prefill_only or only_prefill:
            # prefill only
            prefill_causal = kwargs.get("prefill_causal", True)
            cu_seqlens_q = kwargs.get("cu_seq_lens_q", cu_seqlens_q)
            cu_seqlens_kv = kwargs.get("cu_seq_lens_kv", cu_seqlens_kv)
            max_seqlen_q = kwargs.get("max_seq_len_q", max_seqlen_q)
            max_seqlen_k = kwargs.get("max_seq_len_kv", max_seqlen_k)
            return_lse = kwargs.get("return_lse", False)
            num_prefill_query_tokens = common_metadata.num_prefill_query_tokens
            num_prefill_kv_tokens =  common_metadata.num_prefill_kv_tokens
            use_f32 = attn_bias is not None and attn_bias.dtype == torch.float32
            if use_f32:
                f32_output = torch.empty_like(output, dtype=torch.float32)
            attn_output_list = mlu_ops.flash_attention(
                q=query[:num_prefill_query_tokens].to(torch.float32) if use_f32 else query[:num_prefill_query_tokens],
                k=key[:num_prefill_kv_tokens].to(torch.float32) if use_f32 else key[:num_prefill_kv_tokens],
                v=value[:num_prefill_kv_tokens].to(torch.float32) if use_f32 else value[:num_prefill_kv_tokens],
                out=f32_output[:num_prefill_query_tokens] if use_f32 else output[:num_prefill_query_tokens],
                cu_seq_lens_q=cu_seqlens_q,
                cu_seq_lens_kv=cu_seqlens_kv,
                alibi_slope=alibi_slopes,
                attn_bias=attn_bias,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_kv=max_seqlen_k,
                softmax_scale=self.scale,
                is_causal=prefill_causal,
                window_size_left=(-1 if self.sliding_window is None else self.sliding_window[0]),
                window_size_right=(-1 if self.sliding_window is None else self.sliding_window[1]),
                compute_dtype=self.prefill_compute_dtype,
                return_lse=return_lse,
                q_quant_dtype=self.prefill_q_dtype,
                k_quant_dtype=self.prefill_k_dtype,
                v_quant_dtype=self.prefill_v_dtype
            )
            if use_f32:
                output[:num_prefill_query_tokens].copy_(f32_output[:num_prefill_query_tokens])

            if return_lse:
                out_lse = attn_output_list[1]
        else:
            batch_size = block_table.shape[0]
            # decode only
            decode_query = query[:num_actual_tokens].view(batch_size, -1, self.num_heads, self.head_size)
            head_size_v = value.shape[-1] if self.use_mla else self.head_size
            decode_output = output[:num_actual_tokens].view(batch_size, -1, self.num_heads, head_size_v)
            q_quant_scale = kwargs.get("q_quant_scale", None)
            if q_quant_scale is not None:
                q_quant_scale = q_quant_scale[:num_actual_tokens].view(batch_size, -1, self.num_heads)
            mlu_ops.single_query_cached_kv_attn(
                q=decode_query,
                k_cache=key_cache,
                v_cache=value_cache,
                out=decode_output,
                block_tables=block_table,
                context_lens=seqused_k,
                k_cache_quant_scale=key_cache_scale,
                v_cache_quant_scale=value_cache_scale,
                alibi_slopes=alibi_slopes,
                max_contxt_len=max_seqlen_k,
                windows_size_left=(-1 if self.sliding_window is None else self.sliding_window[0]),
                windows_size_right=(-1 if self.sliding_window is None else self.sliding_window[0]),
                softmax_scale=self.scale,
                head_size_v=(-1 if not self.use_mla else head_size_v),
                compute_dtype=attn_metadata.decode.compute_dtype,
                q_quant_scale=q_quant_scale,
                decoder_attn_dtype=self.decoder_attn_dtype,
                mask=attn_bias,
            )

        return output if out_lse is None else (output, out_lse)
