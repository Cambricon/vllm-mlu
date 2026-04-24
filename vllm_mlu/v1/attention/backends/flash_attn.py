# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache,
                                              MultipleOf,)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.config.vllm import VllmConfig
from vllm.v1.worker.block_table import BlockTable

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    _get_sliding_window_configs
)
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm_mlu.v1.worker.gpu_model_runner import MLUModelRunner

if current_platform.is_cuda():
    from vllm.attention.utils.fa_utils import get_scheduler_metadata

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata,
    MLUInferMode,
    get_common_metadata,
)
from vllm_mlu.model_executor.layers.quantization.utils.common_utils import attn_str_dtype_to_torch

logger = init_logger(__name__)


class MLUFlashAttentionBackend(FlashAttentionBackend):
    
    supported_kernel_block_sizes: ClassVar[list[int | MultipleOf]] = [1, 16, 32, 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256, 512, 576]

    @staticmethod
    def get_impl_cls() -> type["MLUFlashAttentionImpl"]:
        return MLUFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MLUFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MLUFlashAttentionMetadataBuilder"]:
        return MLUFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_scale_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size)


@dataclass
class MLUChunkFlashAttentionMetadata:
    """
    Chunked prefill metadata for MLU backend, which splits both
    input and metadata into prefill and decode phases. With splitting,
    the MLU backend can invoke FA and single_query_cached_kv_attn kerels
    seperately, thus yields better performance.
    """
    @dataclass
    class ChunkContextMetadata:
        """
        ChunkContextMetadata for prefill chunks and decode tokens.
        """
        batch_size: int
        num_actual_tokens: int
        cu_seqlens_q: torch.Tensor
        cu_seqlens_kv: torch.Tensor
        max_query_len: int
        max_seq_len: int
        total_seqlens: int = 0

    prefill_ctx: ChunkContextMetadata
    decode_ctx: ChunkContextMetadata

    @classmethod
    def build(
        cls,
        common_attn_metadata: MLUCommonAttentionMetadata,
        uniform_decode_query_len: int = 1,
    ):
        assert common_attn_metadata.infer_mode.is_chunked
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(common_attn_metadata,
                                       uniform_decode_query_len,
                                       require_uniform=True)
        # split cu_seqlens_q and cu_seqlens_kv
        query_start_loc = common_attn_metadata.query_start_loc
        d_cu_seqlens_q = query_start_loc[:num_decodes + 1]
        p_cu_seqlens_q = query_start_loc[num_decodes:] - query_start_loc[num_decodes]
        seq_start_loc = common_attn_metadata.seq_start_loc
        d_cu_seqlens_kv = seq_start_loc[:num_decodes + 1]
        p_cu_seqlens_kv = seq_start_loc[num_decodes:] - seq_start_loc[num_decodes]
        # compute max_query_len and max_seq_len after split
        # NOTE: use cpu tensor to avoid d2h copy.
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_len_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_len_cpu = common_attn_metadata.seq_lens_cpu
        d_max_query_len = 0
        d_max_seq_len = 0
        p_max_query_len = 0
        p_max_seq_len = 0
        p_total_seqlens = 0
        if num_decodes > 0:
            d_max_query_len = query_len_cpu[:num_decodes].max().item()
            d_max_seq_len = seq_len_cpu[:num_decodes].max().item()
        if num_prefills > 0:
            p_max_query_len = query_len_cpu[num_decodes:].max().item()
            p_max_seq_len = seq_len_cpu[num_decodes:].max().item()
            p_total_seqlens = seq_len_cpu[num_decodes:].sum().item()

        return MLUChunkFlashAttentionMetadata(
            prefill_ctx=MLUChunkFlashAttentionMetadata.
            ChunkContextMetadata(
                batch_size=num_prefills,
                num_actual_tokens=num_prefill_tokens,
                cu_seqlens_q=p_cu_seqlens_q,
                cu_seqlens_kv=p_cu_seqlens_kv,
                max_query_len=p_max_query_len,
                max_seq_len=p_max_seq_len,
                total_seqlens=p_total_seqlens,
            ),
            decode_ctx=MLUChunkFlashAttentionMetadata.
            ChunkContextMetadata(
                batch_size=num_decodes,
                num_actual_tokens=num_decode_tokens,
                cu_seqlens_q=d_cu_seqlens_q,
                cu_seqlens_kv=d_cu_seqlens_kv,
                max_query_len=d_max_query_len,
                max_seq_len=d_max_seq_len,
            ),
        )

@dataclass
class MLUFlashAttentionMetadata(FlashAttentionMetadata):
    # For mlu infer
    seq_start_loc: torch.Tensor | None = None
    infer_mode: MLUInferMode | None = None
    num_input_tokens: int = 0  # Number of tokens including padding.
    compute_dtype: torch.dtype = torch.float32
    chunk_fa_metadata: MLUChunkFlashAttentionMetadata | None = None

    @property
    def num_decode_tokens(self):
        assert self.infer_mode is not None, (
            f"MLUFlashAttentionMetadata infer_mode is not set."
        )

        if self.infer_mode == MLUInferMode.PREFILL_ONLY:
            return 0

        if self.infer_mode == MLUInferMode.DECODE_ONLY:
            return self.num_actual_tokens

        assert self.chunk_fa_metadata is not None, (
            f"chunk_fa_metadata must be set under chunked infer mode."
        )
        return self.chunk_fa_metadata.decode_ctx.num_actual_tokens


def pad_attn_metadata(
    attn_metadata: MLACommonMetadata | FlashAttentionMetadata,
    common_metadata: MLUCommonAttentionMetadata,
    block_table: BlockTable,
    runner: "MLUModelRunner",
    num_scheduled_tokens: int,
    num_input_tokens: int,
    num_reqs: int,
    num_paded_reqs: int,
) -> None:
    is_mla = isinstance(attn_metadata, MLACommonMetadata)
    if is_mla:
        assert attn_metadata.prefill is None and attn_metadata.decode is not None

    pad_token_num = num_input_tokens - num_scheduled_tokens
    pad_req_num = num_paded_reqs - num_reqs
    if pad_token_num == 0:
        return

    query_start_loc_cpu = runner.query_start_loc.cpu[:num_paded_reqs + 1]
    query_start_loc = runner.query_start_loc.gpu[:num_paded_reqs + 1]
    seq_lens_cpu = runner.seq_lens.cpu[:num_paded_reqs]
    seq_lens = runner.seq_lens.gpu[:num_paded_reqs]
    if pad_req_num > 0:
        query_lens = torch.diff(query_start_loc_cpu[:num_reqs + 1])
        pad_lens = torch.full(
            (pad_req_num,),
            pad_token_num // pad_req_num,
            dtype=query_lens.dtype,
            device=query_lens.device)
        query_lens = torch.cat([query_lens, pad_lens])
        torch.cumsum(query_lens, dim=0, out=query_start_loc_cpu[1:])
        query_start_loc.copy_(query_start_loc_cpu, non_blocking=True)
        seq_lens_cpu[num_reqs:].fill_(common_metadata.max_query_len)
        seq_lens[num_reqs:].fill_(common_metadata.max_query_len)

    seq_start_loc_cpu = runner.seq_start_loc.cpu[:(num_paded_reqs + 1)]
    seq_start_loc = runner.seq_start_loc.gpu[:(num_paded_reqs + 1)]
    torch.cumsum(seq_lens, dim=0, out=seq_start_loc[1:])
    torch.cumsum(seq_lens_cpu, dim=0, out=seq_start_loc_cpu[1:])

    slot_mapping_org_num = attn_metadata.slot_mapping.numel()
    slot_mapping = block_table.slot_mapping.gpu[:(slot_mapping_org_num + pad_token_num)]
    slot_mapping[slot_mapping_org_num:] = PAD_SLOT_ID

    block_table = block_table.get_device_tensor(num_paded_reqs)

    attn_metadata.slot_mapping = slot_mapping
    attn_metadata.query_start_loc = query_start_loc
    if is_mla:
        attn_metadata.decode.query_start_loc = query_start_loc
        attn_metadata.decode.seq_lens = seq_lens
        attn_metadata.decode.block_table = block_table
    else:
        attn_metadata.seq_lens = seq_lens
        attn_metadata.seq_start_loc = seq_start_loc
        attn_metadata.block_table = block_table

    common_metadata.num_input_tokens = num_input_tokens
    common_metadata.seq_start_loc = seq_start_loc
    common_metadata.seq_start_loc_cpu = seq_start_loc_cpu
    common_metadata.query_start_loc = query_start_loc
    common_metadata.query_start_loc_cpu = query_start_loc_cpu
    common_metadata.seq_lens = seq_lens
    common_metadata.seq_lens_cpu = seq_lens_cpu
    common_metadata.num_reqs = num_paded_reqs
    common_metadata.block_table_tensor = block_table
    common_metadata.slot_mapping = slot_mapping


class MLUFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    cudagraph_support = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add class member - uniform_decode_query_len
        '''
        self.uniform_decode_query_len = (
            1 if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: MLUCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MLUFlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add seq_start_loc for chunk fa
        '''
        seq_start_loc = common_attn_metadata.seq_start_loc
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            dcp_context_kv_lens_cpu = seq_lens_cpu - query_kv_lens_cpu

            dcp_context_kv_lens_cpu = get_dcp_local_seq_lens(
                dcp_context_kv_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )
            dcp_context_kv_lens = dcp_context_kv_lens_cpu.to(self.device)
            max_dcp_context_kv_len = dcp_context_kv_lens.max().item()

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True
            )
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: 1. build MLUChunkFlashAttentionMetadata to split prefill and decode;
                2. replace metadata with MLUFlashAttnetionMetadta.
        '''
        chunk_fa_metadata = None
        if common_attn_metadata.infer_mode.is_chunked:
            chunk_fa_metadata = MLUChunkFlashAttentionMetadata.build(
                common_attn_metadata,
                self.uniform_decode_query_len,
            )

        attn_metadata = MLUFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
            # For mlu infer
            seq_start_loc=common_attn_metadata.seq_start_loc,
            infer_mode=common_attn_metadata.infer_mode,
            chunk_fa_metadata=chunk_fa_metadata,
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        return attn_metadata


class MLUFlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: 1. move alibi_slopes to mlu,
                2. sliding_window_right only support -1.
                3. add self.use_fused_mla_qkv.
        '''
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32).mlu()
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.is_mla = extra_impl_args.get("is_mla", False)
        self.use_fused_mla_qkv = extra_impl_args.get("use_fused_mla_qkv", False)

        self.decoder_attn_dtype = extra_impl_args.get("decoder_attn_dtype", None)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MLUFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: set mlu infer mode.
        '''
        infer_mode = attn_metadata.infer_mode
        assert not attn_metadata.use_cascade, (
            f"MLU not support use_cascade={attn_metadata.use_cascade}, " +
            f"attn_metadata={attn_metadata}."
        )
        assert self.dcp_world_size <= 1, (
            f"MLU not support dcp_world_size={self.dcp_world_size}."
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: kv_cache[0] is [key_cache, value_cache], and
        kv_cache[1] is [key_cache_scale, value_cache_scale].
        '''
        key_cache, value_cache = kv_cache[0].unbind(0)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache_scale, value_cache_scale = kv_cache[1].unbind(0)
        else:
            key_cache_scale = None
            value_cache_scale = None
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: skip store key/value to kv cache in mla prefill phase.
        @brief: support value is None.
        '''
        skip_process_cache = (
            self.is_mla
            and (infer_mode.is_prefill_only or self.use_fused_mla_qkv)
        )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and not skip_process_cache
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: store key/value cache with mlu ops.
            '''
            if is_quantized_kv_cache(self.kv_cache_dtype):
                mlu_ops.quant_to_paged_cache(
                    k=key[:num_actual_tokens],
                    v=(None if self.is_mla else value[:num_actual_tokens]),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    k_cache_quant_scale=key_cache_scale,
                    v_cache_quant_scale=value_cache_scale,
                    slot_mapping=attn_metadata.slot_mapping.flatten(),
                )
            else:
                mlu_ops.reshape_paged_cache(
                    k=key[:num_actual_tokens],
                    v=(None if self.is_mla else value[:num_actual_tokens]),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping.flatten(),
                )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: skip cascade attention for mlu platform.
        '''
        if attn_metadata.use_cascade:
            raise RuntimeError(
                f"mlu v1 not support use_cascade={attn_metadata.use_cascade}, " +
                f"attn_metadata={attn_metadata}."
            )
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_kv = attn_metadata.seq_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        alibi_slopes = (
            None if self.alibi_slopes is None
            else self.alibi_slopes.repeat(seqused_k.shape[0], 1)
        )
        head_size_v = value.shape[-1] if self.is_mla else self.head_size
        q_quant_scale = kwargs.get("q_quant_scale", None)

        if infer_mode.is_prefill_only:
            num_prefill_query_tokens = num_actual_tokens
            num_prefill_kv_tokens = num_actual_tokens
            mlu_ops.flash_attention(
                q=query[:num_prefill_query_tokens],
                k=key[:num_prefill_kv_tokens],
                v=value[:num_prefill_kv_tokens],
                out=output[:num_prefill_query_tokens],
                cu_seq_lens_q=cu_seqlens_q,
                cu_seq_lens_kv=cu_seqlens_kv,
                alibi_slope=alibi_slopes,
                attn_bias=None,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_kv=max_seqlen_k,
                softmax_scale=self.scale,
                is_causal=True,
                window_size_left=self.sliding_window[0],
                window_size_right=self.sliding_window[1],
                compute_dtype=attn_metadata.compute_dtype,
                return_lse=False,
            )
        elif infer_mode.is_chunked:
            # prefill & decode mixed
            # NOTE: Split prefill chunks and decode tokens will
            # get better performance on MLU devices.
            chunk_fa_metadata = attn_metadata.chunk_fa_metadata
            prefill_ctx = chunk_fa_metadata.prefill_ctx
            decode_ctx = chunk_fa_metadata.decode_ctx
            num_decodes = decode_ctx.batch_size
            num_decode_tokens = decode_ctx.num_actual_tokens
            num_prefills = prefill_ctx.batch_size
            if num_prefills > 0:
                self._forward_prefill_chunk(
                    query=query[num_decode_tokens:],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=output[num_decode_tokens:],
                    block_table=block_table[num_decodes:],
                    seqused_k=seqused_k[num_decodes:],
                    compute_dtype=attn_metadata.compute_dtype,
                    prefill_ctx=prefill_ctx,
                    alibi_slopes=alibi_slopes,
                    key_cache_scale=key_cache_scale,
                    value_cache_scale=value_cache_scale,
                )
            if num_decodes > 0:
                if q_quant_scale is not None:
                    q_quant_scale = q_quant_scale[:num_decode_tokens]
                self._forward_decode_only(
                    query=query[:num_decode_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=output[:num_decode_tokens],
                    block_table=block_table[:num_decodes],
                    seqused_k=seqused_k[:num_decodes],
                    max_seqlen_k=decode_ctx.max_seq_len,
                    head_size_v=head_size_v,
                    compute_dtype=attn_metadata.compute_dtype,
                    alibi_slopes=alibi_slopes,
                    key_cache_scale=key_cache_scale,
                    value_cache_scale=value_cache_scale,
                    q_quant_scale=q_quant_scale,
                )
        else:
            # decode only
            if q_quant_scale is not None:
                q_quant_scale = q_quant_scale[:num_actual_tokens]
            self._forward_decode_only(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                block_table=block_table,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                head_size_v=head_size_v,
                compute_dtype=attn_metadata.compute_dtype,
                alibi_slopes=alibi_slopes,
                key_cache_scale=key_cache_scale,
                value_cache_scale=value_cache_scale,
                q_quant_scale=q_quant_scale,
            )
        return output

    def _forward_prefill_chunk(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        compute_dtype: torch.dtype,
        prefill_ctx: MLUChunkFlashAttentionMetadata.ChunkContextMetadata,
        alibi_slopes: torch.Tensor | None = None,
        key_cache_scale: torch.Tensor | None = None,
        value_cache_scale: torch.Tensor | None = None,
    ):
        '''
        Compute prefill chunks when enable chunked_prefill.
        NOTE: If the kv_cache is quantized,
        will first be dequantized, and return continuous key and value.
        '''
        if is_quantized_kv_cache(self.kv_cache_dtype):
            total_seqlens = prefill_ctx.total_seqlens
            key_cache_dequant = torch.zeros(
                (total_seqlens, self.num_kv_heads, self.head_size),
                dtype=query.dtype,
                device=key_cache.device
            )
            value_cache_dequant = None
            if value_cache is not None:
                value_cache_dequant = torch.zeros(
                    (total_seqlens, self.num_kv_heads, self.head_size),
                    dtype=query.dtype,
                    device=key_cache.device
                )
            mlu_ops.dequant_from_paged_cache(
                key=key_cache_dequant,
                value=value_cache_dequant,
                key_cache=key_cache,
                value_cache=value_cache,
                key_cache_quant_scale=key_cache_scale,
                value_cache_quant_scale=value_cache_scale,
                context_lengths=seqused_k,
                max_context_len=prefill_ctx.max_seq_len,
                context_seq_offset=None,
                block_tables=block_table,
                quant_mode=1,
                quant_bit=8
            )
            block_table_dequant = None
        else:
            key_cache_dequant = key_cache
            value_cache_dequant = value_cache
            block_table_dequant = block_table
        mlu_ops.flash_attention(
            q=query,
            k=key_cache_dequant,
            v=value_cache_dequant,
            out=output,
            cu_seq_lens_q=prefill_ctx.cu_seqlens_q,
            cu_seq_lens_kv=prefill_ctx.cu_seqlens_kv,
            alibi_slope=alibi_slopes,
            attn_bias=None,
            max_seq_len_q=prefill_ctx.max_query_len,
            max_seq_len_kv=prefill_ctx.max_seq_len,
            softmax_scale=self.scale,
            is_causal=True,
            window_size_left=self.sliding_window[0],
            window_size_right=self.sliding_window[1],
            compute_dtype=compute_dtype,
            return_lse=False,
            block_tables=block_table_dequant,
        )

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        head_size_v: int,
        compute_dtype: torch.dtype,
        alibi_slopes: torch.Tensor | None = None,
        key_cache_scale: torch.Tensor | None = None,
        value_cache_scale: torch.Tensor | None = None,
        q_quant_scale: torch.Tensor | None = None,
    ):
        '''
        Compute decode tokens only.
        NOTE: Query only support pad mode, be careful when using MTP model.
        '''
        batch_size = block_table.shape[0]
        decode_query = query.view(batch_size, -1, self.num_heads, self.head_size)
        decode_output = output.view(batch_size, -1, self.num_heads, head_size_v)
        if q_quant_scale is not None:
            q_quant_scale = q_quant_scale.view(batch_size, -1, self.num_heads)
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
            windows_size_left=self.sliding_window[0],
            windows_size_right=self.sliding_window[1],
            softmax_scale=self.scale,
            head_size_v=(-1 if not self.is_mla else head_size_v),
            compute_dtype=compute_dtype,
            q_quant_scale=q_quant_scale,
            decoder_attn_dtype=self.decoder_attn_dtype,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        mlu_ops.flash_attention(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seq_lens_q=cu_seqlens_q,
            cu_seq_lens_kv=cu_seqlens_k,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_kv=max_seqlen_k,
            softmax_scale=self.scale,
            is_causal=False,  # Encoder attention is bidirectional
            window_size_left=self.sliding_window[0],
            window_size_right=self.sliding_window[1],
            compute_dtype=attn_metadata.compute_dtype,
            return_lse=False,
        )

        return output



