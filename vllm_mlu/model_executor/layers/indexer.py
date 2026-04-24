# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.model_executor.layers.compressor import (
    Compressor,
    rotate_activation,
)
from vllm_mlu.v1.attention.backends.utils import get_common_metadata

logger = init_logger(__name__)


class Indexer(torch.nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rope,
        compress_ratio: int = 4,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_local_heads = config.index_n_heads // self.tp_size
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.window_size = config.window_size
        self.block_size = vllm_config.cache_config.block_size

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wq_b",
        )

        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype = torch.bfloat16,
            prefix=f"{prefix}.weights_proj",
        )

        self.softmax_scale = self.head_dim ** -0.5
        self.merged_softmax_scale = (self.head_dim ** -0.5) * (self.n_heads ** -0.5)
        self.compress_ratio = compress_ratio
        self.max_model_len = vllm_config.model_config.max_model_len

        self.rotary_emb = rope
        self.tp_group = get_tp_group()

        self.compressor = Compressor(vllm_config, self.rotary_emb, compress_ratio, self.head_dim, True, f"{prefix}.compressor")

        self.freqs_cis = None

    def forward_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: AttentionMetadata,
        k_full: torch.Tensor,
        context_lens: torch.Tensor,
    ):
        assert attn_metadata.prefill.chunked_context is None, \
            f"Prefill chunked context is not supported."

        query_start_loc = attn_metadata.prefill.query_start_loc
        cu_seq_q_lens = query_start_loc
        cu_seq_k_lens = torch.zeros(
            context_lens.size(0) + 1, dtype=torch.int32, device=q.device,
        )
        torch.cumsum(context_lens, dim=0, out=cu_seq_k_lens[1:])
        attn_metadata.prefill.query_start_loc
        seq_lens = torch.diff(cu_seq_k_lens)

        batch_size = seq_lens.shape[0]

        new_block_tables = torch.empty(
            [attn_metadata.num_prefill_tokens, self.index_topk],
            dtype=torch.int32,
            device=q.device,
        )
        new_context_lens = torch.empty(
            [attn_metadata.num_prefill_tokens],
            dtype=torch.int32,
            device=q.device,
        )

        q_seq_lens = cu_seq_q_lens[1:]-cu_seq_q_lens[:-1]
        max_seq_len = q_seq_lens.max().item()
        batch_size = q_seq_lens.size(0)
        max_compressed_kv_len = max_seq_len // self.compress_ratio
        kv_cache_block_table = torch.zeros([batch_size, max_compressed_kv_len], dtype=torch.int32, device=q.device)

        # The layout of linear kv is as follows:
        # | bs0_origin_kv | bs1_origin_kv | bs0_compressed_kv | bs1_compressed_kv |
        for i in range(batch_size):
            start = cu_seq_k_lens[i].item()
            kv_cache_block_table[i] = torch.arange(
                start, start + max_compressed_kv_len,
                dtype=torch.int32,
                device=q.device,
            )
        # offset total origin_kv len
        kv_cache_block_table = kv_cache_block_table + cu_seq_q_lens[-1]

        # query: (tokens, index_head, index_head_dim)
        # k_full: (tokens, index_head_dim)
        # weights: (tokens, index_head, 1)
        mlu_ops.masked_indexer_select_paged_kv_prefill(
            query=q,
            key_value=k_full,
            weights=weights.unsqueeze(-1),
            kv_cache_block_table=kv_cache_block_table,
            cu_seq_q_lens=cu_seq_q_lens,
            cu_seq_k_lens=cu_seq_k_lens,
            index_topk=self.index_topk,
            kv_cache_block_size=self.block_size,
            softmax_scale=self.merged_softmax_scale,
            q_scale=None,
            k_scale_cache=None,
            sparse_block_table=new_block_tables,
            sparse_context_lens=new_context_lens,
            compress_ratio=self.compress_ratio,
            kv_cache_block_table_offset=None,
        )

        return new_block_tables, new_context_lens

    def forward_decode(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        block_table = attn_metadata.decode.block_table
        batch_size = block_table.shape[0]
        seq_len = x.shape[0] // batch_size
        q = q.view(batch_size, seq_len, *q.shape[1:])
        weights = weights.view(batch_size, seq_len, *weights.shape[1:])

        seq_lens = attn_metadata.decode.seq_lens
        k_block_table = block_table

        seq_len = x.shape[0] // batch_size
        new_block_tables = torch.empty(
            [batch_size, seq_len, self.index_topk],
            dtype=torch.int32,
            device=block_table.device,
        )
        new_context_lens = torch.empty(
            [attn_metadata.num_decode_tokens],
            dtype=torch.int32,
            device=block_table.device,
        )

        kv_cache_block_table_offset=torch.empty(
            [attn_metadata.num_decode_tokens],
            dtype=torch.int32,
            device=block_table.device,
        )
        kv_cache_block_table_offset.fill_(self.window_size)
        mlu_ops.masked_indexer_select_paged_kv_decode(
            query=q,
            k_cache=k_cache,
            weights=weights.unsqueeze(-1), # (bsz, seq_q, head_num, 1)
            kv_cache_block_table=block_table,
            k_context_lens=seq_lens // self.compress_ratio,
            k_cache_block_table=k_block_table,
            index_topk=self.index_topk,
            kv_cache_block_size=self.block_size,
            softmax_scale=self.merged_softmax_scale,
            q_scale=None,
            k_scale_cache=None,
            sparse_block_table=new_block_tables,
            sparse_context_lens=new_context_lens,
            compress_ratio=self.compress_ratio,
            kv_cache_block_table_offset=kv_cache_block_table_offset,
        )

        # [batch, seq_q, index_topk] -> [batch, index_topk]
        new_block_tables = new_block_tables.squeeze(1)
        return new_block_tables, new_context_lens

    def forward(self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        offsets: torch.Tensor,
        attn_metadata: AttentionMetadata,
        batch_to_kv_state: torch.Tensor,
        indexer_kv_cache: torch.Tensor,
        compressor_slot_mapping: torch.Tensor,
    ):
        common_metadata = get_common_metadata()
        query_start_loc = common_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        rd = self.rope_head_dim
        q = self.wq_b(qr)[0]
        q = q.unflatten(-1, (self.n_heads, self.head_dim))

        self.rotary_emb(positions, q[..., -rd:], None, only_prefill=False)
        q_pack = rotate_activation(q)
        weights_pack = self.weights_proj(x)[0] # (tokens, index_local_head)

        num_decode_tokens = attn_metadata.num_decode_tokens
        compressed_kv = self.compressor(
            x,
            positions,
            attn_metadata,
            batch_to_kv_state,
            indexer_kv_cache,
            0,
            compressor_slot_mapping,
        )
        if attn_metadata.prefill:
            assert compressed_kv is not None and compressed_kv.dim() == 3
            compressed_kv = compressed_kv.squeeze(-2)
            compressed_context_lens = query_lens // self.compress_ratio

            prefill_q = q_pack[num_decode_tokens:, ...]
            prefill_weights = weights_pack[num_decode_tokens:, ...]
            prefill_block_tables, prefill_context_lens = self.forward_prefill(
                prefill_q,
                indexer_kv_cache,
                prefill_weights,
                attn_metadata,
                compressed_kv,
                compressed_context_lens,
            )

        if attn_metadata.decode:
            decode_x = x[:num_decode_tokens, ...]
            decode_q = q_pack[:num_decode_tokens, ...]
            decode_weights = weights_pack[attn_metadata.num_prefills:]
            decode_block_tables, decode_context_lens = self.forward_decode(
                decode_q,
                decode_x,
                indexer_kv_cache,
                decode_weights,
                attn_metadata,
            )

        if attn_metadata.prefill and attn_metadata.decode:
            new_block_tables = torch.cat([prefill_block_tables, decode_block_tables], dim=0)
            new_context_lens = torch.cat([prefill_context_lens, decode_context_lens], dim=0)
        elif attn_metadata.prefill:
            new_block_tables = prefill_block_tables
            new_context_lens = prefill_context_lens
        else:
            new_block_tables = decode_block_tables
            new_context_lens = decode_context_lens
        return new_block_tables, new_context_lens