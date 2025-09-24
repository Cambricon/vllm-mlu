# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy.linalg import hadamard
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_mlu import _mlu_ops as mlu_ops
import torch_mlu_ops as tmo

def hadamard_transform(x, scale=1.0):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


class Indexer(nn.Module):
    def __init__(self,
                 dim,
                 index_n_heads,
                 index_head_dim,
                 qk_rope_head_dim,
                 index_topk,
                 q_lora_rank,
                 rotary_emb,
                 prefix: str = ""):
        super().__init__()
        self.dim: int = dim
        self.n_heads: int = index_n_heads
        self.n_local_heads = index_n_heads // get_tensor_model_parallel_world_size()
        self.head_dim: int = index_head_dim
        self.rope_head_dim: int = qk_rope_head_dim
        self.index_topk: int = index_topk
        self.q_lora_rank: int = q_lora_rank
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.softmax_scale = self.head_dim ** -0.5
        self.rotary_emb = rotary_emb

        self.wq_b = ReplicatedLinear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, quant_config=None, prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(self.dim, self.head_dim, bias=False, quant_config=None, prefix=f"{prefix}.wk")
        self.weights_proj = ReplicatedLinear(self.dim, self.n_heads, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj")


    def bf16_index(self, q, q_s, k, k_s):
        B, M, H, K = q.shape
        N = k.size(1)
        q = q.transpose(-1, -2) # [B, M, K, H]
        # qk^T ： [B, M, K, H] @ [B, K, N] -> [B, M, H, N]
        logits = torch.einsum('bmkh,bnk->bmhn', q, k)
        logits = logits.float()
        relu_out = F.relu(logits) # [B, M, H, N]
        relu_out = relu_out * q_s  # [B, M, H, N]
        o = relu_out.sum(dim=2)
        return o


    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        q = self.wq_b(qr)[0]

        q = rearrange(q, 's (h d) -> s h d', d=self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)[0]
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        if mask is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1), only_prefill=True)
        else:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1), only_decode=True)

        k_pe = k_pe.squeeze(1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)

        weights = self.weights_proj(x)[0] * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1).float() * self.softmax_scale

        # reshape to kv_cache
        mlu_ops.reshape_paged_cache(k.unsqueeze(1),
                                    None,
                                    k_cache,
                                    None,
                                    attn_metadata.slot_mapping.flatten())

        block_size = k_cache.shape[-2]
        metadata = attn_metadata.prefill if mask is not None else attn_metadata.decode
        cu_seq_q_lens = metadata.query_start_loc
        index_mask = tmo.masked_indexer_select_paged_kv(
            x, # q
            cu_seq_q_lens, # cu_seq_q_lens
            None, # q_scale
            weights,
            self.softmax_scale,
            k_cache,
            metadata.seq_lens, # k_context_lens
            metadata.block_table, # k_cache_block_table
            None, # k_scale_cache
            self.index_topk, # index_topk
            block_size,
        )

        return index_mask
