# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import math
from typing import Callable
from scipy.linalg import hadamard

import torch
from torch import nn
import torch.nn.functional as F

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.v1.attention.backends.utils import get_common_metadata


def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(
        x,
        torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device),
    )
    out = out * scale
    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size ** -0.5)


class Compressor(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rope,
                 compress_ratio: int = 4,
                 head_dim: int = 512,
                 rotate: bool = False,
                 prefix: str = "",
                 **kwargs,):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.dim = config.dim
        self.head_dim = head_dim
        self.rope_head_dim =config.rope_head_dim
        self.nope_head_dim = head_dim - config.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.norm_eps = config.norm_eps
        self.window_size = config.window_size

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # The first half of dimensions for overlapping compression and second half for normal compression.

        self.wkv = ReplicatedLinear(
            self.dim,
            coff * self.head_dim,
            bias=False,
            quant_config=None,
            params_dtype = torch.float32,
            prefix=f"{prefix}.wkv",
        )

        self.wgate = ReplicatedLinear(
            self.dim,
            coff * self.head_dim,
            bias=False,
            quant_config=None,
            params_dtype = torch.float32,
            prefix=f"{prefix}.wgate",
        )

        self.norm = RMSNorm(self.head_dim, self.norm_eps)

        self.rotary_emb = rope

        hf_config = vllm_config.model_config.hf_config
        assert hasattr(hf_config, "cached_state_num"), \
            f"cached_state_num is not set in hf_config"
        cached_state_num = hf_config.cached_state_num
        self.register_buffer(
            "kv_state",
            torch.zeros(cached_state_num, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (cached_state_num, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.hadamard_matrix = torch.tensor(
            hadamard(self.head_dim, dtype=float), dtype=torch.get_default_dtype(), device="mlu")

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        batch_to_kv_state: torch.Tensor,
        kv_cache: torch.Tensor,
        window_offset: int,
        compressor_slot_mapping: torch.Tensor,
    ):
        x = x.float()
        kv_pack, _ = self.wkv(x)
        score_pack, _ = self.wgate(x)

        mlu_ops.fused_compress_single_kv(
            kv=kv_pack.unsqueeze(1), # (token, D) -> (B, S, D)
            score=score_pack.unsqueeze(1), # (token, D) -> (B, S, D)
            position=positions,
            ape=self.ape,
            kv_state=self.kv_state,
            score_state=self.score_state,
            gamma=self.norm.weight,
            sin=self.rotary_emb.sin_,
            cos=self.rotary_emb.cos_,
            hadamard_matrix=self.hadamard_matrix,
            slot_mapping=compressor_slot_mapping,
            kv_cache=kv_cache,
            kv_cache_scale=None,
            eps=self.norm_eps,
            overlap=self.overlap,
            rotate=self.rotate,
            state_idx=batch_to_kv_state,
        )

        # Here, return fake compressed_kv.
        return None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        batch_to_kv_state: torch.Tensor,
        kv_cache: torch.Tensor,
        window_offset: int,
        compressor_slot_mapping: torch.Tensor,
    ):
        common_metadata = get_common_metadata()
        forward_func: Callable = (
            self.forward_prefill if common_metadata.is_prefill_only
            else self.forward_decode
        )
        return forward_func(
            x,
            positions,
            attn_metadata,
            batch_to_kv_state,
            kv_cache,
            window_offset,
            compressor_slot_mapping,
        )

    def forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        batch_to_kv_state: torch.Tensor,
        kv_cache: torch.Tensor,
        window_offset: int,
        compressor_slot_mapping: torch.Tensor,
    ):
        common_metadata = get_common_metadata()
        seq_lens = common_metadata.seq_lens
        query_start_loc = common_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype
        x = x.float()
        kv_pack, _ = self.wkv(x)
        score_pack, _ = self.wgate(x)

        compress_lens = query_lens // self.compress_ratio
        cu_compress_lens = torch.cat([
            torch.tensor([0], dtype=compress_lens.dtype, device=compress_lens.device),
            torch.cumsum(compress_lens, dim=0)],
        )

        compress_positions = []
        for i in range(len(seq_lens)):
            seqlen = (query_start_loc[i+1] - query_start_loc[i]).item()
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            pos = positions[query_start_loc[i]: query_start_loc[i+1]]
            positions_ = pos[:cutoff:ratio].contiguous()
            compress_positions.append(positions_)
        kv_positions = torch.cat(compress_positions, dim=0)


        total_compress_len = cu_compress_lens[-1].item()
        kv = torch.empty(
            [total_compress_len, self.head_dim],
            dtype=kv_pack.dtype,
            device=kv_pack.device,
        )

        mlu_ops.fused_compress_multi_kv(
            kv = kv_pack,
            score = score_pack,
            kv_state = self.kv_state,
            score_state = self.score_state,
            state_batch_idx = batch_to_kv_state,
            cu_seqlens = query_start_loc,
            ape = self.ape,
            max_seqlen = common_metadata.max_query_len,
            overlap = overlap,
            compressed_kv = kv,
        )

        if kv.size(0) == 0:
            return kv.unsqueeze(-2).to(dtype) # (compress_token_num, 1, head_size)


        kv = self.norm(kv.to(dtype))

        kv_rope = kv[..., -self.rope_head_dim:].unsqueeze(-2)
        # use compressed cu_seqlens here, so can not call rotary_emb directly
        kv_rope = mlu_ops.rotary_embedding(
            kv_rope,
            self.rotary_emb.sin_,
            self.rotary_emb.cos_,
            kv_positions,
            torch.tensor([0, kv_positions.size(0)], dtype=torch.int32, device=kv_positions.device), # cu_seqlens
            True, # interleaved
            True, # discrete
            False,
            common_metadata.max_query_len,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        mlu_ops.reshape_paged_cache(
            kv.unsqueeze(1),
            None,
            kv_cache,
            None,
            compressor_slot_mapping,
        )

        return kv.unsqueeze(-2)  # (compress_token_num, 1, head_size)