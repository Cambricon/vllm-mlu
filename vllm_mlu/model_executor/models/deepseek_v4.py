# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import re
from typing import Iterable, Set, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_data_parallel_group_world_size,
    get_tp_group,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsEagle
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.attention import AttentionMetadata
from vllm_mlu.model_executor.layers.feed_forward import FeedForward
from vllm_mlu.v1.attention.backends.utils import (
    MLUCommonAttentionMetadata,
    get_common_metadata,
)
from vllm_mlu.model_executor.layers.indexer import Indexer
from vllm_mlu.model_executor.layers.compressor import Compressor
from vllm_mlu import _mlu_ops as mlu_ops
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention
from vllm_mlu.model_executor.layers.sparse_moe_mlp import MoeGroupInfo, SparseMoeMlp
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class HCHead(torch.nn.Module):

    def __init__(
        self,
        hc_mult,
        dim,
        hc_eps,
        norm_eps,
        prefix: str = "",
    ):
        super().__init__()
        self.hc_mult: int = hc_mult
        self.dim: int = dim
        self.hc_dim: int = hc_mult * dim
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps

        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, self.hc_dim, dtype=torch.float), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float), requires_grad=False)

    def forward(self, x: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype)


class HCPre(torch.nn.Module):

    def __init__(
        self,
        hc_mult,
        dim,
        hc_sinkhorn_iters,
        hc_eps,
        norm_eps,
        prefix: str = "",
    ):
        super().__init__()
        self.hc_mult: int = hc_mult
        self.dim: int = dim
        self.hc_dim: int = hc_mult * dim
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        self.norm_eps = norm_eps

        self.hc_fn = nn.Parameter(torch.empty(mix_hc, self.hc_dim, dtype=torch.float), requires_grad=False)
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float), requires_grad=False)
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        rsqrt: torch.Tensor | None = None,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()
        if rsqrt is None:
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)

        mixes = F.linear(x, self.hc_fn)
        pre, post, comb = mlu_ops.hc_split_sinkhorn(
            mixes.unsqueeze(0),
            self.hc_scale,
            self.hc_base,
            rsqrt.squeeze(-1).unsqueeze(0),
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        pre, post, comb = pre.squeeze(0), post.squeeze(0), comb.squeeze(0)


        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype), post, comb


class HCPost(torch.nn.Module):
    def __init__(
        self,
        norm_eps: float,
        prefix: str = "",
    ):
        self.norm_eps = norm_eps
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        compute_rms: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor| None]:
        # x: [bs, dim], residual: [bs, hc, dim], post: [bs, hc], comb: [bs, hc, hc]
        # return
        # y: [bs, hc, dim]
        # [bs, hc, 1] * [bs, 1, dim] + torch.sum([bs, hc, hc, 1] * [bs, hc, 1, dim], -2)
        # rsqrt: Optional, [bs, 1]
        use_tmo = True
        if use_tmo:
            y, rsqrt = mlu_ops.fused_mhc_post(x, residual, post, comb, compute_rms, self.norm_eps)
            return y, (rsqrt.unsqueeze(-1) if rsqrt is not None else None)

        y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
            torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)

        rsqrt = (
            torch.rsqrt(y.type_as(x).flatten(-2).float().square().mean(-1, keepdim=True) + self.norm_eps)
            if compute_rms
            else None
        )

        return y.type_as(x), rsqrt


class MLUDeepseekV4MoE(SparseMoeMlp):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ):
        layer_id = int(prefix.split(sep=".")[-2])
        self.layer_id = layer_id
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        SparseMoeMlp.__init__(
            self,
            num_experts=config.n_routed_experts,
            top_k=config.n_activated_experts,
            hidden_size=config.dim,
            intermediate_size=config.moe_inter_dim,
            up_proj_name='w13',
            is_gated=True,
            down_proj_name='w2',
            has_bias=False,
            hidden_act='silu',
            params_dtype=torch.float,
            quant_config=quant_config,
            is_use_fused_moe=True,
            expert_group=1,
            topk_group=1,
            scoring_func=config.score_func,
            topk_method='',
            routed_scaling_factor=config.route_scale,
            use_hash=(layer_id < config.n_hash_layers),
            vocab_size=config.vocab_size,
            prefix=prefix,
        )

        self.dim = config.dim
        world_size = get_ep_group().world_size
        self.world_size = world_size
        assert config.n_routed_experts % world_size == 0, \
            f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = self.n_routed_experts // world_size
        self.n_activated_experts = config.n_activated_experts
        self.experts_start_idx = get_ep_group().rank_in_group * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        assert config.n_shared_experts == 1
        self.shared_experts = FeedForward(
            hidden_size=config.dim,
            intermediate_size=config.moe_inter_dim,
            hidden_act='silu',
            up_proj_name='w13',
            is_gated=True,
            down_proj_name='w2',
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        shape = hidden_states.size()
        shared_output = self.shared_experts(hidden_states)
        router_logits, _ = self.gate(hidden_states.float())
        hidden_states = self.forward_experts(
            hidden_states,
            router_logits,
            shared_output=shared_output,
            input_ids=input_ids,
        )
        hidden_states = self.reduce_results(hidden_states)
        return hidden_states.view(shape)


class MLUDeepseekV4Attention(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-2])
        self.layer_id = layer_id
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.attn_data_parallel_size = get_data_parallel_group_world_size()
        self.attn_tensor_parallel_size = get_tensor_model_parallel_world_size()

        self.num_heads = vllm_config.model_config.hf_config.n_heads
        assert self.num_heads % self.tp_size == 0
        self.num_local_heads = self.num_heads // self.attn_tensor_parallel_size
        self.model_type = config.model_type
        self.use_indexer = hasattr(config, 'index_n_heads')

        self.hidden_size = config.dim
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.rope_head_dim = config.rope_head_dim
        self.eps = config.norm_eps
        self.o_groups = config.o_groups
        self.o_local_groups = self.o_groups // self.attn_tensor_parallel_size
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = config.compress_ratios[layer_id]
        self.window_size = config.window_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.original_seq_len = config.original_seq_len
        self.index_topk = config.index_topk

        self.o_lora_rank = config.o_lora_rank
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.rope_scaling = getattr(config, "rope_scaling", None)

        tp_group = get_tp_group()

        # disable YaRN and use base rope_theta in pure sliding-window attention
        if self.compress_ratio > 1:
            max_position_embeddings = self.rope_scaling.get("original_max_position_embeddings", 65536)
            self.rope_scaling["rope_type"] = 'deepseek_yarn'
        else:
            max_position_embeddings = 0
            self.rope_scaling["rope_type"] = 'default'
            if self.rope_scaling is not None:
                self.rope_scaling["original_max_position_embeddings"] = 0

        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
        )

        self.output_rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta,
            rope_scaling=self.rope_scaling,
            dtype=torch.float32,
            is_neox_style=False,
            inverse=True,
        )

        if self.q_lora_rank is not None:
            self.wq_a = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.wq_a",
                )
            self.q_norm = RMSNorm(
                self.q_lora_rank,
                eps=self.eps,
            )
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wq_b",
                tp_group=tp_group,
            )

        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(
            self.head_dim,
            eps=self.eps,
        )
        if get_tensor_model_parallel_world_size() <= self.o_groups:
            self.wo_a = ColumnParallelLinear(
                self.num_heads * self.head_dim // self.o_groups,
                self.o_groups * self.o_lora_rank,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wo_a",
            )
            self.wo_b = RowParallelLinear(
                self.o_groups * self.o_lora_rank,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.wo_b",
                tp_group=tp_group,
            )
        else:
            self.wo_a = ReplicatedLinear(
                self.num_heads * self.head_dim // self.o_groups,
                self.o_groups * self.o_lora_rank,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wo_a",
            )
            self.wo_b = ReplicatedLinear(
                self.o_groups * self.o_lora_rank,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wo_b",
            )

        self.attn = MLAAttention(
            self.num_local_heads, # num_heads
            self.softmax_scale, # scale
            self.head_dim - self.rope_head_dim, # qk_nope_head_dim
            self.rope_head_dim, # qk_rope_head_dim
            self.head_dim, # v_head_dim
            self.q_lora_rank, # q_lora_rank
            self.head_dim, # kv_lora_rank
            self.wkv, # kv_b_proj
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            # extra_impl_args
            num_kv_heads=1,
            prefix=f"{prefix}.attn",
            use_fused_mla_qkv=False,
        )

        if self.compress_ratio:
            self.compressor = Compressor(vllm_config, self.rotary_emb, self.compress_ratio, self.head_dim, False, f"{prefix}.compressor")
            if self.compress_ratio == 4:
                self.indexer = Indexer(vllm_config, self.rotary_emb, self.compress_ratio, f"{prefix}.indexer")
            else:
                self.indexer = None

        self.attn_sink = nn.Parameter(torch.empty(self.num_local_heads, dtype=torch.float32))

    def forward_sparse_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        batch_to_kv_state: torch.Tensor,
        window_compress_params: dict | None,
        window_slot_mapping: torch.Tensor,
        compressor_slot_mapping: dict | None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if self.q_lora_rank is not None:
            q = self.wq_a(hidden_states)[0]

            q = self.q_norm(q)
            qr = q
            q = self.wq_b(q)[0].view(-1, self.num_local_heads, self.head_dim)

            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        _, q_pe = q.split([self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1)

        kv = self.wkv(hidden_states)[0]
        kv = self.kv_norm(kv)
        kv = kv.unsqueeze(-2)
        kv_pe = kv[..., -self.rope_head_dim :]

        q_pe, kv_pe = self.rotary_emb(positions, q_pe, kv_pe, only_prefill=False)

        common_metadata = get_common_metadata()
        query_start_loc = common_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        key_cache = kv_cache[0][0]
        mlu_ops.reshape_paged_cache(
            kv,
            None,
            key_cache,
            None,
            window_slot_mapping,
        )

        if self.compress_ratio:
            offsets = query_lens if common_metadata.is_prefill_only else torch.full_like(query_lens, self.window_size)
            if self.indexer is not None:
                indexer_kv_cache = kv_cache[2]
                compress_block_tables, compress_context_lens = self.indexer(
                    hidden_states,
                    qr,
                    positions,
                    offsets,
                    attn_metadata,
                    batch_to_kv_state,
                    indexer_kv_cache,
                    compressor_slot_mapping[(0, self.compress_ratio)],
                )

        if self.compress_ratio:
            compress_kv = self.compressor(
                hidden_states,
                positions,
                attn_metadata,
                batch_to_kv_state,
                key_cache,
                self.window_size,
                compressor_slot_mapping[(self.window_size, self.compress_ratio)],
            )

            if common_metadata.is_prefill_only:
                kv = torch.cat([kv, compress_kv], dim=0)

        assert window_compress_params != None
        if self.compress_ratio:
            if self.indexer is not None:
                window_block_tables = window_compress_params.get("window_block_tables", None)
                window_context_lens = window_compress_params.get("window_context_lens", None)
                new_block_tables = torch.empty([num_tokens, self.window_size + self.index_topk], dtype=torch.int32, device=hidden_states.device)
                new_context_lens = torch.empty([num_tokens], dtype=torch.int32, device=hidden_states.device)
                mlu_ops.concat_block_table(
                    window_block_tables,
                    window_context_lens,
                    compress_block_tables,
                    compress_context_lens,
                    new_block_tables,
                    new_context_lens,
                )
                max_contxt_len = self.window_size + self.index_topk
            else:
                new_block_tables = window_compress_params.get("compress_block_tables", None)
                new_context_lens = window_compress_params.get("compress_context_lens", None)
                max_contxt_len = self.window_size + (self.max_model_len // self.compress_ratio)
        else:
            new_block_tables = window_compress_params.get("window_block_tables", None)
            new_context_lens = window_compress_params.get("window_context_lens", None)
            max_contxt_len = self.window_size


        attn_output = torch.zeros_like(q)
        total_token = q.size(0)
        assert total_token == new_block_tables.size(0)
        q_ = q.view(total_token, -1, self.num_local_heads, self.head_dim)
        attn_output = attn_output.view(total_token, -1, self.num_local_heads, self.head_dim)
        if common_metadata.is_prefill_only:
            kv_cache_ = kv.unsqueeze(1) # insert block_size, [total_token, 1, head_dim] -> [total_token, 1, 1, head_dim]
        else:
            kv_cache_ = kv_cache[0].view(-1, 1, 1, self.head_dim)

        mlu_ops.single_query_cached_kv_attn(
            q=q_,
            k_cache=kv_cache_,
            v_cache=None,
            out=attn_output,
            block_tables=new_block_tables,
            context_lens=new_context_lens,
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_contxt_len=max_contxt_len,
            windows_size_left=-1,
            windows_size_right=-1,
            softmax_scale=self.softmax_scale,
            compute_dtype=torch.float,
            learnable_sink=self.attn_sink,
        )

        attn_output = attn_output.reshape(-1, self.num_local_heads, self.head_dim).to(torch.float)

        attn_output_pe = attn_output[..., -self.rope_head_dim:]
        attn_output_pe, _ = self.output_rotary_emb(positions, attn_output_pe, None, only_prefill=False)

        attn_output = attn_output.to(dtype=torch.bfloat16)

        if get_tensor_model_parallel_world_size() <= self.o_groups:
            attn_output = attn_output.reshape(num_tokens, self.o_local_groups, -1)
            wo_a = self.wo_a.weight.view(self.o_local_groups, self.o_lora_rank, -1)

            o = torch.einsum("ngd,grd->ngr", attn_output, wo_a)
            output = self.wo_b(o.flatten(-2))[0]

            output = tensor_model_parallel_all_reduce(output)
        else:
            # (token, 64/tp, head_dim) -> (64/tp, token, head_dim)
            attn_output = attn_output.flatten(-2).contiguous()
            attn_output = tensor_model_parallel_all_gather(attn_output, dim=-1)
            # (token, 64 * head_dim) -> (token, 64, head_dim)
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim).contiguous() # t, 64
            wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
            attn_output = attn_output.reshape(num_tokens, self.o_groups, -1)
            o = torch.einsum("ngd,grd->ngr", attn_output, wo_a)
            output = self.wo_b(o.flatten(-2))[0]
        return output


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_to_kv_state: torch.Tensor,
        window_compress_params: dict | None,
        window_slot_mapping: torch.Tensor,
        compressor_slot_mapping: dict | None,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return torch.empty_like(hidden_states)

        # self.attn and self.attn_decoder always have the same attn_metadata
        # and share the same kv cache for each layer
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.attn.layer_name]
        kv_cache = self.attn.kv_cache[forward_context.virtual_engine]

        output = self.forward_sparse_attn(
            positions,
            hidden_states,
            kv_cache,
            attn_metadata,
            batch_to_kv_state,
            window_compress_params,
            window_slot_mapping,
            compressor_slot_mapping,
        )

        return output


class MLUDeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: PretrainedConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        self.config = config

        self.dim = config.dim
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.attn = MLUDeepseekV4Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.attn",
        )

        self.hc_mult = config.hc_mult
        self.mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.hc_dim = self.hc_mult * config.dim
        self.norm_eps = config.norm_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps

        self.hc_attn_pre = HCPre(
            self.hc_mult,
            config.dim,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
            prefix=f"{prefix}.hc_attn_pre"
        )
        self.hc_attn_post = HCPost(
            self.norm_eps,
        )

        self.hc_ffn_pre = HCPre(
            self.hc_mult,
            config.dim,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
            prefix=f"{prefix}.hc_attn_pre"
        )
        self.hc_ffn_post = HCPost(
            self.norm_eps,
        )

        self.attn_norm = RMSNorm(config.dim, config.norm_eps)

        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = MLUDeepseekV4MoE(
            vllm_config=vllm_config,
            prefix=f"{prefix}.ffn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        residual: torch.Tensor | None,
        batch_to_kv_state: torch.Tensor,
        window_compress_params: dict | None = None,
        hc_attn_pre_norm: torch.Tensor | None = None,
        window_slot_mapping: torch.Tensor | None = None,
        compressor_slot_mapping: dict | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = self.hc_attn_pre(hidden_states, rsqrt=hc_attn_pre_norm)
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            positions,
            hidden_states,
            batch_to_kv_state,
            window_compress_params,
            window_slot_mapping,
            compressor_slot_mapping,
        )
        hidden_states, hc_ffn_pre_norm = self.hc_attn_post(
            hidden_states,
            residual,
            post,
            comb,
            compute_rms=True,
        )
        residual = hidden_states

        is_last_layer = (self.layer_idx == self.config.n_layers - 1)
        hidden_states, post, comb = self.hc_ffn_pre(hidden_states, rsqrt=hc_ffn_pre_norm)
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, input_ids)
        hidden_states, hc_attn_pre_norm = self.hc_ffn_post(
            hidden_states,
            residual,
            post,
            comb,
            compute_rms=(not is_last_layer),
        )

        return hidden_states, hc_attn_pre_norm

@support_torch_compile
class MLUDeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.compress_ratio = 128 # only compressor layer 128
        self.window_size = config.window_size
        self.max_model_len = vllm_config.model_config.max_model_len

        self.vocab_size = config.vocab_size
        self.norm_eps = config.norm_eps
        self.hc_eps = config.hc_eps

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.dim,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(MLUDeepseekV4DecoderLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_id}",
                config=config,
            ))

        self.hc_mult = config.hc_mult
        self.dim = config.dim
        self.hc_head = HCHead(
            self.hc_mult,
            self.dim,
            self.hc_eps,
            self.norm_eps,
            prefix=f"{prefix}.hc_head",
        )

        self.norm = RMSNorm(config.dim, self.norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_to_kv_state: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        window_slot_mapping: torch.Tensor | None = None,
        compressor_slot_mapping: dict | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.config.hc_mult, 1)

        common_metadata = get_common_metadata()
        if common_metadata is not None:
            total_token_num = hidden_states.size(0)
            window_block_tables = torch.empty([total_token_num, self.window_size], dtype=torch.int32, device=hidden_states.device)
            window_context_lens = torch.empty([total_token_num], dtype=torch.int32, device=hidden_states.device)
            kv_cache_size = self.window_size + (self.max_model_len // self.compress_ratio if self.compress_ratio else 0)
            compress_block_tables = torch.empty([total_token_num, kv_cache_size], dtype=torch.int32, device=hidden_states.device)
            compress_context_lens = torch.empty([total_token_num], dtype=torch.int32, device=hidden_states.device)

            mlu_ops.get_window_block_tables(
                window_size = self.window_size,
                block_size = 1,
                seq_k_lens = common_metadata.seq_lens,
                query_start_loc = common_metadata.query_start_loc,
                block_table = common_metadata.block_table_tensor,
                window_block_tables = window_block_tables,
                window_context_lens = window_context_lens
            )

            # get_compress_block_tables
            query_start_loc = common_metadata.query_start_loc
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            compress_lens = query_lens // self.compress_ratio
            cu_compress_lens = torch.cat([
                    torch.tensor([0], dtype=compress_lens.dtype, device=compress_lens.device),
                    torch.cumsum(compress_lens, dim=0)
                ])
            offsets = cu_compress_lens[: -1] + total_token_num if common_metadata.is_prefill_only else torch.full_like(query_lens, self.window_size)

            mlu_ops.get_compress_block_tables(
                ratio = self.compress_ratio,
                block_size = 1,
                seq_k_lens = common_metadata.seq_lens,
                query_start_loc = common_metadata.query_start_loc,
                offset = offsets,
                block_table = common_metadata.block_table_tensor,
                compress_block_tables = compress_block_tables,
                compress_context_lens = compress_context_lens,
            )

            win_comp_block_tables = torch.empty([total_token_num, kv_cache_size], dtype=torch.int32, device=hidden_states.device)
            win_comp_context_lens = torch.empty([total_token_num], dtype=torch.int32, device=hidden_states.device)
            mlu_ops.concat_block_table(
                window_block_tables,
                window_context_lens,
                compress_block_tables,
                compress_context_lens,
                win_comp_block_tables,
                win_comp_context_lens,
            )

            window_compress_params = {
                "window_block_tables": window_block_tables,
                "window_context_lens": window_context_lens,
                "compress_block_tables": win_comp_block_tables,
                "compress_context_lens": win_comp_context_lens,
            }
        else:
            window_compress_params = None

        hc_attn_pre_norm = None
        for layer in self.layers:
            hidden_states, hc_attn_pre_norm = layer(
                positions,
                hidden_states,
                input_ids,
                None,
                batch_to_kv_state,
                window_compress_params,
                hc_attn_pre_norm=hc_attn_pre_norm,
                window_slot_mapping=window_slot_mapping,
                compressor_slot_mapping=compressor_slot_mapping,
            )
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states).to(dtype=torch.float)

        return hidden_states

class MLUDeepseekV4ForCausalLM(nn.Module, SupportsEagle):
    packed_modules_mapping = {
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = MLUDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.lm_head = ColumnParallelLinear(
            config.dim,
            config.vocab_size,
            params_dtype=torch.float32,
            quant_config=quant_config,
            bias=False,
            skip_bias_add=True,
            return_bias=False,
        )

    def update_forward_args(self, args, kwargs):
        window_size = self.config.window_size

        # Part 1. window slot mapping.
        common_metadata: MLUCommonAttentionMetadata = get_common_metadata()
        if common_metadata is None or common_metadata.block_table_tensor is None:
            window_slot_mapping = None
        elif common_metadata.is_prefill_only:
            block_table = common_metadata.block_table_tensor
            query_start_loc = common_metadata.query_start_loc
            window_slot_mapping = torch.empty([query_start_loc[-1]], dtype=torch.int32, device=block_table.device)
            window_slot_mapping.fill_(-1)
            for i, seq_len in enumerate(common_metadata.seq_lens):
                if seq_len < window_size:
                    window_slot_mapping[query_start_loc[i]: query_start_loc[i+1]].copy_(block_table[i, :seq_len])
                else:
                    # | <------- seqlen--------> |
                    # | other | window size      |
                    # | other | tail | head      |
                    # move head to the front of window, and move tail to the latter.
                    tail_pos = query_start_loc[i].item() + seq_len - window_size
                    head_size = seq_len % window_size
                    tail_size = window_size - head_size
                    window_slot_mapping[tail_pos: tail_pos + tail_size].copy_(
                        block_table[i, head_size:window_size],
                    )
                    window_slot_mapping[tail_pos + tail_size: tail_pos + window_size].copy_(
                        block_table[i, :head_size]
                    )
        else:
            block_table = common_metadata.block_table_tensor
            window_pos = (common_metadata.seq_lens - 1) % window_size
            window_slot_mapping = torch.gather(block_table, 1, window_pos.unsqueeze(1)).squeeze(1)

        kwargs["window_slot_mapping"] = window_slot_mapping

        # Part 2. compressor slot mapping
        assert set(self.config.compress_ratios) == {0, 4, 128}
        # The pairs <window_offset, compress_ratio> <128, 128> <128, 4> <0, 4> contain all cases.
        # <128, 128> and <128, 4> indicate attn.compressor, and
        # <0, 4> indicates attn.indexer.compressor.
        window_offsets = [128, 128, 0]
        compress_ratios = [128, 4, 4]
        # dict key: (window_size, compress_ratio)
        compressor_slot_mapping = dict()
        if common_metadata is None or common_metadata.block_table_tensor is None:
            pass
        elif common_metadata.is_prefill_only:
            block_tables = common_metadata.block_table_tensor
            query_start_loc = common_metadata.query_start_loc
            query_start_loc = common_metadata.query_start_loc
            query_lens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()

            for compress_ratio, window_offset in zip(compress_ratios, window_offsets):
                slot_lens = [q // compress_ratio for q in query_lens]
                cu_slot_lens = torch.cat([
                    torch.tensor([0], dtype=torch.int32, device='cpu'),
                    torch.cumsum(torch.tensor(slot_lens, dtype=torch.int32, device='cpu'), dim=0)],
                )
                slot_mapping = torch.empty(sum(slot_lens), dtype=torch.int32, device=block_table.device)
                for i in range(len(query_lens)):
                    slot_mapping[cu_slot_lens[i]: cu_slot_lens[i+1]] = \
                        block_tables[i, window_offset: window_offset + slot_lens[i]]
                compressor_slot_mapping[(window_offset, compress_ratio)] = slot_mapping
        else:
            block_tables = common_metadata.block_table_tensor
            seq_lens = common_metadata.seq_lens
            query_start_loc = common_metadata.query_start_loc
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            for compress_ratio, window_offset in zip(compress_ratios, window_offsets):
                offset = window_offset + (seq_lens - query_lens) // compress_ratio
                slot_mapping = torch.gather(block_tables, 1, offset.unsqueeze(1)).squeeze(1)
                compressor_slot_mapping[(window_offset, compress_ratio)] = slot_mapping

        kwargs["compressor_slot_mapping"] = compressor_slot_mapping

        return args, kwargs

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_to_kv_state: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        window_slot_mapping: torch.Tensor | None = None,
        compressor_slot_mapping: dict | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            batch_to_kv_state,
            inputs_embeds,
            window_slot_mapping,
            compressor_slot_mapping,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('w13', 'w1', 0),
            ('w13', 'w3', 1),
        ]

        for name, m in self.model.named_modules():
            if isinstance(m, SparseMoeMlp):
                m.pack_params()

        moe_group_info = MoeGroupInfo()
        moe_ep_size = moe_group_info.moe_ep_size
        moe_ep_rank = moe_group_info.moe_ep_rank
        num_total_experts = self.config.n_routed_experts
        start_expert_id = moe_ep_rank * ((num_total_experts + moe_ep_size - 1) // moe_ep_size)
        expert_num_per_rank = (num_total_experts + moe_ep_size - 1) // moe_ep_size
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            pattern = r'layers\.([0-9]*)\.'
            match = re.search(pattern, name)
            if match:
                layer_id = int(match.group(1))
                if layer_id >= self.config.n_layers:
                    continue

            # The following parameters are not included yet.
            skiped_parameters = ['mtp']
            if any(param in name for param in skiped_parameters):
                continue

            name = name.replace("embed.weight", "embed_tokens.weight")
            name = "model." + name
            name = name.replace("model.head.weight", "lm_head.weight")

            if "ffn.experts." in name:
                expert_id = int(name.split(".")[-3])
                if expert_id < start_expert_id or expert_id >= start_expert_id + ((num_total_experts + moe_ep_size - 1) // moe_ep_size):
                    continue
                new_expert_id = expert_id - start_expert_id
                name = name.replace(f"experts.{expert_id}", f"experts.{new_expert_id}")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if "w1.weight" not in name and \
                        "w3.weight" not in name:
                    continue

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # remap parameter name for hc pre
                name = name.replace("hc_attn_base", "hc_attn_pre.hc_base")
                name = name.replace("hc_attn_fn", "hc_attn_pre.hc_fn")
                name = name.replace("hc_attn_scale", "hc_attn_pre.hc_scale")
                name = name.replace("hc_ffn_base", "hc_ffn_pre.hc_base")
                name = name.replace("hc_ffn_fn", "hc_ffn_pre.hc_fn")
                name = name.replace("hc_ffn_scale", "hc_ffn_pre.hc_scale")

                # remap parameter name for hc head
                name = name.replace("hc_head_base", "hc_head.hc_head_base")
                name = name.replace("hc_head_fn", "hc_head.hc_head_fn")
                name = name.replace("hc_head_scale", "hc_head.hc_head_scale")

                name = name.replace("gate.tid2eid", "deepseekv4_topk.tid2eid")
                name = name.replace("ffn.gate.bias", "ffn.deepseekv4_topk.bias")

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if 'attn_sink' in name:
                    num_heads = self.config.n_heads
                    tp_size = get_tensor_model_parallel_world_size()
                    tp_rank = get_tensor_model_parallel_rank()
                    assert num_heads % tp_size == 0
                    num_local_heads = num_heads // tp_size
                    loaded_weight = loaded_weight[tp_rank * num_local_heads: (tp_rank + 1) * num_local_heads]

                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if diff := set(params_dict.keys()) - loaded_params:
            logger.error(f"The following params are not loaded: {diff}")

        for name, m in self.model.named_modules():
            if isinstance(m, SparseMoeMlp):
                m.pack_params_after_loading()


        return set(loaded_params)
