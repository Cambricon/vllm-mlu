# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
import re
import types
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.backends.abstract import is_quantized_kv_cache
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention,
    DeepseekV2DecoderLayer,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2ForCausalLM,
    get_spec_layer_idx_from_weight_name,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.model_executor.layers.feed_forward import FeedForward
from vllm_mlu.model_executor.layers.indexer import Indexer
from vllm_mlu.model_executor.layers.sparse_moe_mlp import (
    SparseMoeMlp,
    MoeGroupInfo,
)
from vllm_mlu.v1.attention.backends.mla.common import MLACommonMetadata
from vllm_mlu.v1.attention.backends.utils import COMMON_METADATA_STR


def get_default_device() -> torch.device:
    return torch.empty([]).device


class MLUDeepseekV2MoE(SparseMoeMlp):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            up_proj_name="gate_up_proj",
            is_gated=True,
            down_proj_name="down_proj",
            has_bias=False,
            skip_bias_add=False,
            renormalize=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            params_dtype=None,
            quant_config=quant_config,
            is_use_fused_moe=True,
            expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            topk_method=config.topk_method,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        self.config = config
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.moe_tp_size > config.n_routed_experts:
            raise ValueError(
                f"Moe Tensor parallel size {self.moe_tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts = FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                up_proj_name='gate_up_proj',
                is_gated=True,
                down_proj_name='down_proj',
                bias=False,
                quant_config=quant_config,
                reduce_results=False,
            )

    def forward_compute(
        self,
        hidden_states: torch.Tensor,
        only_compute_routed: bool = False,
        use_tp_weight: bool = False,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None and not only_compute_routed:
            assert shared_output is None
            shared_output = self.shared_experts(
                hidden_states, use_tp_weight=use_tp_weight)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.forward_experts(
            hidden_states, router_logits, shared_output=shared_output)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_communication(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.reduce_results(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        only_compute_routed: bool = False,
        use_tp_weight: bool = False,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.forward_compute(
            hidden_states,
            only_compute_routed=only_compute_routed,
            use_tp_weight=use_tp_weight,
            shared_output=shared_output,
        )

        hidden_states = self.forward_communication(hidden_states)
        return hidden_states


class MLUDeepseekV2MLAAttention(DeepseekV2MLAAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(DeepseekV2MLAAttention, self).__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.attn_tensor_parallel_size = get_tensor_model_parallel_world_size()
        tp_size = self.attn_tensor_parallel_size
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # 1) skip q_a_proj, kv_a_proj_with_mqa, kv_b_proj weight quant,
        # split kv_b_proj weight if not is_fp8_block_wise
        # 2) do all reduce outside when use data parallel
        # 3) add self.quant_config
        assert envs.VLLM_USE_V1, "DeepseekV2MLAAttention only support use_v1=True"
        self.quant_config = quant_config

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj")
        kv_b_proj_weight = self.kv_b_proj.weight
        w_kc, w_vc = kv_b_proj_weight.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
            ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc = w_kc
        self.w_vc = w_vc
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.use_normal_rope = not rope_scaling
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            v_head_dim=self.v_head_dim,
        )
        self.attn_decoder = Attention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mla_attn",
            use_mla=True,
            v_head_dim=self.kv_lora_rank,
            use_fused_mla_qkv=False,
        )
        self.prefix = prefix
        self.pack_params_done = False
        self.pack_params_after_loading_done = False

        self.indexer = Indexer(
            dim=self.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            index_topk=config.index_topk,
            q_lora_rank=self.q_lora_rank,
            rotary_emb=self.rotary_emb,
            prefix=prefix,
        )

    def forward_mla_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_cache: List[torch.Tensor],
    ) -> torch.Tensor:
        num_decode_tokens = attn_metadata.num_decode_tokens
        if attn_metadata.prefill:
            attn_metadata.prefill.compute_dtype = torch.float32
            prefill_positions = positions[num_decode_tokens:, ...]
            prefill_hidden_states = hidden_states[num_decode_tokens:, ...]
            prefill_output = self.forward_prefill(
                prefill_positions,
                prefill_hidden_states,
                kv_cache,
                attn_metadata,
            )
        decode_output = None
        if attn_metadata.decode:
            attn_metadata.decode.compute_dtype = torch.float32
            decode_positions = positions[:num_decode_tokens, ...]
            decode_hidden_states = hidden_states[:num_decode_tokens, ...]
            decode_output = self.forward_decoder(
                decode_positions,
                decode_hidden_states,
                kv_cache,
                attn_metadata,
            )

        if attn_metadata.prefill is not None and attn_metadata.decode is not None:
            return torch.cat([decode_output, prefill_output], dim=0)
        elif attn_metadata.prefill is not None:
            return prefill_output

        return decode_output

    def forward_decoder(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input_dtype = hidden_states.dtype
        q_input = torch.empty(
            q_len,
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            device=hidden_states.device,
            dtype=q_input_dtype,
        )
        q = self.q_a_proj(hidden_states)[0]
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q = self.q_a_layernorm(q)
        qr = q
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        torch.bmm(q_nope.transpose(0, 1), self.w_kc, out=q_input[..., :self.kv_lora_rank].transpose(0, 1))
        q_pe, _ = self.rotary_emb(positions, q_pe, None, only_decode=True)
        q_input[..., self.kv_lora_rank:] = q_pe

        forward_decoder_fused_mla_kv(self, latent_cache, positions, kv_cache, attn_metadata)

        k_input = latent_cache
        k_input = k_input.unsqueeze(1)

        v_input = latent_cache[..., : self.kv_lora_rank]

        q_input = q_input.reshape(q_input.shape[0], -1)
        k_input = k_input.reshape(k_input.shape[0], -1)
        v_input = v_input.reshape(v_input.shape[0], -1)

        attn_bias = self.indexer(
            hidden_states,
            qr,
            positions=positions,
            k_cache=kv_cache[2],
            mask=None,
            attn_metadata=attn_metadata,
        )
        decode_kwargs = {"only_decode": True, "q_quant_scale": None, "attn_bias": attn_bias}
        attn_output = self.attn_decoder(q_input, k_input, v_input, kwargs=decode_kwargs)
        attn_output = attn_output.reshape(-1, self.num_local_heads,
                                          self.kv_lora_rank)
        attn_bmm_output = torch.empty(
            q_len, self.num_local_heads, self.v_head_dim, device=attn_output.device, dtype=attn_output.dtype)
        torch.bmm(attn_output.transpose(0, 1), self.w_vc.transpose(1, 2), out=attn_bmm_output.transpose(0, 1))
        attn_output = attn_bmm_output.flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output


    def forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        qr = q
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
                                        self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        # Update rotary status.
        if isinstance(attn_metadata, dict):
            layer_metadata = next((v for k, v in attn_metadata.items() if k != COMMON_METADATA_STR), None)
        else:
            layer_metadata = attn_metadata
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe, only_prefill=True)

        # MLA save cache before flashattn
        if len(kv_cache) != 0  and kv_cache[0].numel() > 0:
            key_cache = kv_cache[0][0]
            key_value = torch.concat((kv_a.unsqueeze(1), k_pe), dim=-1)
            if isinstance(attn_metadata, MLACommonMetadata):
                slot_mapping = attn_metadata.slot_mapping[attn_metadata.num_decode_tokens:]
            else:
                slot_mapping = attn_metadata.slot_mapping
            updated_slot_mapping = slot_mapping[:key_value.size(0)]
            mlu_ops.reshape_paged_cache(
                key_value,
                None,
                key_cache,
                None,
                updated_slot_mapping.flatten(),
            )

        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        q = q.reshape(-1, self.num_local_heads * self.qk_head_dim)
        k = k.reshape(-1, self.num_local_heads * self.qk_head_dim)
        v = v.reshape(-1, self.num_local_heads * self.v_head_dim)

        prefill_metadata = layer_metadata.prefill
        seqlen = prefill_metadata.max_query_len
        batch = prefill_metadata.query_start_loc.shape[0] - 1
        mask = torch.full((seqlen, seqlen), float("-inf"), device=q.device).triu_(1)
        attn_bias = self.indexer(
            hidden_states,
            qr,
            positions=positions,
            k_cache=kv_cache[2],
            mask=mask,
            attn_metadata=attn_metadata,
        )

        cu_seq_lens_q = prefill_metadata.query_start_loc
        max_seq_len_q = prefill_metadata.max_query_len
        max_seq_len_kv = prefill_metadata.max_query_len
        cu_seqlens_kv = cu_seq_lens_q

        prefill_kwargs = {
            "only_prefill": True,
            "prefill_causal": True,
            "cu_seq_lens_q": cu_seq_lens_q,
            "cu_seq_lens_kv": cu_seqlens_kv,
            "max_seq_len_q": max_seq_len_q,
            "max_seq_len_kv": max_seq_len_kv,
            "return_lse": prefill_metadata.context_chunk_seq_tot is not None,
            "attn_bias": attn_bias,
        }
        attn_output = self.attn(q, k, v, kwargs=prefill_kwargs)
        # chunked prefill
        assert prefill_metadata.context_chunk_seq_tot is None, "We do not supoort chunk prefill yet."
        if isinstance(attn_output, (list, tuple)):
            attn_output = attn_output[0]

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # use normal computation for prefill and use weight absorption for extend/decode.
        # pack_params() is called for dummy model
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return torch.empty_like(hidden_states)

        # self.attn and self.attn_decoder always have the same attn_metadata
        # and share the same kv cache for each layer
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.attn.layer_name]
        kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
        self.pack_params()
        self.pack_params_after_loading()

        return self.forward_mla_attn(
            positions,
            hidden_states,
            attn_metadata,
            kv_cache,
        )

    def pack_params(self):
        if self.pack_params_done:
            return

        self.pack_params_done = True

    def pack_params_after_loading(self):
        if self.pack_params_after_loading_done:
            return

        self.pack_params_after_loading_done = True


class MLUDeepseekV2DecoderLayer(DeepseekV2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super(DeepseekV2DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.vllm_config = vllm_config
        self.torch_dtype = model_config.dtype
        self.quant_config = quant_config

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                        8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.layer_idx = int(prefix.split(sep='.')[-1])
        if model_config.use_mla:
            attn_cls = MLUDeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0):
            self.mlp = MLUDeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                up_proj_name='gate_up_proj',
                is_gated=True,
                down_proj_name='down_proj',
                bias=False,
                reduce_results=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.mlp.forward_compute = self.mlp.forward
            self.mlp.forward_communication = types.MethodType(
                lambda self, x: tensor_model_parallel_all_reduce(x),
                self.mlp
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Add tensor model group all reduce here.
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp.forward_compute(hidden_states)
        hidden_states = self.mlp.forward_communication(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MLUDeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        assert vllm_config.model_config.use_mla, "attn data parallel for deepseek must enable mla"
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MLUDeepseekV2DecoderLayer(
                config,
                prefix,
                vllm_config=vllm_config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def update_vllm_config(vllm_config: VllmConfig):
    vllm_config.model_config.hf_config.index_n_heads = 64
    vllm_config.model_config.hf_config.index_head_dim = 128
    vllm_config.model_config.hf_config.index_topk = 2048

class MLUDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = ""):
        update_vllm_config(vllm_config)
        super(DeepseekV2ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        assert not isinstance(self.quant_config, Fp8Config), "We do not support fp8 yet."

        self.vllm_config = vllm_config
        self.model = MLUDeepseekV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        # pack params and calculate start expert id
        for name, m in self.model.named_modules():
            if isinstance(m, SparseMoeMlp) or isinstance(m, MLUDeepseekV2MLAAttention):
                m.pack_params()

        moe_group_info = MoeGroupInfo()
        moe_ep_size = moe_group_info.moe_ep_size
        moe_ep_rank = moe_group_info.moe_ep_rank
        num_total_experts = self.config.n_routed_experts
        start_expert_id = moe_ep_rank * ((num_total_experts + moe_ep_size - 1) // moe_ep_size)

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = {}
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            pattern = r'layers\.([0-9]*)\.'
            match = re.search(pattern, name)
            if match:
                layer_id = int(match.group(1))
                if layer_id >= self.config.num_hidden_layers:
                    continue
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # replace expert_id in weight to named_expert_id in params_dict
            if start_expert_id > 0 and "mlp.experts." in name:
                expert_str = re.search(r'experts\.\d+', name).group(0)
                expert_id = int(expert_str.split(".")[1])
                named_expert_id = expert_id - start_expert_id
                if named_expert_id < 0:
                    continue
                old_expert_name = f"experts.{expert_id}"
                new_expert_name = f"experts.{named_expert_id}"
                name = name.replace(old_expert_name, new_expert_name)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.

                # add expert skiped condition and delete useless if name not in params_dict: continue condition
                name = name.replace(weight_name, param_name)
                if (("mlp.experts." in name or "mlp.shared_experts." in name or "mlp.shared_expert_gate." in name)
                        and name not in params_dict):
                    continue

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id)
                    break
                else:
                    # add expert skiped condition
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if (("mlp.experts." in name or "mlp.shared_experts." in name or "mlp.shared_expert_gate." in name)
                            and name not in params_dict):
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # pack params after loading
        for name, m in self.model.named_modules():
            if isinstance(m, SparseMoeMlp) or isinstance(m, MLUDeepseekV2MLAAttention):
                m.pack_params_after_loading()

        return loaded_params
