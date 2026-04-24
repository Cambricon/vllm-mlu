# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Tuple
import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.v1.attention.backends.utils import (
    get_common_metadata,
    MLUCommonAttentionMetadata,
)
from vllm_mlu.v1.attention.backends.mla.flashmla import MLACommonMetadata
from vllm_mlu.model_executor.models.sp_utils import get_sp_forward_context

logger = init_logger(__name__)


@CustomOp.register("rotary_embedding_mlu")
class MLURotaryEmbedding(RotaryEmbedding, CustomOp):

    cu_seq_lens : torch.Tensor = None
    max_seq_len : int = None
    max_model_len : int = None
    is_prompt : bool = False
    is_chunked : bool = False
    positions_: torch.Tensor = None
    chunked_prefill_enabled: bool = False
    prefill_cu_seq_lens: torch.Tensor = None
    prefill_max_seq_len: int = None
    decode_cu_seq_lens: torch.Tensor = None
    decode_max_seq_len: int = None

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        inverse: bool = False,
    ) -> None:
        CustomOp.__init__(self)
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        # TODO(mgoin): disabled for now due to failures
        # Flashinfer only supports head_size=64, 128, 256, 512.
        # https://github.com/flashinfer-ai/flashinfer/blob/ebfd655efe830048dba5d582aaa61d61d1cf9a87/include/flashinfer/utils.cuh#L174-L202
        # self.use_flashinfer = (self.enabled()
        #                        and dtype in (torch.float16, torch.bfloat16)
        #                        and current_platform.is_cuda()
        #                        and has_flashinfer()
        #                        and self.head_size in [64, 128, 256, 512])
        self.use_flashinfer = False
        self.inverse = inverse

        # For vlm v1
        # 1. mlu rope run in eager mode
        # 2. all layer use layer0's rope to inference
        prefix = "global_rope"
        vllm_config = get_current_vllm_config()
        self.use_direct_call = False
        if not self.use_direct_call:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                pass
            else:
                compilation_config.static_forward_context[prefix] = self
            self.layer_name = prefix

        from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
        from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YaRNScalingRotaryEmbedding

        if MLURotaryEmbedding.max_seq_len != None \
                and self.max_position_embeddings < MLURotaryEmbedding.max_seq_len and \
                not isinstance(self, (YaRNScalingRotaryEmbedding, DeepseekScalingRotaryEmbedding)):
            logger.warning(f"User-specified max_model_len ({MLURotaryEmbedding.max_seq_len}) is different with " +
                            f"max_position_embedding ({max_position_embeddings}) from model's config.json, " +
                            f"This may lead to incorrect model outputs or MLU errors. " +
                            f"Make sure the value is correct and within the model context size. " +
                            f"Set max_position_embedding={MLURotaryEmbedding.max_seq_len}.")
            self.max_position_embeddings = MLURotaryEmbedding.max_seq_len
        cache = self._compute_cos_sin_cache()

        from vllm_mlu.model_executor.layers.rotary_embedding.linear_scaling_rope import MLULinearScalingRotaryEmbedding
        if isinstance(self, MLULinearScalingRotaryEmbedding):
            logger.debug(f"Using mlu defining _compute_cos_sin_cache due to the special tensor composition")
        elif is_neox_style:
            cache_pos = cache.shape[0]
            cache = cache.reshape(cache_pos, 2, -1)
            cache = torch.tile(cache, (1, 1, 2)).reshape(cache_pos, -1)
        else:
            cache = cache.repeat_interleave(2, dim=-1)

        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.cos_, self.sin_ = self._get_cos_sin()
    @classmethod
    def set_mlu_var_v1(
        cls,
        common_metadata: MLUCommonAttentionMetadata
    ) -> None:
        cls.unset_mlu_var()
        cls.cu_seq_lens = common_metadata.query_start_loc
        cls.max_seq_len = common_metadata.max_query_len
        cls.is_prompt = common_metadata.is_prefill_only
        cls.is_chunked = common_metadata.is_chunked

        # for MLA
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            _, attn_metadata = next(iter(attn_metadata.items()))
        if isinstance(attn_metadata, MLACommonMetadata):
            prefill_metadata = attn_metadata.prefill
            decode_metadata = attn_metadata.decode
            if prefill_metadata:
                cls.prefill_max_seq_len = prefill_metadata.max_query_len
                cls.prefill_cu_seq_lens = prefill_metadata.query_start_loc
            else:
                cls.prefill_max_seq_len = cls.max_seq_len
                cls.prefill_cu_seq_lens = cls.cu_seq_lens

            if decode_metadata:
                cls.decode_max_seq_len = decode_metadata.max_query_len
                cls.decode_cu_seq_lens = decode_metadata.query_start_loc
            else:
                cls.decode_max_seq_len = cls.max_seq_len
                cls.decode_cu_seq_lens = cls.cu_seq_lens

        # for sp
        sp_context = get_sp_forward_context()
        if sp_context is not None and sp_context.is_v32:
            prefill_metadata = sp_context.sp_attn_metadata.prefill
            cls.is_chunked = True
            cls.prefill_max_seq_len = prefill_metadata.max_query_len
            cls.prefill_cu_seq_lens = prefill_metadata.query_start_loc

    @classmethod
    def unset_mlu_var(cls):
        cls.cu_seq_lens = None
        cls.max_seq_len = None
        cls.is_prompt = False
        cls.is_chunked = False
        cls.positions_ = None
        cls.chunked_prefill_enabled = False
        cls.prefill_cu_seq_lens = None
        cls.prefill_max_seq_len = None
        cls.decode_cu_seq_lens = None
        cls.decode_max_seq_len = None

    def _get_cos_sin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
        sin = sin.view(-1, self.rotary_dim)
        cos = cos.view(-1, self.rotary_dim)
        return cos, sin

    def _get_positions_with_offsets_mlu(
        self,
        positions: torch.Tensor,
        offsets: torch.Tensor
    ) -> torch.Tensor:
        if offsets.numel() != positions.numel():
            raise Exception("rope offsets numel mismatch with positions, "
                            f"positions: {positions.numel()}, offsets: {offsets.numel()}")
        return (positions + offsets).to(torch.int32)

    def forward_impl(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_metadata: MLUCommonAttentionMetadata = get_common_metadata()
        if common_metadata is None:
            num_tokens, head_num, head_size = x.shape
            x = mlu_ops.rotary_embedding(
                x.view(1, num_tokens, head_num, head_size),
                self.sin_,
                self.cos_,
                positions,
                None,
                not self.is_neox_style,
                True,
                False,
                num_tokens
            )
            return x
        else:
            cu_seq_lens_ = common_metadata.query_start_loc

        if offsets is not None:
            if MLURotaryEmbedding.positions_ is None:
                MLURotaryEmbedding.positions_ = (
                    self._get_positions_with_offsets_mlu(positions, offsets))
            position_ids = MLURotaryEmbedding.positions_
            discrete = True
        elif MLURotaryEmbedding.is_chunked or not MLURotaryEmbedding.is_prompt:
            position_ids = positions
            discrete = True
        else:
            position_ids = None
            discrete = False

        x = mlu_ops.rotary_embedding(
            x,
            self.sin_,
            self.cos_,
            position_ids,
            cu_seq_lens_,
            not self.is_neox_style,
            discrete,
            False,
            MLURotaryEmbedding.max_seq_len
        )
        return x

    def get_param(self, positions, discrete=False):
        interleaved = True
        if self.is_neox_style:
            interleaved = False

        if discrete:
            position_ids = positions
            discrete = discrete
        else:
            if MLURotaryEmbedding.is_chunked or not MLURotaryEmbedding.is_prompt:
                position_ids = positions
                discrete = True
            else:
                position_ids = None
                discrete = False

        return position_ids, interleaved, discrete

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.outer(t, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        cos = freqs_cis.real
        sin = freqs_cis.imag * (-1 if self.inverse else 1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        only_prefill: bool | None = False,
        only_decode: bool | None = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.forward_impl(positions, query, offsets)
        if key is not None:
            self.forward_impl(positions, key, offsets)
        return query, key


def rope_forward(
    positions: torch.Tensor,
    x: torch.Tensor,
    layer_name: str,
    offsets: torch.Tensor | None = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    self.forward_impl(positions, x, offsets)


def rope_forward_fake(
    positions: torch.Tensor,
    x: torch.Tensor,
    layer_name: str,
    offsets: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="rope_forward",
    op_func=rope_forward,
    mutates_args=["x"],
    fake_impl=rope_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)
