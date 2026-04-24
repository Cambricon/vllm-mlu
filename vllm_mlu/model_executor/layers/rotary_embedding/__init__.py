# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import math
from typing import Any

import torch

from vllm.logger import init_logger
import vllm.model_executor.layers.rotary_embedding as rotary_embedding
from vllm.model_executor.layers.rotary_embedding import (
    _ROPE_DICT,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding import (
    _ROPE_DICT,
    DualChunkRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Llama4VisionRotaryEmbedding,
    MRotaryEmbedding,
    NTKScalingRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)

from .base import MLURotaryEmbedding
from .deepseek_scaling_rope import MLUDeepseekScalingRotaryEmbedding
from .dynamic_ntk_alpha_rope import MLUDynamicNTKAlphaRotaryEmbedding
from .dynamic_ntk_scaling_rope import MLUDynamicNTKScalingRotaryEmbedding
from .linear_scaling_rope import MLULinearScalingRotaryEmbedding
from .llama3_rope import MLULlama3RotaryEmbedding
from .mrope import MLUMRotaryEmbedding
from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


def get_long_max_model_max_position_emb(max_position_embeddings, scaling_factor):
    if MLURotaryEmbedding.max_seq_len != None and \
            MLURotaryEmbedding.max_seq_len > max_position_embeddings * scaling_factor:
        logger.warning(f"User-specified max_model_len ({MLURotaryEmbedding.max_seq_len}) is different with " +
                        f"max_position_embedding ({max_position_embeddings}) * scaling_factor ({scaling_factor}) " +
                        "from model's config.json, This may lead to incorrect model outputs or MLU errors. " +
                        f"Make sure the value is correct and within the model context size. " +
                        f"Set max_position_embedding={MLURotaryEmbedding.max_seq_len}.")
        return math.ceil(MLURotaryEmbedding.max_seq_len / scaling_factor)
    return max_position_embeddings

def vllm__model_executor__layers__rotary_embedding__get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: dict[str, Any] | None = None,
    inverse: bool = False
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dual_chunk_attention_args,
        dtype,
        inverse,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }
        rotary_emb = DualChunkRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            **extra_kwargs,
        )
    elif not rope_scaling:
        rotary_emb = MLURotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype,
            inverse=inverse,
        )
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = MLULlama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "mllama4":
            rotary_emb = Llama4VisionRotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style, dtype
            )
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MLUMRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                )
            else:
                rotary_emb = MLURotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    inverse=inverse,
                )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = MLULinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "ntk":
            scaling_factor = rope_scaling["factor"]
            mixed_b = rope_scaling.get('mixed_b', None)
            rotary_emb = NTKScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                mixed_b,
            )
        elif scaling_type == "dynamic":
            if "alpha" in rope_scaling:
                scaling_alpha = rope_scaling["alpha"]
                rotary_emb = MLUDynamicNTKAlphaRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_alpha,
                    dtype,
                )
            elif "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
                rotary_emb = MLUDynamicNTKScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
            else:
                raise ValueError(
                    "Dynamic rope scaling must contain either 'alpha' or 'factor' field"
                )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "apply_yarn_scaling",
                )
            }
            if "mrope_section" in rope_scaling:
                extra_kwargs.pop("apply_yarn_scaling", None)
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                    scaling_factor=scaling_factor,
                    **extra_kwargs,
                )
            else:
                '''
                =============================
                Modify by vllm_mlu
                =============================
                @brief: update original_max_position
                '''
                original_max_position = get_long_max_model_max_position_emb(
                    original_max_position, scaling_factor,
                )
                '''
                ==================
                End of MLU Hijack
                ==================
                '''
                rotary_emb = YaRNScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                    **extra_kwargs,
                )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: update original_max_position
            '''
            original_max_position = get_long_max_model_max_position_emb(
                original_max_position, scaling_factor,
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            rotary_emb = MLUDeepseekScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                inverse,
                **extra_kwargs,
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                short_factor,
                long_factor,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

MluHijackObject.apply_hijack(
    rotary_embedding,
    rotary_embedding.get_rope,
    vllm__model_executor__layers__rotary_embedding__get_rope,
)
