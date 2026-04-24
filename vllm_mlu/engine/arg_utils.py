# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import get_args

from vllm.platforms import current_platform
from vllm.config import (
    ModelConfig,
    VllmConfig,
    SchedulerConfig,
)
from vllm.config.cache import CacheDType
from vllm.engine.arg_utils import (
    EngineArgs,
    _raise_unsupported_error,
)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


@classmethod
def vllm__engine__arg_utils__EngineArgs__get_chunked_prefill_prefix_caching_defaults(
    cls,
    model_config: ModelConfig,
) -> tuple[bool, bool]:
    if model_config.runner_type != "pooling":
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: mlu-v1 default use unchunked scheduler
        '''
        if mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED:
            default_chunked_prefill = False
        else:
            default_chunked_prefill = True
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Disable prefix caching default for hybrid models
        # since the feature is still experimental.
        default_prefix_caching = not model_config.is_hybrid
    else:
        assert model_config.pooler_config is not None

        pooling_type = model_config.pooler_config.pooling_type
        incremental_prefill_supported = (
            pooling_type is not None
            and pooling_type.lower() == "last"
            and getattr(model_config.hf_config, "is_causal", True)
        )

        default_chunked_prefill = incremental_prefill_supported
        default_prefix_caching = incremental_prefill_supported

    return default_chunked_prefill, default_prefix_caching

def vllm__engine__arg_utils__EngineArgs___set_default_args(
    self, usage_context: UsageContext, model_config: ModelConfig
) -> None:
    """Set Default Arguments for V1 Engine."""
    (
        default_chunked_prefill,
        default_prefix_caching,
    ) = self.get_chunked_prefill_prefix_caching_defaults(model_config)

    if self.enable_chunked_prefill is None:
        self.enable_chunked_prefill = default_chunked_prefill

        logger.debug(
            "%s chunked prefill by default",
            "Enabling" if default_chunked_prefill else "Disabling",
        )
    elif (
        model_config.runner_type == "pooling"
        and self.enable_chunked_prefill
        and not default_chunked_prefill
    ):
        logger.warning(
            "This model does not officially support chunked prefill. "
            "Enabling this manually may cause the engine to crash "
            "or produce incorrect outputs.",
        )

    if self.enable_prefix_caching is None:
        self.enable_prefix_caching = default_prefix_caching

        logger.debug(
            "%s prefix caching by default",
            "Enabling" if default_prefix_caching else "Disabling",
        )
    elif (
        model_config.runner_type == "pooling"
        and self.enable_prefix_caching
        and not default_prefix_caching
    ):
        logger.warning(
            "This model does not officially support prefix caching. "
            "Enabling this manually may cause the engine to crash "
            "or produce incorrect outputs.",
        )

    world_size = self.pipeline_parallel_size * self.tensor_parallel_size
    (
        default_max_num_batched_tokens,
        default_max_num_seqs,
    ) = self.get_batch_defaults(world_size)

    orig_max_num_batched_tokens = self.max_num_batched_tokens
    orig_max_num_seqs = self.max_num_seqs

    if self.max_num_seqs is None:
        self.max_num_seqs = default_max_num_seqs.get(
            usage_context,
            SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
        )

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: only set max_num_batched_tokens when enable chunked_prefill
    '''
    if self.max_num_batched_tokens is None:
        self.max_num_batched_tokens = default_max_num_batched_tokens.get(
            usage_context,
            SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
        )

    if orig_max_num_batched_tokens is None:
        if not self.enable_chunked_prefill:
            # If max_model_len is too short, use the default for higher throughput.
            self.max_num_batched_tokens = max(
                model_config.max_model_len,
                self.max_num_batched_tokens,
            )

        # When using default settings,
        # Ensure max_num_batched_tokens does not exceed model limit.
        # Some models (e.g., Whisper) have embeddings tied to max length.
        self.max_num_batched_tokens = min(
            self.max_num_seqs * model_config.max_model_len,
            self.max_num_batched_tokens,
        )

        logger.debug(
            "Defaulting max_num_batched_tokens to %d for %s usage context.",
            self.max_num_batched_tokens,
            usage_context.value if usage_context else None,
        )

    if orig_max_num_seqs is None:
        if self.max_num_batched_tokens is not None: # For type checking
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

        logger.debug(
            "Defaulting max_num_seqs to %d for %s usage context.",
            self.max_num_seqs,
            usage_context.value if usage_context else None,
        )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


_VALID_QUANT_ATTN_QKV_DTYPE = ['int8', 'fp8', 'fp8_e4m3']

def vllm__engine__arg_utils__EngineArgs__create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """
    Create the VllmConfig.

    NOTE: If VllmConfig is incompatible, we raise an error.
    """
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add data parallel params to parallel config.
    '''
    if self.mlu_config and "decoder_attn_dtype" in self.mlu_config:
        if self.mlu_config.get("decoder_attn_dtype") in ["int8", "fp8", "fp8_e4m3"]:
            self.kv_cache_dtype = self.mlu_config.get("decoder_attn_dtype")

    engine_config = vllm__engine__arg_utils__EngineArgs__create_engine_config_org(
        self, usage_context, headless)

    world_size = engine_config.parallel_config.world_size_across_dp
    tensor_parallel_size = engine_config.parallel_config.tensor_parallel_size
    embedding_tp_size = engine_config.mlu_config.layer_embedding_logit_tp_size
    if embedding_tp_size:
        assert embedding_tp_size >= tensor_parallel_size and embedding_tp_size <= world_size, (
            f"embedding_tp_size = {embedding_tp_size} out of bounds. "
            f"Require {tensor_parallel_size} ≤ size ≤ {world_size}")
    dense_mlp_tp_size = engine_config.mlu_config.layer_dense_mlp_tp_size
    if dense_mlp_tp_size:
        assert dense_mlp_tp_size >= 1 and dense_mlp_tp_size <= world_size, (
            f"dense_mlp_tp_size = {dense_mlp_tp_size} out of bounds. Require 1 ≤ size ≤ {world_size}")
        if dense_mlp_tp_size != world_size:
            assert not engine_config.mlu_config.is_dpsk_mcc_enabled, (
                "dense_mlp_tp_size is not supported when dpsk mcc is enabled.")
        if engine_config.model_config.is_longcat_flash and tensor_parallel_size > 1:
            raise ValueError("For now, for longcat model, custom dense mlp tp split in data parallel requires dpXtp1. "
                             "Necessity of this constraint requires further investigation.")
        if engine_config.model_config.is_longcat_flash and dense_mlp_tp_size < tensor_parallel_size:
            raise ValueError(f"For longcat model, custom dense mlp tp_size {dense_mlp_tp_size} "
                             f"must be greater than or equal to tensor_parallel_size {tensor_parallel_size}")
        if engine_config.model_config.is_deepseek_mla and dense_mlp_tp_size % tensor_parallel_size != 0:
            raise ValueError(f"For deepseek mla model, custom mlp tp size {dense_mlp_tp_size} must "
                             f"be divisible by {tensor_parallel_size}")

    if ((engine_config.parallel_config.data_parallel_size > 1 or engine_config.speculative_config is not None
         or engine_config.mlu_config.prefill_use_sequence_parallel) and engine_config.mlu_config.prefill_enable_mlugraph):
        logger.info("Data parallel or sequence parallel or speculative is enabled, forcing context mlugraph to be disabled.")
        engine_config.mlu_config.prefill_enable_mlugraph = False
    if engine_config.mlu_config.decoder_attn_dtype:
        if engine_config.mlu_config.decoder_attn_dtype not in get_args(CacheDType):
            raise ValueError(f"MLU backend does not support {engine_config.mlu_config.decoder_attn_dtype} "
                            f"decoder_attn_dtype for now")
        is_glm4_moe = (hasattr(engine_config.model_config.hf_text_config, "model_type") and
                       engine_config.model_config.hf_text_config.model_type == "glm4_moe")
        if (not (engine_config.model_config.is_deepseek_mla or is_glm4_moe)
              and engine_config.mlu_config.decoder_attn_dtype != "auto"):
            raise ValueError(f"mlu_config.decoder_attn_dtype only support deepseek_mla and glm4_moe model")

    # sequence parallel checks
    if (engine_config.mlu_config.prefill_use_sequence_parallel
        and engine_config.model_config.hf_text_config.model_type not in ["deepseek_v32", "deepseek_v3"]):
        raise ValueError("Prefill sequence parallel can only use in deepseek model.")
    if engine_config.mlu_config.prefill_use_sequence_parallel and engine_config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Prefill sequence parallel can not use with chunked prefill for now.")
    if engine_config.mlu_config.prefill_use_sequence_parallel and engine_config.mlu_config.is_dpsk_mcc_enabled:
        raise ValueError("Prefill sequence parallel can not use with mcc.")
    if engine_config.mlu_config.prefill_use_sequence_parallel and engine_config.parallel_config.data_parallel_size > 1:
        raise ValueError("Prefill sequence parallel can not use with data parallel.")
    if (engine_config.mlu_config.prefill_use_sequence_parallel
        and engine_config.model_config.hf_text_config.model_type == "deepseek_v3"
        and engine_config.quant_config.get_name() != "SmoothQuant"):
        raise ValueError("Prefill sequence parallel can only use SmoothQuant for deepseek_v3.")

    # disagg constraint
    # 1、only support deepseek-v3/r1
    # 2、unsupport kv8
    if self.kv_transfer_config is not None:
        if engine_config.model_config.hf_config.model_type != "deepseek_v3":
            raise ValueError("Disagg only support DeepDeek-V3/R1")
        if engine_config.cache_config.cache_dtype == "int8":
            raise ValueError("Disagg does not support KV cache dtype is int8")
        if engine_config.cache_config.enable_prefix_caching:
            raise ValueError("Disagg does not support prefix caching")

        if isinstance(self.kv_transfer_config, dict):
            kv_connector = self.kv_transfer_config.get("kv_connector")
            kv_role = self.kv_transfer_config.get("kv_role")
        else:
            kv_connector = self.kv_transfer_config.kv_connector
            kv_role = self.kv_transfer_config.kv_role

        if kv_connector != "LMCacheConnectorV1":
            raise ValueError("Disagg only support LMCacheConnectorV1 connector")
        if kv_role == "kv_consumer":
            if not self.enable_chunked_prefill:
                raise ValueError("Disagg decoder only support chunk scheduler")

    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return engine_config


MluHijackObject.apply_hijack(EngineArgs,
                             EngineArgs._set_default_args,
                             vllm__engine__arg_utils__EngineArgs___set_default_args)
MluHijackObject.apply_hijack(EngineArgs,
                             EngineArgs.create_engine_config,
                             vllm__engine__arg_utils__EngineArgs__create_engine_config)
MluHijackObject.apply_hijack(EngineArgs,
                             EngineArgs.get_chunked_prefill_prefix_caching_defaults,
                             vllm__engine__arg_utils__EngineArgs__get_chunked_prefill_prefix_caching_defaults)
