# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
import logging
from logging import Logger
from vllm_mlu._mlu_utils import *


def mlu_init_logger(name: str) -> Logger:
    """Initialize loggers for vllm_mlu module,
    and keep the configuration consistent with the vllm module"""
    mlu_logger = logging.getLogger(name)
    vllm_logger = logging.Logger.manager.loggerDict.get('vllm', None)
    if vllm_logger:
        mlu_logger.setLevel(vllm_logger.level)
        mlu_logger.propagate = vllm_logger.propagate
        mlu_logger.handlers = vllm_logger.handlers
    return mlu_logger


from vllm import logger
logger.init_logger = mlu_init_logger
from vllm.logger import init_logger

logger = init_logger(__name__)


# Apply v1 hijack
import vllm_mlu.v1.engine.core
import vllm_mlu.v1.engine.core_client
import vllm_mlu.v1.engine.llm_engine
import vllm_mlu.v1.core.sched.scheduler
import vllm_mlu.v1.core.single_type_kv_cache_manager
import vllm_mlu.v1.executor.abstract
import vllm_mlu.v1.sample.rejection_sampler
import vllm_mlu.v1.worker.lora_model_runner_mixin
import vllm_mlu.v1.worker.block_table
import vllm_mlu.compilation.fix_functionalization

# Apply common hijack
import vllm_mlu.config
import vllm_mlu.utils
import vllm_mlu.attention.layer
import vllm_mlu.distributed.parallel_state
import vllm_mlu.distributed.kv_transfer.kv_connector.factory
import vllm_mlu.engine.arg_utils
import vllm_mlu.entrypoints.llm
import vllm_mlu.executor.multiproc_worker_utils
import vllm_mlu.executor.executor_base
import vllm_mlu.executor.ray_distributed_executor
import vllm_mlu.lora.fully_sharded_layers
import vllm_mlu.lora.layers
import vllm_mlu.model_executor.parameter
import vllm_mlu.model_executor.guided_decoding.xgrammar_decoding
import vllm_mlu.model_executor.layers.linear
import vllm_mlu.model_executor.layers.rotary_embedding
import vllm_mlu.model_executor.layers.quantization.utils.w8a8_utils
import vllm_mlu.model_executor.layers.quantization.fp8
import vllm_mlu.model_executor.layers.activation
import vllm_mlu.model_executor.layers.layernorm
import vllm_mlu.model_executor.layers.fused_moe.layer
import vllm_mlu.model_executor.model_loader.tensorizer_loader
import vllm_mlu.model_executor.models.registry
import vllm_mlu.model_executor.models.deepseek_v3_2_exp
import vllm_mlu.worker.cache_engine
