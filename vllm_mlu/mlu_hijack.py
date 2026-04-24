# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import importlib.util
from vllm_mlu._mlu_utils import *
from vllm_mlu.logger import logger


def is_module_available(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def check_environ_compatibility():
    if is_module_available('apex'):
        logger.error(f"The `apex` package is currently present in your environment, "
                     f"which may cause model accuracy issues or other problems. It is "
                     f"strongly recommended that you uninstall it before using vLLM.")

# Check environment compatibility first before applying mlu hijack.
check_environ_compatibility()

logger.info(f"[MLU] Apply Monkey Patch.")

# Apply v1 hijack
import vllm_mlu.v1.engine.core
import vllm_mlu.v1.engine.core_client
import vllm_mlu.v1.engine.llm_engine
import vllm_mlu.v1.engine.async_llm
import vllm_mlu.v1.core.sched.scheduler
import vllm_mlu.v1.core.single_type_kv_cache_manager
import vllm_mlu.v1.core.kv_cache_utils
import vllm_mlu.v1.core.kv_cache_manager
import vllm_mlu.v1.executor.abstract
import vllm_mlu.v1.executor.ray_executor
import vllm_mlu.v1.executor.multiproc_executor
import vllm_mlu.v1.sample.rejection_sampler
import vllm_mlu.v1.worker.lora_model_runner_mixin
import vllm_mlu.v1.worker.block_table
import vllm_mlu.v1.worker.gpu_input_batch
import vllm_mlu.v1.worker.kv_connector_model_runner_mixin
import vllm_mlu.v1.attention.backends.gdn_attn
import vllm_mlu.v1.attention.backends.mla.flashmla
import vllm_mlu.compilation.fix_functionalization

# Apply common hijack
import vllm_mlu.attention.layer
import vllm_mlu.benchmarks.datasets
import vllm_mlu.config.model
import vllm_mlu.config.scheduler
import vllm_mlu.config.speculative
import vllm_mlu.config.vllm
import vllm_mlu.utils
import vllm_mlu.distributed.parallel_state
import vllm_mlu.distributed.kv_transfer.kv_connector.factory
import vllm_mlu.engine.arg_utils
import vllm_mlu.entrypoints.llm
import vllm_mlu.lora.layers.base_linear
import vllm_mlu.lora.layers.row_parallel_linear
import vllm_mlu.lora.layers.column_parallel_linear
import vllm_mlu.model_executor.parameter
import vllm_mlu.model_executor.layers.linear
import vllm_mlu.model_executor.layers.rotary_embedding
import vllm_mlu.model_executor.layers.quantization.utils.w8a8_utils
import vllm_mlu.model_executor.layers.quantization.fp8
import vllm_mlu.model_executor.layers.activation
import vllm_mlu.model_executor.layers.layernorm
import vllm_mlu.model_executor.layers.fused_moe.layer
import vllm_mlu.model_executor.model_loader.tensorizer_loader
import vllm_mlu.model_executor.models.registry
import vllm_mlu.model_executor.models.config
import vllm_mlu.multimodal.utils
if is_module_available('lmcache'):
    import vllm_mlu.distributed.kv_transfer.kv_connector.v1.lmcache_connector

if VLLM_CI_ACCURACY_TEST:
    import vllm_mlu.model_executor.model_loader.dummy_loader

if VLLM_SCHEDULER_PROFILE:
    import vllm_mlu.entrypoints.openai.api_server
