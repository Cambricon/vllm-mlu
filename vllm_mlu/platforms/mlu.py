# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import contextlib
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import os
import torch

from vllm.logger import init_logger

import vllm.envs as envs
from vllm.platforms.interface import (
    DeviceCapability,
    Platform,
    PlatformEnum,
)

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu.logger import logger

if TYPE_CHECKING:
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object


envs.environment_variables.update({
    "MLU_VISIBLE_DEVICES":
    lambda: os.environ.get("MLU_VISIBLE_DEVICES", None)
})

logger = init_logger(__name__)


class MLUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "mlu"
    device_type: str = "mlu"
    dispatch_key: str = "MLU"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "MLU_VISIBLE_DEVICES"
    simple_compile_backend: str = "inductor"
    dist_backend: str = "cncl"

    supported_quantization: list[str] = ["weightonly", "smoothquant",
                                         "awq_mlu", "gptq_mlu", "fp8"]
    additional_env_vars: list[str] = ["VLLM_LATENCY_DEBUG",
                                      "VLLM_LATENCY_DEBUG_NO_DEVICE",
                                      "MLU_GRAPH_CAPTURE_LIST",
                                      "VLLM_LOGITS_USE_ALL_GATHER",
                                      "VLLM_V1_USE_FULL_GRAPH",
                                      "VLLM_MTP_FIXED_ACCEPTANCE_RATE"]

    @classmethod
    def import_kernels(cls) -> None:
        """Import any platform-specific C kernels."""
        try:
            import torch_mlu_ops
        except ImportError as e:
            logger.warning("Failed to import from torch_mlu_ops with %r", e)

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        from vllm_mlu.model_executor.layers.quantization import (
            register_real_mlu_quantization_methods
        )
        register_real_mlu_quantization_methods()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:

        if use_mla:
            logger.info(f"[MLU-V1][MLA] Select FlashMLABackend.")
            return "vllm_mlu.v1.attention.backends.mla.flashmla.FlashMLABackend"
        else:
            logger.info(f"[MLU-V1] Select FlashAttentionBackend.")
            return "vllm_mlu.v1.attention.backends.flash_attn.MLUFlashAttentionBackend"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        try:
            major, minor = torch.mlu.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.mlu.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.mlu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def set_device(cls, device: torch.device):
        torch.mlu.set_device(device)

    @classmethod
    def empty_cache(cls):
        torch.mlu.empty_cache()

    @classmethod
    def synchronize(cls):
        torch.mlu.synchronize()

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:
        return torch.mlu.mem_get_info()

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        speculative_config = vllm_config.speculative_config
        kv_transfer_config = vllm_config.kv_transfer_config
        mlu_config = vllm_config.mlu_config

        # Decode use full mlugraph: V1 mode + VLLM_V1_USE_FULL_GRAPH=true
        use_full_mlugraph = mlu_envs.VLLM_V1_USE_FULL_GRAPH

        # Check compilation config
        from vllm.config import CompilationMode, CUDAGraphMode
        logger.info(
            "[MLU] Force select CompilationMode.None, CUDAGraphMode.FULL_DECODE_ONLY."
        )
        compilation_config.level = None
        compilation_config.mode = CompilationMode.NONE
        compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

        # Dispatch worker
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_mlu.v1.worker.gpu_worker.MLUWorker"

        cls.simple_compile_backend = "inductor"
        # Activate custom ops for v1.
        compilation_config.custom_ops = ["all"]
        if compilation_config.splitting_ops is None:
            compilation_config.splitting_ops = []
        compilation_config.splitting_ops.extend(["vllm.rope_forward"])

        # FIXME: support cascade attention in VLLM-1710
        model_config = vllm_config.model_config
        if model_config:
            model_config.disable_cascade_attn = True

        # Select v1 scheduler type
        if scheduler_config:
            if not scheduler_config.async_scheduling:
                if (mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED
                        and not scheduler_config.enable_chunked_prefill):
                    vllm_config.scheduler_config.scheduler_cls = \
                    "vllm_mlu.v1.core.sched.scheduler.MLUUnchunkScheduler"
                    logger.info(f"[MLU-V1] Select UnchunkScheduler.")
                else:
                    vllm_config.scheduler_config.scheduler_cls = \
                    "vllm_mlu.v1.core.sched.scheduler.SchedulerWithProfiler"
                    logger.info(f"[MLU-V1] Select ChunkScheduler.")
            else:
                if (mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED
                        and not scheduler_config.enable_chunked_prefill):
                    vllm_config.scheduler_config.scheduler_cls = \
                    "vllm_mlu.v1.core.sched.async_scheduler.MLUUnchunkAsyncScheduler"
                    logger.info(f"[MLU-V1] Select UnchunkAsyncScheduler.")


        # Check cache config
        if cache_config:
            logger.info(
                f"[MLU] Select kv_cache_dtype={cache_config.cache_dtype}."
            )
            if cache_config.block_size is None:
                cache_config.block_size = 16

        # Check mla config
        if model_config and model_config.use_mla:
            if (mlu_config.is_dpsk_mcc_enabled or not use_full_mlugraph):
                scheduler_config.enable_chunked_prefill = False
                scheduler_config.chunked_prefill_enabled = False
                logger.warning(
                    "[MLA] Chunked prefill is disabled when deepseek mcc is enabled, "
                    "or not use full mlugraph.")

            if mlu_config.is_dpsk_mcc_enabled:
                cache_config.enable_prefix_caching = False
                logger.warning("[MLA] Prefix Caching is disabled when deepseek mcc is enabled.")


        # For mlu benchmark, we allow max_num_batched_tokens < max_model_len
        # in certain scenarios.
        if (
            model_config
            and scheduler_config.max_num_batched_tokens < model_config.max_model_len
            and not scheduler_config.chunked_prefill_enabled
        ):
            msg = f"max_num_batched_tokens ({scheduler_config.max_num_batched_tokens}) is " + \
                  f"smaller than max_model_len ({model_config.max_model_len}). " + \
                    "This effectively limits the maximum sequence length to " + \
                    "max_num_batched_tokens and makes vLLM reject longer " + \
                    "sequences. Please increase max_num_batched_tokens or " + \
                    "decrease max_model_len."
            if not mlu_envs.VLLM_V1_BENCHMARK:
                raise ValueError(msg)
            else:
                logger.warning(msg)

        if (mlu_config.dispatch_shared_expert_parallel
            and parallel_config.data_parallel_size <= 1
            and not mlu_config.prefill_use_sequence_parallel):
            mlu_config.dispatch_shared_expert_parallel = False
            logger.info(
                "Disabling `mlu_config.dispatch_shared_expert_parallel` when "
                "data_parallel_size == 1 or not using sequence parallel."
            )

        # Check kv_transfer config
        if kv_transfer_config:
            # Register mlu kv_connectors
            import vllm_mlu.distributed.kv_transfer.kv_connector.factory

    @classmethod
    def get_current_memory_usage(
        cls, device: Optional[torch.types.Device] = None
    ) -> float:
        torch.mlu.reset_peak_memory_stats(device)
        return torch.mlu.max_memory_allocated(device)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_mlu.lora.punica_wrapper.punica_mlu.PunicaWrapperMLU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_mlu.distributed.device_communicators.mlu_communicator.MLUCommunicator"

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm_mlu.compilation.mlu_graph.MLUGraphWrapper"

    @classmethod
    def can_update_inplace(cls) -> bool:
        """
        Checks if the platform allows inplace memory updates
        """
        return True

    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C
        with contextlib.suppress(ImportError):
            import vllm._moe_C  # noqa: F401

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """
        Returns if the graph mode is supported by the current platform.
        """
        return True
