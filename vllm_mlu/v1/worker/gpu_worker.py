# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import copy
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group, get_pp_group
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          has_kv_transfer_group)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.utils.mem_constants import GiB_bytes

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm_mlu.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm_mlu.profiler.mlu_profiler import MluProfilerWrapper
from vllm_mlu.utils import MemorySnapshot, memory_profiling
from vllm_mlu._mlu_utils import VLLM_DUMP_MLU_INFO_EN
from vllm_mlu.device_allocator.cnmem import CnMemAllocator
from vllm_mlu.v1.worker.mlu_quant import MLUWorkerQuant
from vllm_mlu.v1.worker.gpu_model_runner import MLUModelRunner
from vllm_mlu.v1.worker.dp_gpu_model_runner import DPMLUModelRunner

logger = init_logger(__name__)


class MLUWorker(Worker, MLUWorkerQuant):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        WorkerBase.__init__(self, vllm_config=vllm_config,
                            local_rank=local_rank,
                            rank=rank,
                            distributed_init_method=distributed_init_method,
                            is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils.import_utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, worker_name=worker_name, use_gzip=True
                ),
            )
        elif envs.VLLM_TORCH_CUDA_PROFILE:
            self.profiler = MluProfilerWrapper()
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.mlu.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CnMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.mlu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CnMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CnMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def init_device(self):
        if self.device_config.device.type == "mlu":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("CNCL_ASYNC_ERROR_HANDLING", None)
            # if (
            #     self.parallel_config.data_parallel_size > 1
            #     and self.parallel_config.data_parallel_size_local > 0
            #     and self.parallel_config.distributed_executor_backend
            #     not in ["ray", "external_launcher"]
            #     and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            # ):
            #     # Use local DP rank if available, otherwise use global DP rank.
            #     dp_local_rank = self.parallel_config.data_parallel_rank_local
            #     if dp_local_rank is None:
            #         dp_local_rank = self.parallel_config.data_parallel_rank

            #     tp_pp_world_size = (
            #         self.parallel_config.pipeline_parallel_size
            #         * self.parallel_config.tensor_parallel_size
            #     )

            #     # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
            #     self.local_rank += dp_local_rank * tp_pp_world_size
            #     assert self.local_rank < torch.mlu.device_count(), (
            #         f"DP adjusted local rank {self.local_rank} is out of bounds. "
            #     )

            self.device = torch.device(f"mlu:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            gc.collect()
            torch.mlu.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        model_runner_cls = (DPMLUModelRunner
                            if self._enable_moe_dp_opt() else MLUModelRunner)
        self.model_runner: MLUModelRunner = model_runner_cls(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much 
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        GiB = lambda b: b / GiB_bytes
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {GiB(self.init_snapshot.free_memory):.2f} "
                f"GiB, reserved {GiB(kv_cache_memory_bytes):.2f} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        torch.mlu.empty_cache()
        torch.mlu.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        GiB = lambda b: b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        gc.collect()

        self.peak_memory = profile_result.non_kv_cache_memory
        self.block_memory = self.available_kv_cache_memory_bytes


        return int(self.available_kv_cache_memory_bytes)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CnMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes
                if x not in self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.
            GiB = lambda b: round(b / GiB_bytes, 2)

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). "
                f"Actual usage is {GiB(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {GiB(self.peak_activation_memory)} GiB "
                f"for peak activation, {GiB(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {GiB(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{GiB(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self.model_runner._get_num_input_tokens(num_scheduled_tokens)
        all_gather_tensors = {
            "residual": not is_residual_scattered_for_sp(
                self.vllm_config, num_input_tokens
            )
        }
        if forward_pass and not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, NoneType)):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def _enable_moe_dp_opt(self):
        '''
        We will enable the MLU-optimized DP scheme for the specified MoE models,
        otherwise the native DP implementation will be used.
        '''
        # case0 enable data parallel
        enable_dp = self.parallel_config.data_parallel_size > 1
        # case1 ds mla
        is_ds_mla = self.model_config.is_deepseek_mla
        # case2 qwen3 moe
        is_supported_moe_model = hasattr(self.model_config.hf_text_config, "model_type") and \
            self.model_config.hf_text_config.model_type in ('qwen3_moe', 'glm4_moe')
        # case 3, private model
        is_private_model = getattr(self.model_config.hf_config, "is_private", False)
        return enable_dp and (is_ds_mla or is_supported_moe_model or is_private_model)

    def execute_dummy_batch(self) -> None:
        if self._enable_moe_dp_opt():
            self.model_runner.moe_dp_execute_dummy_batch(1)
        else:
            self.model_runner._dummy_run(1, uniform_decode=True)
            
    def response_remote_alloc_once(self) -> None:
        self.model_runner.response_remote_alloc_once()

    def _eplb_before_scale_down(self, old_ep_size: int, new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Starting expert resharding before scaling down..."
            )
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_load=None,
            rank_mapping=rank_mapping,
        )
        torch.mlu.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            get_ep_group,
        )

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = (
            reconfig_request.new_data_parallel_size
            * get_tp_group().world_size
            * get_pp_group().world_size
        )
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

        global_expert_loads = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_loads is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size, global_expert_loads)

    def get_hfu_info(self, batch, input_len, output_len):
        try:
            self.model_runner.model.collect_hfu_io_effciency_info(batch, input_len, output_len)
            if VLLM_DUMP_MLU_INFO_EN:
                return self.model_runner.model.hfu_info, self.model_runner.model.io_efficiency
            else:
                return self.model_runner.model.flops_info, 0.0
        except Exception as e:
            raise RuntimeError(
                "Model match failure when get HFU info, please check if an init method was registed."
            )

    def _get_latency(self, time_markers):
        total_latency = 0
        if not isinstance(time_markers, list):
            time_markers = [time_markers]
        for time_marker in time_markers:
            start, end = time_marker
            latency = start.elapsed_time(end)
            total_latency += latency
        return total_latency

    def get_latency(self):
        return self._get_latency(self.model_runner.time_markers)

    def get_mm_encoder_latency(self):
        if not hasattr(self.model_runner, "mm_time_markers"):
            return None
        mm_time_markers = self.model_runner.mm_time_markers
        return None if len(mm_time_markers) == 0 else\
                self._get_latency(mm_time_markers)

    def get_memory_usage(self):
        return (self.peak_memory, self.block_memory)

    def recapture_model(self,
                        prefill_enable_mlugraph: bool,
                        batch_size: int,
                        input_len: int):
        # Reset history capture context
        self.model_runner.reset_capture_context(
            prefill_enable_mlugraph, batch_size, input_len)
        # Re-capture decode graph(full graph or peicewise graph)
        self.compile_or_warm_up_model()
