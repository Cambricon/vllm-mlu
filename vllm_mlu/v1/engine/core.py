# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0
from collections import deque
import signal
from typing import Any, Callable, cast
from concurrent.futures import Future

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import (
    EngineCore,
    EngineCoreProc,
    DPEngineCoreProc,
)
from vllm.v1.executor.abstract import Executor
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION
from logging import DEBUG

import vllm_mlu._mlu_utils as mlu_envs
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu.mlu_metric import LLMMetric


class EngineCore_MluHijack(EngineCore):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: load_general_plugins in run_engine_core
        '''
        # # plugins need to be loaded at the engine/scheduler level too
        # from vllm.plugins import load_general_plugins
        # load_general_plugins()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: v1 support offline benchmark
        '''
        self.step_latency = []
        self.model_exec_latency = []
        self.mm_encoder_latency = []
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: v1 support offline benchmark
        '''
        if mlu_envs.VLLM_LATENCY_DEBUG_EN:
            step_start = LLMMetric.get_mlu_cost_time()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with self.log_error_detail(scheduler_output):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
                
        if self.use_spec_decode and \
            self.vllm_config.kv_transfer_config is not None and \
               self.vllm_config.kv_transfer_config.kv_role == "kv_producer":
            draft_token_ids = self.model_executor.take_draft_token_ids()
            self.scheduler.draft_token_ids = draft_token_ids

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: v1 support offline benchmark
        '''
        has_sched_reqs = (scheduler_output.total_num_scheduled_tokens > 0)
        if mlu_envs.VLLM_LATENCY_DEBUG_EN and has_sched_reqs:
            self.step_latency.append(LLMMetric.get_mlu_cost_time() - step_start)
        if mlu_envs.VLLM_LATENCY_DEBUG_WITH_DEVICE_EN and has_sched_reqs:
            self.model_exec_latency.append(self.get_model_exec_latency())
            mm_encoder_latency = self.get_mm_encoder_latency()
            if mm_encoder_latency:
                self.mm_encoder_latency.append(mm_encoder_latency)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
    
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()
            
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: supoort disagg for mlu.
        '''
        if self.use_spec_decode and \
            self.vllm_config.kv_transfer_config is not None and \
               self.vllm_config.kv_transfer_config.kv_role == "kv_producer":
            draft_token_ids = self.model_executor.take_draft_token_ids()
            self.scheduler.draft_token_ids = draft_token_ids
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        return engine_core_outputs, model_executed

    def get_model_exec_latency(self):
        latency = self.model_executor.get_latency()
        return latency

    def get_mm_encoder_latency(self):
        return self.model_executor.get_mm_encoder_latency()

    def get_hfu_info(self, batch, input_len, output_len):
        return self.model_executor.get_hfu_info(batch, input_len, output_len)

    def get_latency(self):
        return (self.step_latency, self.model_exec_latency, self.mm_encoder_latency)

    def get_memory_usage(self):
        peak_memory, block_memory = self.model_executor.get_memory_usage()
        return (peak_memory, block_memory,
                self.num_gpu_blocks, self.num_cpu_blocks)

    def recapture_model(self,
                        prefill_enable_mlugraph: bool,
                        batch_size: int,
                        input_len: int):
        self.model_executor.recapture_model(
            prefill_enable_mlugraph, batch_size, input_len)

    def init_metric(self, use_unchunk_sched: bool, min_prefill_batch: int):
        self.step_latency = []
        self.model_exec_latency = []
        self.mm_encoder_latency = []
        mlu_envs.VLLM_V1_USE_UNCHUNK_SCHED = use_unchunk_sched
        mlu_envs.VLLM_V1_MIN_PREFILL_BATCH = min_prefill_batch

    def start_scheduler_profile(self):
        self.scheduler.start_scheduler_profile()

    def stop_scheduler_profile(self):
        self.scheduler.stop_scheduler_profile()
        
    def response_remote_alloc_once(self):
        self.model_executor.response_remote_alloc_once()


class EngineCoreProc_MluHijack(EngineCoreProc):

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: load_general_plugins for mp backend engine
        '''
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()
                
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
                
            if self.vllm_config.kv_transfer_config is not None and \
                self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                self.response_remote_alloc_once()
                if self.input_queue.empty():
                    continue
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)    
            else:
                req = self.input_queue.get()
                self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")
            
        if self.vllm_config.kv_transfer_config is not None and \
            self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
            self.response_remote_alloc_once()

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)


MluHijackObject.apply_hijack(EngineCore,
                             "get_mm_encoder_latency",
                             EngineCore_MluHijack.get_mm_encoder_latency)
MluHijackObject.apply_hijack(EngineCore,
                             "get_model_exec_latency",
                             EngineCore_MluHijack.get_model_exec_latency)
MluHijackObject.apply_hijack(EngineCore,
                             "get_hfu_info",
                             EngineCore_MluHijack.get_hfu_info)
MluHijackObject.apply_hijack(EngineCore,
                             "get_latency",
                             EngineCore_MluHijack.get_latency)
MluHijackObject.apply_hijack(EngineCore,
                             "get_memory_usage",
                             EngineCore_MluHijack.get_memory_usage)
MluHijackObject.apply_hijack(EngineCore,
                             "recapture_model",
                             EngineCore_MluHijack.recapture_model)
MluHijackObject.apply_hijack(EngineCore,
                             "init_metric",
                             EngineCore_MluHijack.init_metric)
MluHijackObject.apply_hijack(EngineCore,
                             "start_scheduler_profile",
                             EngineCore_MluHijack.start_scheduler_profile)
MluHijackObject.apply_hijack(EngineCore,
                             "stop_scheduler_profile",
                             EngineCore_MluHijack.stop_scheduler_profile)
MluHijackObject.apply_hijack(EngineCore,
                             EngineCore.__init__,
                             EngineCore_MluHijack.__init__)
MluHijackObject.apply_hijack(EngineCore,
                             EngineCore.step,
                             EngineCore_MluHijack.step)
MluHijackObject.apply_hijack(EngineCore,
                             "response_remote_alloc_once",
                             EngineCore_MluHijack.response_remote_alloc_once)
MluHijackObject.apply_hijack(EngineCore,
                             EngineCore.step_with_batch_queue,
                             EngineCore_MluHijack.step_with_batch_queue)
MluHijackObject.apply_hijack(EngineCoreProc,
                             EngineCoreProc.run_engine_core,
                             EngineCoreProc_MluHijack.run_engine_core)
MluHijackObject.apply_hijack(EngineCoreProc,
                             EngineCoreProc._process_input_queue,
                             EngineCoreProc_MluHijack._process_input_queue)
