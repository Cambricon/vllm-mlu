# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0
from vllm.v1.engine.core_client import (
    EngineCoreClient,
    InprocClient,
    SyncMPClient,
    AsyncMPClient,
    DPAsyncMPClient,
    DPLBAsyncMPClient,
)
from vllm.v1.engine import EngineCoreRequest
from vllm.config import VllmConfig
from vllm.v1.executor import Executor

from vllm_mlu.mlu_hijack_utils import MluHijackObject

class EngineCoreClient_MluHiack(EngineCoreClient):
    
    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "MPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: disagg use DPAsyncMPClient instead of DPLBAsyncMPClient.
        '''
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb or vllm_config.kv_transfer_config is not None:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        return AsyncMPClient(*client_args)


class InprocClient_MluHiack(InprocClient):

    def get_hfu_info(self, batch, input_len, output_len):
        return self.engine_core.get_hfu_info(batch, input_len, output_len)

    def get_latency(self):
        return self.engine_core.get_latency()

    def get_memory_usage(self):
        return self.engine_core.get_memory_usage()

    def recapture_model(
        self,
        prefill_enable_mlugraph: bool,
        batch_size: int,
        input_len: int,
    ):
        return self.engine_core.recapture_model(
            prefill_enable_mlugraph, batch_size, input_len
        )

    def init_metric(self, use_unchunk_sched: bool, min_prefill_batch: int):
        return self.engine_core.init_metric(
            use_unchunk_sched, min_prefill_batch,
        )

    def start_scheduler_profile(self):
        self.engine_core.start_scheduler_profile()

    def stop_scheduler_profile(self):
        self.engine_core.stop_scheduler_profile()
        
    def response_remote_alloc_once(self) -> None:
        self.engine_core.response_remote_alloc_once()


class SyncMPClient_MluHiack(SyncMPClient):

    def get_hfu_info(self, batch, input_len, output_len):
        try:
            return self.call_utility("get_hfu_info", batch, input_len, output_len)
        except Exception as e:
            raise RuntimeError(f"Failed to get HFU info: {str(e)}")

    def get_latency(self):
        return self.call_utility("get_latency")

    def get_memory_usage(self):
        return self.call_utility("get_memory_usage")

    def recapture_model(self,
                        prefill_enable_mlugraph: bool,
                        batch_size: int,
                        input_len: int):
        return self.call_utility("recapture_model",
                                 prefill_enable_mlugraph, batch_size, input_len)

    def init_metric(self, use_unchunk_sched: bool, min_prefill_batch: int):
        return self.call_utility("init_metric",
                                 use_unchunk_sched,
                                 min_prefill_batch)

    def start_scheduler_profile(self):
        self.call_utility("start_scheduler_profile")

    def stop_scheduler_profile(self):
        self.call_utility("stop_scheduler_profile")
        
    def response_remote_alloc_once(self) -> None:
        self.call_utility("response_remote_alloc_once")


class AsyncMPClient_MluHijack(AsyncMPClient):

    async def start_scheduler_profile(self) -> None:
        await self.call_utility_async("start_scheduler_profile")

    async def stop_scheduler_profile(self) -> None:
        await self.call_utility_async("stop_scheduler_profile")
        
    async def response_remote_alloc_once(self) -> None:
        await self.call_utility_async("response_remote_alloc_once")
        
        
class DPAsyncMPClient_MluHijack(DPAsyncMPClient):

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: disagg need proxy to assign dp_rank
        '''
        if request.data_parallel_rank is not None:
            # engines are already in rank order
            return self.core_engines[request.data_parallel_rank]
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        
        return self.core_engine


MluHijackObject.apply_hijack(EngineCoreClient,
                             EngineCoreClient.make_async_mp_client,
                             EngineCoreClient_MluHiack.make_async_mp_client)
MluHijackObject.apply_hijack(InprocClient,
                             "get_hfu_info",
                             InprocClient_MluHiack.get_hfu_info)
MluHijackObject.apply_hijack(InprocClient,
                             "get_latency",
                             InprocClient_MluHiack.get_latency)
MluHijackObject.apply_hijack(InprocClient,
                             "get_memory_usage",
                             InprocClient_MluHiack.get_memory_usage)
MluHijackObject.apply_hijack(InprocClient,
                             "recapture_model",
                             InprocClient_MluHiack.recapture_model)
MluHijackObject.apply_hijack(InprocClient,
                             "init_metric",
                             InprocClient_MluHiack.init_metric)
MluHijackObject.apply_hijack(InprocClient,
                             "start_scheduler_profile",
                             InprocClient_MluHiack.start_scheduler_profile)
MluHijackObject.apply_hijack(InprocClient,
                             "stop_scheduler_profile",
                             InprocClient_MluHiack.stop_scheduler_profile)
MluHijackObject.apply_hijack(InprocClient,
                             "response_remote_alloc_once",
                             InprocClient_MluHiack.response_remote_alloc_once)
MluHijackObject.apply_hijack(SyncMPClient,
                             "get_hfu_info",
                             SyncMPClient_MluHiack.get_hfu_info)
MluHijackObject.apply_hijack(SyncMPClient,
                             "get_latency",
                             SyncMPClient_MluHiack.get_latency)
MluHijackObject.apply_hijack(SyncMPClient,
                             "get_memory_usage",
                             SyncMPClient_MluHiack.get_memory_usage)
MluHijackObject.apply_hijack(SyncMPClient,
                             "recapture_model",
                             SyncMPClient_MluHiack.recapture_model)
MluHijackObject.apply_hijack(SyncMPClient,
                             "init_metric",
                             SyncMPClient_MluHiack.init_metric)
MluHijackObject.apply_hijack(SyncMPClient,
                             "start_scheduler_profile",
                             SyncMPClient_MluHiack.start_scheduler_profile)
MluHijackObject.apply_hijack(SyncMPClient,
                             "stop_scheduler_profile",
                             SyncMPClient_MluHiack.stop_scheduler_profile)
MluHijackObject.apply_hijack(SyncMPClient,
                             "response_remote_alloc_once",
                             SyncMPClient_MluHiack.response_remote_alloc_once)
MluHijackObject.apply_hijack(AsyncMPClient,
                             "start_scheduler_profile",
                             AsyncMPClient_MluHijack.start_scheduler_profile)
MluHijackObject.apply_hijack(AsyncMPClient,
                             "stop_scheduler_profile",
                             AsyncMPClient_MluHijack.stop_scheduler_profile)
MluHijackObject.apply_hijack(AsyncMPClient,
                             "response_remote_alloc_once",
                             AsyncMPClient_MluHijack.response_remote_alloc_once)
MluHijackObject.apply_hijack(DPAsyncMPClient,
                             DPAsyncMPClient.get_core_engine_for_request,
                             DPAsyncMPClient_MluHijack.get_core_engine_for_request)
