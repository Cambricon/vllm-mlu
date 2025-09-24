# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from vllm.v1.engine.core_client import (InprocClient,
                                        SyncMPClient,
                                        AsyncMPClient)

from vllm_mlu.mlu_hijack_utils import MluHijackObject


class InprocClient_MluHiack(InprocClient):

    def get_latency(self):
        return self.engine_core.get_latency()

    def get_memory_usage(self):
        return self.engine_core.get_memory_usage()

    def recapture_model(self,
                        prefill_enable_mlugraph: bool,
                        batch_size: int,
                        input_len: int):
        return self.engine_core.recapture_model(
            prefill_enable_mlugraph, batch_size, input_len)

    def init_metric(self, use_unchunk_sched: bool, min_prefill_batch: int):
        return self.engine_core.init_metric(
                    use_unchunk_sched, min_prefill_batch)


class SyncMPClient_MluHiack(SyncMPClient):

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
