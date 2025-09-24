# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from vllm.utils import run_method
from vllm.executor.executor_base import ExecutorBase
from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__executor__executor_base__ExecutorBase__get_latency(self) -> float:
    """
    requires that torch.mlu.synchronize() be executed before this function
    for getting an accurate reading
    """
    latency = run_method(self.driver_worker,
                         "get_latency", args=[], kwargs={})
    return latency


def vllm__executor__executor_base__ExecutorBase__recapture_model(
    self, prefill_enable_mlugraph: bool, batch_size: int, input_len: int
) -> None:
    return self.collective_rpc("recapture_model",
                               args=(prefill_enable_mlugraph, batch_size, input_len))


def vllm__executor__executor_base__ExecutorBase__get_memory_usage(self):
    memory_usage = run_method(self.driver_worker,
                              "get_memory_usage", args=[], kwargs={})
    return memory_usage


MluHijackObject.apply_hijack(ExecutorBase,
                             "get_latency",
                             vllm__executor__executor_base__ExecutorBase__get_latency)
MluHijackObject.apply_hijack(ExecutorBase,
                             "recapture_model",
                             vllm__executor__executor_base__ExecutorBase__recapture_model)
MluHijackObject.apply_hijack(ExecutorBase,
                             "get_memory_usage",
                             vllm__executor__executor_base__ExecutorBase__get_memory_usage)
