# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0
from vllm.v1.engine.llm_engine import LLMEngine
from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__engine__llm_engine__LLMEngine__get_hfu_info(self, batch, input_len, output_len):
    return self.engine_core.get_hfu_info(batch, input_len, output_len)


def vllm__engine__llm_engine__LLMEngine__get_latency(self):
    return self.engine_core.get_latency()


def vllm__engine__llm_engine__LLMEngine__get_memory_usage(self):
    return self.engine_core.get_memory_usage()


def vllm__engine__llm_engine__LLMEngine__start_scheduler_profile(self):
    self.engine_core.start_scheduler_profile()


def vllm__engine__llm_engine__LLMEngine__stop_scheduler_profile(self):
    self.engine_core.stop_scheduler_profile()


MluHijackObject.apply_hijack(LLMEngine,
                             "get_hfu_info",
                             vllm__engine__llm_engine__LLMEngine__get_hfu_info)
MluHijackObject.apply_hijack(LLMEngine,
                             "get_latency",
                             vllm__engine__llm_engine__LLMEngine__get_latency)
MluHijackObject.apply_hijack(LLMEngine,
                             "get_memory_usage",
                             vllm__engine__llm_engine__LLMEngine__get_memory_usage)
MluHijackObject.apply_hijack(LLMEngine,
                             "start_scheduler_profile",
                             vllm__engine__llm_engine__LLMEngine__start_scheduler_profile)
MluHijackObject.apply_hijack(LLMEngine,
                             "stop_scheduler_profile",
                             vllm__engine__llm_engine__LLMEngine__stop_scheduler_profile)
