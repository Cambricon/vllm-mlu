# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.v1.engine.llm_engine import LLMEngine
from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__engine__llm_engine__LLMEngine__get_latency(self):
    return self.engine_core.get_latency()


def vllm__engine__llm_engine__LLMEngine__get_memory_usage(self):
    return self.engine_core.get_memory_usage()


MluHijackObject.apply_hijack(LLMEngine,
                             "get_latency",
                             vllm__engine__llm_engine__LLMEngine__get_latency)
MluHijackObject.apply_hijack(LLMEngine,
                             "get_memory_usage",
                             vllm__engine__llm_engine__LLMEngine__get_memory_usage)
