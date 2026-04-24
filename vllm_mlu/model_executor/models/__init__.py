# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm import ModelRegistry


def register_model():
    from .deepseek_v4 import MLUDeepseekV4ForCausalLM  # noqa: F401

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_mlu.model_executor.models.deepseek_v4:MLUDeepseekV4ForCausalLM")
