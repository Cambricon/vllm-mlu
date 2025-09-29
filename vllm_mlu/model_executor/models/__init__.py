# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from vllm import ModelRegistry


def register_model():
    from .deepseek_v3_2_exp import MLUDeepseekV2ForCausalLM  # noqa: F401

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM",
        "vllm_mlu.model_executor.models.deepseek_v3_2_exp:MLUDeepseekV2ForCausalLM")
