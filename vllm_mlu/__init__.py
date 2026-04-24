# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project


def register_mlu_platform():
    """Register the MLU platform."""
    return "vllm_mlu.platforms.mlu.MLUPlatform"


def register_mlu_hijack():
    """Register the MLU models and hijack."""
    from vllm_mlu import mlu_hijack
    from vllm_mlu.model_executor.models import register_model
    register_model()
    return