# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
import torch
from vllm.model_executor.layers.activation import QuickGELU
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu import _mlu_ops as mlu_ops

def vllm__model_executor__activation__QuickGELU__forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: implement forward_oot
    '''
    return mlu_ops.active(x, 'quick_gelu', False)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

MluHijackObject.apply_hijack(QuickGELU,
                             QuickGELU.forward_oot,
                             vllm__model_executor__activation__QuickGELU__forward_oot)
