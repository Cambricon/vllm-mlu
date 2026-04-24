# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.lora.layers.column_parallel_linear import ColumnParallelLinearWithLoRA
from vllm_mlu.mlu_hijack_utils import MluHijackObject


vllm__lora__layers__column_parallel_linear__ColumnParallelLinearWithLoRA__forward_org = ColumnParallelLinearWithLoRA.forward


'''
=============================
Modify by vllm_mlu
=============================
@brief: add smooth_quant_scale and use_tp_weight parameters.
'''
def vllm__lora__layers__column_parallel_linear__ColumnParallelLinearWithLoRA__forward(
    self,
    input_,
    smooth_quant_scale: torch.Tensor | None = None,
    use_tp_weight: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    assert not use_tp_weight, "LoRa does not support use_tp_weight yet."
    assert smooth_quant_scale is None, "LoRA does not support smooth quant yet."
    return vllm__lora__layers__column_parallel_linear__ColumnParallelLinearWithLoRA__forward_org(self, input_)
'''
==================
End of MLU Hijack
==================
'''


MluHijackObject.apply_hijack(
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithLoRA.forward,
    vllm__lora__layers__column_parallel_linear__ColumnParallelLinearWithLoRA__forward,
)