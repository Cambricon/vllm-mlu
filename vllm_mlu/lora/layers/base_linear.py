# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.platforms import current_platform

from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__lora__layers__row_parallel_linear__BaseLinearLayerWithLoRA__apply(
    self,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add residual in matmul
    '''
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias, residual)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    # In transformers backend, x and output have extra batch dimension like
    # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
    # therefore we need to flatten the batch dimensions.
    if x.ndim == 3 and output.ndim == 3:
        output = output.flatten(0, 1)
        x = x.flatten(0, 1)

    lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
        output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
    )
    if not current_platform.can_update_inplace():
        output = lora_output

    return output


MluHijackObject.apply_hijack(
    BaseLinearLayerWithLoRA,
    BaseLinearLayerWithLoRA.apply,
    vllm__lora__layers__row_parallel_linear__BaseLinearLayerWithLoRA__apply,
)