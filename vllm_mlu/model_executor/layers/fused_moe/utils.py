# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from math import prod
from typing import List, Optional, Tuple

import torch

from vllm_mlu.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.utils import cdiv


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    assert block_shape is not None
    assert len(block_shape) == 2
    _, block_k = block_shape[0], block_shape[1]
    A, A_scale = per_token_group_quant_fp8(A, block_k)
    assert cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    return A, A_scale


