# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Callable, Any

import torch

from vllm.model_executor.parameter import BasevLLMParameter
from vllm.distributed import (
    get_parallel_rank_with_group,
    get_parallel_world_size_with_group,
)

from vllm_mlu.mlu_hijack_utils import MluHijackObject


vllm__model_executor__parameter__BasevLLMParameter____init__org = BasevLLMParameter.__init__


def vllm__model_executor__parameter__BasevLLMParameter____init__(
    self,
    data: torch.Tensor,
    weight_loader: Callable,
    tp_group: Any = None
):
    vllm__model_executor__parameter__BasevLLMParameter____init__org(
        self, data, weight_loader
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add self.tp_group, world_size and tp_rank
    '''
    if tp_group is not None:
        self.tp_group = tp_group
        self.tp_world_size = get_parallel_world_size_with_group(self.tp_group)
        self.tp_rank = get_parallel_rank_with_group(self.tp_group)
    '''
    =================
    End of MLU Hijack
    =================
    '''


MluHijackObject.apply_hijack(BasevLLMParameter,
                             BasevLLMParameter.__init__,
                             vllm__model_executor__parameter__BasevLLMParameter____init__)
