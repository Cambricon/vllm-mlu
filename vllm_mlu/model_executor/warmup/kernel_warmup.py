# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker"):
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: skip deep GEMM warmup, flashinfer autotune, and
    flash infer attention warmup
    '''

    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    pass