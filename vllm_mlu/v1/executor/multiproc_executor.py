# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm_mlu.mlu_hijack_utils import MluHijackObject

class MultiprocExecutor_MluHijack(MultiprocExecutor):
    
    def response_remote_alloc_once(self) -> None:
        self.collective_rpc("response_remote_alloc_once", unique_reply_rank=self.output_rank)
        

MluHijackObject.apply_hijack(MultiprocExecutor,
                             "response_remote_alloc_once",
                             MultiprocExecutor_MluHijack.response_remote_alloc_once)