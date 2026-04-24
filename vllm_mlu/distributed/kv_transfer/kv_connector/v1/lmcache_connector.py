# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1
from vllm_mlu.mlu_hijack_utils import MluHijackObject

class LMCacheConnectorV1_MluHijack(LMCacheConnectorV1):
    
    def response_remote_alloc_once(self) -> None:
        self._lmcache_engine.response_remote_alloc_once()
        
    def request_remote_memory_send(self) -> None:
        self._lmcache_engine.request_remote_memory_send()
        
        
MluHijackObject.apply_hijack(LMCacheConnectorV1,
                             "response_remote_alloc_once",
                             LMCacheConnectorV1_MluHijack.response_remote_alloc_once)
MluHijackObject.apply_hijack(LMCacheConnectorV1,
                             "request_remote_memory_send",
                             LMCacheConnectorV1_MluHijack.request_remote_memory_send)