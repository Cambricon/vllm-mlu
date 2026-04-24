# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import (
    TYPE_CHECKING,  # noqa: UP035
)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm_mlu.mlu_hijack_utils import MluHijackObject
logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin_MluHijack(KVConnectorModelRunnerMixin):
    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())
            
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: supoort disagg for mlu.
            '''
            kv_connector.request_remote_memory_send()
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput", wait_for_save: bool = True
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context())
        
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: supoort disagg for mlu.
        '''
        kv_connector.request_remote_memory_send()
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        
        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = (
                KVConnectorModelRunnerMixin.get_kv_connector_stats()
            )


MluHijackObject.apply_hijack(KVConnectorModelRunnerMixin,
                             KVConnectorModelRunnerMixin.maybe_setup_kv_connector,
                             KVConnectorModelRunnerMixin_MluHijack.maybe_setup_kv_connector)
MluHijackObject.apply_hijack(KVConnectorModelRunnerMixin,
                             KVConnectorModelRunnerMixin._get_kv_connector_output,
                             KVConnectorModelRunnerMixin_MluHijack._get_kv_connector_output)