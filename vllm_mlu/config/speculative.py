# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger

from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)

@staticmethod
def vllm__config__speculative__SpeculativeConfig__create_draft_parallel_config(
    target_parallel_config: ParallelConfig,
    speculative_draft_tensor_parallel_size: int,
) -> ParallelConfig:
    """Create a parallel config for use by the draft worker.

    This is mostly a copy of the target parallel config, except the tp_size.
    """
    '''
    =============================
    Modify by vllm_mlu
    @brief: add draft data parallel parameters
    =============================
    '''
    draft_parallel_config = ParallelConfig(
        pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
        tensor_parallel_size=speculative_draft_tensor_parallel_size,
        distributed_executor_backend=target_parallel_config.distributed_executor_backend,
        max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
        disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
        ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
        placement_group=target_parallel_config.placement_group,
        # add draft data parallel parameters
        data_parallel_size=target_parallel_config.data_parallel_size,
        data_parallel_size_local=target_parallel_config.data_parallel_size_local,
        data_parallel_master_ip=target_parallel_config.data_parallel_master_ip,
        data_parallel_rpc_port=target_parallel_config.data_parallel_rpc_port,
    )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    return draft_parallel_config


vllm__config__speculative__SpeculativeConfig____post_init___org = SpeculativeConfig.__post_init__
def vllm__config__speculative__SpeculativeConfig____post_init__(self):
    if self.model is None and self.num_speculative_tokens is not None and self.method is None:
        self.method = "mtp"
    vllm__config__speculative__SpeculativeConfig____post_init___org(self)


MluHijackObject.apply_hijack(
    SpeculativeConfig,
    SpeculativeConfig.create_draft_parallel_config,
    vllm__config__speculative__SpeculativeConfig__create_draft_parallel_config,
)
MluHijackObject.apply_hijack(
    SpeculativeConfig,
    SpeculativeConfig.__post_init__,
    vllm__config__speculative__SpeculativeConfig____post_init__,
)