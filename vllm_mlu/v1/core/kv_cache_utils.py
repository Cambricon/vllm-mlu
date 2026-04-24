# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.logger import init_logger
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (may_override_num_blocks,
                                         get_uniform_page_size,
                                         get_num_blocks)

logger = init_logger(__name__)

def vllm__v1__core__kv_cache_utils__get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_specs: dict[str, KVCacheSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        kv_cache_specs: The KV cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(kv_cache_specs)
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: support qwen3-next
        '''
        if (vllm_config.mlu_config.enable_mamba_split_page_size):
            # Note(wulingchao): 预留出linear attention的内存不参与系统调度
            # 当前的 page size是小page，需要扩展到完整的linear attention的page
            mamba_page_size = (page_size \
                * vllm_config.mlu_config.mamba_to_attn_block_ratio
                * vllm_config.mlu_config.mamba_support_max_batch_size \
                * group_size * 3)
            logger.warning(f"all available memory {available_memory}, mamba mem used {mamba_page_size}")
            available_memory = available_memory - mamba_page_size
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

MluHijackObject.apply_hijack(kv_cache_utils,
                             kv_cache_utils.get_kv_cache_config_from_groups,
                             vllm__v1__core__kv_cache_utils__get_kv_cache_config_from_groups)


