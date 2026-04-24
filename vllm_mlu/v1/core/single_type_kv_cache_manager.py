# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
    spec_manager_map,
)

from vllm_mlu.v1.kv_cache_interface import (
    MLUFullAttentionSpec,
    MLUMLAAttentionSpec,
    MLUSlidingWindowSpec,
)


spec_manager_map.update({
    MLUFullAttentionSpec: FullAttentionManager,
    MLUSlidingWindowSpec: SlidingWindowManager,
    MLUMLAAttentionSpec: FullAttentionManager,
})