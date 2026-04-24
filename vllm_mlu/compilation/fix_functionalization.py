# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0

import operator
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.platforms import current_platform
from vllm.logger import init_logger

from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fx_utils import is_func

from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


class FixFunctionalizationPass_MluHijack(FixFunctionalizationPass):

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        # XPU does not support auto-functionalization yet.
        # Will enable this when switch to vllm-xpu-kernels.
        if current_platform.is_xpu():
            logger.debug(
                "XPU platform does not support fix functionalizationpass currently."
            )
            return

        self.nodes_to_remove: list[torch.fx.Node] = []
        count = 0
        for node in graph.nodes:
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: skip custom op on mlu
            '''
            if current_platform.is_out_of_tree():
                continue # skip the count on mlu
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            if not is_func(node, auto_functionalized):
                continue  # Avoid deep if-elif nesting

            kwargs = node.kwargs
            at_target = node.args[0]

            if at_target == torch.ops._C.rotary_embedding.default:
                query = kwargs["query"]
                key = kwargs["key"]
                getitem_nodes = self.getitem_users(node)

                if (
                    is_func(query, operator.getitem)
                    and is_func(key, operator.getitem)
                    and query.args[0] == key.args[0]
                    and is_func(query.args[0], torch.ops.aten.split_with_sizes.default)
                    and all(
                        is_func(user, torch.ops.aten.slice_scatter.default)
                        for getitem_node in getitem_nodes.values()
                        for user in getitem_node.users
                    )
                ):
                    # Pattern where query and key are slices of an mm_node.
                    # While functionalized, results at [1] and [2] are scattered
                    # back into mm_node. So after de-functionalization, we can
                    # just use mm_node directly.

                    mm_node = query.args[0].args[0]
                    for user in getitem_nodes.values():
                        for user_of_getitem in user.users:
                            if is_func(
                                user_of_getitem, torch.ops.aten.slice_scatter.default
                            ):
                                user_of_getitem.replace_all_uses_with(mm_node)
                                self._remove(user_of_getitem)
                        self._remove(user)

                    self.insert_defunctionalized(graph, node)
                    self._remove(node)

                else:
                    # Directly replace the auto_functionalize(rotary_embedding)
                    # with the inplace rotary_embedding. In theory, we shouldn't
                    # do this blindly, but in practice in vLLM it's ok. The best
                    # solution is to use auto_functionalization_v2 and then use
                    # inductor's builtin defunctionalization (reinplacing) pass.
                    mutated_args = {1: "query", 2: "key"}
                    self.defunctionalize(graph, node, mutated_args)

            # rms_norm replacements avoid the most copies for LLaMa.
            elif at_target == torch.ops._C.fused_add_rms_norm.default:
                mutated_args = {1: "input", 2: "residual"}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.fused_add_rms_norm_static_fp8_quant.default:  # noqa: E501
                mutated_args = {1: "result", 2: "residual"}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.rms_norm_dynamic_per_token_quant.default:  # noqa: E501
                mutated_args = {1: "result", 2: "scale", 3: "residual"}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target in [
                torch.ops._C.rms_norm.default,
                torch.ops._C.rms_norm_static_fp8_quant.default,
            ]:
                mutated_args = {1: "result"}
                self.defunctionalize(graph, node, mutated_args)
            # For some reason we need to specify the args for both
            # silu_and_mul and silu_and_mul_quant. The kwargs
            # pathway gets the wrong answer.
            elif at_target == torch.ops._C.silu_and_mul.default:
                mutated_args = {1: "result"}
                self.defunctionalize(
                    graph, node, mutated_args, args=("result", "input")
                )
            elif at_target == torch.ops._C.silu_and_mul_quant.default:
                mutated_args = {1: "result"}
                self.defunctionalize(
                    graph, node, mutated_args, args=("result", "input", "scale")
                )
            elif (
                hasattr(torch.ops._C, "silu_and_mul_nvfp4_quant")
                and at_target == torch.ops._C.silu_and_mul_nvfp4_quant.default
            ):
                mutated_args = {1: "result", 2: "result_block_scale"}
                self.defunctionalize(
                    graph,
                    node,
                    mutated_args,
                    args=(
                        "result",
                        "result_block_scale",
                        "input",
                        "input_global_scale",
                    ),
                )
            # Defunctionalize fused_qk_norm_rope to remove higher-order wrapper.
            elif at_target == torch.ops._C.fused_qk_norm_rope.default:
                mutated_args = {1: "qkv"}
                args = (
                    "qkv",
                    "num_heads_q",
                    "num_heads_k",
                    "num_heads_v",
                    "head_dim",
                    "eps",
                    "q_weight",
                    "k_weight",
                    "cos_sin_cache",
                    "is_neox",
                    "position_ids",
                )
                self.defunctionalize(graph, node, mutated_args=mutated_args, args=args)
            else:
                continue  # skip the count

            count += 1

        self.dump_graph(graph, "before_cleanup")

        # Remove the nodes all at once
        count_removed = len(self.nodes_to_remove)
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        logger.debug(
            "De-functionalized %s nodes, removed %s nodes", count, count_removed
        )
        self.nodes_to_remove.clear()


MluHijackObject.apply_hijack(
    FixFunctionalizationPass,
    FixFunctionalizationPass.__call__,
    FixFunctionalizationPass_MluHijack.__call__
)