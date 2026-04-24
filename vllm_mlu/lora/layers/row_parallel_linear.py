# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch

from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.layers.row_parallel_linear import (
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from vllm.platforms import current_platform

from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__lora__layers__row_parallel_linear__RowParallelLinearWithShardedLoRA__apply(
    self,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add residual and bias in matmul
    '''
    output = self.base_layer.quant_method.apply(
        self.base_layer, x, bias, residual)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape
    buffer = torch.zeros(
        (self.n_slices, x.shape[0], self.lora_a_stacked[0].shape[2]),
        dtype=torch.float32,
        device=x.device,
    )

    shrunk_buffer: torch.Tensor | None = self.punica_wrapper.add_shrink(
        buffer, x, self.lora_a_stacked, 1.0
    )
    if not current_platform.can_update_inplace():
        buffer = shrunk_buffer
    if self.tp_size > 1:
        buffer = tensor_model_parallel_all_reduce(buffer)

    # following S-LoRA, allows the fusing of all_gather and all_reduce
    # by adding the column partitioned lora output to a slice of output
    # tensor, which is a partial sum due to row parallel. All that
    # remains is a standard all_reduce. User should be aware though that
    # the output is not the same as a normal row_parallel, it should be
    # reduced before being used
    # NOTE offset are based on the rank.
    shard_size = self.lora_b_stacked[0].shape[2]
    offset_start = self.tp_rank * shard_size
    lora_output: torch.Tensor | None = self.punica_wrapper.add_expand(
        output,
        buffer,
        self.lora_b_stacked,
        self.output_slices,
        offset_start=offset_start,
        add_input=True,
    )

    if not current_platform.can_update_inplace():
        output = lora_output

    output = output.view(*out_orig_shape)
    return output


def vllm__lora__layers__row_parallel_linear__RowParallelLinearWithLoRA__forward(
    self,
    input_: torch.Tensor,
    residual: torch.Tensor | None = None,
    smooth_quant_scale: torch.Tensor | None = None,
    use_tp_weight: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add parameters `residual`, `smooth_quant_scale`, `use_tp_weight` and `output`
    to keep parameters consistent with RowParallelLinear.forward.
    '''
    assert (not use_tp_weight) and output is None, (
        f"RowParallelLinearWithLoRA.forward does not support use_tp_wight=True"
        f" or pass output parameters.")
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    # Set up backprop all-reduce.
    if self.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        # TODO: simplify code below
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.base_layer.tp_size
        )
        input_parallel = splitted_input[self.tp_rank].contiguous()

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: 1) apply residual fusion in matmul like RowParallelLinear
    2) add bias in matmul, not after all reduce
    '''
    # Matrix multiply.
    bias_ = (
        None if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)
        else self.base_layer.bias
    )
    residual_ = None if self.base_layer.tp_rank > 0 else residual
    output_parallel = self.apply(input_parallel, bias_, residual_)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    if self.base_layer.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: do not add bias after all_reduce
    '''
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    if not self.base_layer.return_bias:
        return output
    return output, output_bias


MluHijackObject.apply_hijack(
    RowParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA.apply,
    vllm__lora__layers__row_parallel_linear__RowParallelLinearWithShardedLoRA__apply,
)
MluHijackObject.apply_hijack(
    RowParallelLinearWithLoRA,
    RowParallelLinearWithLoRA.forward,
    vllm__lora__layers__row_parallel_linear__RowParallelLinearWithLoRA__forward,
)