# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Optional, Any

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (divide, split_tensor_along_last_dim,
    get_parallel_rank_with_group, get_parallel_world_size_with_group,
    get_tp_world_group, get_tp_world_world_size, get_tp_world_rank)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED, UnquantizedLinearMethod, LinearBase,
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear)
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger

from vllm_mlu.model_executor.layers.quantization.smoothquant import SmoothQuantLinearMethod
from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu import _mlu_ops as mlu_ops

logger = init_logger(__name__)


WEIGHT_LOADER_V2_SUPPORTED.extend([
    "GPTQMluLinearMethod",
    "AWQMluLinearMethod"
])

vllm__module_executor__layers__linear__LinearBase____init__org = LinearBase.__init__
vllm__module_executor__layers__linear__MergedColumnParallelLinear__weight_loader_org = MergedColumnParallelLinear.weight_loader
vllm__module_executor__layers__linear__RowParallelLinear__weight_loader_org = RowParallelLinear.weight_loader

'''
=============================
Modify by vllm_mlu
=============================
@brief: add residual parameter.
@brief: dispatch unquantized_gemm to mlu ops.
'''
def vllm__module_executor__layers__linear__UnquantizedLinearMethod__apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None
) -> torch.Tensor:
    beta = 0.0
    if residual is not None:
         beta = 1.0
         residual = residual.view(-1, residual.shape[-1])
    res_shape = x.shape[0:-1] + (layer.weight.shape[0], )
    return mlu_ops.matmul(x.reshape(x.numel() // x.shape[-1], x.shape[-1]),
                          layer.weight,
                          bias, residual, 'none', 1.0, beta).view(res_shape)
'''
==================
End of MLU Hijack
==================
'''

'''
=============================
Modify by vllm_mlu
=============================
@brief: add tp_group and keep_full_weights parameters.
'''
def vllm__module_executor__layers__linear__LinearBase____init__(
    self,
    input_size: int,
    output_size: int,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    *,
    tp_group: Any = None,
    keep_full_weights: bool = False,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    vllm__module_executor__layers__linear__LinearBase____init__org(
        self=self,
        input_size=input_size,
        output_size=output_size,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=return_bias,
        disable_tp=disable_tp)
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add self.tp_group, world_size and tp_rank to support data parallel and moe expert parallel
    '''
    self.tp_group = tp_group
    self.tp_world_size = get_parallel_world_size_with_group(self.tp_group)
    self.tp_size = self.tp_world_size
    self.tp_rank = get_parallel_rank_with_group(self.tp_group)

    self.keep_full_weights = keep_full_weights
    if self.keep_full_weights or disable_tp:
        self.tp_group = None
        self.tp_world_size = 1
        self.tp_size = self.tp_world_size
        self.tp_rank = 0
        self.tp_world_size_org = get_tp_world_world_size()
        self.tp_rank_org = get_tp_world_rank()
    '''
    =================
    End of MLU Hijack
    =================
    '''
'''
=================
End of MLU Hijack
=================
'''

'''
=============================
Modify by vllm_mlu
=============================
@brief: add tp_group and keep_full_weights parameters.
'''
def vllm__module_executor__layers__linear__ColumnParallelLinear____init__(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    output_sizes: list[int] | None = None,
    prefix: str = "",
    *,
    tp_group: Any = None,
    keep_full_weights: bool = False,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    super(ColumnParallelLinear, self).__init__(
        input_size,
        output_size,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        prefix=prefix,
        tp_group=tp_group,
        keep_full_weights=keep_full_weights,
        return_bias=return_bias,
        disable_tp=disable_tp,
    )

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: self.tp_size and self.tp_rank has been initialized in LinearBase.__init__
    '''
    # Divide the weight matrix along the last dimension.
    # self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
    # self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
    '''
    =================
    End of MLU Hijack
    =================
    '''
    self.input_size_per_partition = input_size
    self.output_size_per_partition = divide(output_size, self.tp_size)
    self.output_partition_sizes = [self.output_size_per_partition]
    # If QKV or MergedColumn, use output size of each partition.
    if hasattr(self, "output_sizes"):
        self.output_partition_sizes = [
            divide(output_size, self.tp_size) for output_size in self.output_sizes
        ]

    self.gather_output = gather_output

    if output_sizes is None:
        output_sizes = [output_size]

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add tp_group in create_weights
    '''
    assert self.quant_method is not None
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2
            if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        ),
        tp_group=self.tp_group,
    )
    '''
    =================
    End of MLU Hijack
    =================
    '''
    if bias:
        self.bias = Parameter(
            torch.empty(self.output_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(
            self.bias,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
    else:
        self.register_parameter("bias", None)
    self.update_param_tp_status()
'''
=================
End of MLU Hijack
=================
'''


'''
=============================
Modify by vllm_mlu
=============================
@brief: add smooth_quant_scale and use_tp_weight parameters.
'''
def vllm__module_executor__layers__linear__ColumnParallelLinear__forward(
    self,
    input_,
    smooth_quant_scale: torch.Tensor | None = None,
    use_tp_weight: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    bias = self.bias if not self.skip_bias_add else None

    # Matrix multiply.
    assert self.quant_method is not None
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add input_scale and use_tp_weight parameter.
    '''
    kwargs = {'bias': bias}
    if use_tp_weight:
        kwargs['use_tp_weight'] = use_tp_weight
    if smooth_quant_scale is not None:
        kwargs['input_scale'] = smooth_quant_scale
    output_parallel = self.quant_method.apply(self, input_, **kwargs)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    if self.gather_output and self.tp_size > 1:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add tp_group param to tensor_model_parallel_all_gather
        '''
        # All-gather across the partitions.
        output = tensor_model_parallel_all_gather(output_parallel, dim=-1, tp_group=self.tp_group)
        '''
        =================
        End of MLU Hijack
        =================
        '''
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    if not self.return_bias:
        return output
    return output, output_bias
'''
=================
End of MLU Hijack
=================
'''

'''
=============================
Modify by vllm_mlu
=============================
@brief: add tp_group and keep_full_weights parameters.
'''
def vllm__module_executor__layers__linear__MergedColumnParallelLinear____init__(
    self,
    input_size: int,
    output_sizes: list[int],
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    *,
    tp_group: Any = None,
    keep_full_weights: bool = False,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    self.output_sizes = output_sizes
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: checkout output_sizes after init to get self.tp_world_size
    @brief: add keep_full_weights for dp parallelize shared expert
    '''
    super(MergedColumnParallelLinear, self).__init__(
        input_size=input_size,
        output_size=sum(output_sizes),
        bias=bias,
        gather_output=gather_output,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        output_sizes=self.output_sizes,
        prefix=prefix,
        tp_group=tp_group,
        keep_full_weights=keep_full_weights,
        return_bias=return_bias,
        disable_tp=disable_tp,
    )

    assert all(output_size % self.tp_size == 0 for output_size in output_sizes)

    if self.keep_full_weights:
        tp_size = self.tp_world_size_org
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            out_dim, in_dim = self.weight.shape
            out_dim_tp = divide(out_dim, tp_size)
            self.tp_weight = Parameter(
                self.weight.data.new_empty((out_dim_tp, in_dim)),
                requires_grad=False,
            )
        elif (isinstance(self.quant_method, SmoothQuantLinearMethod)
              and quant_config.input_quant_method == "per_token"):
            out_dim, in_dim = self.qweight.shape
            out_dim_tp = divide(out_dim, tp_size)
            self.tp_qweight = Parameter(
                self.qweight.data.new_empty((out_dim_tp, in_dim)),
                requires_grad=False,
            )
            self.tp_per_channel_scale = Parameter(
                self.per_channel_scale.data.new_empty((out_dim_tp)),
                requires_grad=False,
            )
        else:
            raise TypeError(f"quant method is expected to be unquantized or smoothquant per-token")
    '''
    =================
    End of MLU Hijack
    =================
    '''
'''
=================
End of MLU Hijack
=================
'''


def vllm__module_executor__layers__linear__MergedColumnParallelLinear__weight_loader(
    self,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: int | None = None,
):
    loaded_weight_orig = loaded_weight
    output_dim = getattr(param, "output_dim", None)

    vllm__module_executor__layers__linear__MergedColumnParallelLinear__weight_loader_org(
        self=self,
        param=param,
        loaded_weight=loaded_weight,
        loaded_shard_id=loaded_shard_id,
    )

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add keep_full_weights for dp parallelize shared expert
    '''
    # load into tp weight
    if self.keep_full_weights:
        tp_size = self.tp_world_size_org
        tp_rank = self.tp_rank_org
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        shard_size = self.output_sizes[loaded_shard_id] // tp_size
        start_idx = tp_rank * shard_size
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            tp_weight = loaded_weight_orig.narrow(output_dim, start_idx, shard_size)
            tp_weight_shard = self.tp_weight.narrow(output_dim, shard_offset, shard_size)
            tp_weight_shard.copy_(tp_weight)
        elif isinstance(self.quant_method, SmoothQuantLinearMethod):
            if output_dim is None:
                return
            tp_weight = loaded_weight_orig.narrow(output_dim, start_idx, shard_size)
            if loaded_weight_orig.ndim == 1:
                tp_weight_shard = self.tp_per_channel_scale.narrow(output_dim, shard_offset, shard_size)
            elif loaded_weight_orig.ndim == 2:
                tp_weight_shard = self.tp_qweight.narrow(output_dim, shard_offset, shard_size)
            else:
                raise ValueError("only support rank 1 and 2 when using tp_weight")

            tp_weight_shard.copy_(tp_weight)
        else:
            raise TypeError(f"quant method is expected to be either unquantized or smoothquant")
    '''
    =================
    End of MLU Hijack
    =================
    '''


def vllm__module_executor__layers__linear__RowParallelLinear____init__(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    input_is_parallel: bool = True,
    skip_bias_add: bool = False,
    params_dtype: Optional[torch.dtype] = None,
    reduce_results: bool = True,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    *,
    tp_group: Any = None,
    keep_full_weights: bool = False,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    super(RowParallelLinear, self).__init__(
        input_size,
        output_size,
        skip_bias_add,
        params_dtype,
        quant_config,
        prefix=prefix,
        tp_group=tp_group,
        keep_full_weights=keep_full_weights,
        return_bias=return_bias,
        disable_tp=disable_tp,
    )

    # Divide the weight matrix along the last dimension
    self.input_size_per_partition = divide(input_size, self.tp_size)
    self.output_size_per_partition = output_size
    self.output_partition_sizes = [output_size]

    self.input_is_parallel = input_is_parallel
    self.reduce_results = reduce_results

    assert self.quant_method is not None
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add tp_group in create_weights
    '''
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2
            if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        ),
        tp_group=self.tp_group,
    )
    '''
    =================
    End of MLU Hijack
    =================
    '''
    if not reduce_results and (bias and not skip_bias_add):
        raise ValueError("When not reduce the results, adding bias to the "
                         "results can lead to incorrect results")

    if bias:
        self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
        set_weight_attrs(
            self.bias,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
    else:
        self.register_parameter("bias", None)
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add keep_full_weights for dp parallelize shared expert
    '''
    if self.keep_full_weights:
        tp_size = self.tp_world_size_org
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            out_dim, in_dim = self.weight.data.shape
            in_dim_tp = divide(in_dim, tp_size)
            self.tp_weight = Parameter(self.weight.data.new_empty((out_dim, in_dim_tp)),
                                       requires_grad=False)
        elif (isinstance(self.quant_method, SmoothQuantLinearMethod)
              and quant_config.input_quant_method == "per_token"):
            out_dim, in_dim = self.qweight.data.shape
            in_dim_tp = divide(in_dim, tp_size)
            self.tp_qweight = Parameter(self.qweight.data.new_empty((out_dim, in_dim_tp)),
                                        requires_grad=False)
            if hasattr(self, "smooth"):
                assert len(self.smooth.shape) == 1, "smooth should be a 1D tensor"
                dim = self.smooth.shape[0]
                dim_tp = divide(dim, tp_size)
                self.tp_smooth = Parameter(self.smooth.data.new_empty((dim_tp)),
                                           requires_grad=False)
        else:
            raise TypeError("quant method expected to be unquantized or smoothquant per-token")
    '''
    =================
    End of MLU Hijack
    =================
    '''
    self.update_param_tp_status()


def vllm__module_executor__layers__linear__RowParallelLinear__weight_loader(
    self, param: Parameter, loaded_weight: torch.Tensor
):
    input_dim = getattr(param, "input_dim", None)
    loaded_weight_orig = loaded_weight

    vllm__module_executor__layers__linear__RowParallelLinear__weight_loader_org(
        self=self,
        param=param,
        loaded_weight=loaded_weight,
    )

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add keep_full_weights for dp parallelize shared expert
    '''
    if self.keep_full_weights:
        if input_dim is None:
            return
        tp_size = self.tp_world_size_org
        tp_rank = self.tp_rank_org
        shard_size = divide(loaded_weight_orig.shape[input_dim], tp_size)
        start_idx = tp_rank * shard_size
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            shard_view = self.weight.narrow(input_dim, start_idx, shard_size)
            self.tp_weight.copy_(shard_view)
        elif isinstance(self.quant_method, SmoothQuantLinearMethod):
            if loaded_weight_orig.ndim == 1:
                shard_view = self.smooth.narrow(input_dim, start_idx, shard_size)
                self.tp_smooth.copy_(shard_view)
            elif loaded_weight_orig.ndim == 2:
                shard_view = self.qweight.narrow(input_dim, start_idx, shard_size)
                self.tp_qweight.copy_(shard_view)
            else:
                raise ValueError("only rank 1 and 2 is supported for tp_weight")
        else:
            raise TypeError("quant method is expected to be UnquantizedLinearMethod and SmoothQuant")
    '''
    =================
    End of MLU Hijack
    =================
    '''


'''
=============================
Modify by vllm_mlu
=============================
@brief: add residual, smooth_quant_scale, use_tp_weight and output parameters.
'''
def vllm__module_executor__layers__linear__RowParallelLinear__forward(
    self,
    input_,
    residual: torch.Tensor | None = None,
    smooth_quant_scale: torch.Tensor | None = None,
    use_tp_weight: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    if self.input_is_parallel:
        input_parallel = input_
    else:
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size)
        input_parallel = splitted_input[self.tp_rank].contiguous()

    # Matrix multiply.
    assert self.quant_method is not None
    # Only fuse bias add into GEMM for rank 0 (this ensures that
    # bias will not get added more than once in TP>1 case)
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add additional matmul parameters.
    '''
    residual_ = None if self.tp_rank > 0 else residual
    kwargs = {'bias': bias_, 'residual': residual_}
    if use_tp_weight:
        kwargs['use_tp_weight'] = use_tp_weight
    if smooth_quant_scale is not None:
        kwargs['input_scale'] = smooth_quant_scale
    if output is not None:
        kwargs['output'] = output
    output_parallel = self.quant_method.apply(self, input_parallel, **kwargs)
    '''
    =================
    End of MLU Hijack
    =================
    '''

    if self.reduce_results and self.tp_size > 1:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add tensor_model_parallel_all_reduce() with self.tp_group
        '''
        output = tensor_model_parallel_all_reduce(output_parallel, tp_group=self.tp_group)
        '''
        =================
        End of MLU Hijack
        =================
        '''
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None

    if not self.return_bias:
        return output
    return output, output_bias
'''
=================
End of MLU Hijack
=================
'''


MluHijackObject.apply_hijack(UnquantizedLinearMethod,
                             UnquantizedLinearMethod.apply,
                             vllm__module_executor__layers__linear__UnquantizedLinearMethod__apply)
MluHijackObject.apply_hijack(LinearBase,
                             LinearBase.__init__,
                             vllm__module_executor__layers__linear__LinearBase____init__)
MluHijackObject.apply_hijack(ColumnParallelLinear,
                             ColumnParallelLinear.__init__,
                             vllm__module_executor__layers__linear__ColumnParallelLinear____init__)
MluHijackObject.apply_hijack(ColumnParallelLinear,
                             ColumnParallelLinear.forward,
                             vllm__module_executor__layers__linear__ColumnParallelLinear__forward)
MluHijackObject.apply_hijack(MergedColumnParallelLinear,
                             MergedColumnParallelLinear.__init__,
                             vllm__module_executor__layers__linear__MergedColumnParallelLinear____init__)
MluHijackObject.apply_hijack(MergedColumnParallelLinear,
                             MergedColumnParallelLinear.weight_loader,
                             vllm__module_executor__layers__linear__MergedColumnParallelLinear__weight_loader)
MluHijackObject.apply_hijack(RowParallelLinear,
                             RowParallelLinear.__init__,
                             vllm__module_executor__layers__linear__RowParallelLinear____init__)
MluHijackObject.apply_hijack(RowParallelLinear,
                             RowParallelLinear.weight_loader,
                             vllm__module_executor__layers__linear__RowParallelLinear__weight_loader)
MluHijackObject.apply_hijack(RowParallelLinear,
                             RowParallelLinear.forward,
                             vllm__module_executor__layers__linear__RowParallelLinear__forward)
