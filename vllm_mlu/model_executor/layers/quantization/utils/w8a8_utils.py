# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Optional, Callable
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, USE_ROWWISE_TORCH_SCALED_MM, cutlass_w8a8_scaled_mm,
    flashinfer_w8a8_scaled_mm, rocm_per_tensor_w8a8_scaled_mm,
    torch_per_tensor_w8a8_scaled_mm, torch_per_token_w8a8_scaled_mm,
    torch_channelwise_w8a8_scaled_mm)
from vllm.platforms import current_platform

from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.mlu_hijack_utils import MluHijackObject


def mlu_w8a8_scaled_mm(
    qinput: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype,
    scale_a: torch.Tensor, scale_b: torch.Tensor, bias: torch.Tensor,
    output_shape: list, **kwargs
) -> torch.Tensor:
    output = mlu_ops.scaled_matmul(
        qinput, # a
        weight, # b
        scale_a, # a_scale
        scale_b, # b_scale
        out_dtype, # output_dtype
        bias, # bias
        c=None, act_mode="none",quant_bit_size=8, alpha=1, beta=1, use_hp_active=False,
                              a_quant_bit_size=8, a_calib=None, b_calib=None
    )
    return output.view(*output_shape)


def dispatch_w8a8_scaled_mm(
    preferred_backend: str, per_tensor_weights: bool, per_tensor_activations: bool,
    weight_per_channel: bool, activation_per_token: bool
) -> Callable[..., torch.Tensor]:
    if per_tensor_weights and per_tensor_activations:
        if preferred_backend == "rocm":
            return rocm_per_tensor_w8a8_scaled_mm
        if preferred_backend == "flashinfer":
            return flashinfer_w8a8_scaled_mm
        if preferred_backend == "cutlass":
            return cutlass_w8a8_scaled_mm
        return torch_per_tensor_w8a8_scaled_mm

    # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
    if preferred_backend == "cutlass" or preferred_backend == "flashinfer":
        return cutlass_w8a8_scaled_mm

    # If torch.scaled_mm supports per-channel (weights) per-token (inputs)
    if (
        not per_tensor_weights
        and not per_tensor_activations
        and USE_ROWWISE_TORCH_SCALED_MM
    ):
        return torch_per_token_w8a8_scaled_mm
    # Normally, torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: dispatch to mlu_w8a8_scaled_mm
    '''
    if weight_per_channel and activation_per_token:
        return mlu_w8a8_scaled_mm
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return torch_channelwise_w8a8_scaled_mm


def vllm__model_executor__layers__quantization__utils__w8a8_util__Fp8LinearOp__apply(
    self,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    input_scale: torch.Tensor | None = None,
    input_scale_ub: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    weight_per_channel: bool = True,
    activation_per_token: bool = True,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add mlu_fp8_supported
    '''
    self.mlu_fp8_supported = False
    if weight_per_channel and activation_per_token:
        self.mlu_fp8_supported = True
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if out_dtype is None:
        out_dtype = input.dtype

    if self.mlu_fp8_supported:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: Add support for activation-per-token weight-per-channel quantization.
        '''
        qinput, x_scale = mlu_ops.scaled_quantize(
            input_2d,# x
            None, # scale
            None, # zero
            None, # scale_ub
            quant_type=torch.float8_e4m3fn,
            quant_mode='dynamic_per_token'
        )
        output_shape = [*input.shape[:-1], weight.shape[0]]
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
    else:
        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        if input.dtype != current_platform.fp8_dtype():
            qinput, x_scale = self.quant_fp8(
                input_2d,
                input_scale,
                input_scale_ub,
            )
        else:
            qinput, x_scale = input_2d, input_scale

    # Must have dim() conditions
    # In per-token quant scenario, when the number of token is 1,
    # the scale will only have 1 elements.
    # Without checking the dim(),
    # we cannot distingushes between per-tensor and per-token quant.
    # Example:
    # When the number of token is 1, per-token scale is [[1]]
    # When per-tensor scale is [1] or ().
    per_tensor_weights = weight_scale.numel() == 1
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

    # TODO(luka) do this dispatch during init (after ScaledMM refactor)
    w8a8_scaled_mm_func = dispatch_w8a8_scaled_mm(
        self.preferred_backend, per_tensor_weights, per_tensor_activations,
        weight_per_channel, activation_per_token)
    return w8a8_scaled_mm_func(
        qinput=qinput,
        weight=weight,
        out_dtype=out_dtype,
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=bias,
        output_shape=output_shape,
    )


MluHijackObject.apply_hijack(
    Fp8LinearOp,
    Fp8LinearOp.apply,
    vllm__model_executor__layers__quantization__utils__w8a8_util__Fp8LinearOp__apply
)