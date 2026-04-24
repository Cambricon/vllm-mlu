# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Tuple

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu import _mlu_ops as mlu_ops
from vllm_mlu.model_executor.models.layer_utils import is_per_token_smoothquant


@CustomOp.register("quant_fusion_rms_norm")
class QuantFusionRMSNorm(RMSNorm):
    def __init__(self, hidden_size: int, variance_epsilon: float, proj: LinearBase):
        super().__init__(hidden_size, variance_epsilon)
        assert not isinstance(
            proj.quant_method, UnquantizedLinearMethod
        ), f"UnquantizedLinearMethod of {proj.__class__.__name__} is not supported"
        proj.quant_method.skip_quant_input = True
        if dynamic_quant := is_per_token_smoothquant(proj.quant_method.quant_config):
            quant_scale = proj.smooth.data
        else:
            quant_scale = proj.scale_to_int.data
        self.dynamic_quant = dynamic_quant
        self.quant_scale = torch.nn.Parameter(quant_scale)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None:
        return mlu_ops.fused_rms_norm(
            x,
            residual,
            self.weight.data,
            None,
            None,
            self.variance_epsilon,
            False,
            self.quant_scale.data,
            self.dynamic_quant,
        )


@CustomOp.register("quant_fusion_layer_norm")
class QuantFusionLayerNorm(torch.nn.LayerNorm, CustomOp):
    def __init__(self, hidden_size: int, variance_epsilon: float, proj: LinearBase):
        super().__init__(hidden_size, variance_epsilon)
        assert not isinstance(
            proj.quant_method, UnquantizedLinearMethod
        ), f"UnquantizedLinearMethod of {proj.__class__.__name__} is not supported"
        proj.quant_method.skip_quant_input = True
        if dynamic_quant := is_per_token_smoothquant(proj.quant_method.quant_config):
            quant_scale = proj.smooth.data
        else:
            quant_scale = proj.scale_to_int.data
        self.dynamic_quant = dynamic_quant
        self.quant_scale = torch.nn.Parameter(quant_scale)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None:
        bias = None if self.bias is None else self.bias.data
        return mlu_ops.fused_layer_norm(
            x,
            residual,
            self.weight.data,
            bias,
            None,
            self.eps,
            False,
            self.quant_scale.data,
            self.dynamic_quant,
        )


def vllm__model_executor__layers__layernorm__RMSNorm__forward_oot(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None:

    org_shape = x.shape
    x = x.reshape(-1, self.weight.data.shape[0])
    if out is not None:
        out = out.view(-1, self.weight.data.shape[0])
    if residual is not None:
        residual = residual.view(-1, self.weight.data.shape[0])
        x = mlu_ops.fused_rms_norm(
            x,
            residual,
            self.weight.data,
            None,
            None,
            self.variance_epsilon,
            True,
            out=out,
        )
    else:
        x = mlu_ops.fused_rms_norm(
            x,
            residual,
            self.weight.data,
            None,
            None,
            self.variance_epsilon,
            False,
            out=out,
        )

    if out is not None:
        return x

    if residual is None:
        assert isinstance(x, torch.Tensor)
        return x.view(org_shape)
    
    assert isinstance(x, tuple)
    assert len(x) == 2
    return x[0].view(org_shape), x[1].view(org_shape)

MluHijackObject.apply_hijack(
    RMSNorm,
    RMSNorm.forward_oot,
    vllm__model_executor__layers__layernorm__RMSNorm__forward_oot,
)
