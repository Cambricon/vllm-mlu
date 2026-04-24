# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import functools
from functools import partial
import importlib.util
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from typing import Any, Dict, List, Optional, Callable
from vllm import envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.quantization.fp8 import (
    get_flashinfer_moe_backend,
    ACTIVATION_SCHEMES,
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoeBackend,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise, cutlass_block_fp8_supported, cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale,
    maybe_create_device_identity, Fp8LinearOp)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter, ChannelQuantScaleParameter,
    ModelWeightParameter, PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.flashinfer import has_flashinfer_moe
from vllm.utils.import_utils import has_deep_gemm

from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu.model_executor.layers.fused_moe.utils import _fp8_quantize
import vllm_mlu._mlu_ops as mlu_ops


logger = init_logger(__name__)


def get_fp8_moe_backend(block_quant: bool) -> Fp8MoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # Prefer FlashInfer backends on supported GPUs; allow SM90 and SM100.
    if (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(100)
            or current_platform.is_device_capability(90)
        )
        and envs.VLLM_USE_FLASHINFER_MOE_FP8
        and has_flashinfer_moe()
    ):
        backend = get_flashinfer_moe_backend()
        if backend == FlashinferMoeBackend.TENSORRT_LLM:
            logger.info_once("Using FlashInfer FP8 MoE TRTLLM backend for SM100")
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "instead."
                )
            logger.info_once("Using FlashInfer FP8 MoE CUTLASS backend for SM90/SM100")
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: disable marlin for MLU backend.
    '''
    if current_platform.is_rocm() or current_platform.is_out_of_tree():
        use_marlin = False
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    if use_marlin:
        logger.info_once("Using Marlin backend for FP8 MoE")
        return Fp8MoeBackend.MARLIN

    # deepGEMM on supported platforms with block-quantized weights
    if envs.VLLM_USE_DEEP_GEMM and envs.VLLM_MOE_USE_DEEP_GEMM and block_quant:
        if not has_deep_gemm():
            logger.warning_once("DeepGEMM backend requested but not available.")
        elif is_deep_gemm_supported():
            logger.info_once("Using DeepGEMM backend for FP8 MoE")
            return Fp8MoeBackend.DEEPGEMM

    # CUTLASS BlockScaled GroupedGemm on SM100 with block-quantized weights
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(100)
        and block_quant
    ):
        logger.info_once("Using Cutlass BlockScaled GroupedGemm backend for FP8 MoE")
        return Fp8MoeBackend.CUTLASS_BLOCK_SCALED_GROUPED_GEMM

    # default to Triton
    logger.info_once("Using Triton backend for FP8 MoE")
    return Fp8MoeBackend.TRITON


Fp8Config____init____org = Fp8Config.__init__

def vllm__model_executor__layers__quantization__fp8__Fp8Config____init__(
    self,
    is_checkpoint_fp8_serialized: bool = False,
    activation_scheme: str = "dynamic",
    ignored_layers: list[str] | None = None,
    weight_block_size: list[int] | None = None,
    activation_quant_method: Optional[str] = None,
    weight_quant_method: Optional[str] = None,
) -> None:
    super(Fp8Config, self).__init__()

    Fp8Config____init____org(
        self,
        is_checkpoint_fp8_serialized,
        activation_scheme,
        ignored_layers,
        weight_block_size
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add class members activation_quant_method and weight_quant_method to
    indicate the granularity of quantization.
    '''
    self.activation_quant_method = activation_quant_method
    self.weight_quant_method = weight_quant_method

    assert (self.weight_block_size or \
        self.activation_quant_method == "per_token" and self.weight_quant_method == "per_channel"
        and self.activation_scheme == "dynamic"), "Only support block-wise quantization, or "\
            "input dynamic per-token weight per-channel quantization yet."
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


@classmethod
def vllm__model_executor__layers__quantization__fp8__Fp8Config__from_config(
    cls, config: Dict[str, Any]
) -> "Fp8Config":
    quant_method = cls.get_from_keys(config, ["quant_method"])
    is_checkpoint_fp8_serialized = "fp8" in quant_method
    activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
    ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
    weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
    if not ignored_layers:
        ignored_layers = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add config members activation_quant_method and weight_quant_method to
    indicate the granularity of quantization.
    '''
    activation_quant_method = cls.get_from_keys_or(config,
                                                   ["activation_quant_method"],
                                                   'per_token')
    weight_quant_method = cls.get_from_keys_or(config,
                                               ["weight_quant_method"],
                                               None)
    return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                activation_scheme=activation_scheme,
                ignored_layers=ignored_layers,
                weight_block_size=weight_block_size,
                activation_quant_method=activation_quant_method,
                weight_quant_method=weight_quant_method)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    maybe_create_device_identity()

    output_size_per_partition = sum(output_partition_sizes)
    weight_loader = extra_weight_attrs.get("weight_loader")
    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    layer.orig_dtype = params_dtype
    layer.weight_block_size = None

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add tp_group.
    '''
    tp_group = extra_weight_attrs.get("tp_group", None)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    if self.block_quant:
        assert self.weight_block_size is not None
        layer.weight_block_size = self.weight_block_size
        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add tp_group.
    '''
    # WEIGHT
    if self.quant_config.is_checkpoint_fp8_serialized:
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
    else:
        # For non-serialized checkpoints, use original dtype
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            tp_group=tp_group,
        )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    layer.register_parameter("weight", weight)

    # If checkpoint is serialized fp8, load them.
    # Otherwise, wait until process_weights_after_loading.
    if self.quant_config.is_checkpoint_fp8_serialized:
        # WEIGHT SCALE
        if not self.block_quant:
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: Support weight per channel quantization.
            @brief: Add tp_group to enable custom split.
            '''
            if self.weight_per_channel:
                scale = ChannelQuantScaleParameter(
                    data=torch.empty(sum(output_partition_sizes), dtype=torch.float32),
                    output_dim=0,
                    weight_loader=weight_loader,
                    tp_group=tp_group,
                )
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                        dtype=torch.float32),
                    weight_loader=weight_loader,
                )
            scale[:] = torch.finfo(torch.float32).min
            set_weight_attrs(scale, {"scale_type": "weight_scale"})
            layer.register_parameter("weight_scale", scale)
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
            )
            set_weight_attrs(scale, {"scale_type": "weight_scale"})
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)
        else:
            layer.register_parameter("input_scale", None)


def vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod____init__(
    self,
    quant_config: Fp8Config
):
    self.quant_config = quant_config
    self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
    self.out_dtype = torch.get_default_dtype()

    # For GPUs that lack FP8 hardware support, we can leverage the Marlin
    # kernel for fast weight-only FP8 quantization
    self.use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    # Disable marlin for rocm
    if current_platform.is_rocm():
        self.use_marlin = False
    if vllm_is_batch_invariant():
        self.use_marlin = False

    # AITER is only supported on ROCm and only for FP8_FNUZ
    # and at the moment are MI300 series
    self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enaled()
    self.use_deep_gemm = is_deep_gemm_supported()

    self.weight_block_size = self.quant_config.weight_block_size
    self.block_quant = self.weight_block_size is not None
    if self.block_quant:
        # Marlin doesn't support block-wise fp8
        self.use_marlin = False

    self.act_q_static = self.quant_config.activation_scheme == "static"
    if self.weight_block_size:
        self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
    else:
        # Use per-token quantization for better perf if dynamic and cutlass
        if not self.act_q_static and cutlass_fp8_supported():
            self.act_q_group_shape = GroupShape.PER_TOKEN
        else:
            self.act_q_group_shape = GroupShape.PER_TENSOR
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Add config members activation_quant_method and weight_quant_method to
    indicate the granularity of quantization.
    '''
    self.weight_per_channel = (self.quant_config.weight_quant_method == 'per_channel')
    self.activation_per_token = (self.quant_config.activation_quant_method == 'per_token')
    if self.weight_per_channel and self.activation_per_token:
        self.use_marlin = False
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    if self.block_quant:
        assert not self.act_q_static
        assert self.weight_block_size is not None
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(*self.weight_block_size),
            act_quant_group_shape=self.act_q_group_shape,
            cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            use_aiter_and_is_supported=self.use_aiter_and_is_supported,
        )
    else:
        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.act_q_static,
            act_quant_group_shape=self.act_q_group_shape,
        )


Fp8LinearMethod__process_weights_after_loading__org = Fp8LinearMethod.process_weights_after_loading


def vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__process_weights_after_loading(
    self,
    layer: Module,
) -> None:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: For dynamic activation and channel-wise weight quantization,
    additional processing is not needed.
    '''
    if (self.quant_config.is_checkpoint_fp8_serialized
            and self.weight_per_channel
            and self.quant_config.activation_scheme == "dynamic"):
        return
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    Fp8LinearMethod__process_weights_after_loading__org(self=self, layer=layer)


def vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert residual is None, "Fp8Linear residual is not supported yet."

    # if batch invariant mode is enabled, prefer DeepGEMM FP8 path
    # we will use BF16 dequant when DeepGEMM is not supported.
    if vllm_is_batch_invariant():
        if self.block_quant:
            assert self.weight_block_size is not None
            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )
        else:
            # per-tensor/channel: dequant to BF16 and run GEMM
            weight_fp8 = layer.weight.to(torch.bfloat16)
            weight_scale = layer.weight_scale.to(torch.bfloat16)
            if weight_scale.numel() == 1:
                # Per-tensor: simple scalar multiplication
                weight_bf16 = weight_fp8 * weight_scale
            else:
                # Multiple scales (fused modules like QKV)
                # Try to infer correct broadcasting
                # weight is [K, N], scale could be [num_logical_weights]
                # Need to figure out how to broadcast - for now just try
                # direct multiplication
                if (
                    weight_scale.dim() == 1
                    and weight_scale.shape[0] == weight_fp8.shape[0]
                ):
                    # Per-row scaling
                    weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                else:
                    # Fallback
                    weight_bf16 = weight_fp8 * weight_scale
            return torch.nn.functional.linear(x, weight_bf16.t(), bias)

    if self.use_marlin:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    if self.block_quant:
        assert self.weight_block_size is not None
        from vllm_mlu.model_executor.layers.quantization.utils.fp8_utils import (
            apply_w8a8_block_fp8_linear)
        return apply_w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
        )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Use activation per token quantization based on quantization config.
    '''
    return self.fp8_linear.apply(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        out_dtype=self.out_dtype,
        input_scale=layer.input_scale,
        bias=bias,
        weight_per_channel=self.weight_per_channel,
        activation_per_token=self.activation_per_token)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__model_executor__layers__quantization__fp8__Fp8MoEMethod____init__(
    self,
    quant_config: Fp8Config,
    layer: torch.nn.Module
):
    super(Fp8MoEMethod, self).__init__(layer.moe_config)
    self.layer = layer
    self.quant_config = quant_config
    self.weight_block_size = self.quant_config.weight_block_size
    self.block_quant: bool = self.weight_block_size is not None
    self.fp8_backend = get_fp8_moe_backend(self.block_quant)

    self.use_marlin = self.fp8_backend == Fp8MoeBackend.MARLIN
    self.flashinfer_moe_backend: FlashinferMoeBackend | None = None
    if self.fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
        self.flashinfer_moe_backend = FlashinferMoeBackend.TENSORRT_LLM
    elif self.fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        self.flashinfer_moe_backend = FlashinferMoeBackend.CUTLASS
        if self.block_quant:
            assert self.weight_block_size == [128, 128], (
                f"Only support weight_block_size == [128, 128], "
                f"got {self.weight_block_size}"
            )
        self.flashinfer_moe_fn = partial(
            flashinfer_cutlass_moe_fp8,
            moe=self.moe,
            use_deepseek_fp8_block_scale=self.block_quant,
        )

    self.allow_deep_gemm = self.fp8_backend == Fp8MoeBackend.DEEPGEMM
    self.allow_cutlass_block_scaled_grouped_gemm = (
        self.fp8_backend == Fp8MoeBackend.CUTLASS_BLOCK_SCALED_GROUPED_GEMM
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: In mlu, always set self.use_marlin as False.
    '''
    self.use_marlin = False
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__model_executor__layers__quantization__fp8__Fp8MoEMethod__apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
) -> torch.Tensor:
    if enable_eplb:
        assert expert_load_view is not None
        assert logical_to_physical_map is not None
        assert logical_replica_count is not None
        assert isinstance(layer, FusedMoE)
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: Use moe_softmax_topk and moe_sigmoid_topk of mlu_ops to implement FusedMoE.select_experts
    '''
    from vllm_mlu.model_executor.layers.fused_moe.fused_moe import fused_experts
    if scoring_func == "softmax":
        topk_weights, topk_ids = mlu_ops.moe_softmax_topk(
            router_logits,
            top_k,
            renormalize,
            num_expert_group,
            topk_group,
            route_scale=routed_scaling_factor,
        )
    elif scoring_func == "sigmoid":
        topk_weights, topk_ids = mlu_ops.moe_sigmoid_topk(
            router_logits,
            top_k,
            renormalize,
            num_expert_group,
            topk_group,
            routed_scaling_factor,
            e_score_correction_bias,
        )
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # gen_idx
    ori_input_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    router_logits = router_logits.reshape(-1, router_logits.size(-1))
    expert_num = router_logits.size(-1)
    tokens_num = x.size(0)
    expert_size = layer.w13_weight.size(0)
    expand_idx, combine_idx, token_count, cumsum_token_count = mlu_ops.moe_gen_idx(
        topk_ids, expert_num
    )

    expand_hidden_states = mlu_ops.moe_expand_input(
        x, expand_idx, cumsum_token_count, 0, expert_size
    )
    quant_input, input_scale = _fp8_quantize(
        expand_hidden_states, A_scale=None, block_shape=self.quant_config.weight_block_size
    )
    gemm1_out = mlu_ops.smooth_quant_group_gemm(
        quant_input,
        layer.w13_weight,
        token_count,
        expand_idx=None,
        c=None,
        alpha=None,
        beta=None,
        a_scale=input_scale.T.contiguous(),
        b_scale=layer.w13_weight_scale_inv,
        dtype=x.dtype,
        max_m=tokens_num,
    )

    act_out = mlu_ops.active(gemm1_out, activation, is_gated=True)
    act_out_quantize, act_out_scale = _fp8_quantize(
        act_out, A_scale=None, block_shape=self.quant_config.weight_block_size
    )

    gemm2_out = mlu_ops.smooth_quant_group_gemm(
        act_out_quantize,
        layer.w2_weight,
        token_count,
        expand_idx=None,
        c=None,
        alpha=None,
        beta=None,
        a_scale=act_out_scale.T.contiguous(),
        b_scale=layer.w2_weight_scale_inv,
        dtype=x.dtype,
        max_m=tokens_num,
    )

    output = mlu_ops.moe_combine_result(
        gemm2_out,
        topk_weights,
        combine_idx,
        residual=None,
        cusum_token_count=cumsum_token_count,
        start_expert_id=0,
        expert_size=expert_size,
        bias=None,
    )
    return output.view(ori_input_shape)

    """
    ==================
    End of MLU Hijack
    ==================
    """


MluHijackObject.apply_hijack(
    Fp8LinearMethod,
    Fp8LinearMethod.apply,
    vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__apply
)
MluHijackObject.apply_hijack(
    Fp8Config,
    Fp8Config.__init__,
    vllm__model_executor__layers__quantization__fp8__Fp8Config____init__
)
MluHijackObject.apply_hijack(
    Fp8Config,
    Fp8Config.from_config,
    vllm__model_executor__layers__quantization__fp8__Fp8Config__from_config
)
MluHijackObject.apply_hijack(
    Fp8LinearMethod,
    Fp8LinearMethod.create_weights,
    vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__create_weights
)
MluHijackObject.apply_hijack(
    Fp8LinearMethod,
    Fp8LinearMethod.__init__,
    vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod____init__
)
MluHijackObject.apply_hijack(
    Fp8LinearMethod,
    Fp8LinearMethod.process_weights_after_loading,
    vllm__model_executor__layers__quantization__fp8__Fp8LinearMethod__process_weights_after_loading
)
MluHijackObject.apply_hijack(
    Fp8MoEMethod,
    Fp8MoEMethod.__init__,
    vllm__model_executor__layers__quantization__fp8__Fp8MoEMethod____init__
)
MluHijackObject.apply_hijack(
    Fp8MoEMethod,
    Fp8MoEMethod.apply,
    vllm__model_executor__layers__quantization__fp8__Fp8MoEMethod__apply
)
