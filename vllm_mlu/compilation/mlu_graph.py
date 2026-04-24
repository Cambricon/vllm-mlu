# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch

from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensors
from vllm.compilation.cuda_graph import (
    CUDAGraphEntry,
    CUDAGraphWrapper,
    CUDAGraphOptions,
)
from vllm_mlu.v1.attention.backends.utils import MLUInferMode

logger = init_logger(__name__)


'''
=============================
Modify by vllm_mlu
=============================
@brief: specialized graph entry for prefill graphs
'''
@dataclasses.dataclass
class PrefillGraphEntry:
    batch_size: int = 0
    seq_len: int = 0
    cudagraph: torch.mlu.MLUGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None
'''
==================
End of MLU Hijack
==================
'''


class MLUGraphWrapper(CUDAGraphWrapper):

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        super().__init__(runnable, vllm_config, runtime_mode, cudagraph_options)
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: add separate dict for prefill graph entries
        '''
        self.prefill_mlugraph_entry: PrefillGraphEntry | None = None
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: check if running in prefill mode
    '''
    def is_running_in_prefill(self, entry: PrefillGraphEntry | None = None) -> bool:
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            return False
        infer_mode = forward_context.attn_metadata['common_metadata'].infer_mode
        seq_lens_cpu = forward_context.attn_metadata['common_metadata'].seq_lens_cpu
        if entry is not None \
            and infer_mode == MLUInferMode.PREFILL_ONLY \
            and seq_lens_cpu.size(0) == entry.batch_size \
            and (seq_lens_cpu == entry.seq_len).all().item():
            return True
        return False
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    def __call__(
            self, 
            is_capturing_prefill: bool = False,
            prefill_enable_mlugraph: bool = False,
            prefill_batch_size: int = 0,
            prefill_seq_len: int = 0,
            is_running_drafter: bool = False,
            *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: handle prefill graph separately
        @brief: skip check in running drafter model
        '''
        if is_capturing_prefill: # PREFILL capture
            self.prefill_mlugraph_entry = PrefillGraphEntry(
                batch_size=prefill_batch_size,
                seq_len=prefill_seq_len)
        else: # FULL/DECODE capture
            if batch_descriptor not in self.concrete_cudagraph_entries:
                # create a new entry for this batch descriptor
                self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                    batch_descriptor=batch_descriptor
                )

        if ((self.is_running_in_prefill(self.prefill_mlugraph_entry) and prefill_enable_mlugraph) 
            or is_capturing_prefill):
            entry = self.prefill_mlugraph_entry
            logger.debug(
                        f"Hitting a prefill cudagraph on {self.runtime_mode.name}, "
                        f"batch_size: {entry.batch_size}, seq_len: {entry.seq_len}")
        else: # FULL/DECODE capture
            entry = self.concrete_cudagraph_entries[batch_descriptor]
            logger.debug(
                        "Hitting a decode cudagraph on (%s, %s)",
                        self.runtime_mode.name,
                        entry.batch_descriptor,
                    )

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                if is_capturing_prefill:
                    logger.debug(
                        "Capturing a prefill cudagraph on (%s, batch_size=%d, seq_len=%d)",
                        self.runtime_mode.name,
                        entry.batch_size,
                        entry.seq_len,
                    )
                else:
                    logger.debug(
                        "Capturing a decode cudagraph on (%s, %s)",
                        self.runtime_mode.name,
                        entry.batch_descriptor,
                    )
            if ((not is_capturing_prefill) and (not is_running_drafter)):
                # validate that cudagraph capturing is legal at this point.
                validate_cudagraph_capturing_enabled()
            '''
            ==================
            End of MLU Hijack
            ==================
            '''

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.mlu.MLUGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.mlu.empty_cache", lambda: None))

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())
                # mind-exploding: carefully manage the reference and memory.
                with torch.mlu.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
