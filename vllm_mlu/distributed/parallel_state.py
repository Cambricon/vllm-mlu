# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from contextlib import contextmanager, nullcontext
from typing import Optional
from dataclasses import dataclass

import torch

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    GraphCaptureContext,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.mlu_parallel_state import(
    get_moe_expert_parallel_world_size,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_group,
)
from vllm.logger import init_logger

from vllm_mlu.mlu_hijack_utils import MluHijackObject
from vllm_mlu import _mlu_ops as mlu_ops

logger = init_logger(__name__)

@dataclass
class MLUGraphCaptureContext:
    stream: torch.mlu.Stream


@contextmanager
def mlu_graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    context = MLUGraphCaptureContext(torch.mlu.Stream(device=device))
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
        yield context


@contextmanager
def vllm__distributed__parallel_state__GroupCoordinator__graph_capture(
    self,
    graph_capture_context: GraphCaptureContext | None = None,
):
    if graph_capture_context is None:
        stream = torch.mlu.Stream()
        graph_capture_context = GraphCaptureContext(stream)
    else:
        stream = graph_capture_context.stream

        # only cuda uses this function,
        # so we don't abstract it into the base class
        maybe_ca_context = nullcontext()
        from vllm_mlu.distributed.device_communicators.mlu_communicator import (
            MLUCommunicator,
        )

        if self.device_communicator is not None:
            assert isinstance(self.device_communicator, MLUCommunicator)
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()  # type: ignore

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.mlu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.mlu.stream(stream), maybe_ca_context:
        yield graph_capture_context

@dataclass
class CnclEPBuffer:
    dispatch_send_token_tensor: torch.Tensor
    dispatch_recv_token_tensor: torch.Tensor
    combine_send_token_tensor: torch.Tensor
    combine_recv_token_tensor: torch.Tensor

class CnclEP:

    def __init__(self,
                 dispatch_token_size: int,
                 combine_token_size: int,
                 max_num_tokens_per_rank: int,
                 num_global_experts: int,
                 use_quant_dispatch: bool = True) -> None:
        nranks = get_moe_expert_parallel_world_size()
        rank = get_moe_expert_parallel_rank()
        moe_ep_group = get_moe_expert_parallel_group()
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.use_quant_dispatch = use_quant_dispatch

        (
            handle,
            exchange_info_size,
            exchange_info,
            dispatch_send_token_tensor,
            dispatch_recv_token_tensor,
            combine_send_token_tensor,
            combine_recv_token_tensor
        ) = mlu_ops.moe_all2all_create(dispatch_token_size,
                          combine_token_size,
                          num_global_experts,
                          max_num_tokens_per_rank,
                          rank,
                          nranks)
        self.handle = handle
        self.buffer = CnclEPBuffer(
            dispatch_send_token_tensor,
            dispatch_recv_token_tensor,
            combine_send_token_tensor,
            combine_recv_token_tensor)

        assert exchange_info.ndim == 1, "exchange_info should be 1D"
        all_exchange_info = torch.empty((nranks, exchange_info.size(0)), 
                                        dtype=exchange_info.dtype, 
                                        device=exchange_info.device)
        exchange_info = exchange_info.unsqueeze(0)
        torch.distributed.all_gather_into_tensor(all_exchange_info,
                                                 exchange_info,
                                                 group=moe_ep_group.cpu_group,
                                                 async_op=False)
        mlu_ops.moe_all2all_init(self.handle, all_exchange_info)
        torch.distributed.barrier(group=moe_ep_group.cpu_group)

    def dispatch(self,
                 token_byte: int,
                 token_num: int,
                 send_layout: torch.Tensor,                   
                 send_token_num: torch.Tensor,                
                 recv_layout: torch.Tensor,                   
                 recv_token_num: torch.Tensor,                
                 send_token: Optional[torch.Tensor] = None,   
                 recv_token: Optional[torch.Tensor] = None,   
    ) -> None:
        '''
        The returned tensors are in-placed modified, we could directly use them
        after dispatch finishes.
        '''
        mlu_ops.moe_all2all_dispatch(self.handle,
                                     token_byte,
                                     token_num,
                                     send_layout,                   
                                     send_token_num,                
                                     recv_layout,                   
                                     recv_token_num,                
                                     send_token,   
                                     recv_token)   

    def combine(self,
                token_byte: int,
                token_num: int,
                send_src_layout: torch.Tensor,
                send_dst_layout: torch.Tensor,
                send_token: Optional[torch.Tensor] = None,
                recv_token: Optional[torch.Tensor] = None,
    ) ->None:
        mlu_ops.moe_all2all_combine(self.handle,
                                    token_byte,
                                    token_num,
                                    send_src_layout,
                                    send_dst_layout,
                                    send_token,
                                    recv_token)

    def destroy(self) -> None:
        mlu_ops.moe_all2all_destroy(self.handle)

_CNCLEP: CnclEP | None = None
_CNCLEP_BF16: CnclEP | None = None

def get_cnclep(use_quant_dispatch: bool = True) -> CnclEP:
    if use_quant_dispatch:
        assert _CNCLEP is not None, "cnclep is not initialized"
        return _CNCLEP
    else:
        assert _CNCLEP_BF16 is not None, "cnclep_bf16 is not initialized"
        return _CNCLEP_BF16

def init_cnclep(dispatch_token_size: int,
                combine_token_size: int,
                max_num_tokens_per_rank: int,
                num_global_experts: int,
                use_quant_dispatch: bool = True):
    if use_quant_dispatch:
        global _CNCLEP
        assert _CNCLEP is None, "cnclep has been initialized"
        _CNCLEP = CnclEP(dispatch_token_size, 
                         combine_token_size,
                         max_num_tokens_per_rank,
                         num_global_experts,
                         use_quant_dispatch)
    else:
        global _CNCLEP_BF16
        assert _CNCLEP_BF16 is None, "cnclep_bf16 has been initialized"
        _CNCLEP_BF16 = CnclEP(dispatch_token_size, 
                              combine_token_size,
                              max_num_tokens_per_rank,
                              num_global_experts,
                              use_quant_dispatch)

def cnclep_dispatch(token_byte: int,
                    token_num: int,
                    send_layout: torch.Tensor,                   
                    send_token_num: torch.Tensor,                
                    recv_layout: torch.Tensor,                   
                    recv_token_num: torch.Tensor,                
                    send_token: Optional[torch.Tensor] = None,   
                    recv_token: Optional[torch.Tensor] = None,   
                    use_quant_dispatch: bool = True,
):
    if use_quant_dispatch:
        _CNCLEP.dispatch(token_byte,
                         token_num,
                         send_layout,                   
                         send_token_num,                
                         recv_layout,                   
                         recv_token_num,                
                         send_token,   
                         recv_token)   
    else:
        _CNCLEP_BF16.dispatch(token_byte,
                              token_num,
                              send_layout,                   
                              send_token_num,                
                              recv_layout,                   
                              recv_token_num,                
                              send_token,   
                              recv_token)   

def cnclep_combine(token_byte: int,
                   token_num: int,
                   send_src_layout: torch.Tensor,
                   send_dst_layout: torch.Tensor,
                   send_token: Optional[torch.Tensor] = None,
                   recv_token: Optional[torch.Tensor] = None,
                   use_quant_dispatch: bool = True,
): 
    if use_quant_dispatch:
        _CNCLEP.combine(token_byte,
                        token_num,
                        send_src_layout,
                        send_dst_layout,
                        send_token,
                        recv_token)
    else:
        _CNCLEP_BF16.combine(token_byte,
                             token_num,
                             send_src_layout,
                             send_dst_layout,
                             send_token,
                             recv_token)


def destroy_cnclep():
    global _CNCLEP

    if _CNCLEP:
        _CNCLEP.destroy()
    _CNCLEP = None

    global _CNCLEP_BF16

    if _CNCLEP_BF16:
        _CNCLEP_BF16.destroy()
    _CNCLEP_BF16 = None


MluHijackObject.apply_hijack(GroupCoordinator,
                             GroupCoordinator.graph_capture,
                             vllm__distributed__parallel_state__GroupCoordinator__graph_capture)

