# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import os

from vllm.config.vllm import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


def vllm__config__vllm__VllmConfig___set_cudagraph_sizes(self):
    """
    vLLM defines the default candidate list of batch sizes for CUDA graph
    capture as:

    ```python
    max_graph_size = min(max_num_seqs * 2, 512)
    # 1, 2, 4, then multiples of 8 up to 256 and then multiples of 16
    # up to max_graph_size
    cuda_graph_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
        range(256, max_graph_size + 1, 16))

    In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
    will be the final sizes to capture cudagraph (in ascending order).

    These sizes are used to capture and reuse CUDA graphs for
    performance-critical paths (e.g., decoding). Capturing enables
    significantly faster kernel dispatch by avoiding Python overhead. The
    list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
    most GPUs), which controls the total allowed number of tokens in a
    batch. Since each sequence may have a variable number of tokens, the
    maximum usable batch size will depend on actual sequence lengths.

    Example:
        With `max_num_batched_tokens = 8192`, and typical sequences
        averaging ~32 tokens, most practical batch sizes fall below 256.
        However, the system will still allow capture sizes up to 512 if
        shape and memory permit.

    Note:
        If users explicitly specify cudagraph capture sizes in the
        compilation config, those will override this default logic.
        At runtime:

        - If batch size <= one of the `cudagraph_capture_sizes`, the closest
        padded CUDA graph will be used.
        - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
        not be used.
    """
    if hasattr(self.compilation_config, "_has_set_capture_list"):
        # avoid set capture list twice while init
        return

    if (
        self.model_config is not None
        and not self.model_config.enforce_eager
        and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        # determine the initial max_cudagraph_capture_size
        max_cudagraph_capture_size = (
            self.compilation_config.max_cudagraph_capture_size
        )
        if max_cudagraph_capture_size is None:
            max_cudagraph_capture_size = min(
                self.scheduler_config.max_num_seqs * 2, 512
            )
        max_num_tokens = self.scheduler_config.max_num_batched_tokens
        max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

        assert max_cudagraph_capture_size >= 1, (
            "Maximum cudagraph size should be greater than or equal to 1 "
            "when using cuda graph."
        )

        # determine the cudagraph_capture_sizes
        if self.compilation_config.cudagraph_capture_sizes is not None:
            assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                "cudagraph_capture_sizes should contain at least one element "
                "when using cuda graph."
            )
            # de-duplicate the sizes provided by the config
            dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
            cudagraph_capture_sizes = [
                i for i in dedup_sizes if i <= max_num_tokens
            ]
            # sort to make sure the sizes are in ascending order
            cudagraph_capture_sizes.sort()
        else:
            cudagraph_capture_sizes = [
                i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
            ]
            if max_cudagraph_capture_size >= 8:
                # Step size 8 for small batch sizes, up to 256(not included)
                cudagraph_capture_sizes += list(
                    range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                )
            if max_cudagraph_capture_size >= 256:
                # Step size 16 for larger batch sizes
                cudagraph_capture_sizes += list(
                    range(256, max_cudagraph_capture_size + 1, 16)
                )

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief:
        1) check batch_size_capture_list when enable mtp because bs * (K + 1)
        may greater than max_num_batched_tokens
        2) capture MLUGraph by given batch list
        '''
        mlu_graph_capture_list = os.getenv("MLU_GRAPH_CAPTURE_LIST", None)
        if mlu_graph_capture_list:
            if "-" in mlu_graph_capture_list:
                batch_info = mlu_graph_capture_list.split("-")
                assert len(batch_info) == 3, \
                    f"Got invalid graph_capture_list={mlu_graph_capture_list}, " + \
                    f"but expected format 'min_bs-max_bs(may not include)-step'."
                start, end, step = mlu_graph_capture_list.split("-")
                cudagraph_capture_sizes = [1, 2, 4] + [
                    i for i in range(int(start), int(end), int(step))
                ]
                cudagraph_capture_sizes = sorted(list(set(cudagraph_capture_sizes)))
            else:
                cudagraph_capture_sizes = [int(x) for x in mlu_graph_capture_list.split(",")]

        if (self.speculative_config is not None
            and self.speculative_config.num_speculative_tokens > 0
        ):
            K = self.speculative_config.num_speculative_tokens
            cudagraph_capture_sizes = [x * (1 + K) for x in cudagraph_capture_sizes]

        cudagraph_capture_sizes = [
            size for size in cudagraph_capture_sizes
            if size <= self.scheduler_config.max_num_batched_tokens
        ]
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        if (
            self.parallel_config.tensor_parallel_size > 1
            and self.compilation_config.pass_config.enable_sequence_parallelism
        ):
            cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                cudagraph_capture_sizes
            )

        # user-specific compilation_config.max_cudagraph_capture_size get
        # truncated to valid_max_size when they are inconsistent.
        valid_max_size = (
            cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
        )
        if (
            self.compilation_config.max_cudagraph_capture_size is not None
            and self.compilation_config.max_cudagraph_capture_size != valid_max_size
        ):
            # raise error only when both two flags are user-specified
            # and they are inconsistent with each other
            if self.compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "customized max_cudagraph_capture_size"
                    f"(={self.compilation_config.max_cudagraph_capture_size}) "
                    "should be consistent with the max value of "
                    f"cudagraph_capture_sizes(={valid_max_size})"
                )

            logger.warning(
                "Truncating max_cudagraph_capture_size to %d",
                valid_max_size,
            )
        # always set the final max_cudagraph_capture_size
        self.compilation_config.max_cudagraph_capture_size = valid_max_size

        if self.compilation_config.cudagraph_capture_sizes is not None and len(
            cudagraph_capture_sizes
        ) < len(self.compilation_config.cudagraph_capture_sizes):
            # If users have specified capture sizes, we only need to
            # compare the lens before and after modification since the modified
            # list is only the subset of the original list.
            logger.warning(
                (
                    "cudagraph_capture_sizes specified in compilation_config"
                    " %s is overridden by config %s"
                ),
                self.compilation_config.cudagraph_capture_sizes,
                cudagraph_capture_sizes,
            )
        # always write back the final sizes
        self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

    else:
        # no cudagraph in use
        self.compilation_config.max_cudagraph_capture_size = 0
        self.compilation_config.cudagraph_capture_sizes = []

    # complete the remaining process.
    self.compilation_config.post_init_cudagraph_sizes()

    setattr(self.compilation_config, "_has_set_capture_list", True)


MluHijackObject.apply_hijack(
    VllmConfig,
    VllmConfig._set_cudagraph_sizes,
    vllm__config__vllm__VllmConfig___set_cudagraph_sizes,
)