# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project


from typing_extensions import Self

from vllm.config.scheduler import SchedulerConfig
from vllm.logger import init_logger

from vllm_mlu._mlu_utils import VLLM_V1_BENCHMARK
from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


def vllm__config__scheduler__SchedulerConfig__verify_max_model_len(
    self, max_model_len: int,
) -> Self:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: This restriction is removed when VLLM_V1_BENCHMARK is set to True
    '''
    if not VLLM_V1_BENCHMARK:
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    if self.max_num_batched_tokens < self.max_num_seqs:
        raise ValueError(
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
            "be greater than or equal to max_num_seqs "
            f"({self.max_num_seqs})."
        )

    if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
        logger.warning(
            "max_num_batched_tokens (%d) exceeds max_num_seqs "
            "* max_model_len (%d). This may lead to unexpected behavior.",
            self.max_num_batched_tokens,
            self.max_num_seqs * max_model_len,
        )

    if self.max_num_partial_prefills > 1:
        if not self.enable_chunked_prefill:
            raise ValueError(
                "Chunked prefill must be enabled to set "
                "max_num_partial_prefills > 1."
            )

        if self.long_prefill_token_threshold > max_model_len:
            raise ValueError(
                "long_prefill_token_threshold "
                f"({self.long_prefill_token_threshold}) cannot be greater "
                f"than the max_model_len ({max_model_len})."
            )

    if self.max_long_partial_prefills > self.max_num_partial_prefills:
        raise ValueError(
            f"{self.max_long_partial_prefills=} must be less than or equal to "
            f"{self.max_num_partial_prefills=}."
        )

    return self


MluHijackObject.apply_hijack(
    SchedulerConfig,
    SchedulerConfig.verify_max_model_len,
    vllm__config__scheduler__SchedulerConfig__verify_max_model_len,
)