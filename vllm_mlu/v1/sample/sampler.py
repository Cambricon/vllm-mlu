# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config.model import LogprobsMode
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS

from vllm_mlu._mlu_utils import *
from vllm_mlu import _mlu_ops as mlu_ops

"""
@brief: use tmo random_sample
"""
def mlu_random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    is_gumbel_max = True
    return mlu_ops.random_sample(probs, is_gumbel_max, generators).view(-1)

class MluSampler(Sampler):
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None
        """
        =============================
        Modify by vllm_mlu
        =============================
        @brief: use tmo topk_topp_sampler to sample.
        """
        use_tmo = (sampling_metadata.top_k is not None) or (sampling_metadata.top_p is not None)
        if use_tmo:
            batch_size, vocab_size = logits.shape
            index_in = torch.arange(vocab_size, dtype=torch.int32, device=logits.device)
            (
                logits_out,
                sorted_logits_out,
                index_out,
                true_select_len,
            ) = mlu_ops.apply_topkp_v2(
                logits,
                index_in,
                sampling_metadata.temperature,
                None,
                sampling_metadata.top_k.to(torch.int32) if sampling_metadata.top_k is not None else None,
                sampling_metadata.top_p,
            )

            processed_logprobs = None
            if logprobs_mode == "processed_logits":
                processed_logprobs = logits
            elif logprobs_mode == "processed_logprobs":
                processed_logprobs = self.compute_logprobs(logits)

            probs = logits_out.softmax(dim=-1, dtype=torch.float32)
            random_sampled = mlu_random_sample(probs, sampling_metadata.generators)
        else:
            # Apply temperature.
            logits = self.apply_temperature(
                logits, sampling_metadata.temperature, sampling_metadata.all_random
            )

            # Apply logits processors that only apply to random sampling
            # (argmax invariant)
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            # Apply top_k and/or top_p.
            random_sampled, processed_logprobs = self.topk_topp_sampler(
                logits,
                sampling_metadata.generators,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )
        """
        =================
        End of MLU Hijack
        =================
        """

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs
