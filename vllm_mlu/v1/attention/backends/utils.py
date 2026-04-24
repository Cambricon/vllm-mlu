# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import os
import numpy as np
import pandas as pd
import torch
from typing import TYPE_CHECKING, Union

from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata


COMMON_METADATA_STR: str = "common_metadata"


class MLUInferMode(Enum):
    CHUNKED      = 1
    PREFILL_ONLY = 2
    DECODE_ONLY  = 3

    @classmethod
    def build(
        cls,
        max_query_len,
        max_computed_tokens,
        uniform_decode_query_len: int = 1,
    ) -> Enum:
        if max_query_len <= uniform_decode_query_len:
            return MLUInferMode.DECODE_ONLY
        elif max_computed_tokens == 0:
            return MLUInferMode.PREFILL_ONLY
        else:
            return MLUInferMode.CHUNKED

    @property
    def is_prefill_only(self):
        return self == MLUInferMode.PREFILL_ONLY

    @property
    def is_decode_only(self):
        return self == MLUInferMode.DECODE_ONLY

    @property
    def is_chunked(self):
        return self == MLUInferMode.CHUNKED


@dataclass
class MLUCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """
    seq_start_loc: torch.Tensor | None = None
    seq_start_loc_cpu: torch.Tensor | None = None
    """(batch_size + 1,), the start location of each request in the input key/value sequence."""
    num_input_tokens: int = 0
    """Number of query tokens with padding."""
    num_prefill_query_tokens: int = 0
    """Number of query tokens in prefill phase."""
    num_prefill_kv_tokens: int = 0
    """Number of key/value tokens in prefill phase."""
    infer_mode: MLUInferMode | None = None
    """Inference mode for flash attention."""

    @property
    def is_prefill_only(self):
        return self.infer_mode == MLUInferMode.PREFILL_ONLY

    @property
    def is_decode_only(self):
        return self.infer_mode == MLUInferMode.DECODE_ONLY

    @property
    def is_chunked(self):
        return self.infer_mode == MLUInferMode.CHUNKED

    @classmethod
    def build(
        cls,
        query_start_loc, query_start_loc_cpu,
        seq_lens, seq_lens_cpu,
        num_computed_tokens_cpu,
        num_reqs, num_actual_tokens, max_query_len,
        block_table_tensor, slot_mapping,
        seq_start_loc, is_start_loc_match,
        num_input_tokens: int = 0,
        num_speculative_tokens: int = 0,
        has_prefill_reqs: bool = False
    ):
        """Build attention metadata for MLU inference.
        
        Args:
            has_prefill_reqs: Whether there are pending prefill requests with chunked.
        """
        infer_mode = None
        if is_start_loc_match:
            infer_mode = MLUInferMode.PREFILL_ONLY
        elif max_query_len <= (1 + num_speculative_tokens) and (not has_prefill_reqs):
            infer_mode = MLUInferMode.DECODE_ONLY
        else:
            infer_mode = MLUInferMode.CHUNKED
        num_input_tokens = (
            num_actual_tokens if num_input_tokens == 0
            else num_input_tokens
        )
        max_seq_len = int(seq_lens_cpu.max())
        return cls(query_start_loc=query_start_loc,
                   query_start_loc_cpu=query_start_loc_cpu,
                   seq_lens=seq_lens,
                   seq_lens_cpu=seq_lens_cpu,
                   num_computed_tokens_cpu=num_computed_tokens_cpu,
                   num_reqs=num_reqs,
                   num_actual_tokens=num_actual_tokens,
                   max_query_len=max_query_len,
                   max_seq_len=max_seq_len,
                   block_table_tensor=block_table_tensor,
                   slot_mapping=slot_mapping,
                   seq_start_loc=seq_start_loc,
                   seq_start_loc_cpu=seq_start_loc.to("cpu", non_blocking=True),
                   num_input_tokens=num_input_tokens,
                   infer_mode=infer_mode,
                   num_prefill_query_tokens=num_actual_tokens,
                   num_prefill_kv_tokens=num_actual_tokens)

    def save(self, infer_phase: str):
        csv_path = os.getenv("VLLM_STEP_INPUT_CSV_PATH", None)
        if not csv_path:
            return

        header = [
            "infer_phase", "infer_mode", "num_reqs", "num_actual_tokens",
            "max_query_len", "max_seq_len", "query_start_loc", "seq_lens"
        ]
        data = [
            infer_phase, self.infer_mode, self.num_reqs,
            self.num_actual_tokens, self.max_query_len, self.max_seq_len,
            str(self.query_start_loc_cpu.tolist()),
            str(self.seq_lens_cpu.tolist())
        ]
        data_dict = dict(zip(header, data))
        df_csv = pd.DataFrame(data_dict, index=[0])

        if infer_phase == "RealInfer":
            print(df_csv.to_string())

        try:
            if dir_path := os.path.dirname(csv_path):
                os.makedirs(dir_path, exist_ok=True)
            append = False
            if os.path.isfile(csv_path):
                try:
                    df_old = pd.read_csv(csv_path)
                    append = (df_old.columns.tolist() == header)
                except Exception as e:
                    raise RuntimeError(f"Existing {csv_path} failed to be read and will be overwritten")
            if append:
                df_csv.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df_csv.to_csv(csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Invalid VLLM_STEP_INPUT_CSV_PATH: {csv_path} to dump step inputs, Error: {e}")


def get_common_metadata_from_attn_metadata(
        attn_metadata) -> Union[MLUCommonAttentionMetadata, None]:
    """
    Get MLUCommonAttentionMetadata for MLU-V1 inference.
    Use outside of set_forward_context().
    """
    if attn_metadata is None:
        return

    assert (isinstance(attn_metadata, dict)
            and COMMON_METADATA_STR in attn_metadata), \
        f"MLU-V1 only support type(attn_metadata)=dict, and " + \
        f"{COMMON_METADATA_STR} in attn_metadata. Now, type(attn_metadata)=" + \
        f"{type(attn_metadata)}, or {COMMON_METADATA_STR} not in attn_metadata."
    return attn_metadata[COMMON_METADATA_STR]


def get_common_metadata() -> Union[MLUCommonAttentionMetadata, None]:
    """
    Get MLUCommonAttentionMetadata for MLU-V1 inference.
    Use inside of set_forward_context().
    """
    attn_metadata = get_forward_context().attn_metadata
    return get_common_metadata_from_attn_metadata(attn_metadata)


def unpad_common_attn_metadata(
    common_metadata: MLUCommonAttentionMetadata,
    num_reqs: int,
    num_scheduled_tokens: int,
):
    """
    Unpad MLUCommonAttentionMetadata by given num_reqs and num_scheduled_tokens.
    """
    common_metadata.num_reqs = num_reqs
    common_metadata.num_input_tokens = num_scheduled_tokens
    common_metadata.query_start_loc = common_metadata.query_start_loc[:num_reqs + 1]
    common_metadata.query_start_loc_cpu = common_metadata.query_start_loc_cpu[:num_reqs + 1]
    common_metadata.seq_start_loc = common_metadata.seq_start_loc[:num_reqs + 1]
    common_metadata.seq_lens = common_metadata.seq_lens[:num_reqs]
    common_metadata.seq_lens_cpu = common_metadata.seq_lens_cpu[:num_reqs]
    common_metadata.block_table_tensor = common_metadata.block_table_tensor[:num_reqs]

def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    Returns:
        True if the batch was modified, False otherwise.
    """
    # We now want to reorder the batch into decode → extend → prefill order
    # where:
    #   decode: request with num_scheduled_tokens <= decode_threshold
    #   extend: non-decode request with existing context
    #   prefill: non-decode request with no existing context
    # NOTE for now we loosely use "decode" to mean requests where attention is
    #  likely memory-bound and "prefill" to mean requests where attention is
    #  likely compute-bound,
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: enhence decode mode condition that all prompt tokens are computed.
    '''
    # is_decode = num_scheduled_tokens_np <= decode_threshold
    is_decode = (
        (num_scheduled_tokens_np <= decode_threshold)
        & (num_computed_tokens_np >= input_batch.num_prompt_tokens[:num_reqs])
    )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    is_extend = (~is_decode) & (num_computed_tokens_np > 0)
    is_prefill = (~is_decode) & (num_computed_tokens_np == 0)

    # Desired order: decode → extend → prefill
    req_regions = np.zeros(is_decode.shape, dtype=np.int32)  # 0 = decode by default
    req_regions[is_extend] = 1
    req_regions[is_prefill] = 2

    num_decodes = int(is_decode.sum())
    num_extends = int(is_extend.sum())

    target_regions = np.zeros(num_reqs, dtype=np.int32)
    target_regions[num_decodes : num_decodes + num_extends] = 1
    target_regions[num_decodes + num_extends :] = 2

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True
