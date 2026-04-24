# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import json
import os

import torch

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.forward_context import get_forward_context

def set_attn_compute_dtype_v1(attn_metadata, dtype: torch.dtype):
    '''
    set attn compute_dtype for v1
    '''
    if isinstance(attn_metadata, dict):
        for _, metadata in attn_metadata.items():
            metadata.compute_dtype = dtype
    else:
        metadata.compute_dtype = dtype

def set_attn_compute_dtype(dtype: torch.dtype):
    '''
    set attn compute_dtype.

    TODO: FA may standardize on half precision computation in the future
          set_attn_compute_dtype might be deprecated and removed
    '''
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    set_attn_compute_dtype_v1(attn_metadata, dtype)

def is_tie_word_embeddings(
    model_config: ModelConfig,
    org_tie_word_embeddings: bool
) -> bool:
    '''
    Vllm language model config for multimodal model may have wrong tie_word_embeddings,
    for example, InternVL3.5-38B, InternVL3.5-30B-A3B, etc.

    This function is a WorkAround.
    '''
    from vllm.lora.utils import get_adapter_absolute_path

    if not model_config.is_multimodal_model:
        return org_tie_word_embeddings

    model_path = get_adapter_absolute_path(model_config.model)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return org_tie_word_embeddings

    tie_word_embeddings = org_tie_word_embeddings
    with open(config_path) as f:
        config = json.load(f)
    # first, we find if tie_word_embeddings config is in overall config
    if config.get("tie_word_embeddings") is not None:
        tie_word_embeddings = config["tie_word_embeddings"]
    # then, we find if tie_word_embeddings config is in language model config
    if (config.get("llm_config") is not None
        and config["llm_config"].get("tie_word_embeddings") is not None):
        tie_word_embeddings = config["llm_config"]["tie_word_embeddings"]

    return tie_word_embeddings
