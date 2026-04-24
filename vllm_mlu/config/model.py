# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.config.model import ModelConfig
from vllm.logger import init_logger

from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


def vllm__config__model__ModelConfig__is_embedding_task(self) -> bool:
    return self.runner_type == "pooling"

def vllm__config__model__ModelConfig__get_head_size(self) -> int:
    # TODO remove hard code
    if self.is_deepseek_mla:
        qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
        if self.use_mla:
            return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
        else:
            qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
            if qk_rope_head_dim and qk_nope_head_dim:
                return qk_rope_head_dim + qk_nope_head_dim

    if hasattr(self.hf_text_config, "model_type") and (
        self.hf_text_config.model_type == "zamba2"
    ):
        return self.hf_text_config.attention_head_dim

    if self.is_attention_free:
        return 0

    # NOTE: Some configs may set head_dim=None in the config
    if getattr(self.hf_text_config, "head_dim", None) is not None:
        return self.hf_text_config.head_dim

    # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
    if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
        return self.hf_text_config.hidden_size_per_head

    # FIXME(woosuk): This may not be true for all models.
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: adjust num_heads and num_attention_heads.
    '''
    if hasattr(self.hf_text_config, "num_heads"):
        num_attention_heads = self.hf_text_config.num_heads
    else:
        num_attention_heads = self.hf_text_config.num_attention_heads

    return (self.hf_text_config.hidden_size // num_attention_heads)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


MluHijackObject.apply_hijack(
    ModelConfig,
    "is_embedding_task",
    vllm__config__model__ModelConfig__is_embedding_task,
)
MluHijackObject.apply_hijack(
    ModelConfig,
    ModelConfig.get_head_size,
    vllm__config__model__ModelConfig__get_head_size,
)