# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_mlu.mlu_hijack_utils import MluHijackObject


def split_decodes_and_prefills(self):
    decodes = 0
    prefills = 0
    for i, req_id in enumerate(self.req_ids):
        req_index = self.req_id_to_index.get(req_id)
        num_prompt_tokens = self.num_prompt_tokens[req_index]
        num_computed_tokens = self.num_computed_tokens_cpu[req_index]
        if num_computed_tokens < num_prompt_tokens:
            prefills += 1
        else:
            decodes += 1
    return decodes, prefills


MluHijackObject.apply_hijack(InputBatch,
                             "split_decodes_and_prefills",
                             split_decodes_and_prefills)